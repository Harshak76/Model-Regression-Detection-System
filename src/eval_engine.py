import os
import json
import sqlite3
from datetime import datetime
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from src.classifier import classify_email, load_prompt_config, ClassificationResult

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "eval_history.db")

class TestCaseResult(BaseModel):
    test_id: str
    input_text: str
    expected_category: str
    predicted_category: str
    passed: bool
    summary: str
    relevance_score: float
    difficulty: str
    latency_ms: float

class EvalRunReport(BaseModel):
    run_id: str
    timestamp: str
    prompt_version: str
    total_cases: int
    passed_cases: int
    accuracy_pct: float
    avg_latency_ms: float
    total_tokens: int
    estimated_cost_usd: float
    category_breakdown: Dict[str, Dict[str, int]]
    results: List[TestCaseResult]

class RunComparison(BaseModel):
    baseline_run_id: str
    baseline_version: str
    current_run_id: str
    current_version: str
    accuracy_delta_pct: float
    regressions: List[TestCaseResult]
    improvements: List[TestCaseResult]
    status: str # PASS, WARN, CRITICAL
    slow_drift_detected: bool
    rolling_avg_accuracy: float
    summary_message: str

def init_db():
    """Initialize SQLite database for tracking evaluation history."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS eval_runs (
            run_id TEXT PRIMARY KEY,
            timestamp TEXT,
            prompt_version TEXT,
            total_cases INTEGER,
            passed_cases INTEGER,
            accuracy_pct REAL,
            avg_latency_ms REAL,
            total_tokens INTEGER,
            estimated_cost_usd REAL,
            raw_json TEXT
        )
    """)
    conn.commit()
    conn.close()

def compute_relevance_score(summary: str, expected_keywords: List[str]) -> float:
    """Calculate keyword coverage relevance score for the summary."""
    if not expected_keywords:
        return 1.0
    summary_lower = summary.lower()
    matches = sum(1 for kw in expected_keywords if kw.lower() in summary_lower)
    return round(matches / len(expected_keywords), 2)

def run_evaluation(prompt_path: str, dataset_path: str, use_mock: Optional[bool] = None) -> EvalRunReport:
    """Run full evaluation suite for a given prompt against the golden dataset."""
    init_db()
    prompt_config = load_prompt_config(prompt_path)
    version = prompt_config.get("version", "unknown")

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    results: List[TestCaseResult] = []
    category_stats: Dict[str, Dict[str, int]] = {}
    total_latency = 0.0
    total_tokens = 0

    for item in dataset:
        exp_cat = item["expected_category"]
        if exp_cat not in category_stats:
            category_stats[exp_cat] = {"total": 0, "passed": 0}
        category_stats[exp_cat]["total"] += 1

        cls_res: ClassificationResult = classify_email(item["input"], prompt_config, use_mock=use_mock)
        passed = (cls_res.category == exp_cat)

        if passed:
            category_stats[exp_cat]["passed"] += 1

        relevance = compute_relevance_score(cls_res.summary, item.get("expected_keywords", []))
        total_latency += cls_res.latency_ms
        total_tokens += cls_res.token_usage.get("total_tokens", 0)

        results.append(TestCaseResult(
            test_id=item["id"],
            input_text=item["input"],
            expected_category=exp_cat,
            predicted_category=cls_res.category,
            passed=passed,
            summary=cls_res.summary,
            relevance_score=relevance,
            difficulty=item.get("difficulty", "medium"),
            latency_ms=cls_res.latency_ms
        ))

    total_cases = len(dataset)
    passed_cases = sum(1 for r in results if r.passed)
    accuracy_pct = round((passed_cases / total_cases) * 100, 2) if total_cases > 0 else 0.0
    avg_latency = round(total_latency / total_cases, 2) if total_cases > 0 else 0.0
    # gpt-4o-mini price approx $0.15/1M input, $0.60/1M output -> ~$0.0003 per 1k tokens
    estimated_cost = round((total_tokens / 1000.0) * 0.0003, 6)

    import uuid
    run_id = f"run_{version}_{int(datetime.now().timestamp() * 1000)}_{uuid.uuid4().hex[:6]}"
    timestamp = datetime.now().isoformat()

    report = EvalRunReport(
        run_id=run_id,
        timestamp=timestamp,
        prompt_version=version,
        total_cases=total_cases,
        passed_cases=passed_cases,
        accuracy_pct=accuracy_pct,
        avg_latency_ms=avg_latency,
        total_tokens=total_tokens,
        estimated_cost_usd=estimated_cost,
        category_breakdown=category_stats,
        results=results
    )

    # Save to SQLite
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO eval_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (run_id, timestamp, version, total_cases, passed_cases, accuracy_pct, avg_latency, total_tokens, estimated_cost, report.model_dump_json()))
    conn.commit()
    conn.close()

    return report

def compare_eval_runs(current_report: EvalRunReport, baseline_report: EvalRunReport, warn_thresh: float = 3.0, crit_thresh: float = 8.0) -> RunComparison:
    """Compare a current evaluation run against a baseline run and compute drift & regressions."""
    baseline_map = {r.test_id: r for r in baseline_report.results}
    current_map = {r.test_id: r for r in current_report.results}

    regressions: List[TestCaseResult] = []
    improvements: List[TestCaseResult] = []

    for test_id, cur_res in current_map.items():
        base_res = baseline_map.get(test_id)
        if base_res:
            if base_res.passed and not cur_res.passed:
                regressions.append(cur_res)
            elif not base_res.passed and cur_res.passed:
                improvements.append(cur_res)

    accuracy_delta = round(current_report.accuracy_pct - baseline_report.accuracy_pct, 2)

    # Check rolling average history (last 7 runs) for slow drift
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT accuracy_pct FROM eval_runs ORDER BY timestamp DESC LIMIT 7")
    history = [row[0] for row in cursor.fetchall()]
    conn.close()

    rolling_avg = round(sum(history) / len(history), 2) if history else current_report.accuracy_pct
    slow_drift = (rolling_avg < (baseline_report.accuracy_pct - warn_thresh))

    # Assign status
    drop = -accuracy_delta
    if drop >= crit_thresh:
        status = "CRITICAL"
        msg = f"CRITICAL REGRESSION DETECTED: Accuracy dropped by {drop:.1f}% (Threshold: {crit_thresh}%). {len(regressions)} test case(s) regressed."
    elif drop >= warn_thresh or slow_drift:
        status = "WARN"
        msg = f"WARNING: Accuracy dropped by {drop:.1f}% or slow drift detected (Rolling avg: {rolling_avg}%). {len(regressions)} regression(s)."
    else:
        status = "PASS"
        msg = f"PASS: Prompt version {current_report.prompt_version} meets quality standards. Accuracy: {current_report.accuracy_pct}% (Delta: {accuracy_delta:+.1f}%)."

    return RunComparison(
        baseline_run_id=baseline_report.run_id,
        baseline_version=baseline_report.prompt_version,
        current_run_id=current_report.run_id,
        current_version=current_report.prompt_version,
        accuracy_delta_pct=accuracy_delta,
        regressions=regressions,
        improvements=improvements,
        status=status,
        slow_drift_detected=slow_drift,
        rolling_avg_accuracy=rolling_avg,
        summary_message=msg
    )
