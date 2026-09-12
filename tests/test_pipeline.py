import os
import json
import pytest
from src.classifier import load_prompt_config, classify_email, ClassificationResult
from src.eval_engine import run_evaluation, compare_eval_runs
from src.reporter import generate_html_report

PROMPT_V1_PATH = "prompts/v1.0.yaml"
PROMPT_V1_1_REGRESSED_PATH = "prompts/v1.1_regressed.yaml"
DATASET_PATH = "data/golden_dataset.json"

def test_load_prompt_config():
    config = load_prompt_config(PROMPT_V1_PATH)
    assert config["version"] == "1.0"
    assert "system_prompt" in config
    assert "temperature" in config

def test_golden_dataset_schema():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    assert len(dataset) >= 15
    for case in dataset:
        assert "id" in case
        assert "input" in case
        assert "expected_category" in case
        assert case["expected_category"] in ["billing", "technical", "account", "general"]

def test_classifier_mock():
    config = load_prompt_config(PROMPT_V1_PATH)
    res = classify_email("I was charged twice on my invoice", config, use_mock=True)
    assert isinstance(res, ClassificationResult)
    assert res.category == "billing"
    assert res.latency_ms > 0

def test_eval_engine_and_regression_detection():
    # Run evaluation on baseline v1.0
    baseline_report = run_evaluation(PROMPT_V1_PATH, DATASET_PATH, use_mock=True)
    assert baseline_report.total_cases >= 15
    assert baseline_report.accuracy_pct >= 90.0

    # Run evaluation on regressed v1.1
    regressed_report = run_evaluation(PROMPT_V1_1_REGRESSED_PATH, DATASET_PATH, use_mock=True)
    assert regressed_report.accuracy_pct < baseline_report.accuracy_pct

    # Compare runs
    comparison = compare_eval_runs(regressed_report, baseline_report)
    assert comparison.status in ["WARN", "CRITICAL"]
    assert len(comparison.regressions) > 0

def test_html_report_generation(tmp_path):
    report = run_evaluation(PROMPT_V1_PATH, DATASET_PATH, use_mock=True)
    html_file = generate_html_report(report, output_dir=str(tmp_path))
    assert os.path.exists(html_file)
    with open(html_file, "r", encoding="utf-8") as f:
        content = f.read()
    assert "Model Regression Report" in content
    assert "1.0" in content
