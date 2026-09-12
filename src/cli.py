import os
import sys
import sqlite3
import click

# Ensure src is in pythonpath
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.eval_engine import run_evaluation, compare_eval_runs, DB_PATH
from src.reporter import generate_html_report, send_slack_alert

@click.group()
def main():
    """Model Regression Detector - CI/CD LLM Quality Pipeline."""
    pass

@main.command()
@click.option("--prompt", required=True, type=click.Path(exists=True), help="Path to target prompt YAML file.")
@click.option("--baseline", default=None, type=click.Path(exists=True), help="Path to baseline prompt YAML file for diff comparison.")
@click.option("--dataset", default="data/golden_dataset.json", type=click.Path(exists=True), help="Path to golden dataset JSON.")
@click.option("--mock/--live", default=True, help="Force free mock mode (default: free mock).")
def run(prompt, baseline, dataset, mock):
    """Run evaluation on a prompt version and detect regressions."""
    click.echo("[START] Running Model Regression Evaluation...")
    click.echo(f"   Target Prompt: {prompt}")
    click.echo(f"   Execution Mode: {'FREE MOCK' if mock else 'LIVE OPENAI API'}")

    current_report = run_evaluation(prompt, dataset, use_mock=mock)
    click.echo(f"\n[RESULTS] Prompt Version: {current_report.prompt_version}")
    click.echo(f"   Accuracy: {current_report.accuracy_pct}% ({current_report.passed_cases}/{current_report.total_cases} passed)")
    click.echo(f"   Avg Latency: {current_report.avg_latency_ms} ms")
    click.echo(f"   Cost: ${current_report.estimated_cost_usd:.6f}")

    comparison = None
    if baseline:
        click.echo(f"\n[COMPARE] Comparing against baseline: {baseline}...")
        baseline_report = run_evaluation(baseline, dataset, use_mock=mock)
        comparison = compare_eval_runs(current_report, baseline_report)

        click.echo(f"\n--------------------------------------------------")
        click.echo(f"STATUS: [{comparison.status}]")
        click.echo(f"Summary: {comparison.summary_message}")
        if comparison.regressions:
            click.echo(f"\n[REGRESSIONS] Regressed Cases ({len(comparison.regressions)}):")
            for reg in comparison.regressions:
                click.echo(f"   - [{reg.test_id}] Exp: {reg.expected_category} | Pred: {reg.predicted_category} | Difficulty: {reg.difficulty}")
        click.echo(f"--------------------------------------------------")

        send_slack_alert(comparison)

    report_path = generate_html_report(current_report, comparison)
    click.echo(f"\n[REPORT] HTML Report generated: file:///{os.path.abspath(report_path)}")

    # Exit code strategy for CI/CD
    if comparison and comparison.status == "CRITICAL":
        click.echo("[FAIL] CI/CD Pipeline FAILED due to critical quality regression.")
        sys.exit(1)
    else:
        click.echo("[SUCCESS] Pipeline PASSED.")
        sys.exit(0)

@main.command()
def history():
    """List historical evaluation runs from SQLite database."""
    if not os.path.exists(DB_PATH):
        click.echo("No eval history found.")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT run_id, timestamp, prompt_version, accuracy_pct, passed_cases, total_cases FROM eval_runs ORDER BY timestamp DESC LIMIT 10")
    rows = cursor.fetchall()
    conn.close()

    click.echo("\n[HISTORY] Evaluation History:")
    click.echo(f"{'Timestamp':<22} {'Version':<18} {'Accuracy':<10} {'Passed/Total'}")
    click.echo("-" * 65)
    for r in rows:
        click.echo(f"{r[1][:19]:<22} {r[2]:<18} {r[3]:<10}% {r[4]}/{r[5]}")

if __name__ == "__main__":
    main()
