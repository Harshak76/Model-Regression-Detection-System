import os
import json
import urllib.request
from typing import Optional
from jinja2 import Template
from src.eval_engine import EvalRunReport, RunComparison

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Model Regression Report - {{ current.prompt_version }}</title>
    <style>
        :root {
            --bg: #0f172a;
            --card-bg: #1e293b;
            --text: #f8fafc;
            --text-muted: #94a3b8;
            --pass: #22c55e;
            --warn: #eab308;
            --fail: #ef4444;
            --accent: #38bdf8;
            --border: #334155;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background-color: var(--bg);
            color: var(--text);
            margin: 0;
            padding: 2rem;
        }
        .container {
            max-width: 1100px;
            margin: 0 auto;
        }
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border);
            padding-bottom: 1rem;
            margin-bottom: 2rem;
        }
        .status-badge {
            padding: 0.5rem 1.25rem;
            border-radius: 9999px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .status-PASS { background-color: rgba(34, 197, 94, 0.2); color: var(--pass); border: 1px solid var(--pass); }
        .status-WARN { background-color: rgba(234, 179, 8, 0.2); color: var(--warn); border: 1px solid var(--warn); }
        .status-CRITICAL { background-color: rgba(239, 68, 68, 0.2); color: var(--fail); border: 1px solid var(--fail); }

        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }
        .metric-card {
            background-color: var(--card-bg);
            padding: 1.5rem;
            border-radius: 12px;
            border: 1px solid var(--border);
        }
        .metric-val {
            font-size: 2rem;
            font-weight: 700;
            margin-top: 0.5rem;
        }
        .delta {
            font-size: 0.9rem;
            margin-left: 0.5rem;
        }
        .delta-positive { color: var(--pass); }
        .delta-negative { color: var(--fail); }

        table {
            width: 100%;
            border-collapse: collapse;
            background-color: var(--card-bg);
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid var(--border);
            margin-bottom: 2rem;
        }
        th, td {
            padding: 1rem;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }
        th {
            background-color: #0f172a;
            color: var(--text-muted);
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.75rem;
            letter-spacing: 0.05em;
        }
        tr:last-child td { border-bottom: none; }
        .badge {
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
        }
        .badge-pass { background: rgba(34, 197, 94, 0.2); color: var(--pass); }
        .badge-fail { background: rgba(239, 68, 68, 0.2); color: var(--fail); }

        .diff-section {
            background: var(--card-bg);
            padding: 1.5rem;
            border-radius: 12px;
            border: 1px solid var(--border);
            margin-bottom: 2rem;
        }
        .diff-section h3 { margin-top: 0; color: var(--accent); }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div>
                <h1>Model Regression Report</h1>
                <p style="color: var(--text-muted); margin: 0;">Target Prompt: <strong>{{ current.prompt_version }}</strong> | Baseline: <strong>{{ comparison.baseline_version if comparison else 'N/A' }}</strong></p>
            </div>
            <div class="status-badge status-{{ comparison.status if comparison else 'PASS' }}">
                {{ comparison.status if comparison else 'EVAL COMPLETED' }}
            </div>
        </div>

        {% if comparison %}
        <div style="background: rgba(56, 189, 248, 0.1); border-left: 4px solid var(--accent); padding: 1rem; border-radius: 4px; margin-bottom: 2rem;">
            <strong>Summary:</strong> {{ comparison.summary_message }}
        </div>
        {% endif %}

        <div class="metrics-grid">
            <div class="metric-card">
                <div style="color: var(--text-muted); font-size: 0.85rem;">Accuracy Score</div>
                <div class="metric-val">
                    {{ current.accuracy_pct }}%
                    {% if comparison %}
                    <span class="delta {{ 'delta-positive' if comparison.accuracy_delta_pct >= 0 else 'delta-negative' }}">
                        ({{ comparison.accuracy_delta_pct }}%)
                    </span>
                    {% endif %}
                </div>
            </div>
            <div class="metric-card">
                <div style="color: var(--text-muted); font-size: 0.85rem;">Passed Cases</div>
                <div class="metric-val">{{ current.passed_cases }} / {{ current.total_cases }}</div>
            </div>
            <div class="metric-card">
                <div style="color: var(--text-muted); font-size: 0.85rem;">Avg Latency</div>
                <div class="metric-val">{{ current.avg_latency_ms }} ms</div>
            </div>
            <div class="metric-card">
                <div style="color: var(--text-muted); font-size: 0.85rem;">Total Cost (Free/Mock)</div>
                <div class="metric-val">$0.00</div>
            </div>
        </div>

        {% if comparison and comparison.regressions %}
        <div class="diff-section">
            <h3 style="color: var(--fail);">⚠️ Regressed Test Cases ({{ comparison.regressions|length }})</h3>
            <p style="color: var(--text-muted); font-size: 0.9rem;">These test cases passed in baseline ({{ comparison.baseline_version }}) but failed in current ({{ comparison.current_version }}):</p>
            <table>
                <thead>
                    <tr>
                        <th>ID</th>
                        <th>Input Email</th>
                        <th>Expected</th>
                        <th>Predicted</th>
                        <th>Difficulty</th>
                    </tr>
                </thead>
                <tbody>
                    {% for reg in comparison.regressions %}
                    <tr>
                        <td><strong>{{ reg.test_id }}</strong></td>
                        <td style="max-width: 350px;">{{ reg.input_text }}</td>
                        <td><span class="badge badge-pass">{{ reg.expected_category }}</span></td>
                        <td><span class="badge badge-fail">{{ reg.predicted_category }}</span></td>
                        <td>{{ reg.difficulty }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
        {% endif %}

        <div class="diff-section">
            <h3>Full Test Suite Results</h3>
            <table>
                <thead>
                    <tr>
                        <th>ID</th>
                        <th>Input Email</th>
                        <th>Expected Category</th>
                        <th>Predicted Category</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
                    {% for r in current.results %}
                    <tr>
                        <td><strong>{{ r.test_id }}</strong></td>
                        <td style="max-width: 400px;">{{ r.input_text }}</td>
                        <td>{{ r.expected_category }}</td>
                        <td>{{ r.predicted_category }}</td>
                        <td>
                            {% if r.passed %}
                            <span class="badge badge-pass">PASS</span>
                            {% else %}
                            <span class="badge badge-fail">FAIL</span>
                            {% endif %}
                        </td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
    </div>
</body>
</html>
"""

def generate_html_report(current_report: EvalRunReport, comparison: Optional[RunComparison] = None, output_dir: str = "reports") -> str:
    """Generate HTML report file."""
    os.makedirs(output_dir, exist_ok=True)
    template = Template(HTML_TEMPLATE)
    html_content = template.render(current=current_report, comparison=comparison)

    filepath = os.path.join(output_dir, f"report_{current_report.run_id}.html")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html_content)

    return filepath

def send_slack_alert(comparison: RunComparison, webhook_url: Optional[str] = None) -> bool:
    """Send structured Slack notification if webhook URL is configured."""
    url = webhook_url or os.getenv("SLACK_WEBHOOK_URL")
    if not url or "YOUR/WEBHOOK" in url:
        print("[INFO] Slack webhook URL not configured. Skipping Slack alert.")
        return False

    status_emoji = "🟢" if comparison.status == "PASS" else ("🟡" if comparison.status == "WARN" else "🔴")
    payload = {
        "blocks": [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": f"{status_emoji} AI Model Eval: {comparison.status}"}
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Version:* `{comparison.current_version}`"},
                    {"type": "mrkdwn", "text": f"*Baseline:* `{comparison.baseline_version}`"},
                    {"type": "mrkdwn", "text": f"*Accuracy Delta:* `{comparison.accuracy_delta_pct:+.1f}%`"},
                    {"type": "mrkdwn", "text": f"*Regressions:* `{len(comparison.regressions)}`"}
                ]
            },
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"*Details:* {comparison.summary_message}"}
            }
        ]
    }

    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode('utf-8'),
            headers={'Content-Type': 'application/json'}
        )
        with urllib.request.urlopen(req) as resp:
            return resp.status == 200
    except Exception as e:
        print(f"[WARN] Failed to send Slack alert: {e}")
        return False
