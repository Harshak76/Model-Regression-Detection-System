# Project 1: Model Regression Detection System

A production-grade CI/CD pipeline that continuously tests LLM-powered features against a versioned golden dataset, detects quality regressions across prompt/model versions, generates HTML diff dashboards, and triggers Slack alerts before bad prompts reach users.

## System Architecture

```
                       +-----------------------+
                       |  Golden Dataset JSON  |
                       +-----------+-----------+
                                   |
                                   v
+------------------+     +---------+----------+     +-----------------------+
|  Prompt Version  | --> |  Evaluation Engine | --> |  SQLite History DB   |
|   (v1.0 / v1.1)  |     +---------+----------+     +-----------------------+
+------------------+               |
                                   v
                         +---------+----------+
                         |  Diff & Drift      |
                         |  Comparator        |
                         +---------+----------+
                                   |
                     +-------------+-------------+
                     |                           |
                     v                           v
          +----------+----------+     +----------+----------+
          | HTML Diff Dashboard |     |  Slack / Webhook    |
          |       Report        |     |      Notifier       |
          +---------------------+     +---------------------+
```

## Features

- **100% Free / Zero API Cost Default**: Built-in deterministic mock classifier allows offline development and zero-cost testing. Live OpenAI mode available by passing `--live`.
- **Golden Dataset**: Hand-curated 15+ test cases covering standard queries and edge cases (typos, sarcasm, mixed-category complaints, French language).
- **Multi-Metric Scoring**: Accuracy %, latency tracking, keyword relevance, token cost estimation, and category-level breakdown.
- **Diff & Drift Detection**: Compares candidate prompt runs against baseline prompts, flags regressed test cases (Pass -> Fail), and monitors rolling 7-run averages to catch slow drift.
- **Automated CI/CD**: Includes GitHub Action workflow to fail PRs when critical quality drops (>8%) occur.
- **HTML Dashboards**: Generates interactive dark-mode HTML reports with old vs. new output side-by-side diff tables.

## Quickstart

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Baseline Evaluation (Free Mock Mode)
```bash
python src/cli.py run --prompt prompts/v1.0.yaml
```

### 3. Compare Regressed Prompt vs Baseline Prompt
```bash
python src/cli.py run --prompt prompts/v1.1_regressed.yaml --baseline prompts/v1.0.yaml
```

### 4. Run Pytest Suite
```bash
pytest tests/
```

### 5. View Evaluation History
```bash
python src/cli.py history
```
