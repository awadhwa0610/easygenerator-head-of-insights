# Easygenerator — Head of Insights Case Study

A product analytics case study analyzing **2.9 million product events** from Easygenerator's platform (May 2025), covering user engagement, feature adoption, and a data strategy for the upcoming Multi-Language Courses (MLC) launch.

## 🎤 View the Presentation

Open [`presentation.html`](./presentation.html) in any browser.

- **Arrow keys** to navigate slides
- **F** for fullscreen
- 22 slides covering Part 1 (EDA) and Part 2 (MLC Measurement Strategy)

## 📁 What's in This Repo

| File | Purpose |
|------|---------|
| `presentation.html` | 22-slide case study deck |
| `charts/` | 15 charts generated from the event data |
| `analysis.py` | Core EDA script — engagement, segmentation, feature adoption, funnel |
| `generate_charts.py` | Chart generation script (Matplotlib) |
| `authors.csv` | 63,279 author records (plan, country, org, registration) |

## 🔍 Analysis Summary

### Part 1: Exploratory Data Analysis
- **Platform Scale:** 63K authors, 14,790 MAU, 2.9M events, 44K courses
- **User Segmentation:** 70% light users (1-3 days/month) — the "missing middle"
- **Feature Adoption:** 58% collaboration, 31% AI, 24% SCORM, 5.5% translation
- **Engagement by Plan:** Team (326 events/user) >> Enterprise (189) >> Free (61)
- **Key Finding:** Trial users behave identically to Free — trial isn't demonstrating paid value

### Part 2: MLC Measurement Strategy
- Pre-launch baselines from 815 existing translation users
- 28-event tracking taxonomy across 3 MLC beta milestones
- Success KPIs: 40% adoption, -30% duplicate courses, >95% SCORM success
- 3 experiment designs including propensity score matching for churn analysis

## 🛠️ Tools Used

- **Python 3.9** — core analysis language
- **Pandas** — data loading, joining, aggregation across 2.9M rows
- **Matplotlib + Seaborn** — visualization and chart generation
- **Shell/zsh** — data profiling and rapid inspection

## 📊 Regenerating Charts

Charts were pre-generated and committed. To regenerate:

```bash
# Requires events.csv (300MB, excluded from repo due to size)
python3 generate_charts.py
```

The event data (2.9M rows, 300MB) is excluded from this repo due to file size limits. Charts were pre-generated and committed directly.
