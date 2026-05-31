# finance-replication

CodeWhale skill for reproducing empirical finance papers. From data cleaning to academic-grade charts — one skill, end-to-end.

[中文](README.zh-CN.md)

## What it covers

- **Data preprocessing** — CRSP/Compustat cleaning, winsorization, fiscal-year alignment, CRSP-Compustat merge (6-month lag)
- **Cross-sectional asset pricing** — Fama-MacBeth, portfolio sorts, factor models
- **Corporate finance** — panel fixed effects, DID / staggered DID, event studies
- **ML in asset pricing** — neural networks, gradient boosting, random forests
- **Volatility modeling** — GARCH / EGARCH
- **Academic visualization** — publication-grade charts (CAR plots, portfolio bar charts, FM coefficient time series, correlation heatmaps) and regression tables with significance stars

## Quick install

```bash
# In CodeWhale:
/skill install github:Echo-Tian1/finance-replication
```

Or manually clone into `~/.codewhale/skills/finance-replication/`.

## Local reference projects

This skill references:
- [OpenSourceAP/CrossSection](https://github.com/OpenSourceAP/CrossSection) — Chen & Zimmermann (2020), 200+ anomaly signals
- [Kratos0806/GKX](https://github.com/Kratos0806/GKX) — Gu, Kelly & Xiu (2020) ML asset pricing

## Companion Python packages

```bash
pip install statsmodels linearmodels pyfixest arch yfinance pandas-datareader scikit-learn xgboost lightgbm pyportfolioopt famafrench wrds seaborn matplotlib formulaic
```

## License

MIT