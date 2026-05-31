# finance-replication

CodeWhale skill for reproducing empirical finance papers.

[中文](README.zh-CN.md)

## What it covers

- **Cross-sectional asset pricing** — Fama-MacBeth, portfolio sorts, factor models
- **Corporate finance** — panel fixed effects, DID, event studies
- **ML in asset pricing** — neural networks, gradient boosting, random forests
- **Volatility modeling** — GARCH / EGARCH

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
pip install statsmodels linearmodels pyfixest arch yfinance pandas-datareader scikit-learn xgboost lightgbm pyportfolioopt famafrench wrds seaborn formulaic
```

## License

MIT
