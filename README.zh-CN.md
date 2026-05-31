# finance-replication

用于复现金融学实证论文的 CodeWhale skill。

[English](README.md)

## 覆盖范围

- **截面资产定价** — Fama-MacBeth、组合排序、因子模型
- **公司金融** — 面板固定效应、双重差分 (DID)、事件研究
- **机器学习资产定价** — 神经网络、梯度提升树、随机森林
- **波动率建模** — GARCH / EGARCH

## 快速安装

```bash
# 在 CodeWhale 中运行：
/skill install github:Echo-Tian1/finance-replication
```

或手动克隆到 `~/.codewhale/skills/finance-replication/`。

## 本地参考项目

此 skill 引用以下开源项目：

- [OpenSourceAP/CrossSection](https://github.com/OpenSourceAP/CrossSection) — Chen & Zimmermann (2020)，200+ 个资产定价异象信号的复现代码
- [Kratos0806/GKX](https://github.com/Kratos0806/GKX) — Gu, Kelly & Xiu (2020) 机器学习资产定价的完整 Python 复现

## 配套 Python 包

```bash
pip install statsmodels linearmodels pyfixest arch yfinance pandas-datareader scikit-learn xgboost lightgbm pyportfolioopt famafrench wrds seaborn formulaic
```

## 许可

MIT License
