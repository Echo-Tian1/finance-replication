---
name: finance-replication
description: >
  复现金融学实证论文的标准化工具包。涵盖数据预处理（CRSP/Compustat 清洗、
  缩尾、合并）、资产定价（Fama-MacBeth、组合排序、因子模型）、公司金融
  （面板固定效应、DID、事件研究）、机器学习资产定价等核心实证范式，以及
  学术发表级可视化模板。提供 Python 管线模板、关键库速查和本地参考项目索引。
  TRIGGER: 论文复现、实证资产定价、Fama-MacBeth、因子模型、事件研究、
  组合排序、DID、面板回归、金融异象、收益率预测、波动率建模。
---

# 金融实证论文复现工具包

## 快速启动

核对环境后，先走「数据预处理」管线清洗数据，再从「标准管线」中选择与目标论文匹配的范式模板开始。

## 1. Python 环境速查

```python
# ── 计量核心 ──
import statsmodels.api as sm          # OLS, Logit, GLM
from statsmodels.regression.rolling import RollingOLS
from linearmodels.panel import PanelOLS, RandomEffects  # 面板 + 聚类标准误
from linearmodels.iv import IV2SLS, IVGMM               # 工具变量
import pyfixest as pf                                      # 高维固定效应（比 linearmodels 快 100×）

# ── 资产定价专用 ──
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb

# ── 时间序列 / 波动率 ──
from arch import arch_model            # GARCH/EGARCH

# ── 数据源 ──
import yfinance as yf
from pandas_datareader import data as pdr
# WRDS 接口: import wrds  # 需要配置 WRDS 账号

# ── 数据预处理 ──
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import StandardScaler

# ── 可视化 ──
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.ticker import FuncFormatter, PercentFormatter
import seaborn as sns
```

## 2. 数据预处理管线（新增）

> 金融数据清洗通常占复现总时间的 60-80%。以下管线覆盖最常见的 CRSP / Compustat 处理流程，直接复用代码模板即可。

### 2.1 CRSP 股票数据清洗

```python
def clean_crsp_monthly(df):
    """
    CRSP 月度股票数据标准清洗。
    
    输入：CRSP Monthly Stock File (msf) 导出的 DataFrame，
          至少包含 PERMNO, date, RET, PRC, SHROUT, SHRCD, EXCHCD
    输出：清洗后的月度收益数据
    """
    # ── 1. 样本筛选 ──
    # 普通股（share code 10/11），剔除 ADR/REIT/封闭式基金
    df = df[df["SHRCD"].isin([10, 11])]
    
    # 主要交易所（NYSE=1, AMEX=2, NASDAQ=3）
    df = df[df["EXCHCD"].isin([1, 2, 3])]
    
    # ── 2. 价格筛选（剔除低价股，避免微观结构噪音）──
    df["PRC"] = df["PRC"].abs()  # CRSP 负价格表示 bid-ask 平均
    df = df[df["PRC"] >= 5]       # 剔除 < $5 的股票（可根据论文要求调整）
    
    # ── 3. 计算市值 ──
    df["me"] = df["PRC"] * df["SHROUT"] / 1000  # 单位：千美元
    
    # ── 4. 退市收益调整 ──
    # CRSP RET 不含退市收益，需合并 delisting return
    # 通常从 CRSP delisting file (dse) 获取 DLRET
    # 合并逻辑: RET_adj = (1+RET)*(1+DLRET) - 1，若无 DLRET 则用 -0.3（NYSE/AMEX）或 -0.55（NASDAQ）
    
    # ── 5. 缺失值处理 ──
    df = df.dropna(subset=["RET"])
    
    return df
```

### 2.2 Compustat 会计数据清洗

```python
def clean_compustat(df):
    """
    Compustat 年度会计数据标准清洗。
    
    输入：Compustat Fundamentals Annual (funda)，
          至少包含 GVKEY, fyear, AT, LT, SEQ, ...
    输出：清洗后的年度财务数据
    """
    # ── 1. 格式标准化（统一为 FYEAR）──
    # Compustat 数据可能使用 datadate / fyear 作为年份标识
    df["fyear"] = df["datadate"].dt.year
    
    # ── 2. 剔除金融/公用事业（SIC 6000-6999, 4900-4999）──
    # 根据论文要求决定是否剔除
    df = df[~(df["SIC"].between(6000, 6999))]
    df = df[~(df["SIC"].between(4900, 4999))]
    
    # ── 3. 总资产 > 0 ──
    df = df[df["AT"] > 0]
    
    # ── 4. 普通股账面价值 > 0 ──
    df["BE"] = df["SEQ"] + df["TXDITC"] - df["PSTKRV"] - df["PSTKL"]
    df["BE"] = df["BE"].clip(lower=0)  # 负账面值设为 0
    df = df[df["BE"] > 0]
    
    # ── 5. 变量缺失处理 ──
    # 投资 = CAPX 缺失时设 0
    df["CAPX"] = df["CAPX"].fillna(0)
    
    return df
```

### 2.3 缩尾（Winsorization）

```python
def winsorize_panel(df, columns, pct=0.01, by_period="date"):
    """
    面板数据分组缩尾（金融标准：1%/99% 分位数）。
    
    df: DataFrame with MultiIndex or columns
    columns: 需要缩尾的变量列表
    pct: 缩尾比例，默认 1%
    by_period: 按哪个维度分组缩尾（"date" 按年/月，"none" 不分组）
    """
    df = df.copy()
    if by_period == "date":
        for col in columns:
            df[col] = df.groupby("date")[col].transform(
                lambda x: winsorize(x, limits=(pct, pct))
            )
    else:
        for col in columns:
            lower = df[col].quantile(pct)
            upper = df[col].quantile(1 - pct)
            df[col] = df[col].clip(lower, upper)
    return df
```

### 2.4 CRSP-Compustat 标准合并

```python
def merge_crsp_compustat(crsp, comp, lag_months=6):
    """
    CRSP 月度数据与 Compustat 年度数据合并。
    
    标准做法：t 年 7 月到 t+1 年 6 月使用 t-1 财年末的会计数据（6 个月滞后）。
    
    输入：
      crsp: 清洗后的 CRSP 月度数据（列: PERMNO, date, RET, me, ...）
      comp: 清洗后的 Compustat 年度数据（列: GVKEY, fyear, AT, BE, ...）
      lag_months: 滞后月数，默认 6
    输出：合并后的月度面板数据
    """
    # PERMNO-GVKEY 链接表（从 CRSP/Compustat Merged - CCM 获取）
    # 实际使用时需先加载 ccmxpf_linktable
    # 这里展示合并逻辑：
    
    # 1. 会计数据对齐：fyear 对应的窗口为 [fyear+1 年 7月, fyear+2 年 6月]
    comp["merge_fyear"] = comp["fyear"] + 1  # 滞后 6 个月 ⇒ 从次年 7 月开始使用
    
    # 2. 按 PERMNO-GVKEY 链接表 merge
    merged = crsp.merge(link_table, on="PERMNO", how="inner")
    merged = merged.merge(comp, on=["GVKEY", "merge_fyear"], how="left")
    
    return merged
```

### 2.5 数据质量检查清单

复现前必须通过的检查：

| 检查项 | 方法 | 预期 |
|--------|------|------|
| 样本量一致性 | `df.groupby("year").size().plot()` | 与论文 Table 1 基本吻合 |
| 变量均值/标准差 | `df.describe()` | 与论文 Summary Stats 表对比 |
| 缺失值比例 | `df.isnull().mean()` | 每个变量 < 20%，关键变量 < 5% |
| 收益率分布 | `df["RET"].hist(bins=100)` | 近似正态，无极端异常值 |
| 面板平衡性 | `df.groupby("permno").size().describe()` | 了解每个股票的时间跨度 |

---

## 3. 学术可视化模块（新增）

> 以下模板可直接生成符合 Journal of Finance / Review of Financial Studies 发表标准的图表。所有配色方案均提供彩色版和灰度安全版。

### 3.1 matplotlib 全局配置（学术发表级）

```python
def set_academic_style(style="jf"):
    """
    一键设置学术发表级的 matplotlib 样式。
    
    style="jf"  → 彩色（JF 风格）
    style="jfe" → 灰度安全（JFE 风格）
    """
    if style == "jf":
        COLOR = "#1f77b4"  # 主色
        PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    else:
        COLOR = "#333333"
        PALETTE = ["#333333", "#888888", "#BBBBBB", "#666666", "#999999"]
    
    plt.rcParams.update({
        "figure.dpi": 300,
        "figure.figsize": (8, 5),
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "lines.linewidth": 1.5,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })
    return PALETTE
```

### 3.2 标准图表示例

#### 累积异常收益（CAR）图 — 事件研究

```python
def plot_car_with_ci(car_mean, car_ci_lower, car_ci_upper, event_window,
                     title="Cumulative Abnormal Returns", save_path=None):
    """
    事件研究的 CAR 图（含 95% 置信区间）。
    
    car_mean: array, 每个事件窗口日的平均 CAR
    car_ci_lower / car_ci_upper: 置信区间上下界
    event_window: list, 事件窗口日（如 range(-5, 6)）
    """
    PALETTE = set_academic_style("jf")
    
    fig, ax = plt.subplots()
    t = list(event_window)
    
    ax.fill_between(t, car_ci_lower, car_ci_upper,
                    alpha=0.15, color=PALETTE[0], label="95% CI")
    ax.plot(t, car_mean, color=PALETTE[0], linewidth=2)
    ax.axhline(y=0, color="black", linewidth=0.5, linestyle="--")
    ax.axvline(x=0, color="red", linewidth=0.5, linestyle="--", alpha=0.5)
    
    ax.set_xlabel("Event Day")
    ax.set_ylabel("Cumulative Abnormal Return")
    ax.set_title(title)
    ax.legend()
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    
    if save_path:
        fig.savefig(save_path)
    return fig, ax
```

#### 组合排序收益对比柱状图

```python
def plot_portfolio_bar(returns, labels=None, title="Portfolio Returns",
                       save_path=None):
    """
    组合排序收益柱状图（如 Fama-French 十分组）。
    
    returns: list/array, 各组合平均月收益率
    labels: 组合标签（如 ["Lo", "2", "3", ..., "Hi"]）
    """
    PALETTE = set_academic_style("jf")
    
    if labels is None:
        labels = [f"P{i+1}" for i in range(len(returns))]
    
    fig, ax = plt.subplots()
    colors = [PALETTE[3] if r < 0 else PALETTE[0] for r in returns]
    bars = ax.bar(range(len(returns)), returns, color=colors, edgecolor="white", linewidth=0.5)
    
    ax.set_xticks(range(len(returns)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Avg Monthly Return")
    ax.set_title(title)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.axhline(y=0, color="black", linewidth=0.5)
    
    # 标注数值
    for bar, val in zip(bars, returns):
        va = "bottom" if val >= 0 else "top"
        ax.text(bar.get_x() + bar.get_width()/2, val,
                f"{val:.2%}", ha="center", va=va, fontsize=8)
    
    if save_path:
        fig.savefig(save_path)
    return fig, ax
```

#### Fama-MacBeth 系数时序图

```python
def plot_fm_coefficients(ts_coef, ts_se, var_name,
                         title="Fama-MacBeth Coefficient Time Series",
                         save_path=None):
    """
    Fama-MacBeth 回归系数的时序图（含 ±2 SE 置信带）。
    
    ts_coef: Series, 每月截面回归系数
    ts_se: Series, 每月系数标准误
    """
    PALETTE = set_academic_style("jf")
    
    fig, ax = plt.subplots()
    dates = ts_coef.index
    
    ax.plot(dates, ts_coef, color=PALETTE[0], linewidth=1)
    ax.fill_between(dates, ts_coef - 2*ts_se, ts_coef + 2*ts_se,
                    alpha=0.15, color=PALETTE[0])
    ax.axhline(y=0, color="black", linewidth=0.5, linestyle="--")
    
    ax.set_xlabel("Year")
    ax.set_ylabel(f"Coefficient: {var_name}")
    ax.set_title(title)
    
    # 添加均值线
    mean_coef = ts_coef.mean()
    ax.axhline(y=mean_coef, color=PALETTE[3], linewidth=0.8, linestyle="--",
               label=f"Mean = {mean_coef:.4f}")
    ax.legend()
    
    if save_path:
        fig.savefig(save_path)
    return fig, ax
```

#### 学术表格 — 回归结果（带显著性星号）

```python
def format_regression_table(models_dict, stars=True):
    """
    将多个回归模型结果组装为学术发表级表格。
    
    models_dict: {"Model 1": sm_result, "Model 2": sm_result, ...}
    
    返回 DataFrame，可以 .to_latex() 导出为 LaTeX 表格。
    """
    rows = []
    for name, result in models_dict.items():
        for var in result.params.index:
            coef = result.params[var]
            se = result.bse[var]
            pval = result.pvalues[var]
            
            # 显著性星号
            if stars and pval < 0.01:
                star = "***"
            elif stars and pval < 0.05:
                star = "**"
            elif stars and pval < 0.10:
                star = "*"
            else:
                star = ""
            
            rows.append({
                "Variable": var,
                f"{name} (Coef)": f"{coef:.4f}{star}",
                f"{name} (SE)": f"({se:.4f})",
            })
    
    table = pd.DataFrame(rows)
    return table
```

### 3.3 热力图 — 相关系数矩阵

```python
def plot_correlation_heatmap(df, columns=None, title="Correlation Matrix",
                              save_path=None):
    """
    学术风格的相关系数热力图。
    """
    PALETTE = set_academic_style("jf")
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    corr = df[columns].corr()
    
    fig, ax = plt.subplots(figsize=(len(columns)*0.8, len(columns)*0.7))
    
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                linewidths=0.5, cbar_kws={"shrink": 0.8},
                ax=ax)
    
    ax.set_title(title)
    
    if save_path:
        fig.savefig(save_path)
    return fig, ax
```

---

## 4. 标准实证管线

### 4.1 截面资产定价（Fama-MacBeth）

```
数据获取 → 数据预处理(§2) → 特征工程 → 每月截面回归(TS) → 系数时间序列均值 & t 检验(FM) → 可视化(§3)
```

- 参考项目：`OpenSourceAP_CrossSection/` — Chen & Zimmermann (2020)，200+ 异象信号
- 参考项目：`GKX_ML_AssetPricing/` — Gu, Kelly & Xiu (2020) ML 方法

### 4.2 组合排序（Portfolio Sort）

```
每月按信号排序 → 分10组/5组 → 等权/市值加权收益 → 多空组合 → Newey-West t 检验 → 柱状图(§3.2)
```

- 关键细节：使用 NYSE 断点、持有期、交易成本调整

### 4.3 面板数据 / 固定效应（公司金融）

```python
# pyfixest 高性能语法（推荐）
model = pf.feols("y ~ x1 + x2 | firm_id + year", data=df, vcov={"CRV1": "firm_id + year"})

# linearmodels 经典语法
model = PanelOLS(y, X, entity_effects=True, time_effects=True).fit(cov_type="clustered", clusters=df[["firm","year"]])
```

### 4.4 双重差分（DID / Staggered DID）

```
事件窗口 → 处理组/控制组 → 平行趋势检验 → 动态 DID (Sun-Abraham / CSDID) → 系数图(§3.2)
```

- 关键：注意处理时间交错的 DID 估计偏差，优先使用最新方法

### 4.5 事件研究（Event Study）

```
事件日对齐 → 估计窗口(-252,-21) → 市场模型 → 异常收益 CAR/BHAR → 截面 t 检验 → CAR 图(§3.2)
```

### 4.6 GARCH 波动率建模

```python
model = arch_model(returns, vol="GARCH", p=1, q=1).fit()
```

---

## 5. 本地参考项目

在 `/Users/echo/Desktop/python/` 下：

| 项目 | 论文 | 用途 |
|------|------|------|
| `OpenSourceAP_CrossSection/` | Chen & Zimmermann (2020) | 200+ 异象复现、组合排序、Fama-MacBeth |
| `GKX_ML_AssetPricing/` | Gu, Kelly & Xiu (2020) | ML 资产定价（NN/GBRT/RF） |
| `famafrench_pkg/` | — | Fama-French 因子数据获取（WRDS） |

## 6. 论文复现工作流

当用户说"复现某论文"时：

1. **理解论文**：确认因变量、自变量、样本期、实证方法
2. **定位数据**：CRSP / Compustat / IBES / OptionMetrics / FRED / 手工数据
3. **数据预处理**：按 §2 管线清洗数据（缩尾、合并、缺失值处理），通过数据质量检查清单
4. **构建因子/变量**：参考 OpenSourceAP 的信号定义方式
5. **复现表**：逐表复现，每张表一个 notebook cell 或脚本
6. **可视化输出**：用 §3 模板生成学术级图表，对比论文原图
7. **核对结果**：与论文报告值对比（系数符号、大小、显著性）

## 7. 常见陷阱

- **Newey-West 滞后阶数**：默认 floor(4×(T/100)^(2/9))，金融常用 6-12
- **NYSE 断点**：组合排序必须使用 NYSE 市值断点，而非全样本
- **微观层聚类**：面板回归的标准误必须聚类到 firm 层面
- **样本筛选**：CRSP 股价 < $1/$5 剔除、金融/公用事业行业剔除
- **超额收益**：确认是无风险利率调整还是市场模型调整
- **数据合并**：财政年度 vs 日历年度对齐（Compustat 滞后 6 个月）
- **退市偏差**：CRSP 月度收益不含退市收益，必须手动合并且处理缺失退市收益
- **存活偏差**：使用 CRSP/Compustat Merged (CCM) 链接表，而非直接按 PERMNO 匹配
- **缩尾分组**：缩尾应按时间截面分组（每年/每月），不能全样本一次性缩尾
- **可视化陷阱**：确保图表在灰度打印下仍可区分，用线型+标记区分而非纯依赖颜色
