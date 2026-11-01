# Quantitative Trading Backtester & ML Market Regime Classifier

This repository holds the **DATA 3401 (Python for Data Science 1)** Honors Contract work. It has two connected projects, tested on the **SPDR S&P 500 ETF Trust (`SPY`)**:

- **Project 1: Quantitative Trading Backtester.** An object-oriented system that simulates trading strategies on historical data.
- **Project 2: ML Market Regime Classifier.** Predicts whether the market will be Bullish or Bearish over the next 5 days. It includes a **from-scratch NumPy logistic regression** and a **PyTorch deep learning suite** (MLP, LSTM/GRU, 1D CNN, small Transformer).

The Project 2 classifier output acts as a trade filter for the Project 1 backtester (`MLHybridStrategy`), so we can test whether regime filtering improves risk-adjusted performance.

```text
Historical OHLCV (SPY)
        │
        ├── Project 1: Strategies + Backtester ──┐
        │                                        ├──> MLHybridStrategy ──> Backtest ──> Metrics & charts
        └── Project 2: Features + Classifiers ───┘
            (NumPy LogReg, MLP, LSTM, GRU, 1D-CNN, Transformer)
```

---

## Repository Structure

```text
DATA-3401-Honors-Project/
├── Market_Regime_Prediction/                # Project 2: ML Market Regime Classifier
│   ├── preprocessing_pipeline.py            # Features (basic/extended), target, splits, scaling, 3D sequences
│   ├── LogisticRegression.py                # Pure-NumPy logistic regression baseline
│   ├── torch_models.py                      # PyTorch MLP / LSTM / GRU / 1D-CNN / Transformer + DeepRegimeClassifier
│   ├── visualizations.py                    # Heatmaps, diagnostics and strategy-comparison charts
│   └── README.md                            # Project 2 documentation
├── Trading Strategies Backtesting System/   # Project 1: Quantitative Trading Backtester
│   ├── pipeline.py                          # yfinance / CSV ingestion, MultiIndex normalization (df.xs)
│   ├── strategy.py                          # Strategy hierarchy (MACrossover, RSIMeanReversion, MLHybridStrategy)
│   ├── backtester.py                        # Daily portfolio simulation, metrics and plots
│   ├── SPY_5_years                          # Cached 5-year daily SPY data (1,254 trading days)
│   ├── TradingNotebook.ipynb                # Executed notebook: Chapters 1-5 (core) and Chapter 6 (deep learning)
│   └── README.md                            # Project 1 documentation
├── tests/                                   # E2E tiers (tests/e2e) plus unit, stress and adversarial tests
├── requirements.txt                         # Python dependencies (includes torch)
└── README.md
```

---

## Project 1: Quantitative Trading Backtester

Details are in [`Trading Strategies Backtesting System/README.md`](Trading%20Strategies%20Backtesting%20System/README.md).

- **`DataPipeline`** loads split-adjusted data from `yfinance` or a cached CSV. It wraps single-string tickers into lists with `isinstance()`, extracts one ticker from MultiIndex data with `df.xs()`, and forward-fills gaps.
- **`Strategy`** is the base class (`generate_signal()`, `execute_trade()`, `calculate_returns()`). Its subclasses:
  - `MACrossover`: long when the 20-day SMA is above the 50-day SMA.
  - `RSIMeanReversion`: NumPy-computed 14-day RSI. Buy below 30, sell above 70.
  - `MLHybridStrategy`: trades only when the technical signal is active and the ML classifier predicts Bullish.
- **`Backtest`** simulates day by day from \$100,000. It logs trades and computes Total Return, Sharpe Ratio, Max Drawdown, Win Rate and Profit Factor. It also draws equity, drawdown and signal-shaded price charts.

## Project 2: ML Market Regime Classifier

Details are in [`Market_Regime_Prediction/README.md`](Market_Regime_Prediction/README.md).

- **Target:** `1` if `Close[t+5] > SMA_50[t]`, else `0`.
- **Features:** a 4-feature `'basic'` set (`ret_5d`, `vol_20d`, `rsi_14`, `sma50_ratio`) and a 14-feature scale-invariant `'extended'` set. The extended set adds OHLCV ratios, multi-horizon returns, rolling volatility, SMA ratios, MACD and a volatility-regime ratio.
- **Splits:** chronological 65% train / 15% validation / 20% test (780 / 180 / 240 days). Standardization is fit on the training set only. Sequence models use 20-day windows with `L-1` warmup rows, so every model is scored on the same 240 test days.
- **Models:**

| Model | Input | Notes |
| :--- | :--- | :--- |
| NumPy `LogisticRegression` | `(B, F)` | Written from scratch: sigmoid, weighted BCE, gradient descent |
| `MLPRegimeNet` | `(B, F)` | Non-linear tabular baseline |
| `RecurrentRegimeNet` (LSTM / GRU) | `(B, 20, F)` | Temporal memory over the lookback window |
| `Conv1DRegimeNet` | `(B, 20, F)` | Local multi-day pattern detection |
| `TransformerRegimeNet` | `(B, 20, F)` | Self-attention with exposed attention weights |

- **`DeepRegimeClassifier`** wraps all PyTorch models with the same `fit` / `predict_proba` / `predict` / `evaluate` API as the NumPy model. It uses `BCEWithLogitsLoss(pos_weight=...)` and validation early stopping.
- **Visualizations:** feature correlation heatmap, class-conditional feature distributions, volatility/regime timeline, training curves, confusion-matrix grid, ROC/PR curves, permutation-importance heatmap, Transformer attention heatmap, regime-agreement barcode and a risk-return frontier.

---

## Key Empirical Results (Out-of-Sample: 240 Trading Days)

All numbers come from the executed [`TradingNotebook.ipynb`](Trading%20Strategies%20Backtesting%20System/TradingNotebook.ipynb) (Chapters 3, 4 and 6). The "always bullish" baseline accuracy is 80.83%.

### Regime classification

| Model | Accuracy | F1 | ROC-AUC |
| :--- | :---: | :---: | :---: |
| LogisticRegression (4F) | 83.75% | 0.905 | 0.856 |
| LogisticRegression (14F) | 85.42% | 0.914 | 0.876 |
| MLP | 83.33% | 0.894 | 0.861 |
| LSTM | 84.17% | 0.907 | 0.885 |
| GRU | 83.75% | 0.898 | 0.870 |
| 1D-CNN | 82.08% | 0.898 | 0.736 |
| Transformer | 82.50% | 0.896 | 0.852 |

### Backtest comparison (\$100,000 initial capital)

| Strategy | Total Return | Sharpe | Max Drawdown | Closed Trades |
| :--- | :---: | :---: | :---: | :---: |
| Buy & Hold (SPY) | 17.76% | 1.39 | 8.88% | 1 |
| Dual MA Crossover (20/50) | 13.72% | 1.27 | 5.05% | 2 |
| RSI Mean Reversion (14-d, 30/70) | 8.92% | 1.50 | 4.55% | 2 |
| Hybrid + LogisticRegression (4F / 14F) | 16.96% | 1.54 | 4.47% | 3 |
| Hybrid + MLP | 7.85% | 0.84 | 7.44% | 8 |
| Hybrid + LSTM | 18.39% | 1.62 | 4.75% | 4 |
| Hybrid + GRU | 19.74% | 1.84 | 4.49% | 7 |
| Hybrid + 1D-CNN | **22.18%** | **1.87** | 5.05% | 2 |
| Hybrid + Transformer | 16.23% | 1.48 | 5.97% | 5 |

> **Interpretation caveat:** This is a single 240-day window on one asset, and each strategy closes only a few trades. The GRU and 1D-CNN hybrids beat Buy & Hold on Sharpe ratio here, and the LSTM hybrid is close behind. The MLP hybrid underperformed. These differences are suggestive, not statistically significant. Classification accuracy is only 1.5 to 4.5 points above the always-bullish baseline.

---

## How to Run

1. Create an environment and install dependencies (Python 3.12+ recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Open [`Trading Strategies Backtesting System/TradingNotebook.ipynb`](Trading%20Strategies%20Backtesting%20System/TradingNotebook.ipynb) in Jupyter. Chapters 1-5 cover the core projects and Chapter 6 covers the PyTorch benchmark. The PyTorch models use Apple `mps` or CUDA when available and fall back to CPU.
3. Run the tests from the repository root:
   ```bash
   python tests/e2e/run_tests.py
   python -m unittest discover -s tests -p "*.py"
   ```
