# Quantitative Trading Backtesting System (Project 1)

This directory contains the object-oriented Quantitative Trading Backtesting Engine and the integrated research notebook (`TradingNotebook.ipynb`) for the DATA 3401 Honors Project. The ML classifiers it consumes live in [`../Market_Regime_Prediction/`](../Market_Regime_Prediction/README.md) (Project 2).

---

## Module Architecture

### 1. `pipeline.py` (`DataPipeline`)
- Downloads split-adjusted historical OHLCV data via `yfinance` (`auto_adjust=True`) or loads cached CSV data (`SPY_5_years`).
- Normalizes ticker inputs (`str` or `list`) using `isinstance()` and exposes `self.interval` / `self.intervals`.
- Detects and extracts single-ticker cross-sections from MultiIndex columns using `df.xs()`, then forward-fills gaps (`.ffill()`).

### 2. `strategy.py` (`Strategy` hierarchy)
All strategies inherit from `Strategy`, which defines `generate_signal()`, `execute_trade()` and `calculate_returns()` / `calculate_return()`:
- **`MACrossover`**: Dual SMA crossover (`short_window=20`, `long_window=50`). Buy (`1`) when `SMA_20 > SMA_50`, otherwise cash (`0`).
- **`RSIMeanReversion`**: Computes the 14-day RSI with NumPy. Buy (`1`) when `RSI < 30`, exit to cash (`0`) when `RSI > 70`, holding the prior state in between.
- **`MLHybridStrategy(asset_data, base_strategy, ml_predictions)`**: Allows long exposure only when the technical trend is active (`MA Crossover == 1` or `Close > SMA_20`) **and** the ML classifier predicts a Bullish regime (`1`).
  - It accepts predictions from **any** Project 2 model: the NumPy `LogisticRegression`, or a PyTorch MLP, LSTM, GRU, 1D-CNN or Transformer (`DeepRegimeClassifier`).
  - If `ml_predictions` holds continuous probabilities, they are thresholded at 0.5 with a `UserWarning`.
  - If `ml_predictions` is shorter than the price series (for example, test-split predictions only), the missing leading days default to cash, with a `UserWarning`.

### 3. `backtester.py` (`Backtest`)
- Simulates daily trading from \$100,000 and tracks mark-to-market net worth in `self.tot_net_worth`.
- Logs each Buy and Sell (`Action`, `Shares`, `Price`, `Profit`) in `self.trade_history`.
- **`cal_metrics()`** returns Total Return, Annualized Sharpe Ratio (√252 scaling), Max Drawdown, Win Rate, Profit Factor and the Buy & Hold benchmark return.
- **`plot_result()`** and **`plot_drawdown()`** draw the equity curve, price with green position shading, and the drawdown curve.

### 4. `TradingNotebook.ipynb`
End-to-end executed notebook:
- **Chapter 1**: Data pipeline and EDA (5-year `SPY` price, 20/50 SMAs, return distribution).
- **Chapter 2**: Standalone strategy backtests (`MACrossover`, `RSIMeanReversion`).
- **Chapter 3**: Project 2 NumPy classifier (features, loss convergence, feature weights, confusion matrix, regime overlay).
- **Chapter 4**: System integration (`MLHybridStrategy` backtest).
- **Chapter 5**: Comparative analysis and written research report.
- **Chapter 6**: PyTorch deep learning benchmark:
  - 14 extended stationary features and a 65/15/20 train/validation/test split with sequence warmup.
  - EDA charts: correlation heatmap, regime feature distributions, volatility timeline.
  - Training curves for MLP, LSTM, GRU, 1D-CNN and Transformer.
  - Confusion matrices and ROC/PR curves, plus a permutation-importance heatmap and the Transformer attention heatmap.
  - Out-of-sample hybrid backtests, with a regime-agreement barcode, comparative equity/drawdown curves and a risk-return frontier.

---

## Running

```bash
pip install -r ../requirements.txt
jupyter notebook TradingNotebook.ipynb
```

The `.py` modules import each other by file name, so run them from inside this directory. The notebook adds `../Market_Regime_Prediction` to its path.
