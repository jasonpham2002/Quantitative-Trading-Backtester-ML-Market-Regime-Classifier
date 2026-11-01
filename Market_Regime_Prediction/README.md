# ML Market Regime Classifier (Project 2)

This directory holds the Project 2 code for the DATA 3401 Honors Project. It predicts whether the market will be **Bullish (1)** or **Bearish (0)** over the next 5 trading days. The label is `Close[t+5] > SMA_50[t]`.

It has two model families, both trained and evaluated on the same data splits:

1. A **from-scratch NumPy `LogisticRegression`**, the course-required baseline.
2. A **PyTorch deep learning suite** made of an MLP baseline, LSTM/GRU, a 1D CNN and a small Transformer.

Every model exposes the same `fit` / `predict_proba` / `predict` / `evaluate` interface. Any of them can be plugged into `MLHybridStrategy` in Project 1 as a regime filter.

---

## Files

| File | Purpose |
| :--- | :--- |
| `preprocessing_pipeline.py` | `Pipeline`: feature engineering, target labeling, chronological splits, zero-leakage scaling, 3D sequence windows |
| `LogisticRegression.py` | Pure-NumPy logistic regression (sigmoid, weighted BCE loss, gradient descent, metrics) |
| `torch_models.py` | PyTorch `RegimeDataset`, `MLPRegimeNet`, `RecurrentRegimeNet` (LSTM/GRU), `Conv1DRegimeNet`, `TransformerRegimeNet`, and the `DeepRegimeClassifier` wrapper |
| `visualizations.py` | Matplotlib/NumPy plotting suite (EDA, model diagnostics, strategy comparison) |

---

## 1. Feature Engineering (`Pipeline`)

`Pipeline(dataset, ticker=None, feature_set='basic' | 'extended')`

| `feature_set` | Features |
| :--- | :--- |
| `'basic'` (4, the original set) | `ret_5d`, `vol_20d`, `rsi_14`, `sma50_ratio` |
| `'extended'` (14, scale-invariant) | `candle_body`, `intraday_range`, `volume_ratio`, `ret_1d`, `ret_5d`, `ret_10d`, `vol_10d`, `vol_20d`, `sma20_ratio`, `sma50_ratio`, `rsi_14`, `macd_norm`, `macd_hist_norm`, `vol_regime` |

All extended features are ratios, returns or bounded oscillators. Raw OHLCV price levels are not used, because they are non-stationary (SPY drifts from about \$420 to \$680+ over the sample) and would not transfer to unseen price ranges.

### Splits and sequence windows

- `train_test_split_scale(train_ratio=0.8)` is the original 80/20 split (960 train / 240 test).
- `train_val_test_split_scale(train_ratio=0.65, val_ratio=0.15)` gives **780 train / 180 validation / 240 test** days. The 240-day test window is the same one used by the NumPy model and the Project 1 backtests.
- Z-score mean and std are fit **only on the training split**, so there is no leakage into validation or test.
- `Pipeline.create_sequences(features_2d, targets_1d, seq_len=20, prev_features_2d=...)` builds `(N, L, F)` sliding windows. Prepending the previous split's last `L-1` rows means sequence models predict all 240 test days.

---

## 2. Models

| Model | Input shape | Idea |
| :--- | :--- | :--- |
| `LogisticRegression` (NumPy) | `(B, F)` | Linear boundary. Written from scratch with a class-weighted BCE loss. |
| `MLPRegimeNet` | `(B, F)` | `Linear → LayerNorm → ReLU → Dropout` stacks. Learns non-linear feature interactions on a single day. |
| `RecurrentRegimeNet` (`cell_type='lstm'` or `'gru'`) | `(B, L, F)` | Reads the 20-day window step by step. The final hidden state `out[:, -1, :]` gives the logit. |
| `Conv1DRegimeNet` | `(B, L, F)` → `(B, F, L)` | Two 1D convolution blocks plus global average pooling detect local multi-day patterns. |
| `TransformerRegimeNet` | `(B, L, F)` | Sinusoidal positional encoding plus self-attention. The `L×L` attention weights are exposed for plotting. |

### `DeepRegimeClassifier`

A single trainer wrapper for all four PyTorch architectures. It mirrors the NumPy `LogisticRegression` API.

```python
from torch_models import DeepRegimeClassifier

clf = DeepRegimeClassifier(model_name='gru')   # 'mlp' | 'lstm' | 'gru' | 'cnn1d' | 'transformer'
clf.fit(X_train, y_train, X_val, y_val)         # 2D for mlp, 3D for the sequence models
probs = clf.predict_proba(X_test)
preds = clf.predict(X_test)
metrics = clf.evaluate(y_test, preds)
```

- Loss is `BCEWithLogitsLoss(pos_weight=N_0/N_1)`, the PyTorch counterpart of `class_weight='balanced'`.
- Early stopping on validation loss restores the best checkpoint.
- `train_loss_history` / `val_loss_history` are stored for the training-curve plots.
- `evaluate()` reports Accuracy, Precision, Recall, F1, Confusion Matrix, the always-bullish baseline, plus ROC-AUC, Balanced Accuracy and Log Loss. These extra metrics are computed in pure NumPy.

---

## 3. Visualization Suite (`visualizations.py`)

| Group | Functions |
| :--- | :--- |
| **A. Feature EDA** | `plot_feature_correlation_heatmap`, `plot_regime_feature_distributions`, `plot_volatility_regime_timeline` |
| **B. Model diagnostics** | `plot_training_curves_grid`, `plot_confusion_matrix_grid`, `plot_roc_pr_curves`, `plot_feature_importance_heatmap` (permutation importance), `plot_transformer_attention_heatmap` |
| **C. Strategy comparison** | `plot_regime_agreement_barcode`, `plot_risk_return_frontier` |

The module uses the headless `Agg` backend, so it runs without a display. The executed notebook shows all figures.

---

## 4. Results (240 out-of-sample trading days, SPY)

Bullish prevalence in the test window is 80.8% (194 bull / 46 bear days).

### Classification

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Bear days caught (TN/46) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| LogisticRegression (4F) | 83.75% | 86.05% | 95.36% | 0.905 | 0.856 | 16 |
| LogisticRegression (14F) | 85.42% | 87.68% | 95.36% | 0.914 | 0.876 | 20 |
| MLP | 83.33% | 91.85% | 87.11% | 0.894 | 0.861 | 31 |
| LSTM | 84.17% | 86.45% | 95.36% | 0.907 | 0.885 | 17 |
| GRU | 83.75% | 91.01% | 88.66% | 0.898 | 0.870 | 29 |
| 1D-CNN | 82.08% | 82.97% | 97.94% | 0.898 | 0.736 | 7 |
| Transformer | 82.50% | 86.54% | 92.78% | 0.896 | 0.852 | 18 |

Baseline "always bullish" accuracy is 80.83%. Every model beats it, but only by a few points. Treat the accuracy differences between models as small, not as a clear ranking.

### Hybrid backtest (`MLHybridStrategy`, \$100,000 start)

| Strategy | Total Return | Sharpe | Max Drawdown | Closed Trades |
| :--- | :---: | :---: | :---: | :---: |
| Buy & Hold (SPY) | 17.76% | 1.39 | 8.88% | 1 |
| Hybrid + LogReg (4F / 14F) | 16.96% | 1.54 | 4.47% | 3 |
| Hybrid + MLP | 7.85% | 0.84 | 7.44% | 8 |
| Hybrid + LSTM | 18.39% | 1.62 | 4.75% | 4 |
| Hybrid + GRU | 19.74% | 1.84 | 4.49% | 7 |
| Hybrid + 1D-CNN | **22.18%** | **1.87** | 5.05% | 2 |
| Hybrid + Transformer | 16.23% | 1.48 | 5.97% | 5 |

> **Caveat:** This is one 240-day test window on one asset, with a few closed trades per strategy. The deep models match or beat the NumPy baseline on risk-adjusted return in this window. That is not statistically significant. The MLP hybrid underperformed. Do not read the table as proof that deeper models are better.

---

## Usage

```python
import sys
sys.path.append("Market_Regime_Prediction")
from preprocessing_pipeline import Pipeline
from torch_models import DeepRegimeClassifier

pipe = Pipeline(spy_df, feature_set='extended')
X_tr, X_val, X_te, y_tr, y_val, y_te = pipe.train_val_test_split_scale()

# Sequence models: 20-day windows with warmup so the test set stays at 240 rows
X_te_3d = Pipeline.create_sequences(X_te, seq_len=20, prev_features_2d=X_val)
```

The end-to-end walkthrough is **Chapter 6** of [`../Trading Strategies Backtesting System/TradingNotebook.ipynb`](../Trading%20Strategies%20Backtesting%20System/TradingNotebook.ipynb).
