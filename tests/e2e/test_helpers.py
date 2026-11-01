"""
Shared test utilities, mathematical reference oracles, synthetic data generators,
and environment capability detectors for the PyTorch Market Regime Benchmark Suite.
"""

import os
import sys
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd

# Configure headless Matplotlib environment
def ensure_headless_matplotlib():
    temp_dir = tempfile.gettempdir()
    os.environ.setdefault('MPLCONFIGDIR', os.path.join(temp_dir, 'matplotlib_test_cache'))
    try:
        import matplotlib
        matplotlib.use('Agg')
    except ImportError:
        pass

ensure_headless_matplotlib()

# Project directory paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BACKTESTER_DIR = PROJECT_ROOT / "Trading Strategies Backtesting System"
MARKET_REGIME_DIR = PROJECT_ROOT / "Market_Regime_Prediction"
SPY_DATA_PATH = BACKTESTER_DIR / "SPY_5_years"

# Add directories to sys.path if not present
for d in [str(PROJECT_ROOT), str(BACKTESTER_DIR), str(MARKET_REGIME_DIR)]:
    if d not in sys.path:
        sys.path.insert(0, d)

# Capability Detectors
try:
    import torch
    HAS_TORCH = True
except (ImportError, ModuleNotFoundError):
    HAS_TORCH = False

try:
    import preprocessing_pipeline
    from preprocessing_pipeline import Pipeline
    HAS_PREPROCESSING = True
except (ImportError, ModuleNotFoundError):
    HAS_PREPROCESSING = False

try:
    import torch_models
    HAS_TORCH_MODELS = True
except (ImportError, ModuleNotFoundError):
    HAS_TORCH_MODELS = False

try:
    import visualizations
    HAS_VISUALIZATIONS = True
except (ImportError, ModuleNotFoundError):
    HAS_VISUALIZATIONS = False

try:
    from strategy import MACrossover, RSIMeanReversion, MLHybridStrategy
    from backtester import Backtest
    from pipeline import DataPipeline
    HAS_BACKTESTER = True
except (ImportError, ModuleNotFoundError):
    HAS_BACKTESTER = False

try:
    from LogisticRegression import LogisticRegression
    HAS_LOGISTIC_REGRESSION = True
except (ImportError, ModuleNotFoundError):
    HAS_LOGISTIC_REGRESSION = False


# Synthetic OHLCV Data Generator
def generate_synthetic_ohlcv(
    n_days=1254,
    start_date='2021-09-02',
    ticker='SPY',
    multi_index=True,
    random_seed=42,
    constant_price=None,
    zero_volume=False,
):
    """
    Generates synthetic daily financial OHLCV price series.
    Guarantees structural validity: High >= max(Open, Close) and Low <= min(Open, Close).
    """
    np.random.seed(random_seed)
    dates = pd.bdate_range(start=start_date, periods=n_days)

    if constant_price is not None:
        close = np.full(n_days, float(constant_price))
        open_p = close.copy()
        high = close.copy()
        low = close.copy()
    else:
        # Generate geometric random walk
        drift = 0.0004
        volatility = 0.012
        returns = np.random.normal(drift, volatility, n_days)
        returns[0] = 0.0
        price_series = 450.0 * np.exp(np.cumsum(returns))
        close = price_series

        # Realistic intraday variations
        open_ratio = np.random.normal(1.0, 0.003, n_days)
        open_p = close * open_ratio

        high_extra = np.abs(np.random.normal(0.005, 0.003, n_days))
        high = np.maximum(open_p, close) * (1.0 + high_extra)

        low_deficit = np.abs(np.random.normal(0.005, 0.003, n_days))
        low = np.minimum(open_p, close) * (1.0 - low_deficit)

    if zero_volume:
        volume = np.zeros(n_days, dtype=float)
    else:
        volume = np.random.uniform(5e7, 1e8, n_days)

    data = {
        'Close': close,
        'High': high,
        'Low': low,
        'Open': open_p,
        'Volume': volume,
    }

    if multi_index:
        cols = pd.MultiIndex.from_tuples(
            [(col, ticker) for col in ['Close', 'High', 'Low', 'Open', 'Volume']],
            names=['Price', 'Ticker']
        )
        df = pd.DataFrame(data, index=dates)
        df.columns = cols
    else:
        df = pd.DataFrame(data, index=dates)

    return df


def load_real_spy_data():
    """Loads the real SPY 5-year dataset from the repository."""
    if not SPY_DATA_PATH.exists():
        raise FileNotFoundError(f"Real SPY dataset not found at {SPY_DATA_PATH}")
    return pd.read_csv(SPY_DATA_PATH, header=[0, 1], index_col=0, parse_dates=True)


def check_pipeline_has_extended():
    """Detects if Pipeline implementation supports feature_set='extended'."""
    if not HAS_PREPROCESSING:
        return False
    try:
        dummy_df = generate_synthetic_ohlcv(n_days=100, multi_index=False)
        pipe = Pipeline(dummy_df, feature_set='extended')
        return getattr(pipe, 'feature_set', None) == 'extended' and len(pipe.feature_names) == 14
    except Exception:
        return False


def check_pipeline_has_train_val_test():
    """Detects if Pipeline implementation has train_val_test_split_scale method."""
    if not HAS_PREPROCESSING:
        return False
    return hasattr(Pipeline, 'train_val_test_split_scale')


def check_pipeline_has_create_sequences():
    """Detects if Pipeline implementation has create_sequences static method."""
    if not HAS_PREPROCESSING:
        return False
    return hasattr(Pipeline, 'create_sequences')


HAS_EXTENDED_PIPELINE = check_pipeline_has_extended()
HAS_TRAIN_VAL_TEST = check_pipeline_has_train_val_test()
HAS_CREATE_SEQUENCES = check_pipeline_has_create_sequences()


# Authoritative Mathematical Reference Oracles
def oracle_candle_body(open_p, close_p):
    """Reference calculation for candle_body: (Close - Open) / Open."""
    return (np.asarray(close_p) - np.asarray(open_p)) / np.asarray(open_p)


def oracle_intraday_range(high_p, low_p, close_p):
    """Reference calculation for intraday_range: (High - Low) / Close."""
    return (np.asarray(high_p) - np.asarray(low_p)) / np.asarray(close_p)


def oracle_volume_ratio(volume_series, window=20):
    """Reference calculation for volume_ratio: Volume / (SMA_20(Volume) + 1e-10)."""
    s = pd.Series(volume_series)
    sma = s.rolling(window=window).mean()
    return (s / (sma + 1e-10)).values


def oracle_ret_1d(close_series):
    """Reference calculation for ret_1d: Close.pct_change(1)."""
    return pd.Series(close_series).pct_change(1).values


def oracle_ret_5d(close_series):
    """Reference calculation for ret_5d: Close.pct_change(5)."""
    return pd.Series(close_series).pct_change(5).values


def oracle_ret_10d(close_series):
    """Reference calculation for ret_10d: Close.pct_change(10)."""
    return pd.Series(close_series).pct_change(10).values


def oracle_vol_10d(close_series):
    """Reference calculation for vol_10d: 10-day rolling std of daily returns."""
    daily_ret = pd.Series(close_series).pct_change()
    return daily_ret.rolling(window=10).std().values


def oracle_vol_20d(close_series):
    """Reference calculation for vol_20d: 20-day rolling std of daily returns."""
    daily_ret = pd.Series(close_series).pct_change()
    return daily_ret.rolling(window=20).std().values


def oracle_sma20_ratio(close_series):
    """Reference calculation for sma20_ratio: Close / SMA_20(Close)."""
    s = pd.Series(close_series)
    return (s / s.rolling(window=20).mean()).values


def oracle_sma50_ratio(close_series):
    """Reference calculation for sma50_ratio: Close / SMA_50(Close)."""
    s = pd.Series(close_series)
    return (s / s.rolling(window=50).mean()).values


def oracle_rsi_14(close_series):
    """Reference calculation for rsi_14: 14-period RSI using NumPy delta separation."""
    s = pd.Series(close_series)
    delta = s.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    roll_gain = pd.Series(gain, index=s.index).rolling(window=14).mean()
    roll_loss = pd.Series(loss, index=s.index).rolling(window=14).mean()
    rs = roll_gain / (roll_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.values


def oracle_macd_norm(close_series):
    """Reference calculation for macd_norm: (EMA_12 - EMA_26) / Close (adjust=False)."""
    s = pd.Series(close_series)
    ema_12 = s.ewm(span=12, adjust=False).mean()
    ema_26 = s.ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    return (macd / s).values


def oracle_macd_hist_norm(close_series):
    """Reference calculation for macd_hist_norm: (MACD - EMA_9(MACD)) / Close (adjust=False)."""
    s = pd.Series(close_series)
    ema_12 = s.ewm(span=12, adjust=False).mean()
    ema_26 = s.ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    macd_sig = macd.ewm(span=9, adjust=False).mean()
    macd_hist = macd - macd_sig
    return (macd_hist / s).values


def oracle_vol_regime(close_series):
    """Reference calculation for vol_regime: vol_10d / (vol_20d + 1e-10)."""
    v10 = oracle_vol_10d(close_series)
    v20 = oracle_vol_20d(close_series)
    return v10 / (v20 + 1e-10)


def oracle_create_sequences(features_2d, targets_1d=None, seq_len=20, prev_features_2d=None):
    """
    Authoritative reference sliding window sequence generator.
    """
    features_2d = np.asarray(features_2d, dtype=float)
    if prev_features_2d is not None:
        prev_features_2d = np.asarray(prev_features_2d, dtype=float)
        context = prev_features_2d[-(seq_len - 1):] if seq_len > 1 else np.empty((0, features_2d.shape[1]))
        full = np.vstack([context, features_2d])
        trim_target = False
    else:
        full = features_2d
        trim_target = True

    n_samples, n_feat = full.shape
    num_windows = n_samples - seq_len + 1
    if num_windows <= 0:
        raise ValueError("Not enough samples to construct sequences")

    shape = (num_windows, seq_len, n_feat)
    strides = (full.strides[0], full.strides[0], full.strides[1])
    X_seq = np.lib.stride_tricks.as_strided(full, shape=shape, strides=strides).copy()

    if targets_1d is not None:
        targets_1d = np.asarray(targets_1d)
        y_seq = targets_1d[seq_len - 1:] if trim_target else targets_1d
        return X_seq, y_seq
    return X_seq


def oracle_train_val_test_split_scale(clean_df, feature_names, train_ratio=0.65, val_ratio=0.15):
    """Authoritative reference implementation of 65/15/20 chronological split and zero-leakage scaling."""
    n = len(clean_df)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_df = clean_df.iloc[:train_end].copy()
    val_df = clean_df.iloc[train_end:val_end].copy()
    test_df = clean_df.iloc[val_end:].copy()

    X_train_raw = train_df[feature_names].values
    X_val_raw = val_df[feature_names].values
    X_test_raw = test_df[feature_names].values

    y_train = train_df['target'].values.reshape(-1, 1)
    y_val = val_df['target'].values.reshape(-1, 1)
    y_test = test_df['target'].values.reshape(-1, 1)

    mean_ = np.mean(X_train_raw, axis=0)
    std_ = np.std(X_train_raw, axis=0)
    std_[std_ == 0] = 1e-10

    X_train_scaled = (X_train_raw - mean_) / std_
    X_val_scaled = (X_val_raw - mean_) / std_
    X_test_scaled = (X_test_raw - mean_) / std_

    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, mean_, std_


# Drop-in Test Double / Mock Deep Classifier for Harness Verification
class MockDeepRegimeClassifier:
    """
    Simulates DeepRegimeClassifier contract using analytical rules
    to allow complete verification of the test harness and downstream backtester.
    """
    def __init__(self, model_name='mock', **kwargs):
        self.model_name = model_name
        self.cost_history = [0.69, 0.55, 0.42, 0.38, 0.35]
        self.train_loss_history = self.cost_history
        self.val_loss_history = [0.70, 0.58, 0.45, 0.40, 0.39]
        self.is_fitted = False

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self.is_fitted = True
        return self

    def predict_proba(self, X):
        X = np.asarray(X)
        n = X.shape[0]
        # Return synthetic probabilities with mild bullish tendency
        np.random.seed(42)
        probs = 0.3 + 0.4 * np.random.rand(n, 1)
        return probs

    def predict(self, X, threshold=0.22, smooth_window=3):
        probs = self.predict_proba(X).ravel()
        smoothed = np.array([
            np.mean(probs[max(0, i - smooth_window + 1):i + 1])
            for i in range(len(probs))
        ])
        return (smoothed >= threshold).astype(int).reshape(-1, 1)

    def evaluate(self, y_true, y_pred):
        y_t = np.asarray(y_true).ravel()
        y_p = np.asarray(y_pred).ravel()
        tp = int(np.sum((y_t == 1) & (y_p == 1)))
        tn = int(np.sum((y_t == 0) & (y_p == 0)))
        fp = int(np.sum((y_t == 0) & (y_p == 1)))
        fn = int(np.sum((y_t == 1) & (y_p == 0)))

        acc = (tp + tn) / max(len(y_t), 1)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * (prec * rec) / max(prec + rec, 1e-10)
        baseline = float(np.mean(y_t == 1))

        return {
            "Accuracy": float(acc),
            "Precision": float(prec),
            "Recall": float(rec),
            "F1-Score": float(f1),
            "Baseline (Always Bullish)": float(baseline),
            "Confusion Matrix": np.array([[tn, fp], [fn, tp]]),
        }
