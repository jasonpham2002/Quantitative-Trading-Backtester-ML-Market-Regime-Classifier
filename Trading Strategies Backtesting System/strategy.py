import pandas as pd
import numpy as np
import warnings

class Strategy:
    """
    Base Strategy class defining standard placeholders for quantitative trading:
    - generate_signal(): computes technical indicator entry/exit rules
    - execute_trade(): tracks position state changes (1=Buy, -1=Sell, 0=Hold)
    - calculate_returns() / calculate_return(): computes benchmark and strategy return series
    """
    def __init__(self, asset_data, full_data=None, ticker=None):
        self.ticker = ticker
        self.data = self._extract_single_ticker(asset_data, ticker=ticker)
        self.full_data = self._extract_single_ticker(full_data, ticker=ticker) if full_data is not None else None

    @staticmethod
    def _extract_single_ticker(df, ticker=None):
        """Extracts a single-ticker OHLCV DataFrame via df.xs() if df has MultiIndex columns."""
        if df is not None and isinstance(df.columns, pd.MultiIndex):
            price_cols = {'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'}
            level_0 = set(df.columns.get_level_values(0))
            ticker_level = 1 if price_cols.intersection(level_0) else 0
            level_tickers = df.columns.get_level_values(ticker_level)
            target = ticker if (ticker is not None and ticker in level_tickers) else level_tickers[0]
            return df.xs(target, axis=1, level=ticker_level).copy()
        return df

    def execute_trade(self):
        # Position change:
        # 1  (0 -> 1): Buy signal triggered
        # -1 (1 -> 0): Sell signal triggered
        # 0  (1 -> 1 or 0 -> 0): Hold current state
        self.data['position'] = self.data['signal'].diff().fillna(self.data['signal'])

    def calculate_return(self):
        self.data['asset_returns'] = self.data['Close'].pct_change()
        # Strategy return: multiply tomorrow's return by today's end-of-day signal
        self.data['strategy_returns'] = self.data['asset_returns'] * self.data['signal'].shift(1).fillna(0)
        return self.data['strategy_returns']

    def calculate_returns(self):
        return self.calculate_return()

    def generate_signal(self):
        pass

    @property
    def signal(self):
        """Returns the signal series if generated in data, or None."""
        if hasattr(self, 'data') and self.data is not None and 'signal' in self.data:
            return self.data['signal']
        return None

class MACrossover(Strategy):
    """
    Dual Moving Average Crossover Strategy:
    Generates a Buy signal (1) when the short-term 20-day moving average crosses
    above the long-term 50-day moving average, and Sell (0) otherwise.
    """
    def __init__(self, asset_data, short_window=20, long_window=50, full_data=None, ticker=None):
        super().__init__(asset_data, full_data=full_data, ticker=ticker)
        self.short_window = short_window
        self.long_window = long_window

    def generate_signal(self):
        source = self.full_data.copy() if self.full_data is not None else self.data
        short_ma = source['Close'].rolling(window=self.short_window).mean()
        long_ma = source['Close'].rolling(window=self.long_window).mean()
        raw_sig = pd.Series(np.where(short_ma > long_ma, 1, 0), index=source.index)

        self.data['short_average_close'] = short_ma.loc[self.data.index]
        self.data['long_average_close'] = long_ma.loc[self.data.index]
        self.data['signal'] = raw_sig.loc[self.data.index].values
        return self.data['signal']

class RSIMeanReversion(Strategy):
    """
    RSI Mean Reversion Strategy:
    Calculates Relative Strength Index manually using NumPy:
    - Entry (Buy = 1): RSI drops below 30 (oversold condition)
    - Exit (Sell = 0): RSI rises above 70 (overbought condition)
    """
    def __init__(self, asset_data, time_window=14, full_data=None, ticker=None):
        super().__init__(asset_data, full_data=full_data, ticker=ticker)
        self.interval = time_window

    def generate_signal(self):
        source = self.full_data.copy() if self.full_data is not None else self.data

        # Daily changes for isolation of daily gains and losses
        daily_changes = source['Close'].diff()

        # Gains and losses columns for Relative Strength
        gains = np.where(daily_changes > 0, daily_changes, 0.0)
        losses = np.where(daily_changes < 0, -daily_changes, 0.0)

        # Average gains and losses
        avg_gains = pd.Series(gains, index=source.index).rolling(window=self.interval).mean()
        avg_losses = pd.Series(losses, index=source.index).rolling(window=self.interval).mean()

        # RSI computation
        rs = avg_gains / (avg_losses + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))

        # Honors rubric rules: Buy when RSI drops below 30; Sell when RSI rises above 70
        conditions = [
            rsi < 30,
            rsi > 70
        ]
        choices = [1, 0]

        sig = pd.Series(np.select(conditions, choices, default=np.nan), index=source.index)
        sig = sig.ffill().fillna(0).astype(int)

        self.data[f'RSI_{self.interval}'] = rsi.loc[self.data.index]
        self.data['signal'] = sig.loc[self.data.index].values
        return self.data['signal']

class MLHybridStrategy(Strategy):
    """
    Hybrid Strategy:
    Integrates the ML Market Regime Classifier as an independent trade filter.
    Only allows trade positions (signal = 1) when technical trend conditions are active
    (either the 20/50 SMA crossover is active OR price is above the fast 20-day SMA)
    AND the ML classifier independently predicts a Bullish regime (1).
    """
    def __init__(self, asset_data, base_strategy, ml_predictions):
        super().__init__(asset_data)
        self.base_strategy = base_strategy
        ml_arr = np.asarray(ml_predictions).ravel()

        if np.issubdtype(ml_arr.dtype, np.floating) and np.any((ml_arr > 0.0) & (ml_arr < 1.0)):
            warnings.warn(
                "ml_predictions contains continuous probabilities in [0, 1] rather than binary integers {0, 1}. "
                "Automatically thresholding at 0.5.",
                UserWarning,
                stacklevel=2,
            )
            ml_arr = (ml_arr >= 0.5).astype(int)
        self.ml_predictions = ml_arr

        # Determine reference length from base_strategy or self.data
        ref_len = None
        if hasattr(self.base_strategy, 'signal') and self.base_strategy.signal is not None:
            ref_len = len(self.base_strategy.signal)
        elif hasattr(self.base_strategy, 'data') and self.base_strategy.data is not None:
            ref_len = len(self.base_strategy.data)
        elif self.data is not None:
            ref_len = len(self.data)

        if ref_len is not None and len(self.ml_predictions) < ref_len:
            warnings.warn(
                f"Length of ml_predictions ({len(self.ml_predictions)}) is less than base strategy length ({ref_len}). "
                f"Missing leading {ref_len - len(self.ml_predictions)} days will default to cash (0).",
                UserWarning,
                stacklevel=2,
            )

    def generate_signal(self):
        # 1. Run underlying technical strategy signal logic
        self.base_strategy.generate_signal()
        base_signal = self.base_strategy.data['signal'].values

        # Allow entry when either the 20/50 MA crossover is active OR price has recovered above the 20-day MA,
        # strictly gated by the ML Classifier predicting a Bullish (1) market regime
        if 'short_average_close' in self.base_strategy.data.columns:
            fast_trend = (self.base_strategy.data['Close'].values > self.base_strategy.data['short_average_close'].values).astype(int)
            tech_active = (base_signal == 1) | (fast_trend == 1)
        else:
            tech_active = (base_signal == 1)

        # 2. Align lengths if operating on a subset (e.g. test set)
        if len(base_signal) == len(self.ml_predictions):
            hybrid_signal = np.where(tech_active & (self.ml_predictions == 1), 1, 0)
        else:
            min_len = min(len(base_signal), len(self.ml_predictions))
            hybrid_signal = np.zeros_like(base_signal)
            hybrid_signal[-min_len:] = np.where(
                tech_active[-min_len:] & (self.ml_predictions[-min_len:] == 1), 1, 0
            )

        self.data['signal'] = hybrid_signal
        return self.data['signal']
