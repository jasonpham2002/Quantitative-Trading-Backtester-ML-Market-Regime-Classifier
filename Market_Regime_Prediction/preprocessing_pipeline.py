import numpy as np
import pandas as pd

class Pipeline:
    """
    Feature engineering and preprocessing pipeline for Market Regime Prediction.
    Prepares financial time series data for machine learning by engineering features,
    generating forward-looking target labels, and performing lookahead-free standardization.
    """
    def __init__(self, dataset, ticker=None, feature_set='basic'):
        self.raw_data = dataset.copy()
        self.feature_set = feature_set
        
        # Handle MultiIndex columns using df.xs() if multi-level columns exist
        if isinstance(self.raw_data.columns, pd.MultiIndex):
            price_cols = {'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'}
            # Determine which level holds ticker symbols vs OHLCV price fields
            level_0_vals = set(self.raw_data.columns.get_level_values(0))
            ticker_level = 1 if price_cols.intersection(level_0_vals) else 0
            if ticker is not None:
                self.df = self.raw_data.xs(ticker, axis=1, level=ticker_level).copy()
            else:
                first_ticker = self.raw_data.columns.get_level_values(ticker_level)[0]
                self.df = self.raw_data.xs(first_ticker, axis=1, level=ticker_level).copy()
        else:
            self.df = self.raw_data.copy()

        # Ensure index is datetime sorted
        if not self.df.index.is_monotonic_increasing:
            self.df = self.df.sort_index()

        if self.feature_set == 'extended':
            self.feature_names = [
                'candle_body', 'intraday_range', 'volume_ratio',
                'ret_1d', 'ret_5d', 'ret_10d',
                'vol_10d', 'vol_20d',
                'sma20_ratio', 'sma50_ratio',
                'rsi_14',
                'macd_norm', 'macd_hist_norm',
                'vol_regime'
            ]
        else:
            self.feature_names = ['ret_5d', 'vol_20d', 'rsi_14', 'sma50_ratio']

        self.clean_df = None
        self.train_df = None
        self.val_df = None
        self.test_df = None
        self.mean_ = None
        self.std_ = None

    def generate_features(self):
        """
        Calculates technical features for market regime classification.
        Supports feature_set='basic' (4 features) and feature_set='extended' (14 stationary features).
        """
        close = self.df['Close']
        open_p = self.df['Open'] if 'Open' in self.df.columns else close
        high = self.df['High'] if 'High' in self.df.columns else close
        low = self.df['Low'] if 'Low' in self.df.columns else close
        volume = self.df['Volume'] if 'Volume' in self.df.columns else pd.Series(1.0, index=self.df.index)

        # Core shared calculations
        daily_returns = close.pct_change()
        self.df['ret_5d'] = close.pct_change(5)
        self.df['vol_20d'] = daily_returns.rolling(window=20).std()

        # 14-period RSI using NumPy delta separation
        delta = close.diff()
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)

        # Compute rolling average gain and loss
        roll_gain = pd.Series(gain, index=self.df.index).rolling(window=14).mean()
        roll_loss = pd.Series(loss, index=self.df.index).rolling(window=14).mean()

        rs = roll_gain / (roll_loss + 1e-10)
        self.df['rsi_14'] = 100.0 - (100.0 / (1.0 + rs))

        # 50-day SMA ratio
        sma_50 = close.rolling(window=50).mean()
        self.df['sma50_ratio'] = close / sma_50

        if self.feature_set == 'extended':
            # 1. OHLCV Stationary ratios
            self.df['candle_body'] = (close - open_p) / open_p
            self.df['intraday_range'] = (high - low) / close
            vol_sma20 = volume.rolling(window=20).mean()
            self.df['volume_ratio'] = volume / (vol_sma20 + 1e-10)

            # 2. Multi-horizon returns
            self.df['ret_1d'] = daily_returns
            self.df['ret_10d'] = close.pct_change(10)

            # 3. Rolling volatility
            self.df['vol_10d'] = daily_returns.rolling(window=10).std()

            # 4. Moving average ratios
            sma_20 = close.rolling(window=20).mean()
            self.df['sma20_ratio'] = close / sma_20

            # 5. MACD normalized
            ema_12 = close.ewm(span=12, adjust=False).mean()
            ema_26 = close.ewm(span=26, adjust=False).mean()
            macd = ema_12 - ema_26
            self.df['macd_norm'] = macd / close

            macd_sig = macd.ewm(span=9, adjust=False).mean()
            macd_hist = macd - macd_sig
            self.df['macd_hist_norm'] = macd_hist / close

            # 6. Volatility regime ratio
            self.df['vol_regime'] = self.df['vol_10d'] / (self.df['vol_20d'] + 1e-10)

        return self.df

    def generate_target(self, horizon=5):
        """
        Creates forward-looking market regime target label over `horizon` trading days:
        Target = 1 (Bullish Regime) if Close[t + horizon] > SMA_50[t], else 0 (Bearish Regime).
        Using the 50-day trend regime prevents short-term 5-day return mean-reversion from
        inverting the learned weights on trend and momentum features.
        """
        future_close = self.df['Close'].shift(-horizon)
        sma_50 = self.df['Close'].rolling(window=50).mean()
        forward_return = (future_close - self.df['Close']) / self.df['Close']

        # Binary regime label: 1 if forward price remains in a bullish trend above SMA_50, 0 otherwise
        # Note: final `horizon` rows will be NaN because future_close is NaN
        self.df['forward_return'] = forward_return
        self.df['target'] = np.where(future_close > sma_50, 1, 0)
        # Explicitly restore NaN for the unobservable future tail or initial warmup
        self.df.loc[future_close.isna() | sma_50.isna(), 'target'] = np.nan

        return self.df

    def clean_and_prepare(self):
        """
        Removes boundary NaN values generated by lookback indicators (first 50 rows)
        and lookahead target generation (last 5 rows).
        """
        self.df = self.df.replace([np.inf, -np.inf], np.nan)
        if not all(col in self.df.columns for col in self.feature_names):
            self.generate_features()
        if 'target' not in self.df.columns:
            self.generate_target()
        all_cols = self.feature_names + ['target', 'Close', 'forward_return']
        self.df = self.df.replace([np.inf, -np.inf], np.nan)
        self.clean_df = self.df.dropna(subset=all_cols).copy()
        self.clean_df['target'] = self.clean_df['target'].astype(int)
        return self.clean_df

    def train_test_split_scale(self, train_ratio=0.8):
        """
        Performs strict chronological train/test split and zero-leakage Z-score standardization.
        Standardization statistics (mean, std) are computed ONLY from training data.
        
        Returns:
            X_train_scaled, X_test_scaled, y_train, y_test
        """
        if self.clean_df is None:
            self.clean_and_prepare()

        split_idx = int(len(self.clean_df) * train_ratio)
        self.train_df = self.clean_df.iloc[:split_idx].copy()
        self.test_df = self.clean_df.iloc[split_idx:].copy()

        # Extract numpy arrays
        X_train_raw = self.train_df[self.feature_names].values
        X_test_raw = self.test_df[self.feature_names].values

        y_train = self.train_df['target'].values.reshape(-1, 1)
        y_test = self.test_df['target'].values.reshape(-1, 1)

        # Zero-leakage standardization: compute mu and sigma on training data only
        self.mean_ = np.mean(X_train_raw, axis=0)
        self.std_ = np.std(X_train_raw, axis=0)
        # Avoid division by zero
        self.std_[self.std_ == 0] = 1e-10

        X_train_scaled = (X_train_raw - self.mean_) / self.std_
        X_test_scaled = (X_test_raw - self.mean_) / self.std_

        return X_train_scaled, X_test_scaled, y_train, y_test

    def train_val_test_split_scale(self, train_ratio=0.65, val_ratio=0.15):
        """
        Performs chronological 65% Train / 15% Val / 20% Test split.
        Fits Z-score standardization parameters strictly on the Training split (first 65%),
        and applies them to Train, Val, and Test sets to prevent forward lookahead leakage.
        
        Returns:
            X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test
        """
        if not (train_ratio > 0 and val_ratio > 0 and (train_ratio + val_ratio) < 1.0):
            raise ValueError(
                f"Invalid split ratios: train_ratio={train_ratio}, val_ratio={val_ratio}. "
                "train_ratio and val_ratio must be positive, and their sum must be strictly less than 1.0."
            )

        if self.clean_df is None:
            self.clean_and_prepare()

        n = len(self.clean_df)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        self.train_df = self.clean_df.iloc[:train_end].copy()
        self.val_df = self.clean_df.iloc[train_end:val_end].copy()
        self.test_df = self.clean_df.iloc[val_end:].copy()

        X_train_raw = self.train_df[self.feature_names].values
        X_val_raw = self.val_df[self.feature_names].values
        X_test_raw = self.test_df[self.feature_names].values

        y_train = self.train_df['target'].values.reshape(-1, 1)
        y_val = self.val_df['target'].values.reshape(-1, 1)
        y_test = self.test_df['target'].values.reshape(-1, 1)

        # Zero-leakage standardization: compute mu and sigma on training data only
        self.mean_ = np.mean(X_train_raw, axis=0)
        self.std_ = np.std(X_train_raw, axis=0)
        self.std_[self.std_ == 0] = 1e-10

        X_train_scaled = (X_train_raw - self.mean_) / self.std_
        X_val_scaled = (X_val_raw - self.mean_) / self.std_
        X_test_scaled = (X_test_raw - self.mean_) / self.std_

        return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test

    @staticmethod
    def create_sequences(features_2d, targets_1d=None, seq_len=20, prev_features_2d=None):
        """
        Converts 2D tabular features (N, F) into 3D sequence tensors (num_windows, seq_len, F).
        
        Parameters:
            features_2d (np.ndarray or pd.DataFrame): 2D array of shape (N, F).
            targets_1d (np.ndarray or pd.Series, optional): 1D array of labels (N,) or (N, 1).
            seq_len (int): Length of lookback window (default 20).
            prev_features_2d (np.ndarray or pd.DataFrame, optional): Historical context rows
                from the immediately prior chronological split (at least seq_len - 1 rows).
                When provided, prepends context so output window count is exactly N.
                
        Returns:
            X_seq (np.ndarray): 3D array of shape (num_windows, seq_len, F).
            y_seq (np.ndarray, optional): Array of targets matching each sequence window.
                Returned if targets_1d is not None.
        """
        if not isinstance(seq_len, int) or isinstance(seq_len, bool) or seq_len < 1:
            raise ValueError(f"seq_len must be a positive integer >= 1, got {seq_len}")

        if isinstance(features_2d, (pd.DataFrame, pd.Series)):
            features_2d = features_2d.values
        if prev_features_2d is not None and isinstance(prev_features_2d, (pd.DataFrame, pd.Series)):
            prev_features_2d = prev_features_2d.values
        if targets_1d is not None and isinstance(targets_1d, (pd.DataFrame, pd.Series)):
            targets_1d = targets_1d.values

        features_2d = np.asarray(features_2d, dtype=float)
        if features_2d.ndim != 2:
            raise ValueError(f"features_2d must be a 2D array, got ndim={features_2d.ndim}")

        N, F = features_2d.shape

        if targets_1d is not None:
            targets_1d = np.asarray(targets_1d)
            if len(targets_1d) != N:
                raise ValueError(f"features_2d length ({N}) and targets_1d length ({len(targets_1d)}) mismatch")

        if prev_features_2d is not None:
            prev_features_2d = np.asarray(prev_features_2d, dtype=float)
            if prev_features_2d.ndim != 2 or prev_features_2d.shape[1] != F:
                raise ValueError(f"prev_features_2d shape mismatch: expected (_, {F}), got {prev_features_2d.shape}")
            if len(prev_features_2d) < seq_len - 1:
                raise ValueError(f"prev_features_2d must have at least {seq_len - 1} rows, got {len(prev_features_2d)}")

            context = prev_features_2d[-(seq_len - 1):] if seq_len > 1 else np.empty((0, F))
            full_features = np.vstack([context, features_2d])
            trim_target = False
        else:
            full_features = features_2d
            trim_target = True

        N_full = len(full_features)
        num_seqs = N_full - seq_len + 1
        if num_seqs <= 0:
            raise ValueError(f"Not enough rows ({N_full}) to create sequences of length {seq_len}")

        # Vectorized sliding window creation via NumPy strided memory view
        shape = (num_seqs, seq_len, F)
        strides = (full_features.strides[0], full_features.strides[0], full_features.strides[1])
        X_seq = np.lib.stride_tricks.as_strided(full_features, shape=shape, strides=strides).copy()

        if targets_1d is not None:
            y_seq = targets_1d[seq_len - 1:] if trim_target else targets_1d
            return X_seq, y_seq
        return X_seq

    def get_test_dataframe(self):
        """Returns the unscaled test DataFrame with intact dates and prices."""
        return self.test_df

    def get_val_dataframe(self):
        """Returns the unscaled validation DataFrame with intact dates and prices."""
        return self.val_df