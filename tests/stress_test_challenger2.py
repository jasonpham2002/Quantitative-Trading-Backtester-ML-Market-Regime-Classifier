"""
Empirical Adversarial Stress Test Suite for Milestone 1 Preprocessing Pipeline.
Author: Challenger 2 Agent
"""

import sys
import unittest
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "Market_Regime_Prediction"))
sys.path.insert(0, str(PROJECT_ROOT / "Trading Strategies Backtesting System"))
sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing_pipeline import Pipeline


class TestStressConstantPriceAndVol(unittest.TestCase):
    """Stress Test Suite 1: Synthetic constant prices, zero volatility, and zero/missing volume."""

    def test_1_1_constant_price_extended_pipeline_finite_features(self):
        """Constant price (100.0) across 150 days produces finite values (no crash, no unhandled inf/nan)."""
        dates = pd.bdate_range('2023-01-01', periods=150)
        df = pd.DataFrame({
            'Open': 100.0,
            'High': 100.0,
            'Low': 100.0,
            'Close': 100.0,
            'Volume': 1e6
        }, index=dates)

        pipe = Pipeline(df, feature_set='extended')
        clean = pipe.clean_and_prepare()

        self.assertGreater(len(clean), 0, "Clean dataframe should have rows after warmup")
        for feat in pipe.feature_names:
            self.assertFalse(np.isnan(clean[feat]).any(), f"Feature {feat} contains NaN for constant prices")
            self.assertFalse(np.isinf(clean[feat]).any(), f"Feature {feat} contains Inf for constant prices")

        np.testing.assert_allclose(clean['vol_10d'].values, 0.0, atol=1e-7)
        np.testing.assert_allclose(clean['vol_20d'].values, 0.0, atol=1e-7)
        np.testing.assert_allclose(clean['ret_1d'].values, 0.0, atol=1e-7)
        np.testing.assert_allclose(clean['ret_5d'].values, 0.0, atol=1e-7)
        np.testing.assert_allclose(clean['ret_10d'].values, 0.0, atol=1e-7)
        np.testing.assert_allclose(clean['candle_body'].values, 0.0, atol=1e-7)
        np.testing.assert_allclose(clean['intraday_range'].values, 0.0, atol=1e-7)
        np.testing.assert_allclose(clean['sma20_ratio'].values, 1.0, atol=1e-7)
        np.testing.assert_allclose(clean['sma50_ratio'].values, 1.0, atol=1e-7)
        np.testing.assert_allclose(clean['macd_norm'].values, 0.0, atol=1e-7)
        np.testing.assert_allclose(clean['macd_hist_norm'].values, 0.0, atol=1e-7)
        np.testing.assert_allclose(clean['vol_regime'].values, 0.0, atol=1e-7)
        np.testing.assert_allclose(clean['rsi_14'].values, 0.0, atol=1e-7)

    def test_1_2_constant_price_split_and_scaling(self):
        """Constant prices have std=0 in train split; Z-score scaling must clamp std to 1e-10 without NaN or Inf."""
        dates = pd.bdate_range('2023-01-01', periods=150)
        df = pd.DataFrame({
            'Open': 100.0,
            'High': 100.0,
            'Low': 100.0,
            'Close': 100.0,
            'Volume': 1e6
        }, index=dates)

        pipe = Pipeline(df, feature_set='extended')
        X_tr, X_val, X_te, y_tr, y_val, y_te = pipe.train_val_test_split_scale(0.65, 0.15)

        self.assertFalse(np.isnan(X_tr).any(), "X_tr contains NaN under constant price scaling")
        self.assertFalse(np.isinf(X_tr).any(), "X_tr contains Inf under constant price scaling")
        self.assertFalse(np.isnan(X_val).any(), "X_val contains NaN under constant price scaling")
        self.assertFalse(np.isnan(X_te).any(), "X_te contains NaN under constant price scaling")
        np.testing.assert_allclose(X_tr, 0.0, atol=1e-5)

    def test_1_3_missing_volume_column(self):
        """Pipeline operates gracefully when Volume column is absent."""
        dates = pd.bdate_range('2023-01-01', periods=100)
        df = pd.DataFrame({
            'Open': np.linspace(100, 110, 100),
            'High': np.linspace(101, 111, 100),
            'Low': np.linspace(99, 109, 100),
            'Close': np.linspace(100, 110, 100)
        }, index=dates)

        pipe = Pipeline(df, feature_set='extended')
        clean = pipe.clean_and_prepare()
        self.assertIn('volume_ratio', clean.columns)
        self.assertFalse(np.isnan(clean['volume_ratio']).any())
        self.assertFalse(np.isinf(clean['volume_ratio']).any())

    def test_1_4_zero_volume_everywhere(self):
        """Volume=0 everywhere does not trigger ZeroDivisionError in volume_ratio."""
        dates = pd.bdate_range('2023-01-01', periods=100)
        df = pd.DataFrame({
            'Open': np.linspace(100, 110, 100),
            'High': np.linspace(101, 111, 100),
            'Low': np.linspace(99, 109, 100),
            'Close': np.linspace(100, 110, 100),
            'Volume': 0.0
        }, index=dates)

        pipe = Pipeline(df, feature_set='extended')
        clean = pipe.clean_and_prepare()
        self.assertFalse(np.isnan(clean['volume_ratio']).any())
        self.assertFalse(np.isinf(clean['volume_ratio']).any())
        np.testing.assert_allclose(clean['volume_ratio'].values, 0.0, atol=1e-7)

    def test_1_5_close_only_dataframe(self):
        """Pipeline handles DataFrame with only 'Close' column (Open, High, Low missing)."""
        dates = pd.bdate_range('2023-01-01', periods=100)
        df = pd.DataFrame({'Close': np.linspace(100, 150, 100)}, index=dates)

        pipe = Pipeline(df, feature_set='extended')
        clean = pipe.clean_and_prepare()
        self.assertEqual(len(clean), 100 - 49 - 5)
        self.assertFalse(clean[pipe.feature_names].isna().any().any())


class TestStressPriceShocksAndOutliers(unittest.TestCase):
    """Stress Test Suite 2: Extreme price shocks, volume spikes, and outliers."""

    def test_2_1_extreme_price_crash_99_percent(self):
        """Single-day 99% crash ($500 -> $5) maintains finite features."""
        n = 120
        dates = pd.bdate_range('2023-01-01', periods=n)
        prices = np.full(n, 500.0)
        prices[60:] = 5.0
        df = pd.DataFrame({
            'Open': prices,
            'High': prices,
            'Low': prices,
            'Close': prices,
            'Volume': 1e6
        }, index=dates)

        pipe = Pipeline(df, feature_set='extended')
        clean = pipe.clean_and_prepare()
        self.assertFalse(np.isnan(clean[pipe.feature_names].values).any())
        self.assertFalse(np.isinf(clean[pipe.feature_names].values).any())

    def test_2_2_extreme_price_jump_1000_percent(self):
        """Single-day 1000% jump ($10 -> $110) maintains finite features."""
        n = 120
        dates = pd.bdate_range('2023-01-01', periods=n)
        prices = np.full(n, 10.0)
        prices[60:] = 110.0
        df = pd.DataFrame({
            'Open': prices,
            'High': prices,
            'Low': prices,
            'Close': prices,
            'Volume': 1e6
        }, index=dates)

        pipe = Pipeline(df, feature_set='extended')
        clean = pipe.clean_and_prepare()
        self.assertFalse(np.isnan(clean[pipe.feature_names].values).any())
        self.assertFalse(np.isinf(clean[pipe.feature_names].values).any())

    def test_2_3_extreme_volume_spike_1000x(self):
        """Volume spike of 1,000x baseline produces finite volume_ratio."""
        n = 120
        dates = pd.bdate_range('2023-01-01', periods=n)
        vol = np.full(n, 1e5)
        vol[70] = 1e8
        df = pd.DataFrame({
            'Open': 100.0,
            'High': 101.0,
            'Low': 99.0,
            'Close': 100.0,
            'Volume': vol
        }, index=dates)

        pipe = Pipeline(df, feature_set='extended')
        clean = pipe.clean_and_prepare()
        self.assertFalse(np.isnan(clean['volume_ratio']).any())
        self.assertFalse(np.isinf(clean['volume_ratio']).any())
        self.assertGreater(clean.loc[dates[70], 'volume_ratio'], 10.0)

    def test_2_4_nan_in_raw_data_interior(self):
        """Interior NaNs in raw data are dropped by clean_and_prepare without corrupting shapes."""
        n = 120
        dates = pd.bdate_range('2023-01-01', periods=n)
        df = pd.DataFrame({
            'Open': 100.0,
            'High': 101.0,
            'Low': 99.0,
            'Close': 100.0,
            'Volume': 1e6
        }, index=dates)
        df.loc[dates[65], 'Close'] = np.nan

        pipe = Pipeline(df, feature_set='extended')
        clean = pipe.clean_and_prepare()
        self.assertFalse(clean.isna().any().any())
        self.assertNotIn(dates[65], clean.index)


class TestStressMultiIndexFormats(unittest.TestCase):
    """Stress Test Suite 3: MultiIndex configurations, level swapping, multi-ticker, and index sorting."""

    def test_3_1_multiindex_price_level_0_ticker_level_1(self):
        """Format 1: ('Close', 'SPY') where Price is level 0, Ticker is level 1."""
        dates = pd.bdate_range('2023-01-01', periods=100)
        cols = pd.MultiIndex.from_tuples(
            [(c, 'SPY') for c in ['Open', 'High', 'Low', 'Close', 'Volume']],
            names=['Price', 'Ticker']
        )
        data = np.ones((100, 5)) * 100.0
        df = pd.DataFrame(data, index=dates, columns=cols)

        pipe = Pipeline(df, ticker='SPY', feature_set='extended')
        clean = pipe.clean_and_prepare()
        self.assertIn('Close', pipe.df.columns)
        self.assertEqual(len(clean), 100 - 49 - 5)

    def test_3_2_multiindex_ticker_level_0_price_level_1(self):
        """Format 2: ('SPY', 'Close') where Ticker is level 0, Price is level 1."""
        dates = pd.bdate_range('2023-01-01', periods=100)
        cols = pd.MultiIndex.from_tuples(
            [('SPY', c) for c in ['Open', 'High', 'Low', 'Close', 'Volume']],
            names=['Ticker', 'Price']
        )
        data = np.ones((100, 5)) * 100.0
        df = pd.DataFrame(data, index=dates, columns=cols)

        pipe = Pipeline(df, ticker='SPY', feature_set='extended')
        clean = pipe.clean_and_prepare()
        self.assertIn('Close', pipe.df.columns)
        self.assertEqual(len(clean), 100 - 49 - 5)

    def test_3_3_multiindex_multi_ticker_selection(self):
        """Multi-ticker DataFrame: Selecting QQQ isolates QQQ data from SPY."""
        dates = pd.bdate_range('2023-01-01', periods=100)
        cols = pd.MultiIndex.from_tuples(
            [('Close', 'SPY'), ('Close', 'QQQ'),
             ('Open', 'SPY'), ('Open', 'QQQ'),
             ('High', 'SPY'), ('High', 'QQQ'),
             ('Low', 'SPY'), ('Low', 'QQQ'),
             ('Volume', 'SPY'), ('Volume', 'QQQ')],
            names=['Price', 'Ticker']
        )
        data = np.zeros((100, 10))
        data[:, 0] = 100.0  # Close SPY
        data[:, 1] = 300.0  # Close QQQ
        data[:, 2] = 100.0  # Open SPY
        data[:, 3] = 300.0  # Open QQQ
        data[:, 4] = 101.0  # High SPY
        data[:, 5] = 301.0  # High QQQ
        data[:, 6] = 99.0   # Low SPY
        data[:, 7] = 299.0  # Low QQQ
        data[:, 8] = 1e6    # Vol SPY
        data[:, 9] = 2e6    # Vol QQQ

        df = pd.DataFrame(data, index=dates, columns=cols)
        pipe_qqq = Pipeline(df, ticker='QQQ', feature_set='extended')
        clean_qqq = pipe_qqq.clean_and_prepare()

        np.testing.assert_allclose(clean_qqq['Close'].values, 300.0)

    def test_3_4_multiindex_ticker_none_defaults_to_first(self):
        """When ticker=None on MultiIndex, defaults to first detected ticker."""
        dates = pd.bdate_range('2023-01-01', periods=100)
        cols = pd.MultiIndex.from_tuples(
            [('Close', 'AAPL'), ('Open', 'AAPL'), ('High', 'AAPL'), ('Low', 'AAPL'), ('Volume', 'AAPL')],
            names=['Price', 'Ticker']
        )
        df = pd.DataFrame(np.ones((100, 5)) * 150.0, index=dates, columns=cols)
        pipe = Pipeline(df, ticker=None)
        clean = pipe.clean_and_prepare()
        self.assertIn('Close', pipe.df.columns)
        np.testing.assert_allclose(clean['Close'].values, 150.0)

    def test_3_5_shuffled_datetime_index_auto_sorted(self):
        """Shuffled datetime index is automatically sorted, matching sequential output."""
        np.random.seed(42)
        n = 100
        dates = pd.bdate_range('2023-01-01', periods=n)
        prices = 100.0 + np.cumsum(np.random.randn(n))
        df_seq = pd.DataFrame({
            'Open': prices,
            'High': prices + 1.0,
            'Low': prices - 1.0,
            'Close': prices,
            'Volume': 1e6
        }, index=dates)

        perm = np.random.permutation(n)
        df_shuffled = df_seq.iloc[perm].copy()
        self.assertFalse(df_shuffled.index.is_monotonic_increasing)

        pipe_seq = Pipeline(df_seq, feature_set='extended')
        clean_seq = pipe_seq.clean_and_prepare()

        pipe_shuffled = Pipeline(df_shuffled, feature_set='extended')
        clean_shuffled = pipe_shuffled.clean_and_prepare()

        self.assertTrue(pipe_shuffled.df.index.is_monotonic_increasing)
        self.assertEqual(len(clean_seq), len(clean_shuffled))
        np.testing.assert_allclose(clean_seq['Close'].values, clean_shuffled['Close'].values)
        np.testing.assert_allclose(clean_seq['vol_20d'].values, clean_shuffled['vol_20d'].values)


class TestStressCreateSequences(unittest.TestCase):
    """Stress Test Suite 4: create_sequences varying sequence lengths, memory independence, and boundaries."""

    def setUp(self):
        np.random.seed(42)
        self.N = 100
        self.F = 14
        self.features = np.arange(self.N * self.F, dtype=float).reshape(self.N, self.F)
        self.targets = np.arange(self.N, dtype=int).reshape(-1, 1)

    def test_4_1_varying_seq_len_no_context(self):
        """Varying seq_len (L=1, 5, 20, 50, 100) without context produces exact window counts (N - L + 1)."""
        for L in [1, 5, 20, 50, 100]:
            X_seq, y_seq = Pipeline.create_sequences(self.features, self.targets, seq_len=L)
            expected_windows = self.N - L + 1
            self.assertEqual(X_seq.shape, (expected_windows, L, self.F), f"Failed for L={L}")
            self.assertEqual(y_seq.shape, (expected_windows, 1), f"Target shape failed for L={L}")
            np.testing.assert_array_equal(X_seq[0, -1, :], self.features[L - 1, :])
            np.testing.assert_array_equal(X_seq[-1, -1, :], self.features[-1, :])
            self.assertEqual(y_seq[0, 0], self.targets[L - 1, 0])
            self.assertEqual(y_seq[-1, 0], self.targets[-1, 0])

    def test_4_2_varying_seq_len_with_context(self):
        """Varying seq_len (L=1, 5, 20, 50) with context produces exactly N windows."""
        for L in [1, 5, 20, 50]:
            prev_context = np.random.randn(L + 10, self.F)
            X_seq, y_seq = Pipeline.create_sequences(self.features, self.targets, seq_len=L, prev_features_2d=prev_context)
            self.assertEqual(X_seq.shape, (self.N, L, self.F), f"Shape failed for L={L} with context")
            self.assertEqual(y_seq.shape, (self.N, 1), f"Target shape failed for L={L} with context")
            np.testing.assert_array_equal(X_seq[0, -1, :], self.features[0, :])
            np.testing.assert_array_equal(X_seq[-1, -1, :], self.features[-1, :])
            np.testing.assert_array_equal(y_seq, self.targets)

    def test_4_3_memory_independence_input_mutation(self):
        """Mutating original features array after create_sequences does NOT mutate X_seq."""
        features_copy = self.features.copy()
        X_seq = Pipeline.create_sequences(features_copy, seq_len=20)
        original_val = X_seq[0, 0, 0]

        features_copy[0, 0] = 999999.0
        self.assertEqual(X_seq[0, 0, 0], original_val, "X_seq was mutated when input features were modified!")

    def test_4_4_memory_independence_internal_overlap_mutation(self):
        """Mutating one sequence window in X_seq does NOT corrupt adjacent overlapping windows."""
        X_seq = Pipeline.create_sequences(self.features, seq_len=20)
        val_in_window_1 = X_seq[1, 0, 0]

        X_seq[0, 1, 0] = 888888.0
        self.assertEqual(
            X_seq[1, 0, 0],
            val_in_window_1,
            "X_seq overlapping memory corruption! Mutating X_seq[0, 1] modified X_seq[1, 0]!"
        )

    def test_4_5_buffer_owndata_and_contiguity(self):
        """X_seq must be C-contiguous and own its memory buffer."""
        X_seq = Pipeline.create_sequences(self.features, seq_len=20)
        self.assertTrue(X_seq.flags.c_contiguous, "X_seq is not C-contiguous")
        self.assertTrue(X_seq.flags.owndata, "X_seq does not own its memory buffer")

    def test_4_6_context_exact_boundary_length(self):
        """Context with exactly L - 1 rows succeeds."""
        L = 20
        exact_context = np.random.randn(L - 1, self.F)
        X_seq = Pipeline.create_sequences(self.features, seq_len=L, prev_features_2d=exact_context)
        self.assertEqual(X_seq.shape, (self.N, L, self.F))

    def test_4_7_context_shorter_than_l_minus_1_raises(self):
        """Context with L - 2 rows raises ValueError."""
        L = 20
        short_context = np.random.randn(L - 2, self.F)
        with self.assertRaises(ValueError):
            Pipeline.create_sequences(self.features, seq_len=L, prev_features_2d=short_context)

    def test_4_8_feature_dimension_mismatch_raises(self):
        """Context with different feature count F raises ValueError."""
        L = 20
        mismatched_context = np.random.randn(L - 1, self.F + 1)
        with self.assertRaises(ValueError):
            Pipeline.create_sequences(self.features, seq_len=L, prev_features_2d=mismatched_context)

    def test_4_9_targets_length_mismatch_raises(self):
        """targets_1d with length != N raises ValueError."""
        with self.assertRaises(ValueError):
            Pipeline.create_sequences(self.features, targets_1d=np.array([1, 2, 3]), seq_len=20)

    def test_4_10_target_1d_and_2d_formats(self):
        """create_sequences accepts both (N,) 1D array and (N, 1) 2D array for targets."""
        y_1d = self.targets.ravel()
        X_seq, y_seq_1d = Pipeline.create_sequences(self.features, targets_1d=y_1d, seq_len=20)
        self.assertEqual(y_seq_1d.ndim, 1)
        self.assertEqual(len(y_seq_1d), len(X_seq))

        X_seq, y_seq_2d = Pipeline.create_sequences(self.features, targets_1d=self.targets, seq_len=20)
        self.assertEqual(y_seq_2d.ndim, 2)
        self.assertEqual(len(y_seq_2d), len(X_seq))

    def test_4_11_negative_seq_len_raises_error(self):
        """Negative sequence length raises ValueError."""
        with self.assertRaises(ValueError):
            Pipeline.create_sequences(self.features, seq_len=-5)

    def test_4_12_seq_len_exceeds_sample_size(self):
        """seq_len > N without context raises ValueError."""
        with self.assertRaises(ValueError):
            Pipeline.create_sequences(self.features, seq_len=self.N + 1)


class TestStressSplitAndScalingBoundaries(unittest.TestCase):
    """Stress Test Suite 5: train_val_test_split_scale edge ratios and zero leakage."""

    def test_5_1_zero_leakage_outlier_isolation(self):
        """Extreme outlier in test set does not alter train split mean_ or std_."""
        np.random.seed(42)
        n = 200
        dates = pd.bdate_range('2023-01-01', periods=n)
        prices = 100.0 + np.cumsum(np.random.randn(n))
        df = pd.DataFrame({
            'Open': prices,
            'High': prices + 1.0,
            'Low': prices - 1.0,
            'Close': prices,
            'Volume': 1e6
        }, index=dates)

        pipe_normal = Pipeline(df, feature_set='extended')
        X_tr1, X_val1, X_te1, _, _, _ = pipe_normal.train_val_test_split_scale(0.65, 0.15)
        mean_normal = pipe_normal.mean_.copy()
        std_normal = pipe_normal.std_.copy()

        df_contaminated = df.copy()
        df_contaminated.iloc[-5:, df_contaminated.columns.get_loc('Close')] = 999999.0

        pipe_outlier = Pipeline(df_contaminated, feature_set='extended')
        X_tr2, X_val2, X_te2, _, _, _ = pipe_outlier.train_val_test_split_scale(0.65, 0.15)

        np.testing.assert_array_equal(mean_normal, pipe_outlier.mean_)
        np.testing.assert_array_equal(std_normal, pipe_outlier.std_)
        np.testing.assert_array_equal(X_tr1, X_tr2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
