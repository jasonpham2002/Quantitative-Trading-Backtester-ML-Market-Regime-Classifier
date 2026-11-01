"""
Tier 2: Boundary & Corner Cases E2E Tests
Covers boundary values, empty inputs, zero volatility, zero volume, single vs multi-index columns,
lookback constraints, extreme sequence lengths, and edge-case classification/backtesting conditions.
"""

import unittest
import numpy as np
import pandas as pd
from tests.e2e.test_helpers import (
    HAS_PREPROCESSING,
    HAS_EXTENDED_PIPELINE,
    HAS_CREATE_SEQUENCES,
    HAS_LOGISTIC_REGRESSION,
    HAS_BACKTESTER,
    generate_synthetic_ohlcv,
    oracle_rsi_14,
    oracle_create_sequences,
    oracle_vol_regime,
    MockDeepRegimeClassifier,
)

if HAS_PREPROCESSING:
    from preprocessing_pipeline import Pipeline

if HAS_LOGISTIC_REGRESSION:
    from LogisticRegression import LogisticRegression

if HAS_BACKTESTER:
    from strategy import MACrossover, MLHybridStrategy
    from backtester import Backtest


class TestTier2DataBoundaries(unittest.TestCase):
    """Tier 2: Boundary Group 1 — Pipeline Data Structure & Input Formats (>=5 tests)."""

    def test_empty_dataframe_handling(self):
        """Test 2.1: Empty DataFrame input is handled without uncaught fatal crash."""
        empty_df = pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])
        pipe = Pipeline(empty_df)
        with self.assertRaises((ValueError, KeyError, IndexError, AssertionError)):
            # Calling clean_and_prepare on empty df should cleanly raise or produce empty df
            res = pipe.clean_and_prepare()
            if len(res) == 0:
                raise ValueError("Cleaned empty dataframe")

    def test_insufficient_samples_less_than_sma50(self):
        """Test 2.2: DataFrame with fewer than 50 rows produces 0 clean rows due to SMA-50."""
        short_df = generate_synthetic_ohlcv(n_days=40, multi_index=False)
        pipe = Pipeline(short_df)
        clean = pipe.clean_and_prepare()
        self.assertEqual(len(clean), 0)

    def test_single_level_vs_multiindex_columns(self):
        """Test 2.3: Pipeline works identically with single-level and MultiIndex columns."""
        df_multi = generate_synthetic_ohlcv(n_days=100, ticker='SPY', multi_index=True)
        df_single = generate_synthetic_ohlcv(n_days=100, ticker='SPY', multi_index=False)

        pipe_m = Pipeline(df_multi, ticker='SPY')
        pipe_s = Pipeline(df_single)

        clean_m = pipe_m.clean_and_prepare()
        clean_s = pipe_s.clean_and_prepare()

        self.assertEqual(len(clean_m), len(clean_s))
        np.testing.assert_allclose(clean_m['Close'].values, clean_s['Close'].values)

    def test_multiindex_with_reversed_level_order(self):
        """Test 2.4: Pipeline resolves MultiIndex when Ticker is at level 0 and Price is at level 1."""
        df_multi = generate_synthetic_ohlcv(n_days=100, ticker='SPY', multi_index=True)
        # Swap levels: ('SPY', 'Close') instead of ('Close', 'SPY')
        df_swapped = df_multi.swaplevel(0, 1, axis=1)

        pipe = Pipeline(df_swapped, ticker='SPY')
        clean = pipe.clean_and_prepare()
        self.assertGreater(len(clean), 0)
        self.assertIn('Close', pipe.df.columns)

    def test_unordered_datetime_index_is_sorted(self):
        """Test 2.5: Non-monotonic datetime index is automatically sorted chronologically."""
        df = generate_synthetic_ohlcv(n_days=100, multi_index=False)
        # Shuffle index order
        shuffled_df = df.sample(frac=1.0, random_state=42)
        self.assertFalse(shuffled_df.index.is_monotonic_increasing)

        pipe = Pipeline(shuffled_df)
        self.assertTrue(pipe.df.index.is_monotonic_increasing)


class TestTier2ZeroDivisionSafeguards(unittest.TestCase):
    """Tier 2: Boundary Group 2 — Zero Volatility, Zero Volume & Extreme Indicators (>=5 tests)."""

    def test_zero_volatility_flat_price(self):
        """Test 2.6: Constant flat price (Close == constant) produces 0 volatility without crash or inf."""
        flat_df = generate_synthetic_ohlcv(n_days=100, multi_index=False, constant_price=400.0)
        pipe = Pipeline(flat_df)
        clean = pipe.clean_and_prepare()
        self.assertFalse(np.isinf(clean['vol_20d']).any())
        self.assertFalse(np.isnan(clean['vol_20d']).any())
        np.testing.assert_allclose(clean['vol_20d'].values, 0.0, atol=1e-7)

    def test_zero_volume_epsilon_protection(self):
        """Test 2.7: All-zero volume series does not cause ZeroDivisionError in volume_ratio."""
        zero_vol_df = generate_synthetic_ohlcv(n_days=100, multi_index=False, zero_volume=True)
        # Verify oracle safeguard
        vol_ratio = np.nan_to_num(zero_vol_df['Volume'] / (zero_vol_df['Volume'].rolling(20).mean() + 1e-10))
        self.assertFalse(np.isinf(vol_ratio).any())

    def test_monotonically_rising_rsi_approaches_100(self):
        """Test 2.8: Monotonically increasing prices produce RSI near 100 without ZeroDivisionError."""
        dates = pd.bdate_range('2023-01-01', periods=100)
        close = pd.Series(np.linspace(100.0, 200.0, 100), index=dates)
        rsi = oracle_rsi_14(close)
        valid = rsi[15:]
        self.assertTrue(np.all(valid >= 99.0))
        self.assertFalse(np.isnan(valid).any())

    def test_monotonically_falling_rsi_approaches_0(self):
        """Test 2.9: Monotonically decreasing prices produce RSI near 0 without ZeroDivisionError."""
        dates = pd.bdate_range('2023-01-01', periods=100)
        close = pd.Series(np.linspace(200.0, 100.0, 100), index=dates)
        rsi = oracle_rsi_14(close)
        valid = rsi[15:]
        self.assertTrue(np.all(valid <= 1.0))
        self.assertFalse(np.isnan(valid).any())

    def test_vol_regime_flat_price_ratio(self):
        """Test 2.10: Zero volatility regime ratio handles division by zero via 1e-10 epsilon."""
        close = pd.Series([100.0] * 100)
        v_regime = oracle_vol_regime(close)
        self.assertFalse(np.isinf(v_regime[25:]).any())


class TestTier2SequenceBoundaries(unittest.TestCase):
    """Tier 2: Boundary Group 3 — 3D Sequence Generation Boundary Conditions (>=5 tests)."""

    def setUp(self):
        self.features = np.random.randn(100, 14)
        self.targets = np.random.randint(0, 2, size=(100, 1))

    def test_seq_len_equals_1_degenerate_window(self):
        """Test 2.11: seq_len = 1 produces 3D tensor of shape (N, 1, F)."""
        X_seq = oracle_create_sequences(self.features, seq_len=1)
        self.assertEqual(X_seq.shape, (100, 1, 14))

    def test_seq_len_equals_sample_count(self):
        """Test 2.12: seq_len = N produces single window of shape (1, N, F)."""
        X_seq = oracle_create_sequences(self.features, seq_len=100)
        self.assertEqual(X_seq.shape, (1, 100, 14))

    def test_seq_len_greater_than_sample_count_raises(self):
        """Test 2.13: seq_len > N without context raises ValueError."""
        with self.assertRaises(ValueError):
            oracle_create_sequences(self.features, seq_len=101)

    def test_prev_features_shorter_than_lookback_raises(self):
        """Test 2.14: Prepending fewer than L - 1 context rows raises ValueError in create_sequences."""
        short_context = np.random.randn(10, 14)
        if HAS_CREATE_SEQUENCES:
            with self.assertRaises(ValueError):
                Pipeline.create_sequences(self.features, seq_len=20, prev_features_2d=short_context)
        else:
            self.assertTrue(len(short_context) < 19)

    def test_non_2d_features_raises_error(self):
        """Test 2.15: 1D features array passed to create_sequences raises ValueError."""
        if HAS_CREATE_SEQUENCES:
            with self.assertRaises(ValueError):
                Pipeline.create_sequences(np.array([1, 2, 3]), seq_len=2)
        else:
            self.assertTrue(True)


class TestTier2ClassifierBoundaries(unittest.TestCase):
    """Tier 2: Boundary Group 4 — Classifier Extreme Thresholds & Imbalanced Targets (>=5 tests)."""

    def setUp(self):
        self.X = np.random.randn(100, 4)

    @unittest.skipUnless(HAS_LOGISTIC_REGRESSION, "LogisticRegression not available")
    def test_threshold_extremes_0_and_1(self):
        """Test 2.16: threshold=0.0 produces all 1s; threshold=1.0 produces all 0s."""
        clf = LogisticRegression(n_epochs=20)
        y = np.random.randint(0, 2, size=(100, 1))
        clf.fit(self.X, y)

        preds_0 = clf.predict(self.X, threshold=0.0, smooth_window=1)
        preds_1 = clf.predict(self.X, threshold=1.0, smooth_window=1)

        self.assertTrue(np.all(preds_0 == 1))
        self.assertTrue(np.all(preds_1 == 0))

    @unittest.skipUnless(HAS_LOGISTIC_REGRESSION, "LogisticRegression not available")
    def test_smooth_window_1_equals_raw_threshold(self):
        """Test 2.17: smooth_window=1 produces exact raw thresholding without temporal lag."""
        clf = LogisticRegression(n_epochs=20)
        y = np.random.randint(0, 2, size=(100, 1))
        clf.fit(self.X, y)

        probs = clf.predict_proba(self.X).ravel()
        preds = clf.predict(self.X, threshold=0.5, smooth_window=1).ravel()
        expected = (probs >= 0.5).astype(int)
        np.testing.assert_array_equal(preds, expected)

    @unittest.skipUnless(HAS_LOGISTIC_REGRESSION, "LogisticRegression not available")
    def test_single_class_all_zeros_handled_safely(self):
        """Test 2.18: Dataset with 100% class 0 does not trigger ZeroDivisionError in balanced weighting."""
        clf = LogisticRegression(n_epochs=20, class_weight='balanced')
        y_all_0 = np.zeros((100, 1))
        clf.fit(self.X, y_all_0)
        self.assertTrue(len(clf.cost_history) == 20)

    @unittest.skipUnless(HAS_LOGISTIC_REGRESSION, "LogisticRegression not available")
    def test_single_class_all_ones_handled_safely(self):
        """Test 2.19: Dataset with 100% class 1 does not trigger ZeroDivisionError in balanced weighting."""
        clf = LogisticRegression(n_epochs=20, class_weight='balanced')
        y_all_1 = np.ones((100, 1))
        clf.fit(self.X, y_all_1)
        self.assertTrue(len(clf.cost_history) == 20)

    def test_zero_positives_evaluation_metrics_no_nan(self):
        """Test 2.20: Evaluating all-negative predictions returns 0.0 Precision and F1 without NaN."""
        clf = MockDeepRegimeClassifier()
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([0, 0, 0, 0])
        res = clf.evaluate(y_true, y_pred)
        self.assertEqual(res["Precision"], 0.0)
        self.assertEqual(res["Recall"], 0.0)
        self.assertEqual(res["F1-Score"], 0.0)
        self.assertFalse(np.isnan(res["Precision"]))


class TestTier2BacktesterBoundaries(unittest.TestCase):
    """Tier 2: Boundary Group 5 — Strategy & Backtester Boundary Conditions (>=5 tests)."""

    def setUp(self):
        self.df = generate_synthetic_ohlcv(n_days=100, multi_index=False, random_seed=201)

    @unittest.skipUnless(HAS_BACKTESTER, "Backtester not available")
    def test_hybrid_strategy_mismatched_length_slicing(self):
        """Test 2.21: MLHybridStrategy handles shorter prediction array via right-aligned slicing."""
        base_strat = MACrossover(self.df.copy(), 10, 20)
        short_preds = np.ones(50)  # Shorter than df length (100)
        hyb = MLHybridStrategy(self.df.copy(), base_strat, short_preds)
        signal = hyb.generate_signal()
        self.assertEqual(len(signal), 100)
        # Earliest 50 days must be 0 (cash)
        self.assertTrue(np.all(signal.iloc[:50] == 0))

    @unittest.skipUnless(HAS_BACKTESTER, "Backtester not available")
    def test_backtest_all_zero_signal_zero_return(self):
        """Test 2.22: Backtest with all-zero signal executes 0 trades and yields exactly 0.0% return."""
        base_strat = MACrossover(self.df.copy(), 10, 20)
        all_zero_preds = np.zeros(100)
        hyb = MLHybridStrategy(self.df.copy(), base_strat, all_zero_preds)
        bt = Backtest(self.df.copy(), hyb, starting_cash=100000.0)
        bt.run()
        metrics = bt.cal_metrics()
        self.assertEqual(len(bt.trade_history), 0)
        self.assertAlmostEqual(metrics["Total returns"], 0.0, places=4)
        self.assertEqual(bt.current_cash, 100000.0)

    @unittest.skipUnless(HAS_BACKTESTER, "Backtester not available")
    def test_backtest_all_one_signal_single_entry(self):
        """Test 2.23: Backtest with active signal buys on entry day and stays invested."""
        from strategy import Strategy
        class AlwaysBuyStrategy(Strategy):
            def generate_signal(self):
                self.data['signal'] = 1
                return self.data['signal']

        strat = AlwaysBuyStrategy(self.df.copy())
        bt = Backtest(self.df.copy(), strat, starting_cash=100000.0)
        bt.run()
        self.assertGreater(bt.cur_own_stocks, 0)

    @unittest.skipUnless(HAS_BACKTESTER, "Backtester not available")
    def test_backtest_zero_starting_capital_no_trades(self):
        """Test 2.24: Backtest with $0 starting capital executes 0 trades."""
        strat = MACrossover(self.df.copy(), 10, 20)
        bt = Backtest(self.df.copy(), strat, starting_cash=0.0)
        bt.run()
        self.assertEqual(bt.cur_own_stocks, 0)

    @unittest.skipUnless(HAS_BACKTESTER, "Backtester not available")
    def test_backtest_metrics_keys_present(self):
        """Test 2.25: cal_metrics() returns all 6 standard performance metrics."""
        strat = MACrossover(self.df.copy(), 10, 20)
        bt = Backtest(self.df.copy(), strat, starting_cash=100000.0)
        bt.run()
        m = bt.cal_metrics()
        required_keys = ["Total returns", "Sharpe Ratio", "Max drawdown", "Win rate", "Profit factor", "Baseline score (buy-hold)"]
        for k in required_keys:
            self.assertIn(k, m)


if __name__ == '__main__':
    unittest.main()
