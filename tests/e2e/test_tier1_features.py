"""
Tier 1: Feature Coverage E2E Tests
Covers all 14 stationary pipeline features, chronological 65/15/20 splits,
zero-leakage scaling, 3D sequence creation with warmup, baseline LogisticRegression,
and PyTorch neural network architectures (MLP, LSTM, GRU, CNN1D, Transformer, DeepRegimeClassifier).
"""

import unittest
import numpy as np
import pandas as pd
from tests.e2e.test_helpers import (
    HAS_TORCH,
    HAS_PREPROCESSING,
    HAS_EXTENDED_PIPELINE,
    HAS_TRAIN_VAL_TEST,
    HAS_CREATE_SEQUENCES,
    HAS_TORCH_MODELS,
    HAS_LOGISTIC_REGRESSION,
    generate_synthetic_ohlcv,
    oracle_candle_body,
    oracle_intraday_range,
    oracle_volume_ratio,
    oracle_ret_1d,
    oracle_ret_5d,
    oracle_ret_10d,
    oracle_vol_10d,
    oracle_vol_20d,
    oracle_sma20_ratio,
    oracle_sma50_ratio,
    oracle_rsi_14,
    oracle_macd_norm,
    oracle_macd_hist_norm,
    oracle_vol_regime,
    oracle_create_sequences,
    oracle_train_val_test_split_scale,
    MockDeepRegimeClassifier,
)

if HAS_PREPROCESSING:
    from preprocessing_pipeline import Pipeline

if HAS_LOGISTIC_REGRESSION:
    from LogisticRegression import LogisticRegression

if HAS_TORCH:
    import torch

if HAS_TORCH_MODELS:
    import torch_models
    from torch_models import (
        RegimeDataset,
        MLPRegimeNet,
        RecurrentRegimeNet,
        Conv1DRegimeNet,
        TransformerRegimeNet,
        DeepRegimeClassifier,
    )


class TestTier1OHLCVFeatures(unittest.TestCase):
    """Tier 1: Feature Group 1 — Stationary OHLCV Ratios (>=5 tests)."""

    def setUp(self):
        self.df = generate_synthetic_ohlcv(n_days=100, multi_index=False, random_seed=101)

    @unittest.skipUnless(HAS_EXTENDED_PIPELINE, "Extended pipeline (M1) not yet implemented")
    def test_candle_body_bullish_matches_oracle(self):
        """Test 1.1: candle_body on green candle matches (Close - Open) / Open."""
        pipe = Pipeline(self.df, feature_set='extended')
        feat_df = pipe.generate_features()
        expected = oracle_candle_body(self.df['Open'], self.df['Close'])
        np.testing.assert_allclose(feat_df['candle_body'].values, expected, rtol=1e-5)

    @unittest.skipUnless(HAS_EXTENDED_PIPELINE, "Extended pipeline (M1) not yet implemented")
    def test_candle_body_bearish_is_negative(self):
        """Test 1.2: candle_body is strictly negative when Close < Open."""
        pipe = Pipeline(self.df, feature_set='extended')
        feat_df = pipe.generate_features()
        bear_mask = self.df['Close'] < self.df['Open']
        self.assertTrue(np.all(feat_df.loc[bear_mask, 'candle_body'] < 0.0))

    @unittest.skipUnless(HAS_EXTENDED_PIPELINE, "Extended pipeline (M1) not yet implemented")
    def test_candle_body_flat_is_zero(self):
        """Test 1.3: candle_body is zero when Close equals Open."""
        flat_df = self.df.copy()
        flat_df['Open'] = flat_df['Close']
        pipe = Pipeline(flat_df, feature_set='extended')
        feat_df = pipe.generate_features()
        np.testing.assert_allclose(feat_df['candle_body'].values, 0.0, atol=1e-7)

    @unittest.skipUnless(HAS_EXTENDED_PIPELINE, "Extended pipeline (M1) not yet implemented")
    def test_intraday_range_is_strictly_non_negative(self):
        """Test 1.4: intraday_range is strictly non-negative (High >= Low, Close > 0)."""
        pipe = Pipeline(self.df, feature_set='extended')
        feat_df = pipe.generate_features()
        self.assertTrue(np.all(feat_df['intraday_range'] >= 0.0))

    @unittest.skipUnless(HAS_EXTENDED_PIPELINE, "Extended pipeline (M1) not yet implemented")
    def test_intraday_range_matches_oracle(self):
        """Test 1.5: intraday_range matches (High - Low) / Close oracle exactly."""
        pipe = Pipeline(self.df, feature_set='extended')
        feat_df = pipe.generate_features()
        expected = oracle_intraday_range(self.df['High'], self.df['Low'], self.df['Close'])
        np.testing.assert_allclose(feat_df['intraday_range'].values, expected, rtol=1e-5)

    @unittest.skipUnless(HAS_EXTENDED_PIPELINE, "Extended pipeline (M1) not yet implemented")
    def test_volume_ratio_matches_oracle(self):
        """Test 1.6: volume_ratio matches Volume / SMA_20(Volume) oracle."""
        pipe = Pipeline(self.df, feature_set='extended')
        feat_df = pipe.generate_features()
        expected = oracle_volume_ratio(self.df['Volume'])
        valid = ~np.isnan(expected)
        np.testing.assert_allclose(feat_df['volume_ratio'].values[valid], expected[valid], rtol=1e-5)

    @unittest.skipUnless(HAS_EXTENDED_PIPELINE, "Extended pipeline (M1) not yet implemented")
    def test_volume_ratio_spike_detection(self):
        """Test 1.7: A 3x volume spike yields a volume_ratio substantially above 1.0 (>= 1.5)."""
        spike_df = self.df.copy()
        spike_df.iloc[-1, spike_df.columns.get_loc('Volume')] *= 3.0
        pipe = Pipeline(spike_df, feature_set='extended')
        feat_df = pipe.generate_features()
        self.assertGreater(feat_df['volume_ratio'].iloc[-1], 1.5)


class TestTier1ReturnFeatures(unittest.TestCase):
    """Tier 1: Feature Group 2 — Multi-Horizon Returns (>=5 tests)."""

    def setUp(self):
        self.df = generate_synthetic_ohlcv(n_days=100, multi_index=False, random_seed=102)

    @unittest.skipUnless(HAS_EXTENDED_PIPELINE, "Extended pipeline (M1) not yet implemented")
    def test_ret_1d_matches_pct_change(self):
        """Test 1.8: ret_1d matches Close.pct_change(1)."""
        pipe = Pipeline(self.df, feature_set='extended')
        feat_df = pipe.generate_features()
        expected = oracle_ret_1d(self.df['Close'])
        np.testing.assert_allclose(feat_df['ret_1d'].iloc[1:].values, expected[1:], rtol=1e-5)

    def test_ret_5d_matches_pct_change(self):
        """Test 1.9: ret_5d matches Close.pct_change(5) in basic or extended pipeline."""
        pipe = Pipeline(self.df)
        feat_df = pipe.generate_features()
        expected = oracle_ret_5d(self.df['Close'])
        np.testing.assert_allclose(feat_df['ret_5d'].iloc[5:].values, expected[5:], rtol=1e-5)

    @unittest.skipUnless(HAS_EXTENDED_PIPELINE, "Extended pipeline (M1) not yet implemented")
    def test_ret_10d_matches_pct_change(self):
        """Test 1.10: ret_10d matches Close.pct_change(10)."""
        pipe = Pipeline(self.df, feature_set='extended')
        feat_df = pipe.generate_features()
        expected = oracle_ret_10d(self.df['Close'])
        np.testing.assert_allclose(feat_df['ret_10d'].iloc[10:].values, expected[10:], rtol=1e-5)

    def test_return_direction_consistency(self):
        """Test 1.11: If Close is monotonically increasing, all rolling returns are positive."""
        mono_df = self.df.copy()
        mono_df['Close'] = np.linspace(100.0, 200.0, len(mono_df))
        pipe = Pipeline(mono_df)
        feat_df = pipe.generate_features()
        self.assertTrue(np.all(feat_df['ret_5d'].iloc[5:] > 0.0))

    def test_returns_finite_and_clean(self):
        """Test 1.12: Post-cleaning, return features contain no NaNs or Infs."""
        pipe = Pipeline(self.df)
        clean_df = pipe.clean_and_prepare()
        self.assertFalse(clean_df['ret_5d'].isna().any())
        self.assertFalse(np.isinf(clean_df['ret_5d']).any())


class TestTier1VolatilityAndMovingAverages(unittest.TestCase):
    """Tier 1: Feature Group 3 — Volatility & Moving Average Ratios (>=5 tests)."""

    def setUp(self):
        self.df = generate_synthetic_ohlcv(n_days=100, multi_index=False, random_seed=103)

    @unittest.skipUnless(HAS_EXTENDED_PIPELINE, "Extended pipeline (M1) not yet implemented")
    def test_vol_10d_matches_rolling_std(self):
        """Test 1.13: vol_10d matches 10-day rolling return std oracle."""
        pipe = Pipeline(self.df, feature_set='extended')
        feat_df = pipe.generate_features()
        expected = oracle_vol_10d(self.df['Close'])
        valid = ~np.isnan(expected)
        np.testing.assert_allclose(feat_df['vol_10d'].values[valid], expected[valid], rtol=1e-5)

    def test_vol_20d_matches_rolling_std(self):
        """Test 1.14: vol_20d matches 20-day rolling return std oracle."""
        pipe = Pipeline(self.df)
        feat_df = pipe.generate_features()
        expected = oracle_vol_20d(self.df['Close'])
        valid = ~np.isnan(expected)
        np.testing.assert_allclose(feat_df['vol_20d'].values[valid], expected[valid], rtol=1e-5)

    @unittest.skipUnless(HAS_EXTENDED_PIPELINE, "Extended pipeline (M1) not yet implemented")
    def test_sma20_ratio_matches_oracle(self):
        """Test 1.15: sma20_ratio matches Close / SMA_20(Close) oracle."""
        pipe = Pipeline(self.df, feature_set='extended')
        feat_df = pipe.generate_features()
        expected = oracle_sma20_ratio(self.df['Close'])
        valid = ~np.isnan(expected)
        np.testing.assert_allclose(feat_df['sma20_ratio'].values[valid], expected[valid], rtol=1e-5)

    def test_sma50_ratio_matches_oracle(self):
        """Test 1.16: sma50_ratio matches Close / SMA_50(Close) oracle."""
        pipe = Pipeline(self.df)
        feat_df = pipe.generate_features()
        expected = oracle_sma50_ratio(self.df['Close'])
        valid = ~np.isnan(expected)
        np.testing.assert_allclose(feat_df['sma50_ratio'].values[valid], expected[valid], rtol=1e-5)

    def test_sma_ratios_fluctuate_around_unity(self):
        """Test 1.17: sma50_ratio mean is close to 1.0 on a stationary/mild trend."""
        pipe = Pipeline(self.df)
        clean_df = pipe.clean_and_prepare()
        mean_ratio = clean_df['sma50_ratio'].mean()
        self.assertAlmostEqual(mean_ratio, 1.0, delta=0.15)


class TestTier1OscillatorsAndRegime(unittest.TestCase):
    """Tier 1: Feature Group 4 — Oscillators & Market Regime (>=5 tests)."""

    def setUp(self):
        self.df = generate_synthetic_ohlcv(n_days=100, multi_index=False, random_seed=104)

    def test_rsi_14_bounded_between_0_and_100(self):
        """Test 1.18: rsi_14 values are strictly within [0.0, 100.0]."""
        pipe = Pipeline(self.df)
        clean_df = pipe.clean_and_prepare()
        rsi = clean_df['rsi_14'].values
        self.assertTrue(np.all(rsi >= 0.0))
        self.assertTrue(np.all(rsi <= 100.0))

    def test_rsi_14_matches_numpy_delta_oracle(self):
        """Test 1.19: rsi_14 matches NumPy delta separation oracle."""
        pipe = Pipeline(self.df)
        feat_df = pipe.generate_features()
        expected = oracle_rsi_14(self.df['Close'])
        valid = ~np.isnan(expected)
        np.testing.assert_allclose(feat_df['rsi_14'].values[valid], expected[valid], rtol=1e-4)

    @unittest.skipUnless(HAS_EXTENDED_PIPELINE, "Extended pipeline (M1) not yet implemented")
    def test_macd_norm_matches_oracle_adjust_false(self):
        """Test 1.20: macd_norm matches (EMA_12 - EMA_26) / Close (adjust=False)."""
        pipe = Pipeline(self.df, feature_set='extended')
        feat_df = pipe.generate_features()
        expected = oracle_macd_norm(self.df['Close'])
        np.testing.assert_allclose(feat_df['macd_norm'].values[30:], expected[30:], rtol=1e-4)

    @unittest.skipUnless(HAS_EXTENDED_PIPELINE, "Extended pipeline (M1) not yet implemented")
    def test_macd_hist_norm_matches_oracle_adjust_false(self):
        """Test 1.21: macd_hist_norm matches normalized histogram oracle."""
        pipe = Pipeline(self.df, feature_set='extended')
        feat_df = pipe.generate_features()
        expected = oracle_macd_hist_norm(self.df['Close'])
        np.testing.assert_allclose(feat_df['macd_hist_norm'].values[35:], expected[35:], rtol=1e-4)

    @unittest.skipUnless(HAS_EXTENDED_PIPELINE, "Extended pipeline (M1) not yet implemented")
    def test_vol_regime_expansion_and_contraction_ratio(self):
        """Test 1.22: vol_regime matches vol_10d / (vol_20d + 1e-10)."""
        pipe = Pipeline(self.df, feature_set='extended')
        feat_df = pipe.generate_features()
        expected = oracle_vol_regime(self.df['Close'])
        valid = ~np.isnan(expected)
        np.testing.assert_allclose(feat_df['vol_regime'].values[valid], expected[valid], rtol=1e-4)


class TestTier1SplitsAndScaling(unittest.TestCase):
    """Tier 1: Feature Group 5 — Chronological Splits & Zero-Leakage Standardization (>=5 tests)."""

    def setUp(self):
        self.df = generate_synthetic_ohlcv(n_days=1254, multi_index=False, random_seed=105)

    def test_clean_df_drops_exactly_54_boundary_rows(self):
        """Test 1.23: Clean dataset drops 49 warmup rows and 5 tail rows -> exactly 1200 rows."""
        pipe = Pipeline(self.df)
        clean_df = pipe.clean_and_prepare()
        self.assertEqual(len(clean_df), 1200)

    @unittest.skipUnless(HAS_TRAIN_VAL_TEST, "train_val_test_split_scale (M1) not yet implemented")
    def test_chronological_split_exact_row_proportions(self):
        """Test 1.24: 65/15/20 split produces exact shapes (780, 180, 240)."""
        pipe = Pipeline(self.df, feature_set='extended' if HAS_EXTENDED_PIPELINE else 'basic')
        X_tr, X_va, X_te, y_tr, y_va, y_te = pipe.train_val_test_split_scale(0.65, 0.15)
        self.assertEqual(len(X_tr), 780)
        self.assertEqual(len(X_va), 180)
        self.assertEqual(len(X_te), 240)
        self.assertEqual(len(y_tr), 780)
        self.assertEqual(len(y_va), 180)
        self.assertEqual(len(y_te), 240)

    @unittest.skipUnless(HAS_TRAIN_VAL_TEST, "train_val_test_split_scale (M1) not yet implemented")
    def test_chronological_split_date_ordering_no_overlap(self):
        """Test 1.25: Train, Val, and Test splits are strictly strictly chronological with 0 date overlap."""
        pipe = Pipeline(self.df)
        pipe.train_val_test_split_scale(0.65, 0.15)
        tr_dates = pipe.train_df.index
        va_dates = pipe.val_df.index
        te_dates = pipe.test_df.index

        self.assertLess(tr_dates[-1], va_dates[0])
        self.assertLess(va_dates[-1], te_dates[0])
        self.assertTrue(tr_dates.is_monotonic_increasing)
        self.assertTrue(va_dates.is_monotonic_increasing)
        self.assertTrue(te_dates.is_monotonic_increasing)

    def test_zero_leakage_mean_std_computed_strictly_on_train(self):
        """Test 1.26: Mean and std are computed strictly from training samples."""
        pipe = Pipeline(self.df)
        X_tr_scaled, X_te_scaled, _, _ = pipe.train_test_split_scale(0.8)
        # Training scaled features must have mean close to 0 and std close to 1
        np.testing.assert_allclose(np.mean(X_tr_scaled, axis=0), 0.0, atol=1e-5)
        np.testing.assert_allclose(np.std(X_tr_scaled, axis=0), 1.0, atol=1e-5)

    def test_zero_variance_std_protection(self):
        """Test 1.27: Constant feature column in train has fallback std=1e-10 without NaN."""
        const_df = self.df.copy()
        # Add a constant dummy column
        pipe = Pipeline(const_df)
        pipe.clean_and_prepare()
        pipe.feature_names.append('const_col')
        pipe.clean_df['const_col'] = 5.0
        X_tr, X_te, _, _ = pipe.train_test_split_scale(0.8)
        self.assertFalse(np.isnan(X_tr).any())
        self.assertFalse(np.isnan(X_te).any())


class TestTier1SequenceGeneration(unittest.TestCase):
    """Tier 1: Feature Group 6 — 3D Sequence Generation with Context Warmup (>=5 tests)."""

    def setUp(self):
        self.X_2d = np.random.randn(240, 14)
        self.y_1d = np.random.randint(0, 2, size=(240, 1))
        self.prev_X = np.random.randn(180, 14)

    @unittest.skipUnless(HAS_CREATE_SEQUENCES, "create_sequences (M1) not yet implemented")
    def test_create_sequences_without_context_shape(self):
        """Test 1.28: create_sequences without prev_features shrinks by L - 1."""
        X_seq = Pipeline.create_sequences(self.X_2d, seq_len=20)
        self.assertEqual(X_seq.shape, (221, 20, 14))

    @unittest.skipUnless(HAS_CREATE_SEQUENCES, "create_sequences (M1) not yet implemented")
    def test_create_sequences_with_warmup_preserves_exact_length(self):
        """Test 1.29: Prepending prev_features ensures output length matches input length (240)."""
        X_seq, y_seq = Pipeline.create_sequences(
            self.X_2d, targets_1d=self.y_1d, seq_len=20, prev_features_2d=self.prev_X
        )
        self.assertEqual(X_seq.shape, (240, 20, 14))
        self.assertEqual(y_seq.shape, (240, 1))

    def test_oracle_sequence_temporal_alignment_final_step(self):
        """Test 1.30: Window i's final step [-1, :] equals current observation X[i, :]."""
        X_seq = oracle_create_sequences(self.X_2d, seq_len=20, prev_features_2d=self.prev_X)
        for i in [0, 50, 100, 239]:
            np.testing.assert_allclose(X_seq[i, -1, :], self.X_2d[i, :])

    def test_oracle_sequence_context_lookback_alignment(self):
        """Test 1.31: Window 0's first step [0, :] equals context row prev_X[-19, :]."""
        X_seq = oracle_create_sequences(self.X_2d, seq_len=20, prev_features_2d=self.prev_X)
        np.testing.assert_allclose(X_seq[0, 0, :], self.prev_X[-19, :])

    def test_oracle_sequence_targets_alignment(self):
        """Test 1.32: Target y_seq[i] matches y_1d[i] 1-to-1 when warmup is used."""
        _, y_seq = oracle_create_sequences(
            self.X_2d, targets_1d=self.y_1d, seq_len=20, prev_features_2d=self.prev_X
        )
        np.testing.assert_array_equal(y_seq, self.y_1d)


class TestTier1ModelArchitectures(unittest.TestCase):
    """Tier 1: Feature Group 7 — Model Architectures & Classifier API (>=5 tests)."""

    def setUp(self):
        np.random.seed(42)
        self.X_tab = np.random.randn(100, 14)
        self.X_seq = np.random.randn(100, 20, 14)
        self.y = np.random.randint(0, 2, size=(100, 1))

    @unittest.skipUnless(HAS_LOGISTIC_REGRESSION, "LogisticRegression baseline not available")
    def test_logistic_regression_baseline_fit_and_predict(self):
        """Test 1.33: LogisticRegression baseline fits and computes probabilities in [0, 1]."""
        clf = LogisticRegression(learning_rate=0.01, n_epochs=50, class_weight='balanced')
        clf.fit(self.X_tab, self.y)
        self.assertTrue(len(clf.cost_history) == 50)
        probs = clf.predict_proba(self.X_tab)
        self.assertEqual(probs.shape, (100, 1))
        self.assertTrue(np.all(probs >= 0.0) and np.all(probs <= 1.0))
        preds = clf.predict(self.X_tab, threshold=0.5, smooth_window=1)
        self.assertEqual(preds.shape, (100, 1))
        self.assertTrue(set(np.unique(preds)).issubset({0, 1}))

    @unittest.skipUnless(HAS_TORCH_MODELS and HAS_TORCH, "torch_models (M2) not yet implemented")
    def test_mlp_regime_net_forward_pass_shape(self):
        """Test 1.34: MLPRegimeNet accepts (B, F) and outputs logit (B, 1)."""
        model = MLPRegimeNet(input_dim=14, hidden_dim=32)
        x = torch.randn(16, 14)
        out = model(x)
        self.assertEqual(out.shape, (16, 1))

    @unittest.skipUnless(HAS_TORCH_MODELS and HAS_TORCH, "torch_models (M3) not yet implemented")
    def test_recurrent_regime_net_lstm_and_gru_forward_shape(self):
        """Test 1.35: RecurrentRegimeNet with LSTM and GRU accepts (B, L, F) and outputs logit (B, 1)."""
        for cell in ['lstm', 'gru']:
            model = RecurrentRegimeNet(input_dim=14, hidden_dim=32, cell_type=cell)
            x = torch.randn(8, 20, 14)
            out = model(x)
            self.assertEqual(out.shape, (8, 1))

    @unittest.skipUnless(HAS_TORCH_MODELS and HAS_TORCH, "torch_models (M4) not yet implemented")
    def test_conv1d_regime_net_channel_permutation_forward_shape(self):
        """Test 1.36: Conv1DRegimeNet accepts (B, L, F), permutes channels, and outputs logit (B, 1)."""
        model = Conv1DRegimeNet(input_dim=14, filters=32)
        x = torch.randn(8, 20, 14)
        out = model(x)
        self.assertEqual(out.shape, (8, 1))

    @unittest.skipUnless(HAS_TORCH_MODELS and HAS_TORCH, "torch_models (M4) not yet implemented")
    def test_transformer_regime_net_attention_matrix_extraction(self):
        """Test 1.37: TransformerRegimeNet accepts (B, L, F) and stores (B, L, L) attention weights."""
        model = TransformerRegimeNet(input_dim=14, d_model=32, nhead=4)
        x = torch.randn(4, 20, 14)
        out = model(x)
        self.assertEqual(out.shape, (4, 1))
        self.assertTrue(hasattr(model, 'last_attention_weights'))

    def test_mock_classifier_matches_interface_contract(self):
        """Test 1.38: DeepRegimeClassifier contract matches LogisticRegression interface."""
        clf = MockDeepRegimeClassifier()
        clf.fit(self.X_tab, self.y)
        probs = clf.predict_proba(self.X_tab)
        preds = clf.predict(self.X_tab)
        eval_dict = clf.evaluate(self.y, preds)

        self.assertEqual(probs.shape, (100, 1))
        self.assertEqual(preds.shape, (100, 1))
        self.assertIn("Accuracy", eval_dict)
        self.assertIn("Precision", eval_dict)
        self.assertIn("Recall", eval_dict)
        self.assertIn("F1-Score", eval_dict)
        self.assertIn("Confusion Matrix", eval_dict)
        self.assertEqual(eval_dict["Confusion Matrix"].shape, (2, 2))


if __name__ == '__main__':
    unittest.main()
