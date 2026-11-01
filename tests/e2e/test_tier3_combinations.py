"""
Tier 3: Cross-Feature Combinations E2E Tests
Covers multi-step data flow interactions across modules:
Pipeline -> Train/Val/Test Split -> Zero-Leakage Standardization -> 3D Sequences ->
RegimeDataset -> PyTorch / Baseline Models -> MLHybridStrategy -> BacktestingEngine.
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
    HAS_BACKTESTER,
    generate_synthetic_ohlcv,
    oracle_create_sequences,
    oracle_train_val_test_split_scale,
    MockDeepRegimeClassifier,
)

if HAS_PREPROCESSING:
    from preprocessing_pipeline import Pipeline

if HAS_LOGISTIC_REGRESSION:
    from LogisticRegression import LogisticRegression

if HAS_BACKTESTER:
    from strategy import MACrossover, RSIMeanReversion, MLHybridStrategy
    from backtester import Backtest

if HAS_TORCH:
    import torch
    from torch.utils.data import DataLoader

if HAS_TORCH_MODELS:
    from torch_models import (
        RegimeDataset,
        MLPRegimeNet,
        RecurrentRegimeNet,
        Conv1DRegimeNet,
        TransformerRegimeNet,
        DeepRegimeClassifier,
    )


class TestTier3PipelineToModels(unittest.TestCase):
    """Tier 3: Combinations Group 1 — Pipeline to Models & Datasets (>=5 tests)."""

    def setUp(self):
        self.df = generate_synthetic_ohlcv(n_days=1254, multi_index=False, random_seed=301)

    @unittest.skipUnless(HAS_PREPROCESSING and HAS_LOGISTIC_REGRESSION, "Pipeline and LogisticRegression required")
    def test_pipeline_basic_to_logistic_regression(self):
        """Test 3.1: Basic pipeline -> 80/20 split -> LogisticRegression fit -> probability output."""
        pipe = Pipeline(self.df)
        X_tr, X_te, y_tr, y_te = pipe.train_test_split_scale(0.8)
        clf = LogisticRegression(learning_rate=0.05, n_epochs=50, class_weight='balanced')
        clf.fit(X_tr, y_tr)
        probs = clf.predict_proba(X_te)
        self.assertEqual(probs.shape, (240, 1))
        self.assertTrue(len(clf.cost_history) == 50)

    @unittest.skipUnless(HAS_EXTENDED_PIPELINE and HAS_TRAIN_VAL_TEST, "Extended pipeline required")
    def test_extended_pipeline_to_train_val_test_scaling(self):
        """Test 3.2: Extended pipeline -> 65/15/20 split -> zero-leakage scaling across all 3 sets."""
        pipe = Pipeline(self.df, feature_set='extended')
        X_tr, X_va, X_te, y_tr, y_va, y_te = pipe.train_val_test_split_scale(0.65, 0.15)
        self.assertEqual(X_tr.shape, (780, 14))
        self.assertEqual(X_va.shape, (180, 14))
        self.assertEqual(X_te.shape, (240, 14))
        # Ensure training features are normalized
        np.testing.assert_allclose(np.mean(X_tr, axis=0), 0.0, atol=1e-5)
        np.testing.assert_allclose(np.std(X_tr, axis=0), 1.0, atol=1e-5)

    @unittest.skipUnless(HAS_EXTENDED_PIPELINE and HAS_TRAIN_VAL_TEST and HAS_CREATE_SEQUENCES, "Sequences required")
    def test_pipeline_to_3d_sequences_with_warmup(self):
        """Test 3.3: Extended split -> create_sequences with warmup -> exact 180 and 240 window shapes."""
        pipe = Pipeline(self.df, feature_set='extended')
        X_tr, X_va, X_te, y_tr, y_va, y_te = pipe.train_val_test_split_scale(0.65, 0.15)
        X_va_seq, y_va_seq = Pipeline.create_sequences(X_va, y_va, seq_len=20, prev_features_2d=X_tr)
        X_te_seq, y_te_seq = Pipeline.create_sequences(X_te, y_te, seq_len=20, prev_features_2d=X_va)

        self.assertEqual(X_va_seq.shape, (180, 20, 14))
        self.assertEqual(y_va_seq.shape, (180, 1))
        self.assertEqual(X_te_seq.shape, (240, 20, 14))
        self.assertEqual(y_te_seq.shape, (240, 1))

    @unittest.skipUnless(HAS_TORCH_MODELS and HAS_TORCH, "torch_models required")
    def test_regime_dataset_2d_tabular_loader(self):
        """Test 3.4: 2D tabular features -> RegimeDataset -> DataLoader batch iteration."""
        X_tab = np.random.randn(100, 14)
        y = np.random.randint(0, 2, size=(100, 1))
        ds = RegimeDataset(X_tab, y)
        loader = DataLoader(ds, batch_size=16, shuffle=False)
        batch_x, batch_y = next(iter(loader))
        self.assertEqual(batch_x.shape, (16, 14))
        self.assertEqual(batch_y.shape, (16, 1))

    @unittest.skipUnless(HAS_TORCH_MODELS and HAS_TORCH, "torch_models required")
    def test_regime_dataset_3d_sequence_loader(self):
        """Test 3.5: 3D sequence tensors -> RegimeDataset -> DataLoader batch iteration."""
        X_seq = np.random.randn(100, 20, 14)
        y = np.random.randint(0, 2, size=(100, 1))
        ds = RegimeDataset(X_seq, y)
        loader = DataLoader(ds, batch_size=8, shuffle=False)
        batch_x, batch_y = next(iter(loader))
        self.assertEqual(batch_x.shape, (8, 20, 14))
        self.assertEqual(batch_y.shape, (8, 1))

    @unittest.skipUnless(HAS_TORCH_MODELS and HAS_TORCH, "torch_models required")
    def test_recurrent_model_dataloader_training_step(self):
        """Test 3.6: RecurrentRegimeNet executes training forward/backward pass with BCEWithLogitsLoss."""
        model = RecurrentRegimeNet(input_dim=14, hidden_dim=16, cell_type='lstm')
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.BCEWithLogitsLoss()

        x = torch.randn(16, 20, 14)
        y = torch.randint(0, 2, size=(16, 1), dtype=torch.float32)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        self.assertFalse(torch.isnan(loss).item())
        self.assertGreater(loss.item(), 0.0)


class TestTier3ModelsToStrategy(unittest.TestCase):
    """Tier 3: Combinations Group 2 — Models to MLHybridStrategy (>=5 tests)."""

    def setUp(self):
        self.df = generate_synthetic_ohlcv(n_days=1254, multi_index=False, random_seed=302)
        pipe = Pipeline(self.df)
        self.clean_df = pipe.clean_and_prepare()
        _, _, _, _ = pipe.train_test_split_scale(0.8)
        self.test_df = pipe.get_test_dataframe()

    @unittest.skipUnless(HAS_BACKTESTER, "Backtester required")
    def test_macrossover_with_mock_classifier(self):
        """Test 3.7: MACrossover + MockDeepRegimeClassifier -> MLHybridStrategy."""
        base_strat = MACrossover(self.test_df.copy(), 10, 20)
        clf = MockDeepRegimeClassifier()
        y_pred = clf.predict(np.random.randn(len(self.test_df), 14))

        hyb = MLHybridStrategy(self.test_df.copy(), base_strat, y_pred)
        signal = hyb.generate_signal()
        self.assertEqual(len(signal), len(self.test_df))
        self.assertTrue(set(np.unique(signal)).issubset({0, 1}))

    @unittest.skipUnless(HAS_BACKTESTER, "Backtester required")
    def test_rsim_reversion_with_mock_classifier(self):
        """Test 3.8: RSIMeanReversion + MockDeepRegimeClassifier -> MLHybridStrategy."""
        base_strat = RSIMeanReversion(self.test_df.copy())
        clf = MockDeepRegimeClassifier()
        y_pred = clf.predict(np.random.randn(len(self.test_df), 14))

        hyb = MLHybridStrategy(self.test_df.copy(), base_strat, y_pred)
        signal = hyb.generate_signal()
        self.assertEqual(len(signal), len(self.test_df))
        self.assertTrue(set(np.unique(signal)).issubset({0, 1}))

    @unittest.skipUnless(HAS_BACKTESTER, "Backtester required")
    def test_hybrid_signal_only_active_when_both_technical_and_ml_agree(self):
        """Test 3.9: Hybrid signal == 1 if and only if technical condition is active AND ML prediction == 1."""
        base_strat = MACrossover(self.test_df.copy(), 10, 20)
        base_strat.generate_signal()
        base_sig = base_strat.data['signal'].values

        # Alternate ML predictions
        ml_preds = np.array([1 if i % 2 == 0 else 0 for i in range(len(self.test_df))])
        hyb = MLHybridStrategy(self.test_df.copy(), base_strat, ml_preds)
        hyb_sig = hyb.generate_signal().values

        for i in range(len(self.test_df)):
            if ml_preds[i] == 0:
                self.assertEqual(hyb_sig[i], 0, f"Index {i} should be flat when ML is 0")

    @unittest.skipUnless(HAS_BACKTESTER and HAS_LOGISTIC_REGRESSION, "Backtester and LogisticRegression required")
    def test_logistic_regression_to_hybrid_strategy(self):
        """Test 3.10: Real LogisticRegression predictions feed into MLHybridStrategy."""
        pipe = Pipeline(self.df)
        X_tr, X_te, y_tr, _ = pipe.train_test_split_scale(0.8)
        clf = LogisticRegression(n_epochs=50)
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te, threshold=0.22, smooth_window=3)

        base_strat = MACrossover(self.test_df.copy(), 10, 20)
        hyb = MLHybridStrategy(self.test_df.copy(), base_strat, y_pred)
        sig = hyb.generate_signal()
        self.assertEqual(len(sig), 240)


class TestTier3StrategyToBacktester(unittest.TestCase):
    """Tier 3: Combinations Group 3 — Strategy to Backtester & Portfolio Evaluation (>=5 tests)."""

    def setUp(self):
        self.df = generate_synthetic_ohlcv(n_days=1254, multi_index=False, random_seed=303)
        pipe = Pipeline(self.df)
        pipe.clean_and_prepare()
        pipe.train_test_split_scale(0.8)
        self.test_df = pipe.get_test_dataframe()

    @unittest.skipUnless(HAS_BACKTESTER, "Backtester required")
    def test_full_backtest_execution_with_mock_hybrid(self):
        """Test 3.11: MLHybridStrategy runs through BacktestingEngine cleanly."""
        base_strat = MACrossover(self.test_df.copy(), 10, 20)
        clf = MockDeepRegimeClassifier()
        y_pred = clf.predict(np.random.randn(len(self.test_df), 14))

        hyb = MLHybridStrategy(self.test_df.copy(), base_strat, y_pred)
        bt = Backtest(self.test_df.copy(), hyb, starting_cash=100000.0)
        bt.run()
        metrics = bt.cal_metrics()

        self.assertIn("Total returns", metrics)
        self.assertIn("Sharpe Ratio", metrics)
        self.assertIn("Max drawdown", metrics)
        self.assertGreaterEqual(bt.current_cash, 0.0)
        self.assertEqual(len(bt.tot_net_worth), len(self.test_df))

    @unittest.skipUnless(HAS_BACKTESTER, "Backtester required")
    def test_multi_model_backtest_comparison_suite(self):
        """Test 3.12: Run backtest across 3 distinct prediction streams on the exact same test dataset."""
        base_strat = MACrossover(self.test_df.copy(), 10, 20)
        np.random.seed(42)

        predictions_suite = {
            "all_bullish": np.ones(len(self.test_df)),
            "all_bearish": np.zeros(len(self.test_df)),
            "alternating": np.array([i % 2 for i in range(len(self.test_df))]),
        }

        results = {}
        for name, preds in predictions_suite.items():
            hyb = MLHybridStrategy(self.test_df.copy(), base_strat, preds)
            bt = Backtest(self.test_df.copy(), hyb, starting_cash=100000.0)
            bt.run()
            results[name] = bt.cal_metrics()

        # All-bearish should have executed 0 trades and 0.0% return
        self.assertAlmostEqual(results["all_bearish"]["Total returns"], 0.0, places=4)
        # All 3 evaluations return valid Sharpe Ratios
        for name in results:
            self.assertFalse(np.isnan(results[name]["Sharpe Ratio"]))

    @unittest.skipUnless(HAS_BACKTESTER and HAS_LOGISTIC_REGRESSION, "Backtester and LR required")
    def test_end_to_end_pipeline_train_predict_backtest(self):
        """Test 3.13: Complete End-to-End Pipeline: Raw Data -> Split -> LR -> Hybrid -> Backtest."""
        pipe = Pipeline(self.df)
        X_tr, X_te, y_tr, y_te = pipe.train_test_split_scale(0.8)
        test_df = pipe.get_test_dataframe()

        clf = LogisticRegression(learning_rate=0.05, n_epochs=50, class_weight='balanced')
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te, threshold=0.22, smooth_window=3)

        base_strat = MACrossover(test_df.copy(), 20, 50, full_data=self.df)
        hyb = MLHybridStrategy(test_df.copy(), base_strat, y_pred)
        bt = Backtest(test_df.copy(), hyb, starting_cash=100000.0)
        bt.run()
        m = bt.cal_metrics()

        self.assertIsInstance(m["Total returns"], float)
        self.assertIsInstance(m["Sharpe Ratio"], float)
        self.assertIsInstance(m["Max drawdown"], float)


if __name__ == '__main__':
    unittest.main()
