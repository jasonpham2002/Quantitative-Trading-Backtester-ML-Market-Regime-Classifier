"""
Adversarial Stress Verification Suite for Milestone 2 Remediation
Author: Teamwork Challenger M2-4 (Empirical Challenger)

Tests deep edge cases:
1. PyTorch Tensor Memory Isolation in RegimeDataset (mutating original input tensor)
2. Gradient Detachment across polymorphic inputs
3. Exact mathematical loss parity between compute_cost and BCEWithLogitsLoss(pos_weight)
4. Strict float equality of compute_cost across NumPy, Pandas DataFrame, and PyTorch Tensor inputs
5. Single-sample inference across all 5 architectures (MLP, LSTM, GRU, CNN1D, Transformer)
6. Smoothing window boundary conditions in predict() (window=1, window=N, window > N)
7. Extreme raw data corruption (Infs in all columns, mixed NaNs, subnormals, extreme magnitudes)
8. Zero-variance feature columns in Pipeline zero-leakage scaler
9. Batch size greater than dataset length
10. Reverse-strided and non-standard memory layouts
"""

import sys
import unittest
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "Market_Regime_Prediction"))
sys.path.insert(0, str(PROJECT_ROOT / "Trading Strategies Backtesting System"))
sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing_pipeline import Pipeline
from torch_models import (
    RegimeDataset,
    MLPRegimeNet,
    RecurrentRegimeNet,
    Conv1DRegimeNet,
    TransformerRegimeNet,
    DeepRegimeClassifier,
    _to_numpy,
)


class TestAdversarialMilestone2(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        torch.manual_seed(42)
        self.n_samples = 48
        self.n_features = 14
        self.seq_len = 12

        self.X_2d = np.random.randn(self.n_samples, self.n_features).astype(np.float32)
        self.y_1d = np.random.randint(0, 2, size=self.n_samples)
        self.y_2d = self.y_1d.reshape(-1, 1)

        self.X_3d = np.random.randn(self.n_samples, self.seq_len, self.n_features).astype(np.float32)

    def test_adv_01_torch_tensor_memory_isolation(self):
        """Mutating original PyTorch tensor inputs MUST NOT mutate RegimeDataset internal state."""
        t_X = torch.tensor(self.X_2d, dtype=torch.float32, requires_grad=True)
        t_y = torch.tensor(self.y_2d, dtype=torch.float32, requires_grad=True)

        ds = RegimeDataset(t_X, t_y)
        orig_x_val = ds.X[0, 0].item()
        orig_y_val = ds.y[0, 0].item()

        # Mutate originals in-place
        with torch.no_grad():
            t_X[0, 0] = 999999.0
            t_y[0, 0] = -88888.0

        self.assertEqual(ds.X[0, 0].item(), orig_x_val)
        self.assertEqual(ds.y[0, 0].item(), orig_y_val)
        self.assertFalse(ds.X.requires_grad)
        self.assertFalse(ds.y.requires_grad)

    def test_adv_02_loss_pos_weight_exact_mathematical_parity(self):
        """
        Validates exact mathematical equivalence between DeepRegimeClassifier.compute_cost
        and explicit PyTorch nn.BCEWithLogitsLoss(pos_weight=...) under balanced class weights.
        """
        # Create an imbalanced dataset: 40 class 0, 8 class 1
        y_imb = np.zeros((self.n_samples, 1), dtype=int)
        y_imb[:8] = 1

        clf = DeepRegimeClassifier(
            model_name='mlp',
            n_epochs=10,
            batch_size=16,
            class_weight='balanced',
            random_state=42
        )
        clf.fit(self.X_2d, y_imb)

        # Expected pos_weight: n0 / max(n1, 1.0) = 40 / 8 = 5.0
        n0 = float(np.sum(y_imb == 0))
        n1 = float(np.sum(y_imb == 1))
        expected_pos_weight = n0 / n1
        self.assertIsNotNone(clf.pos_weight_tensor)
        self.assertAlmostEqual(clf.pos_weight_tensor.item(), expected_pos_weight, places=5)

        # Compute cost via classifier
        clf_cost = clf.compute_cost(self.X_2d, y_imb)

        # Manually compute loss using the model's forward logits
        clf.model.eval()
        with torch.no_grad():
            t_X = torch.tensor(self.X_2d, dtype=torch.float32).to(clf.device)
            t_y = torch.tensor(y_imb, dtype=torch.float32).to(clf.device)
            logits = clf.model(t_X)
            criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([expected_pos_weight], device=clf.device))
            manual_loss = criterion(logits, t_y).item()

        self.assertAlmostEqual(clf_cost, manual_loss, places=5)

    def test_adv_03_compute_cost_polymorphic_float_identity(self):
        """compute_cost returns identical results across NumPy, Pandas DataFrame, and Torch Tensor."""
        clf = DeepRegimeClassifier(model_name='mlp', n_epochs=5, random_state=42)
        clf.fit(self.X_2d, self.y_2d)

        df_X = pd.DataFrame(self.X_2d)
        s_y = pd.Series(self.y_1d)
        t_X = torch.tensor(self.X_2d, dtype=torch.float32)
        t_y = torch.tensor(self.y_2d, dtype=torch.float32)

        cost_np = clf.compute_cost(self.X_2d, self.y_2d)
        cost_df = clf.compute_cost(df_X, s_y)
        cost_torch = clf.compute_cost(t_X, t_y)

        self.assertAlmostEqual(cost_np, cost_df, places=6)
        self.assertAlmostEqual(cost_np, cost_torch, places=6)

    def test_adv_04_single_sample_inference_all_architectures(self):
        """All 5 architectures successfully handle N=1 single-sample inference."""
        models_2d = ['mlp']
        for m in models_2d:
            clf = DeepRegimeClassifier(model_name=m, n_epochs=2, random_state=42)
            clf.fit(self.X_2d, self.y_2d)

            x_single = self.X_2d[:1]
            prob = clf.predict_proba(x_single)
            pred = clf.predict(x_single)

            self.assertEqual(prob.shape, (1, 1))
            self.assertEqual(pred.shape, (1, 1))
            self.assertTrue(0.0 <= prob[0, 0] <= 1.0)
            self.assertIn(pred[0, 0], [0, 1])

        models_3d = ['lstm', 'gru', 'cnn1d', 'transformer']
        for m in models_3d:
            clf = DeepRegimeClassifier(model_name=m, n_epochs=2, random_state=42)
            clf.fit(self.X_3d, self.y_2d)

            x_single_3d = self.X_3d[:1]
            prob = clf.predict_proba(x_single_3d)
            pred = clf.predict(x_single_3d)

            self.assertEqual(prob.shape, (1, 1))
            self.assertEqual(pred.shape, (1, 1))
            self.assertTrue(0.0 <= prob[0, 0] <= 1.0)
            self.assertIn(pred[0, 0], [0, 1])

    def test_adv_05_predict_smoothing_window_boundaries(self):
        """predict() handles smooth_window=1, smooth_window=N, and smooth_window > N cleanly."""
        clf = DeepRegimeClassifier(model_name='mlp', n_epochs=3, random_state=42)
        clf.fit(self.X_2d, self.y_2d)

        # smooth_window = 1 (unsmoothed raw thresholding)
        pred_w1 = clf.predict(self.X_2d, threshold=0.5, smooth_window=1)
        self.assertEqual(pred_w1.shape, (self.n_samples, 1))
        self.assertTrue(set(np.unique(pred_w1)).issubset({0, 1}))

        # smooth_window = N (full sequence smoothing)
        pred_wN = clf.predict(self.X_2d, threshold=0.5, smooth_window=self.n_samples)
        self.assertEqual(pred_wN.shape, (self.n_samples, 1))
        self.assertTrue(set(np.unique(pred_wN)).issubset({0, 1}))

        # smooth_window = 1000 (> N)
        pred_w_large = clf.predict(self.X_2d, threshold=0.5, smooth_window=1000)
        self.assertEqual(pred_w_large.shape, (self.n_samples, 1))
        self.assertTrue(set(np.unique(pred_w_large)).issubset({0, 1}))

    def test_adv_06_reverse_strided_and_permuted_inputs(self):
        """RegimeDataset cleanly handles negative striding (X[::-1]) and permuted 3D arrays."""
        # Reverse strided 2D
        X_rev = self.X_2d[::-1]
        y_rev = self.y_2d[::-1]
        self.assertFalse(X_rev.flags.c_contiguous)
        ds_rev = RegimeDataset(X_rev, y_rev)
        self.assertTrue(ds_rev.X.is_contiguous())
        self.assertTrue(ds_rev.y.is_contiguous())
        np.testing.assert_allclose(ds_rev.X[0].numpy(), self.X_2d[-1], atol=1e-5)

        # Transposed 3D tensor: (L, N, F) is non-contiguous
        t_perm = torch.tensor(self.X_3d).permute(1, 0, 2)
        self.assertFalse(t_perm.is_contiguous())
        # To align with y_2d of length self.n_samples, permute back with non-contiguous strides
        # e.g. slice strided along dim 0 or use t_perm directly with matching y
        y_perm = self.y_2d[:self.seq_len]
        ds_perm = RegimeDataset(t_perm, y_perm)
        self.assertTrue(ds_perm.X.is_contiguous())
        self.assertEqual(ds_perm.X.shape, (self.seq_len, self.n_samples, self.n_features))

    def test_adv_07_batch_size_larger_than_dataset(self):
        """DeepRegimeClassifier fits and evaluates when batch_size > len(dataset)."""
        clf = DeepRegimeClassifier(model_name='mlp', n_epochs=3, batch_size=256, random_state=42)
        clf.fit(self.X_2d[:16], self.y_2d[:16])
        self.assertTrue(clf.is_fitted)
        self.assertEqual(len(clf.train_loss_history), 3)

        cost = clf.compute_cost(self.X_2d[:16], self.y_2d[:16])
        self.assertGreater(cost, 0.0)

    def test_adv_08_pipeline_extreme_corruption(self):
        """
        Pipeline.clean_and_prepare handles extreme corruption across all OHLCV columns:
        mixed +inf, -inf, NaNs, subnormals, and extreme price spikes without crashing or leaving NaNs.
        """
        dates = pd.bdate_range('2023-01-01', periods=120)
        df_corrupt = pd.DataFrame({
            'Open': 100.0 + np.random.randn(120),
            'High': 102.0 + np.random.randn(120),
            'Low': 98.0 + np.random.randn(120),
            'Close': 100.0 + np.random.randn(120),
            'Volume': 1e6 + np.random.randint(0, 1000, 120),
        }, index=dates)

        # Inject extreme anomalies
        df_corrupt.loc[dates[10], 'High'] = np.inf
        df_corrupt.loc[dates[15], 'Low'] = -np.inf
        df_corrupt.loc[dates[20], 'Close'] = np.nan
        df_corrupt.loc[dates[25], 'Open'] = 1e-300  # Subnormal
        df_corrupt.loc[dates[30], 'Volume'] = 1e12  # Extreme volume
        df_corrupt.loc[dates[40], 'Close'] = np.inf
        df_corrupt.loc[dates[50], 'Close'] = -np.inf

        pipe = Pipeline(df_corrupt, feature_set='extended')
        clean = pipe.clean_and_prepare()

        self.assertFalse(np.isinf(clean.values).any(), "Found Inf in clean_df!")
        self.assertFalse(np.isnan(clean.values).any(), "Found NaN in clean_df!")
        self.assertNotIn(dates[10], clean.index)
        self.assertNotIn(dates[15], clean.index)
        self.assertNotIn(dates[20], clean.index)
        self.assertNotIn(dates[40], clean.index)
        self.assertNotIn(dates[50], clean.index)
        self.assertTrue(set(clean['target'].unique()).issubset({0, 1}))

    def test_adv_09_pipeline_zero_variance_feature_scaling(self):
        """
        Pipeline handles a constant feature (std == 0) without ZeroDivisionError
        or NaN production during Z-score standardization.
        """
        dates = pd.bdate_range('2023-01-01', periods=120)
        # Flat prices -> returns and volatility will be zero
        df_flat = pd.DataFrame({
            'Open': np.full(120, 100.0),
            'High': np.full(120, 100.0),
            'Low': np.full(120, 100.0),
            'Close': np.full(120, 100.0),
            'Volume': np.full(120, 1000000),
        }, index=dates)

        pipe = Pipeline(df_flat, feature_set='basic')
        X_tr, X_val, X_te, y_tr, y_val, y_te = pipe.train_val_test_split_scale(0.65, 0.15)

        self.assertFalse(np.isnan(X_tr).any(), "Train scaled contains NaNs")
        self.assertFalse(np.isnan(X_val).any(), "Val scaled contains NaNs")
        self.assertFalse(np.isnan(X_te).any(), "Test scaled contains NaNs")
        self.assertFalse(np.isinf(X_tr).any(), "Train scaled contains Infs")
        self.assertFalse(np.isinf(X_val).any(), "Val scaled contains Infs")
        self.assertFalse(np.isinf(X_te).any(), "Test scaled contains Infs")


if __name__ == '__main__':
    unittest.main(verbosity=2)
