"""
Empirical Challenger 3 Verification and Adversarial Stress Test Suite for Milestone 2:
- Polymorphic Input Conformance: NumPy (all dtypes, layouts), PyTorch Tensors, Pandas DataFrames/Series
- Mathematical Correctness of pos_weight, compute_cost(), and compute_gradient() with and without class_weight='balanced'
- Numerical Gradient Oracle: Finite Difference check verifying compute_gradient() computes exact gradient of compute_cost()
- Pipeline & Real SPY Downstream Backtesting Integration

Author: Teamwork Challenger 3 Agent (teamwork_challenger_m2_3)
"""

import copy
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

from Market_Regime_Prediction.preprocessing_pipeline import Pipeline
from Market_Regime_Prediction.torch_models import (
    RegimeDataset,
    MLPRegimeNet,
    DeepRegimeClassifier,
    _to_numpy,
)
from strategy import MACrossover, MLHybridStrategy
from backtester import Backtest


class TestPolymorphicInputsChallenger3(unittest.TestCase):
    """
    Stress-tests polymorphic input handling in RegimeDataset and DeepRegimeClassifier.
    """

    def setUp(self):
        np.random.seed(42)
        torch.manual_seed(42)
        self.n_samples = 48
        self.n_features = 14
        self.seq_len = 8

        self.X_np = np.random.randn(self.n_samples, self.n_features).astype(np.float64)
        self.y_np = np.random.randint(0, 2, size=(self.n_samples, 1)).astype(np.float64)

    def test_numpy_dtypes_and_memory_layouts(self):
        """Verify RegimeDataset accepts varied NumPy dtypes and memory layouts (C, F, strided, reversed)."""
        dtypes = [np.float64, np.float32, np.int64, np.int32, np.uint8, bool]
        for dt in dtypes:
            X = self.X_np.astype(dt)
            y = self.y_np.astype(dt)
            ds = RegimeDataset(X, y)
            self.assertEqual(len(ds), self.n_samples)
            x_val, y_val = ds[0]
            self.assertEqual(x_val.dtype, torch.float32)
            self.assertEqual(y_val.dtype, torch.float32)
            self.assertTrue(ds.X.is_contiguous())
            self.assertTrue(ds.y.is_contiguous())

        # Fortran contiguous
        X_f = np.asfortranarray(self.X_np)
        ds_f = RegimeDataset(X_f, self.y_np)
        self.assertTrue(ds_f.X.is_contiguous())

        # Negative stride (reversed array)
        X_rev = self.X_np[::-1]
        ds_rev = RegimeDataset(X_rev, self.y_np)
        self.assertTrue(ds_rev.X.is_contiguous())
        np.testing.assert_allclose(ds_rev.X[0].numpy(), X_rev[0].astype(np.float32), atol=1e-5)

    def test_pytorch_tensor_polymorphism(self):
        """Verify RegimeDataset and DeepRegimeClassifier accept PyTorch tensors with requires_grad, slicing, and devices."""
        # Tensor with requires_grad
        t_X = torch.tensor(self.X_np, dtype=torch.float32, requires_grad=True)
        t_y = torch.tensor(self.y_np, dtype=torch.float32, requires_grad=True)
        ds = RegimeDataset(t_X, t_y)
        self.assertFalse(ds.X.requires_grad)
        self.assertFalse(ds.y.requires_grad)
        self.assertTrue(ds.X.is_contiguous())
        self.assertTrue(ds.y.is_contiguous())

        # Transposed / non-contiguous tensor
        t_trans = torch.randn(self.n_features, self.n_samples).t()
        self.assertFalse(t_trans.is_contiguous())
        ds_trans = RegimeDataset(t_trans, t_y)
        self.assertTrue(ds_trans.X.is_contiguous())

        # Sliced tensor with stride 2
        t_sliced = torch.tensor(self.X_np, dtype=torch.float32)[::2]
        t_y_sliced = torch.tensor(self.y_np, dtype=torch.float32)[::2]
        ds_sliced = RegimeDataset(t_sliced, t_y_sliced)
        self.assertEqual(len(ds_sliced), len(t_sliced))
        self.assertTrue(ds_sliced.X.is_contiguous())

    def test_pandas_polymorphism(self):
        """Verify RegimeDataset and DeepRegimeClassifier accept DataFrames and Series with varied indices."""
        dates = pd.bdate_range('2023-01-01', periods=self.n_samples)
        df_X = pd.DataFrame(self.X_np, index=dates, columns=[f"f_{i}" for i in range(self.n_features)])
        s_y = pd.Series(self.y_np.ravel(), index=dates, name="regime")

        ds = RegimeDataset(df_X, s_y)
        self.assertEqual(len(ds), self.n_samples)
        x_0, y_0 = ds[0]
        self.assertEqual(x_0.shape, (self.n_features,))
        self.assertEqual(y_0.shape, (1,))
        np.testing.assert_allclose(x_0.numpy(), df_X.iloc[0].values.astype(np.float32), atol=1e-5)
        self.assertEqual(y_0.item(), float(s_y.iloc[0]))

    def test_deep_regime_classifier_polymorphic_api(self):
        """Verify DeepRegimeClassifier fit, predict_proba, predict, compute_cost, compute_gradient accept polymorphic inputs."""
        # 1. Fit with PyTorch Tensors
        t_X_tr = torch.tensor(self.X_np, dtype=torch.float32)
        t_y_tr = torch.tensor(self.y_np, dtype=torch.float32)
        clf = DeepRegimeClassifier(model_name='mlp', n_epochs=3, batch_size=16, random_state=42)
        clf.fit(t_X_tr, t_y_tr)
        self.assertTrue(clf.is_fitted)

        # 2. predict_proba & predict with PyTorch Tensor
        t_X_test = torch.tensor(self.X_np[:10], dtype=torch.float32, requires_grad=True)
        probs_t = clf.predict_proba(t_X_test)
        self.assertEqual(probs_t.shape, (10, 1))
        preds_t = clf.predict(t_X_test)
        self.assertEqual(preds_t.shape, (10, 1))

        # 3. predict_proba & predict with Pandas DataFrame
        df_X = pd.DataFrame(self.X_np[:10])
        probs_df = clf.predict_proba(df_X)
        self.assertEqual(probs_df.shape, (10, 1))
        preds_df = clf.predict(df_X)
        self.assertEqual(preds_df.shape, (10, 1))

        # 4. compute_cost & compute_gradient with PyTorch Tensors and Pandas
        cost_t = clf.compute_cost(t_X_test, t_y_tr[:10])
        self.assertIsInstance(cost_t, float)
        self.assertGreater(cost_t, 0.0)

        grads_t = clf.compute_gradient(t_X_test, t_y_tr[:10])
        self.assertIsInstance(grads_t, list)
        self.assertGreater(len(grads_t), 0)

        cost_df = clf.compute_cost(df_X, pd.Series(self.y_np[:10].ravel()))
        self.assertIsInstance(cost_df, float)
        self.assertGreater(cost_df, 0.0)

        # Verify numerical equivalence between tensor and dataframe inputs
        self.assertAlmostEqual(cost_t, cost_df, places=5)


class TestCostGradientBalancedLossChallenger3(unittest.TestCase):
    """
    Stress-tests compute_cost() and compute_gradient() with class_weight='balanced',
    verifying pos_weight calculation, exact BCE loss parity, and gradient correctness.
    """

    def setUp(self):
        np.random.seed(42)
        torch.manual_seed(42)
        self.n_features = 8

    def test_pos_weight_oracle(self):
        """Verify pos_weight matches n0 / max(n1, 1.0) across extreme and balanced ratios."""
        ratios = [
            (50, 50),   # 1:1 -> pos_weight = 1.0
            (80, 20),   # 4:1 -> pos_weight = 4.0
            (10, 90),   # 1:9 -> pos_weight = 10/90 = 0.1111...
            (100, 0),   # monolithic 0s -> pos_weight = 100 / 1.0 = 100.0
            (0, 100),   # monolithic 1s -> pos_weight = 0 / 100 = 0.0
        ]

        for n0, n1 in ratios:
            y = np.array([0] * n0 + [1] * n1, dtype=np.float32).reshape(-1, 1)
            X = np.random.randn(n0 + n1, self.n_features).astype(np.float32)

            clf = DeepRegimeClassifier(model_name='mlp', class_weight='balanced', n_epochs=1, random_state=42)
            clf.fit(X, y)

            expected_pos_weight = float(n0) / max(float(n1), 1.0)
            self.assertIsNotNone(clf.pos_weight_tensor)
            self.assertAlmostEqual(clf.pos_weight_tensor.item(), expected_pos_weight, places=6)

    def test_compute_cost_exact_parity_with_analytical_formula(self):
        """
        Verify compute_cost() matches both nn.BCEWithLogitsLoss(pos_weight) and
        analytical NumPy weighted binary cross entropy calculation.
        """
        # Create an imbalanced dataset: 75% class 0, 25% class 1
        n_samples = 100
        n0 = 75
        n1 = 25
        y = np.array([0] * n0 + [1] * n1, dtype=np.float32).reshape(-1, 1)
        X = np.random.randn(n_samples, self.n_features).astype(np.float32)

        clf = DeepRegimeClassifier(model_name='mlp', class_weight='balanced', n_epochs=2, random_state=42)
        clf.fit(X, y)

        cost_val = clf.compute_cost(X, y)

        # 1. Oracle via PyTorch native BCEWithLogitsLoss
        clf.model.eval()
        with torch.no_grad():
            t_X = torch.tensor(X, dtype=torch.float32, device=clf.device)
            t_y = torch.tensor(y, dtype=torch.float32, device=clf.device)
            logits = clf.model(t_X)
            crit = nn.BCEWithLogitsLoss(pos_weight=clf.pos_weight_tensor)
            expected_loss_torch = crit(logits, t_y).item()

        self.assertAlmostEqual(cost_val, expected_loss_torch, places=5)

        # 2. Oracle via pure NumPy analytical formula
        # loss_i = - [ pos_weight * y_i * log(sigmoid(z_i)) + (1 - y_i) * log(1 - sigmoid(z_i)) ]
        # Mathematically equivalent to: max(z_i, 0) - z_i * y_i + log(1 + exp(-|z_i|)) with pos_weight on y_i term:
        # For pos_weight: (1 - y) * log(1 + exp(z)) + y * pos_weight * log(1 + exp(-z))
        z = logits.cpu().numpy()
        pos_w = n0 / float(n1)
        # Numerically stable BCEWithLogitsLoss formula:
        term_pos = y * pos_w * np.log1p(np.exp(-np.abs(z))) + y * pos_w * np.maximum(-z, 0)
        term_neg = (1 - y) * np.log1p(np.exp(-np.abs(z))) + (1 - y) * np.maximum(z, 0)
        expected_loss_numpy = float(np.mean(term_pos + term_neg))

        self.assertAlmostEqual(cost_val, expected_loss_numpy, places=4)

    def test_compute_cost_unweighted_when_class_weight_is_none(self):
        """Verify compute_cost() is unweighted when class_weight=None."""
        n_samples = 80
        y = np.array([0] * 60 + [1] * 20, dtype=np.float32).reshape(-1, 1)
        X = np.random.randn(n_samples, self.n_features).astype(np.float32)

        clf_unweighted = DeepRegimeClassifier(model_name='mlp', class_weight=None, n_epochs=2, random_state=42)
        clf_unweighted.fit(X, y)
        self.assertIsNone(clf_unweighted.pos_weight_tensor)

        cost_unweighted = clf_unweighted.compute_cost(X, y)

        with torch.no_grad():
            t_X = torch.tensor(X, dtype=torch.float32, device=clf_unweighted.device)
            t_y = torch.tensor(y, dtype=torch.float32, device=clf_unweighted.device)
            logits = clf_unweighted.model(t_X)
            crit_unweighted = nn.BCEWithLogitsLoss()
            expected_unweighted = crit_unweighted(logits, t_y).item()

        self.assertAlmostEqual(cost_unweighted, expected_unweighted, places=5)

    def test_finite_difference_numerical_gradient_oracle(self):
        """
        Specialist Oracle: Verify compute_gradient() against finite-difference numerical gradients of compute_cost().
        This empirically proves that compute_gradient() computes the exact mathematical gradient of compute_cost(),
        including the balanced pos_weight multiplier.
        """
        n_samples = 30
        n0 = 20
        n1 = 10
        y = np.array([0] * n0 + [1] * n1, dtype=np.float32).reshape(-1, 1)
        X = np.random.randn(n_samples, self.n_features).astype(np.float32)

        clf = DeepRegimeClassifier(model_name='mlp', hidden_dim=16, num_layers=1, dropout=0.0,
                                   class_weight='balanced', n_epochs=1, random_state=42, device='cpu')
        clf.fit(X, y)

        # Compute analytical gradients from compute_gradient
        analytical_grads = clf.compute_gradient(X, y)
        self.assertIsNotNone(analytical_grads)

        # Compute numerical gradient for each parameter tensor using central differences
        # For float32 single precision, optimal epsilon is ~ sqrt(eps_machine) ~ 1e-3
        eps = 1e-3
        param_list = list(clf.model.parameters())

        for p_idx, p in enumerate(param_list):
            grad_anal = analytical_grads[p_idx]
            grad_num = np.zeros_like(grad_anal)

            p_data = p.data.numpy()
            orig_data = p_data.copy()

            # Check a sample of elements (first 10 elements to keep test execution fast)
            it = np.nditer(p_data, flags=['multi_index'])
            count = 0
            while not it.finished and count < 10:
                idx = it.multi_index

                # theta + eps
                p_data[idx] = orig_data[idx] + eps
                cost_plus = clf.compute_cost(X, y)

                # theta - eps
                p_data[idx] = orig_data[idx] - eps
                cost_minus = clf.compute_cost(X, y)

                # restore original
                p_data[idx] = orig_data[idx]

                grad_num[idx] = (cost_plus - cost_minus) / (2.0 * eps)
                anal_val = grad_anal[idx]
                num_val = grad_num[idx]

                diff = abs(anal_val - num_val)
                denom = max(abs(anal_val) + abs(num_val), 1e-4)
                rel_error = diff / denom

                self.assertTrue(
                    diff < 5e-4 or rel_error < 0.05,
                    f"Gradient mismatch for parameter {p_idx} at index {idx}: analytical={anal_val}, numerical={num_val}, diff={diff}, rel_err={rel_error}"
                )

                count += 1
                it.iternext()


class TestSPYBacktestIntegrationChallenger3(unittest.TestCase):
    """
    Empirically verifies end-to-end integration on real SPY 5-year data.
    """

    def test_real_spy_backtesting_pipeline(self):
        """Runs the entire pipeline on SPY_5_years data and asserts valid trade execution and metrics."""
        spy_path = PROJECT_ROOT / "Trading Strategies Backtesting System" / "SPY_5_years"
        self.assertTrue(spy_path.exists(), f"SPY_5_years dataset missing at {spy_path}")

        raw_df = pd.read_csv(spy_path, index_col=0, parse_dates=True, header=[0, 1])
        pipe = Pipeline(raw_df, ticker='SPY', feature_set='extended')
        X_tr, X_va, X_te, y_tr, y_va, y_te = pipe.train_val_test_split_scale(0.65, 0.15)

        self.assertEqual(X_tr.shape, (780, 14))
        self.assertEqual(X_va.shape, (180, 14))
        self.assertEqual(X_te.shape, (240, 14))

        clf = DeepRegimeClassifier(model_name='mlp', n_epochs=30, random_state=42)
        clf.fit(X_tr, y_tr, X_val=X_va, y_val=y_va)

        probs = clf.predict_proba(X_te)
        self.assertEqual(probs.shape, (240, 1))
        self.assertTrue(np.all((probs >= 0.0) & (probs <= 1.0)))

        preds = clf.predict(X_te, threshold=0.22, smooth_window=3)
        self.assertEqual(preds.shape, (240, 1))
        self.assertTrue(set(np.unique(preds)).issubset({0, 1}))

        test_df = pipe.get_test_dataframe()
        base_strat = MACrossover(test_df)
        hybrid_strat = MLHybridStrategy(test_df, base_strat, preds)
        engine = Backtest(test_df, hybrid_strat, starting_cash=100000)
        engine.run()

        metrics = engine.cal_metrics()
        required_keys = ["Total returns", "Sharpe Ratio", "Max drawdown", "Win rate", "Profit factor"]
        for k in required_keys:
            self.assertIn(k, metrics)
            self.assertFalse(np.isnan(metrics[k]), f"Metric {k} is NaN")

class TestEdgeCasesAndAdversarialAttacksChallenger3(unittest.TestCase):
    """
    Adversarial attacks and boundary conditions for DeepRegimeClassifier.
    """

    def test_unfitted_model_contract(self):
        """Calling predict/predict_proba before fit raises RuntimeError; compute_cost returns 0.0."""
        clf = DeepRegimeClassifier(model_name='mlp')
        with self.assertRaises(RuntimeError):
            clf.predict_proba(np.random.randn(10, 14))
        with self.assertRaises(RuntimeError):
            clf.predict(np.random.randn(10, 14))
        self.assertEqual(clf.compute_cost(np.random.randn(10, 14), np.random.randint(0, 2, (10, 1))), 0.0)
        self.assertIsNone(clf.compute_gradient(np.random.randn(10, 14), np.random.randint(0, 2, (10, 1))))

    def test_empty_input_shapes(self):
        """Passing empty inputs (0, F) returns empty (0, 1) predictions without crashing."""
        clf = DeepRegimeClassifier(model_name='mlp', n_epochs=1, random_state=42)
        clf.fit(np.random.randn(20, 14), np.random.randint(0, 2, (20, 1)))

        empty_X = np.empty((0, 14), dtype=np.float32)
        empty_probs = clf.predict_proba(empty_X)
        self.assertEqual(empty_probs.shape, (0, 1))

        empty_preds = clf.predict(empty_X)
        self.assertEqual(empty_preds.shape, (0, 1))

    def test_monolithic_class_weight_balanced_stability(self):
        """Fitting on 100% negative or 100% positive data does not cause ZeroDivisionError or NaNs."""
        X = np.random.randn(40, 14).astype(np.float32)

        # Monolithic 0s
        y_zeros = np.zeros((40, 1), dtype=np.float32)
        clf_zeros = DeepRegimeClassifier(model_name='mlp', class_weight='balanced', n_epochs=3, random_state=42)
        clf_zeros.fit(X, y_zeros)
        self.assertEqual(clf_zeros.pos_weight_tensor.item(), 40.0)
        self.assertFalse(np.isnan(clf_zeros.train_loss_history).any())
        probs_z = clf_zeros.predict_proba(X)
        self.assertTrue(np.all((probs_z >= 0.0) & (probs_z <= 1.0)))

        # Monolithic 1s
        y_ones = np.ones((40, 1), dtype=np.float32)
        clf_ones = DeepRegimeClassifier(model_name='mlp', class_weight='balanced', n_epochs=3, random_state=42)
        clf_ones.fit(X, y_ones)
        self.assertEqual(clf_ones.pos_weight_tensor.item(), 0.0)
        self.assertFalse(np.isnan(clf_ones.train_loss_history).any())
        probs_o = clf_ones.predict_proba(X)
        self.assertTrue(np.all((probs_o >= 0.0) & (probs_o <= 1.0)))

    def test_extreme_inputs_numerical_stability(self):
        """Extreme feature inputs (+/- 1e5) produce finite, bounded probabilities in [0.0, 1.0]."""
        clf = DeepRegimeClassifier(model_name='mlp', n_epochs=2, random_state=42)
        clf.fit(np.random.randn(30, 14), np.random.randint(0, 2, (30, 1)))

        extreme_X = np.array([
            [1e5] * 14,
            [-1e5] * 14,
            [0.0] * 14,
        ], dtype=np.float32)

        probs = clf.predict_proba(extreme_X)
        self.assertEqual(probs.shape, (3, 1))
        self.assertFalse(np.isnan(probs).any(), "Probabilities contained NaN for extreme inputs")
        self.assertFalse(np.isinf(probs).any(), "Probabilities contained Inf for extreme inputs")
        self.assertTrue(np.all((probs >= 0.0) & (probs <= 1.0)))

    def test_causal_rolling_filter_no_lookahead(self):
        """Verify the causal smoothing in predict() has zero lookahead leakage."""
        clf = DeepRegimeClassifier(model_name='mlp', n_epochs=2, random_state=42)
        clf.fit(np.random.randn(40, 14), np.random.randint(0, 2, (40, 1)))

        X_orig = np.random.randn(20, 14).astype(np.float32)
        preds_orig = clf.predict(X_orig, threshold=0.5, smooth_window=5)

        # Mutate future row 15
        X_mutated = X_orig.copy()
        X_mutated[15] = 9999.0
        preds_mutated = clf.predict(X_mutated, threshold=0.5, smooth_window=5)

        # Days 0..14 MUST be 100% identical
        np.testing.assert_array_equal(
            preds_orig[:15],
            preds_mutated[:15],
            err_msg="Causal filter lookahead leak: altering day 15 changed earlier predictions!"
        )


if __name__ == "__main__":
    unittest.main()
