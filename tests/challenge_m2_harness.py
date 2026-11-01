"""
Challenger 1 Empirical Verification Harness for Milestone 2
(PyTorch Dataset, MLP Baseline & DeepRegimeClassifier Core)

Comprehensive empirical test suites:
1. Exact API Parity and Contract Equivalence with LogisticRegression.
2. Gradient and Loss Convergence, Optimization & pos_weight.
3. In-Memory Checkpoint Restoration on Early Stopping.
4. Probability Bounds [0, 1], Binary Integer Predictions {0, 1}, Output Shapes, and Causal Smoothing.
5. RegimeDataset and MLPRegimeNet Component Stress Tests.
6. Real SPY End-to-End Backtesting Pipeline Integration.
"""

import copy
import inspect
import math
import os
import sys
import unittest
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Include relevant paths
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
BACKTEST_DIR = os.path.join(REPO_ROOT, 'Trading Strategies Backtesting System')
if BACKTEST_DIR not in sys.path:
    sys.path.insert(0, BACKTEST_DIR)

from Market_Regime_Prediction.LogisticRegression import LogisticRegression
from Market_Regime_Prediction.torch_models import (
    RegimeDataset,
    MLPRegimeNet,
    RecurrentRegimeNet,
    Conv1DRegimeNet,
    TransformerRegimeNet,
    DeepRegimeClassifier,
    _to_numpy,
)
from Market_Regime_Prediction.preprocessing_pipeline import Pipeline
from strategy import MACrossover, MLHybridStrategy
from backtester import Backtest


class Milestone2EmpiricalChallenge(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Set seeds for reproducibility
        np.random.seed(42)
        torch.manual_seed(42)

    # =========================================================================
    # SUITE 1: EXACT API PARITY WITH LogisticRegression
    # =========================================================================
    def test_01_public_methods_and_signatures_parity(self):
        """Every public method in LogisticRegression must exist in DeepRegimeClassifier with compatible signatures."""
        lr_methods = {
            name: getattr(LogisticRegression, name)
            for name, _ in inspect.getmembers(LogisticRegression, predicate=inspect.isfunction)
            if not name.startswith('_')
        }
        drc_methods = {
            name: getattr(DeepRegimeClassifier, name)
            for name, _ in inspect.getmembers(DeepRegimeClassifier, predicate=inspect.isfunction)
            if not name.startswith('_')
        }

        # Check for missing methods
        missing = set(lr_methods.keys()) - set(drc_methods.keys())
        self.assertEqual(len(missing), 0, f"DeepRegimeClassifier is missing public methods: {missing}")

        # Check call compatibility for key methods
        # 1. predict
        lr_pred_sig = inspect.signature(lr_methods['predict'])
        drc_pred_sig = inspect.signature(drc_methods['predict'])
        self.assertEqual(lr_pred_sig.parameters['threshold'].default, drc_pred_sig.parameters['threshold'].default)
        self.assertEqual(lr_pred_sig.parameters['smooth_window'].default, drc_pred_sig.parameters['smooth_window'].default)

        # 2. evaluate, confusion_matrix, accuracy_test, precision_score, recall_score, f1_score, baseline_accuracy
        for m_name in ['evaluate', 'confusion_matrix', 'accuracy_test', 'precision_score', 'recall_score', 'f1_score']:
            self.assertIn(m_name, drc_methods)

    def test_02_metric_calculations_exact_equivalence(self):
        """All evaluation metrics in DeepRegimeClassifier must produce results identical to LogisticRegression."""
        lr = LogisticRegression()
        drc = DeepRegimeClassifier(model_name='mlp')

        test_cases = [
            # Case A: Standard balanced
            (np.array([1, 0, 1, 0, 1, 0, 1, 0]), np.array([1, 0, 1, 1, 0, 0, 1, 0])),
            # Case B: All ones true
            (np.array([1, 1, 1, 1]), np.array([1, 0, 1, 1])),
            # Case C: All zeros true
            (np.array([0, 0, 0, 0]), np.array([0, 1, 0, 0])),
            # Case D: Empty
            (np.array([]), np.array([])),
            # Case E: Perfect predictions
            (np.array([1, 0, 1, 1, 0]), np.array([1, 0, 1, 1, 0])),
            # Case F: Perfectly inverted predictions
            (np.array([1, 1, 0, 0]), np.array([0, 0, 1, 1])),
            # Case G: 2D column vector shapes
            (np.array([[1], [0], [1]]), np.array([[1], [1], [0]])),
            # Case H: Pandas Series
            (pd.Series([1, 0, 1, 1]), pd.Series([1, 0, 0, 1])),
        ]

        for idx, (y_t, y_p) in enumerate(test_cases):
            with self.subTest(case_idx=idx):
                # 1. Confusion Matrix
                cm_lr = lr.confusion_matrix(y_t, y_p)
                cm_drc = drc.confusion_matrix(y_t, y_p)
                np.testing.assert_array_equal(cm_lr, cm_drc, err_msg=f"Confusion matrix mismatch in case {idx}")

                # 2. Accuracy
                acc_lr = lr.accuracy_test(y_t, y_p)
                acc_drc = drc.accuracy_test(y_t, y_p)
                self.assertAlmostEqual(acc_lr, acc_drc, places=9, msg=f"Accuracy mismatch in case {idx}")

                # 3. Precision
                prec_lr = lr.precision_score(y_t, y_p)
                prec_drc = drc.precision_score(y_t, y_p)
                self.assertAlmostEqual(prec_lr, prec_drc, places=9, msg=f"Precision mismatch in case {idx}")

                # 4. Recall
                rec_lr = lr.recall_score(y_t, y_p)
                rec_drc = drc.recall_score(y_t, y_p)
                self.assertAlmostEqual(rec_lr, rec_drc, places=9, msg=f"Recall mismatch in case {idx}")

                # 5. F1 Score
                f1_lr = lr.f1_score(y_t, y_p)
                f1_drc = drc.f1_score(y_t, y_p)
                self.assertAlmostEqual(f1_lr, f1_drc, places=9, msg=f"F1 mismatch in case {idx}")

                # 6. Baseline Accuracy
                base_lr = lr.baseline_accuracy(y_t)
                base_drc = drc.baseline_accuracy(y_t)
                self.assertAlmostEqual(base_lr, base_drc, places=9, msg=f"Baseline accuracy mismatch in case {idx}")

                # 7. Evaluate dictionary
                eval_lr = lr.evaluate(y_t, y_p)
                eval_drc = drc.evaluate(y_t, y_p)
                self.assertEqual(set(eval_lr.keys()), set(eval_drc.keys()))
                for k in eval_lr:
                    if isinstance(eval_lr[k], np.ndarray):
                        np.testing.assert_array_equal(eval_lr[k], eval_drc[k])
                    else:
                        self.assertAlmostEqual(eval_lr[k], eval_drc[k], places=9)

    def test_03_cost_history_attribute_alias(self):
        """cost_history must exist as a list and alias train_loss_history after fit."""
        X = np.random.randn(50, 4).astype(np.float32)
        y = np.random.randint(0, 2, size=(50, 1)).astype(np.float32)

        clf = DeepRegimeClassifier(model_name='mlp', n_epochs=5, batch_size=16, random_state=42)
        # Prior to fit, cost_history is an empty list
        self.assertEqual(clf.cost_history, [])

        clf.fit(X, y)
        self.assertIsInstance(clf.cost_history, list)
        self.assertEqual(len(clf.cost_history), 5)
        self.assertEqual(clf.cost_history, clf.train_loss_history)

    def test_04_sigmoid_numerical_equivalence(self):
        """sigmoid_ method in DeepRegimeClassifier must match LogisticRegression on scalar, vector, and extremes."""
        lr = LogisticRegression()
        drc = DeepRegimeClassifier()

        test_inputs = [
            0.0,
            1.0,
            -1.0,
            50.0,
            -50.0,
            1000.0,   # extreme positive clipping
            -1000.0,  # extreme negative clipping
            np.array([-500, -10, 0, 10, 500]),
            [-10.0, 0.0, 10.0],
        ]

        for z in test_inputs:
            res_lr = lr.sigmoid_(z)
            res_drc = drc.sigmoid_(z)
            np.testing.assert_allclose(res_lr, res_drc, rtol=1e-12, atol=1e-12)

    # =========================================================================
    # SUITE 2: LOSS CONVERGENCE & GRADIENT COMPUTATION
    # =========================================================================
    def test_05_gradient_descent_loss_convergence_on_separable_data(self):
        """MLPRegimeNet must decrease BCE loss by at least 70% on linearly separable data."""
        # Generate 200 samples with 4 features, clearly separable on feature 0
        N = 200
        X = np.random.randn(N, 4).astype(np.float32)
        # Class 1 if feature 0 > 0 else 0
        y = (X[:, 0] > 0.0).astype(np.float32).reshape(-1, 1)

        clf = DeepRegimeClassifier(
            model_name='mlp',
            hidden_dim=32,
            num_layers=2,
            dropout=0.0,
            learning_rate=0.05,
            n_epochs=60,
            batch_size=32,
            random_state=42,
            device='cpu'
        )
        clf.fit(X, y)

        losses = clf.train_loss_history
        self.assertEqual(len(losses), 60)
        initial_loss = losses[0]
        final_loss = losses[-1]

        # Verify loss decreased monotonically or significantly
        self.assertLess(final_loss, initial_loss * 0.30, f"Loss did not decrease sufficiently: {initial_loss} -> {final_loss}")

        # Check accuracy on training data
        preds = clf.predict(X, threshold=0.5, smooth_window=1)
        acc = clf.accuracy_test(y, preds)
        self.assertGreaterEqual(acc, 0.95, f"Training accuracy should be >= 95%, got {acc}")

    def test_06_compute_gradient_and_compute_cost(self):
        """compute_gradient and compute_cost must return finite values and valid gradient tensors."""
        X = np.random.randn(30, 4).astype(np.float32)
        y = np.random.randint(0, 2, size=(30, 1)).astype(np.float32)

        clf = DeepRegimeClassifier(model_name='mlp', n_epochs=5, random_state=42, device='cpu')
        clf.fit(X, y)

        cost = clf.compute_cost(X, y)
        self.assertIsInstance(cost, float)
        self.assertTrue(math.isfinite(cost))
        self.assertGreater(cost, 0.0)

        grads = clf.compute_gradient(X, y)
        self.assertIsInstance(grads, list)
        self.assertGreater(len(grads), 0)
        for g in grads:
            self.assertIsInstance(g, np.ndarray)
            self.assertTrue(np.all(np.isfinite(g)))
            self.assertGreater(np.sum(np.abs(g)), 0.0, "Gradient should be non-zero")

    def test_07_class_weighting_balanced(self):
        """Balanced class weighting must adjust pos_weight in BCEWithLogitsLoss."""
        # 90% class 0, 10% class 1
        N = 100
        y = np.zeros((N, 1), dtype=np.float32)
        y[:10] = 1.0  # 10 ones, 90 zeros
        X = np.random.randn(N, 4).astype(np.float32)

        clf = DeepRegimeClassifier(model_name='mlp', class_weight='balanced', n_epochs=5, device='cpu')
        clf.fit(X, y)

        # Expected pos_weight is N0 / N1 = 90 / 10 = 9.0
        # Check by fitting and ensuring the model learns to predict class 1
        preds = clf.predict(X, threshold=0.5, smooth_window=1)
        self.assertIn(1, preds, "Balanced weighting should allow predictions of minority class 1")

    # =========================================================================
    # SUITE 3: IN-MEMORY CHECKPOINT RESTORATION ON EARLY STOPPING
    # =========================================================================
    def test_08_early_stopping_restores_best_weights(self):
        """When validation loss worsens, early stopping must stop early and restore the best epoch weights."""
        N_tr = 40
        N_va = 40
        F = 10

        # Create training set where y depends heavily on feature 0
        X_tr = np.random.randn(N_tr, F).astype(np.float32)
        y_tr = (X_tr[:, 0] > 0).astype(np.float32).reshape(-1, 1)

        # Create validation set with inverted or random labels so validation loss quickly diverges after initial epochs
        X_va = np.random.randn(N_va, F).astype(np.float32)
        # Independent random noise labels
        y_va = np.random.choice([0.0, 1.0], size=(N_va, 1)).astype(np.float32)

        patience = 5
        n_epochs = 80
        clf = DeepRegimeClassifier(
            model_name='mlp',
            hidden_dim=64,
            num_layers=3,
            dropout=0.0,
            learning_rate=0.05,
            n_epochs=n_epochs,
            patience=patience,
            random_state=42,
            device='cpu'
        )
        clf.fit(X_tr, y_tr, X_val=X_va, y_val=y_va)

        val_losses = clf.val_loss_history
        # Verify it stopped early: len(val_losses) < n_epochs
        self.assertLess(len(val_losses), n_epochs, f"Early stopping did not halt training: ran {len(val_losses)} epochs")

        best_val_idx = int(np.argmin(val_losses))
        best_val_loss = val_losses[best_val_idx]
        final_val_loss = val_losses[-1]

        # Ensure validation loss indeed worsened before halting
        self.assertLess(best_val_loss, final_val_loss, "Validation loss should have worsened before early stopping")

        # Verify that the RESTORED model achieves the best epoch validation loss (not final_val_loss)
        # Use the exact training criterion (including pos_weight if class_weight='balanced')
        y_flat = y_tr.ravel()
        n0 = float(np.sum(y_flat == 0))
        n1 = float(np.sum(y_flat == 1))
        pos_weight = torch.tensor([n0 / max(n1, 1.0)], device='cpu')
        crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        val_ds = RegimeDataset(X_va, y_va)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=clf.batch_size, shuffle=False)
        clf.model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                out = clf.model(bx)
                val_running_loss += crit(out, by).item() * len(bx)
        restored_cost = val_running_loss / len(val_ds)

        self.assertAlmostEqual(
            restored_cost,
            best_val_loss,
            delta=1e-4,
            msg=f"Model does not match best epoch validation loss! Restored: {restored_cost}, Best: {best_val_loss}, Final: {final_val_loss}"
        )


    # =========================================================================
    # SUITE 4: PROBABILITY BOUNDS, INTEGER PREDICTIONS, SHAPES & CAUSALITY
    # =========================================================================
    def test_09_predict_proba_strict_bounds_and_extremes(self):
        """predict_proba must be strictly bounded in [0.0, 1.0] across normal and extreme numerical inputs."""
        X_tr = np.random.randn(60, 4).astype(np.float32)
        y_tr = np.random.randint(0, 2, size=(60, 1)).astype(np.float32)

        clf = DeepRegimeClassifier(model_name='mlp', n_epochs=10, random_state=42, device='cpu')
        clf.fit(X_tr, y_tr)

        # Test A: Standard test samples
        X_test = np.random.randn(100, 4).astype(np.float32)
        probs = clf.predict_proba(X_test)
        self.assertEqual(probs.shape, (100, 1))
        self.assertTrue(np.all(probs >= 0.0), f"Found probability < 0.0: min={probs.min()}")
        self.assertTrue(np.all(probs <= 1.0), f"Found probability > 1.0: max={probs.max()}")
        self.assertTrue(np.all(np.isfinite(probs)), "Found non-finite probability values")

        # Test B: Extreme values: 1e5, -1e5, 0.0, 1e-8
        X_extreme = np.array([
            [1e5, 1e5, 1e5, 1e5],
            [-1e5, -1e5, -1e5, -1e5],
            [0.0, 0.0, 0.0, 0.0],
            [1e-8, -1e-8, 1e-8, -1e-8],
        ], dtype=np.float32)
        probs_extreme = clf.predict_proba(X_extreme)
        self.assertEqual(probs_extreme.shape, (4, 1))
        self.assertTrue(np.all(probs_extreme >= 0.0))
        self.assertTrue(np.all(probs_extreme <= 1.0))
        self.assertTrue(np.all(np.isfinite(probs_extreme)))

    def test_10_predict_binary_integer_outputs_and_shapes(self):
        """predict must return integer array of {0, 1} with exact shape (N, 1)."""
        X_tr = np.random.randn(60, 4).astype(np.float32)
        y_tr = np.random.randint(0, 2, size=(60, 1)).astype(np.float32)

        clf = DeepRegimeClassifier(model_name='mlp', n_epochs=10, random_state=42, device='cpu')
        clf.fit(X_tr, y_tr)

        test_sizes = [240, 50, 7, 1]
        for N in test_sizes:
            X_eval = np.random.randn(N, 4).astype(np.float32)
            preds = clf.predict(X_eval, threshold=0.22, smooth_window=3)
            self.assertEqual(preds.shape, (N, 1), f"Shape mismatch for N={N}: got {preds.shape}")
            self.assertTrue(np.issubdtype(preds.dtype, np.integer), f"predict dtype must be integer, got {preds.dtype}")
            unique_vals = set(np.unique(preds))
            self.assertTrue(unique_vals.issubset({0, 1}), f"Predictions must be subset of {{0, 1}}, got {unique_vals}")

    def test_11_threshold_extremes_and_smoothing_parameters(self):
        """predict must correctly handle threshold=0.0 (all 1s), threshold=1.0 (all 0s), and smoothing options."""
        X_tr = np.random.randn(60, 4).astype(np.float32)
        y_tr = np.random.randint(0, 2, size=(60, 1)).astype(np.float32)
        clf = DeepRegimeClassifier(model_name='mlp', n_epochs=10, random_state=42, device='cpu')
        clf.fit(X_tr, y_tr)

        X_eval = np.random.randn(30, 4).astype(np.float32)

        # Threshold 0.0 -> all 1s
        preds_t0 = clf.predict(X_eval, threshold=0.0, smooth_window=None)
        np.testing.assert_array_equal(preds_t0, np.ones((30, 1), dtype=int))

        # Threshold 1.1 -> all 0s
        preds_t1 = clf.predict(X_eval, threshold=1.1, smooth_window=None)
        np.testing.assert_array_equal(preds_t1, np.zeros((30, 1), dtype=int))

        # smooth_window = 1 vs smooth_window = None (must be identical)
        p_none = clf.predict(X_eval, threshold=0.5, smooth_window=None)
        p_one = clf.predict(X_eval, threshold=0.5, smooth_window=1)
        np.testing.assert_array_equal(p_none, p_one)

        # smooth_window > len(X) must not crash
        p_large = clf.predict(X_eval, threshold=0.5, smooth_window=100)
        self.assertEqual(p_large.shape, (30, 1))

    def test_12_causal_rolling_smoothing_strictly_no_future_leakage(self):
        """Causal rolling smoothing on probabilities must NOT depend on future time steps."""
        X_tr = np.random.randn(50, 4).astype(np.float32)
        y_tr = np.random.randint(0, 2, size=(50, 1)).astype(np.float32)
        clf = DeepRegimeClassifier(model_name='mlp', n_epochs=5, random_state=42, device='cpu')
        clf.fit(X_tr, y_tr)

        # Create sequential test samples
        X_test = np.random.randn(20, 4).astype(np.float32)
        preds_orig = clf.predict(X_test, threshold=0.5, smooth_window=3)

        # Perturb only the final 5 samples (t = 15..19)
        X_perturbed = X_test.copy()
        X_perturbed[15:] = X_perturbed[15:] + 100.0
        preds_perturbed = clf.predict(X_perturbed, threshold=0.5, smooth_window=3)

        # Predictions at t = 0..14 MUST be strictly identical because future perturbations cannot leak backward!
        np.testing.assert_array_equal(
            preds_orig[:15],
            preds_perturbed[:15],
            err_msg="Future data perturbation leaked into past smoothed predictions!"
        )

    # =========================================================================
    # SUITE 5: COMPONENT STRESS TESTS (RegimeDataset & MLPRegimeNet)
    # =========================================================================
    def test_13_regime_dataset_memory_safety_and_shapes(self):
        """RegimeDataset must own its memory copy and enforce strict shape/length contracts."""
        # 2D Tabular
        X_raw = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        y_raw = np.array([0, 1], dtype=np.float32)
        ds = RegimeDataset(X_raw, y_raw)
        self.assertEqual(len(ds), 2)
        x_item, y_item = ds[0]
        self.assertEqual(x_item.shape, torch.Size([2]))
        self.assertEqual(y_item.shape, torch.Size([1]))

        # Memory isolation test: modifying X_raw must NOT affect ds.X
        X_raw[0, 0] = 999.0
        self.assertNotEqual(float(ds.X[0, 0]), 999.0, "RegimeDataset failed copy-on-construction memory isolation!")

        # Length mismatch must raise ValueError
        with self.assertRaises(ValueError):
            RegimeDataset(np.random.randn(10, 4), np.random.randn(8, 1))

        # Unsupervised / Inference mode (y=None)
        ds_inf = RegimeDataset(X_raw)
        self.assertEqual(len(ds_inf), 2)
        x_inf_item = ds_inf[0]
        self.assertEqual(x_inf_item.shape, torch.Size([2]))

    def test_14_mlp_regimenet_forward_and_gradient_flow(self):
        """MLPRegimeNet must process forward pass and propagate gradients to all layers."""
        for num_layers in [1, 2, 3]:
            net = MLPRegimeNet(input_dim=14, hidden_dim=32, num_layers=num_layers, dropout=0.2)
            x = torch.randn(8, 14, dtype=torch.float32)
            out = net(x)
            self.assertEqual(out.shape, torch.Size([8, 1]))

            # Backward pass
            loss = out.sum()
            loss.backward()
            for name, param in net.named_parameters():
                self.assertIsNotNone(param.grad, f"Gradient missing for {name}")
                self.assertTrue(torch.all(torch.isfinite(param.grad)), f"Non-finite grad in {name}")

    # =========================================================================
    # SUITE 6: REAL SPY END-TO-END BENCHMARK & BACKTEST INTEGRATION
    # =========================================================================
    def test_15_real_spy_e2e_pipeline_and_backtest(self):
        """Real SPY 5-year data must train MLP baseline and execute MLHybridStrategy backtest."""
        csv_path = os.path.join(BACKTEST_DIR, 'SPY_5_years')
        self.assertTrue(os.path.exists(csv_path), f"SPY dataset not found at {csv_path}")

        raw_df = pd.read_csv(csv_path, header=[0, 1], index_col=0, parse_dates=True)
        pipe = Pipeline(raw_df, ticker='SPY', feature_set='extended')
        X_tr, X_va, X_te, y_tr, y_va, y_te = pipe.train_val_test_split_scale(0.65, 0.15)

        self.assertEqual(X_tr.shape, (780, 14))
        self.assertEqual(X_va.shape, (180, 14))
        self.assertEqual(X_te.shape, (240, 14))

        clf = DeepRegimeClassifier(
            model_name='mlp',
            hidden_dim=32,
            num_layers=2,
            dropout=0.2,
            n_epochs=40,
            batch_size=32,
            random_state=42
        )
        clf.fit(X_tr, y_tr, X_val=X_va, y_val=y_va)

        probs = clf.predict_proba(X_te)
        self.assertEqual(probs.shape, (240, 1))
        self.assertTrue(np.all((probs >= 0.0) & (probs <= 1.0)))

        preds = clf.predict(X_te, threshold=0.22, smooth_window=3)
        self.assertEqual(preds.shape, (240, 1))

        eval_metrics = clf.evaluate(y_te, preds)
        self.assertIn("Accuracy", eval_metrics)
        self.assertIn("Confusion Matrix", eval_metrics)
        self.assertGreater(eval_metrics["Accuracy"], 0.60)

        # Plug into MLHybridStrategy and Backtest
        test_df = pipe.get_test_dataframe()
        self.assertEqual(len(test_df), 240)

        base_strat = MACrossover(test_df)
        hybrid_strat = MLHybridStrategy(test_df, base_strat, preds)
        engine = Backtest(test_df, hybrid_strat, starting_cash=100000)
        engine.run()
        metrics = engine.cal_metrics()

        required_backtest_keys = ["Total returns", "Sharpe Ratio", "Max drawdown", "Win rate", "Profit factor"]
        for k in required_backtest_keys:
            self.assertIn(k, metrics)
            val = float(metrics[k].replace('%', '')) if isinstance(metrics[k], str) else float(metrics[k])
            self.assertTrue(math.isfinite(val), f"Metric {k} is not finite: {val}")

    def test_16_audit_compute_cost_and_gradient_pos_weight(self):
        """Audits compute_cost and compute_gradient parity under balanced vs unweighted regimes."""
        X = np.random.randn(30, 4).astype(np.float32)
        y = np.random.choice([0.0, 1.0], size=(30, 1), p=[0.8, 0.2]).astype(np.float32)

        # 1. Unweighted: compute_cost must return finite positive scalar
        clf_unweighted = DeepRegimeClassifier(model_name='mlp', class_weight=None, n_epochs=5, random_state=42, device='cpu')
        clf_unweighted.fit(X, y)
        cost_unw = clf_unweighted.compute_cost(X, y)
        self.assertIsInstance(cost_unw, float)
        self.assertTrue(math.isfinite(cost_unw))
        self.assertGreater(cost_unw, 0.0)

        # 2. Balanced: Check that compute_cost and compute_gradient execute without crash,
        # return finite values, and non-empty gradient list
        clf_balanced = DeepRegimeClassifier(model_name='mlp', class_weight='balanced', n_epochs=5, random_state=42, device='cpu')
        clf_balanced.fit(X, y)
        cost_bal = clf_balanced.compute_cost(X, y)
        self.assertIsInstance(cost_bal, float)
        self.assertTrue(math.isfinite(cost_bal))
        self.assertGreater(cost_bal, 0.0)

        grads = clf_balanced.compute_gradient(X, y)
        self.assertIsInstance(grads, list)
        self.assertGreater(len(grads), 0)
        for g in grads:
            self.assertTrue(np.all(np.isfinite(g)))



if __name__ == '__main__':
    unittest.main(verbosity=2)

