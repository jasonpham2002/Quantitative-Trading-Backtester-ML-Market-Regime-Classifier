"""
Milestone 4 Adversarial Challenger Test Suite
DATA 3401 Honors Project

Independent Empirical Challenge:
- Parameter counting oracles and non-standard architecture configurations
- Defensive boundary checking (1D, 2D, 4D input rejections)
- Self-attention normalization (row sum == 1.0 across sequence lengths)
- Pure NumPy ROC and PR curve mathematical oracles vs. analytical truth & edge cases
- Headless diagnostic visualization robustness under boundary conditions
- Real SPY 5-year downstream backtesting invariants and threshold sensitivity
"""

import math
import os
import shutil
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Repository paths
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
BACKTEST_DIR = os.path.join(REPO_ROOT, 'Trading Strategies Backtesting System')
if BACKTEST_DIR not in sys.path:
    sys.path.insert(0, BACKTEST_DIR)

from Market_Regime_Prediction.preprocessing_pipeline import Pipeline
from Market_Regime_Prediction.torch_models import (
    Conv1DRegimeNet,
    TransformerRegimeNet,
    DeepRegimeClassifier,
    SinusoidalPositionalEncoding,
)
from Market_Regime_Prediction.visualizations import (
    _compute_roc_curve,
    _compute_pr_curve,
    plot_training_curves_grid,
    plot_confusion_matrix_grid,
    plot_roc_pr_curves,
    plot_feature_importance_heatmap,
    plot_transformer_attention_heatmap,
)
from strategy import MACrossover, MLHybridStrategy
from backtester import Backtest
from tests.e2e.test_helpers import load_real_spy_data


class AdversarialMilestone4Challenger(unittest.TestCase):
    """
    Adversarial challenge suite for Milestone 4.
    Executes independent empirical stress tests.
    """

    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        torch.manual_seed(42)

        cls.temp_dir = tempfile.mkdtemp()

        # Load real SPY 5-year dataset and prepare sequences
        cls.df_raw = load_real_spy_data()
        cls.pipe = Pipeline(cls.df_raw, ticker='SPY', feature_set='extended')
        (
            cls.X_tr,
            cls.X_val,
            cls.X_te,
            cls.y_tr,
            cls.y_val,
            cls.y_te,
        ) = cls.pipe.train_val_test_split_scale(0.65, 0.15)

        cls.seq_len = 20
        cls.X_tr_s, cls.y_tr_s = cls.pipe.create_sequences(cls.X_tr, cls.y_tr, seq_len=cls.seq_len)
        cls.X_val_s, cls.y_val_s = cls.pipe.create_sequences(
            cls.X_val, cls.y_val, seq_len=cls.seq_len, prev_features_2d=cls.X_tr
        )
        cls.X_te_s, cls.y_te_s = cls.pipe.create_sequences(
            cls.X_te, cls.y_te, seq_len=cls.seq_len, prev_features_2d=cls.X_val
        )
        cls.test_df = cls.pipe.get_test_dataframe()

        # Baseline MACrossover metrics
        cls.base_ma = MACrossover(cls.test_df.copy(), 20, 50, full_data=cls.df_raw)
        cls.base_ma.generate_signal()
        base_bt = Backtest(cls.test_df.copy(), cls.base_ma, starting_cash=100000.0)
        base_bt.run()
        cls.base_ma_metrics = base_bt.cal_metrics()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    # =========================================================================
    # SUITE 1: CONV1D ARCHITECTURAL INVARIANTS & STRESS HARNESS
    # =========================================================================

    def test_adv_01_conv1d_parameter_oracle(self):
        """
        Oracle test: Verify exact parameter count formula for Conv1D across configurations.
        Formula:
          conv1: filters * input_dim * kernel_size + filters
          bn1: 2 * filters
          conv2: (filters * 2) * filters * kernel_size + (filters * 2)
          bn2: 2 * (filters * 2)
          fc: 1 * (filters * 2) + 1
        """
        configs = [
            (14, 32, 3),   # Default M4 specification
            (10, 16, 5),   # Smaller custom
            (20, 64, 3),   # Larger custom
        ]
        for input_dim, filters, kernel_size in configs:
            net = Conv1DRegimeNet(input_dim=input_dim, filters=filters, kernel_size=kernel_size)
            actual_params = sum(p.numel() for p in net.parameters())

            p_conv1 = filters * input_dim * kernel_size + filters
            p_bn1 = 2 * filters
            p_conv2 = (filters * 2) * filters * kernel_size + (filters * 2)
            p_bn2 = 2 * (filters * 2)
            p_fc = 1 * (filters * 2) + 1
            expected_params = p_conv1 + p_bn1 + p_conv2 + p_bn2 + p_fc

            self.assertEqual(actual_params, expected_params)

        # Standard M4 contract: exactly 7,841 parameters
        default_net = Conv1DRegimeNet(input_dim=14, filters=32, kernel_size=3)
        self.assertEqual(sum(p.numel() for p in default_net.parameters()), 7841)

    def test_adv_02_conv1d_sequence_length_invariance(self):
        """
        Stress test: AdaptiveAvgPool1d(1) must yield (B, 1) output for arbitrary lookback lengths.
        """
        net = Conv1DRegimeNet(input_dim=14, filters=32)
        net.eval()
        for L in [1, 2, 5, 13, 20, 42, 100, 250]:
            x = torch.randn(3, L, 14)
            with torch.no_grad():
                out = net(x)
            self.assertEqual(out.shape, (3, 1), f"Failed for lookback length L={L}")
            self.assertTrue(torch.all(torch.isfinite(out)), f"Non-finite output for L={L}")

    def test_adv_03_conv1d_defensive_dimension_guards(self):
        """
        Adversarial test: Reject tensors of rank 1, 2, 4, and 5 with clear ValueError.
        """
        net = Conv1DRegimeNet(input_dim=14, filters=32)
        bad_inputs = [
            torch.randn(14),                # 1D
            torch.randn(10, 14),            # 2D
            torch.randn(2, 4, 20, 14),      # 4D
            torch.randn(1, 2, 3, 4, 5),     # 5D
        ]
        for x in bad_inputs:
            with self.assertRaises(ValueError) as ctx:
                net(x)
            self.assertIn("3D sequential input", str(ctx.exception))

    def test_adv_04_conv1d_extreme_inputs(self):
        """
        Stress test: Verify behavior under all-zeros, constant, and extreme values.
        """
        net = Conv1DRegimeNet(input_dim=14, filters=32)
        net.eval()

        # All zeros
        zeros = torch.zeros(4, 20, 14)
        with torch.no_grad():
            out_zero = net(zeros)
        self.assertEqual(out_zero.shape, (4, 1))
        self.assertTrue(torch.all(torch.isfinite(out_zero)))

        # All large constants
        const = torch.full((4, 20, 14), fill_value=1e4)
        with torch.no_grad():
            out_const = net(const)
        self.assertEqual(out_const.shape, (4, 1))
        self.assertTrue(torch.all(torch.isfinite(out_const)))

    # =========================================================================
    # SUITE 2: TRANSFORMER ARCHITECTURAL INVARIANTS & ATTENTION STRESS
    # =========================================================================

    def test_adv_05_transformer_parameter_oracle(self):
        """
        Oracle test: Verify exact parameter count formula for TransformerRegimeNet.
        Formula:
          input_proj: (d_model * input_dim) + d_model
          mha: in_proj (3 * d_model * d_model + 3 * d_model) + out_proj (d_model * d_model + d_model)
          norm1: 2 * d_model
          norm2: 2 * d_model
          ffn: linear1 (dim_ff * d_model + dim_ff) + linear2 (d_model * dim_ff + d_model)
          fc: (1 * d_model) + 1
        """
        configs = [
            (14, 32, 4, 64, 1),   # Default M4 specification
            (7, 16, 2, 32, 1),    # Smaller custom
            (20, 48, 6, 96, 1),   # Larger custom
        ]
        for input_dim, d_model, nhead, dim_ff, num_layers in configs:
            net = TransformerRegimeNet(
                input_dim=input_dim,
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_ff,
                num_layers=num_layers,
            )
            actual_params = sum(p.numel() for p in net.parameters())

            p_proj = (d_model * input_dim) + d_model
            p_mha = (3 * d_model * d_model + 3 * d_model) + (d_model * d_model + d_model)
            p_norm1 = 2 * d_model
            p_norm2 = 2 * d_model
            p_ffn = (dim_ff * d_model + dim_ff) + (d_model * dim_ff + d_model)
            p_fc = (1 * d_model) + 1
            expected_params = p_proj + p_mha + p_norm1 + p_norm2 + p_ffn + p_fc

            self.assertEqual(actual_params, expected_params)

        # Standard M4 contract: exactly 9,057 parameters
        default_net = TransformerRegimeNet(input_dim=14, d_model=32, nhead=4, dim_feedforward=64)
        self.assertEqual(sum(p.numel() for p in default_net.parameters()), 9057)

    def test_adv_06_transformer_attention_normalization_across_lengths(self):
        """
        Invariant test: In eval() mode, attention matrix must have shape (B, L, L),
        all values in [0, 1], and every row summing to 1.0 +- 1e-5 across keys for any L.
        """
        net = TransformerRegimeNet(input_dim=14, d_model=32, nhead=4)
        net.eval()

        for L in [2, 5, 10, 20, 35, 50]:
            x = torch.randn(4, L, 14)
            with torch.no_grad():
                _ = net(x)
            attn = net.last_attention_weights
            self.assertEqual(attn.shape, (4, L, L))
            self.assertTrue(torch.all(attn >= 0.0) and torch.all(attn <= 1.0))

            row_sums = attn.sum(dim=-1)
            expected_sums = torch.ones_like(row_sums)
            self.assertTrue(torch.allclose(row_sums, expected_sums, atol=1e-5),
                            f"Attention row sum != 1.0 for L={L}")

    def test_adv_07_transformer_defensive_dimension_guards(self):
        """
        Adversarial test: Reject tensors of rank 1, 2, 4, and 5 with clear ValueError.
        """
        net = TransformerRegimeNet(input_dim=14, d_model=32, nhead=4)
        bad_inputs = [
            torch.randn(14),                # 1D
            torch.randn(10, 14),            # 2D
            torch.randn(2, 4, 20, 14),      # 4D
            torch.randn(1, 2, 3, 4, 5),     # 5D
        ]
        for x in bad_inputs:
            with self.assertRaises(ValueError) as ctx:
                net(x)
            self.assertIn("3D sequential input", str(ctx.exception))

    def test_adv_08_get_attention_weights_contract(self):
        """
        Validates DeepRegimeClassifier.get_attention_weights API contract:
        - Rejects non-transformer model types with informative ValueError.
        - Handles X=None (cached), 2D single sequence, 3D batch, and aggregate='mean'.
        """
        cnn_clf = DeepRegimeClassifier(model_name='cnn1d', n_epochs=1)
        with self.assertRaises(ValueError) as ctx:
            cnn_clf.get_attention_weights()
        self.assertIn("Transformer", str(ctx.exception))

        tf_clf = DeepRegimeClassifier(model_name='transformer', n_epochs=1)
        tf_clf.fit(self.X_tr_s[:40], self.y_tr_s[:40])

        # 1. Single 2D sequence (L, F)
        attn_single = tf_clf.get_attention_weights(self.X_te_s[0])
        self.assertEqual(attn_single.shape, (20, 20))
        np.testing.assert_allclose(attn_single.sum(axis=-1), 1.0, atol=1e-5)

        # 2. 3D batch (N, L, F) without aggregate
        attn_batch = tf_clf.get_attention_weights(self.X_te_s[:5], aggregate=None)
        self.assertEqual(attn_batch.shape, (5, 20, 20))
        np.testing.assert_allclose(attn_batch.sum(axis=-1), 1.0, atol=1e-5)

        # 3. 3D batch with aggregate='mean'
        attn_mean = tf_clf.get_attention_weights(self.X_te_s[:5], aggregate='mean')
        self.assertEqual(attn_mean.shape, (20, 20))
        np.testing.assert_allclose(attn_mean.sum(axis=-1), 1.0, atol=1e-5)

    # =========================================================================
    # SUITE 3: MATHEMATICAL ORACLES FOR ROC & PR CURVES
    # =========================================================================

    def test_adv_09_roc_analytical_oracles(self):
        """
        Verify _compute_roc_curve against exact analytical values and edge cases.
        """
        # Case 1: Perfect classifier (AUC = 1.0)
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9])
        fpr, tpr, auc = _compute_roc_curve(y_true, y_prob)
        self.assertAlmostEqual(auc, 1.0, places=5)
        self.assertEqual(fpr[0], 0.0)
        self.assertEqual(tpr[-1], 1.0)

        # Case 2: Completely inverted classifier (AUC = 0.0)
        y_prob_inv = np.array([0.9, 0.8, 0.2, 0.1])
        _, _, auc_inv = _compute_roc_curve(y_true, y_prob_inv)
        self.assertAlmostEqual(auc_inv, 0.0, places=5)

        # Case 3: Identical / constant probabilities (AUC = 0.5)
        y_prob_const = np.array([0.5, 0.5, 0.5, 0.5])
        _, _, auc_const = _compute_roc_curve(y_true, y_prob_const)
        self.assertAlmostEqual(auc_const, 0.5, places=5)

        # Case 4: Single class: All 0s (no positives) -> safe fallback to AUC = 0.5
        y_all_0 = np.array([0, 0, 0, 0])
        _, _, auc_0 = _compute_roc_curve(y_all_0, y_prob)
        self.assertEqual(auc_0, 0.5)

        # Case 5: Single class: All 1s (no negatives) -> safe fallback to AUC = 0.5
        y_all_1 = np.array([1, 1, 1, 1])
        _, _, auc_1 = _compute_roc_curve(y_all_1, y_prob)
        self.assertEqual(auc_1, 0.5)

        # Case 6: Tied probabilities with mixed labels
        y_tied = np.array([0, 0, 1, 1])
        p_tied = np.array([0.3, 0.7, 0.7, 0.9])
        fpr, tpr, auc_tied = _compute_roc_curve(y_tied, p_tied)
        self.assertAlmostEqual(auc_tied, 0.875, places=5)

        # Monotonicity check
        self.assertTrue(np.all(np.diff(fpr) >= -1e-12))
        self.assertTrue(np.all(np.diff(tpr) >= -1e-12))

    def test_adv_10_pr_analytical_oracles(self):
        """
        Verify _compute_pr_curve against exact analytical values and edge cases.
        """
        # Case 1: Perfect classifier (AP = 1.0)
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9])
        recall, precision, ap = _compute_pr_curve(y_true, y_prob)
        self.assertAlmostEqual(ap, 1.0, places=5)
        self.assertEqual(recall[0], 0.0)
        self.assertEqual(precision[0], 1.0)
        self.assertEqual(recall[-1], 1.0)

        # Case 2: Constant probabilities
        y_prob_const = np.array([0.5, 0.5, 0.5, 0.5])
        rec, prec, ap_const = _compute_pr_curve(y_true, y_prob_const)
        self.assertAlmostEqual(ap_const, 0.5, places=5)

        # Case 3: Single class: All 0s (no positives) -> safe fallback to AP = 0.0
        y_all_0 = np.array([0, 0, 0, 0])
        rec0, prec0, ap0 = _compute_pr_curve(y_all_0, y_prob)
        self.assertEqual(ap0, 0.0)

        # Case 4: Single class: All 1s -> AP = 1.0
        y_all_1 = np.array([1, 1, 1, 1])
        rec1, prec1, ap1 = _compute_pr_curve(y_all_1, y_prob)
        self.assertAlmostEqual(ap1, 1.0, places=5)

        # Case 5: 2D column input polymorphism (N, 1)
        rec_2d, prec_2d, ap_2d = _compute_pr_curve(y_true.reshape(-1, 1), y_prob.reshape(-1, 1))
        self.assertAlmostEqual(ap_2d, ap, places=5)

    # =========================================================================
    # SUITE 4: DIAGNOSTIC VISUALIZATIONS BOUNDARY RESILIENCE
    # =========================================================================

    def test_adv_11_plot_training_curves_boundary_checks(self):
        """
        Test plot_training_curves_grid with edge-case configurations.
        """
        # Empty dict must raise ValueError
        with self.assertRaises(ValueError):
            plot_training_curves_grid({})

        # Single model with missing val_loss
        single_dict = {
            "OnlyTrain": {"train": [0.8, 0.6, 0.5, 0.4]}
        }
        out_path = os.path.join(self.temp_dir, "single_train_curve.png")
        fig = plot_training_curves_grid(single_dict, save_path=out_path)
        self.assertTrue(os.path.exists(out_path))
        self.assertGreater(os.path.getsize(out_path), 5000)

    def test_adv_12_plot_confusion_matrix_boundary_checks(self):
        """
        Test plot_confusion_matrix_grid with zero-division cases.
        """
        # Empty dict must raise ValueError
        with self.assertRaises(ValueError):
            plot_confusion_matrix_grid({})

        # Zero TP and FP (all predictions 0)
        cm_zero_tp = {
            "AllNegative": np.array([[50, 0], [50, 0]])
        }
        out_path = os.path.join(self.temp_dir, "cm_zero_tp.png")
        fig = plot_confusion_matrix_grid(cm_zero_tp, save_path=out_path)
        self.assertTrue(os.path.exists(out_path))
        self.assertGreater(os.path.getsize(out_path), 5000)

    def test_adv_13_plot_feature_importance_mixed_2d_3d_models(self):
        """
        Stress test: Pass a dict containing both a 2D tabular classifier (LogisticRegression)
        and 3D sequential classifier (CNN1D) into plot_feature_importance_heatmap simultaneously.
        """
        from Market_Regime_Prediction.torch_models import DeepRegimeClassifier
        from Market_Regime_Prediction.LogisticRegression import LogisticRegression

        lr = LogisticRegression(learning_rate=0.01, n_epochs=100)
        lr.fit(self.X_tr, self.y_tr)

        cnn = DeepRegimeClassifier(model_name='cnn1d', n_epochs=5, patience=5)
        cnn.fit(self.X_tr_s, self.y_tr_s)

        models_dict = {
            "LogisticRegression": lr,
            "CNN1D": cnn,
        }
        # Provide dict with both 2d and 3d representation
        X_test_combo = {
            "2d": self.X_te,
            "3d": self.X_te_s,
        }
        out_path = os.path.join(self.temp_dir, "mixed_importance.png")
        fig = plot_feature_importance_heatmap(
            models_dict,
            X_test_combo,
            self.y_te_s,
            self.pipe.feature_names,
            metric='prob_shift',
            n_repeats=2,
            save_path=out_path,
        )
        self.assertTrue(os.path.exists(out_path))
        self.assertGreater(os.path.getsize(out_path), 5000)

    # =========================================================================
    # SUITE 5: REAL SPY 5-YEAR DOWNSTREAM BACKTESTING INVARIANTS
    # =========================================================================

    def test_adv_14_real_spy_backtesting_superiority_and_threshold_sensitivity(self):
        """
        Empirically verifies:
        1. CNN1D and Transformer hybrid strategies on 240 out-of-sample days:
           - Total Return >= 0.15 (15%)
           - Sharpe Ratio >= 1.40
           - Max Drawdown >= -0.08 (better than -8%)
           - Exceeds base MACrossover return (13.72%) and Sharpe (1.27)
        2. Sensitivity analysis across thresholds [0.18, 0.22, 0.25]:
           All produce positive returns and valid trades without crashing.
        """
        # Train fresh models with seed 42
        cnn_clf = DeepRegimeClassifier(
            model_name='cnn1d',
            filters=32,
            dropout=0.2,
            n_epochs=50,
            patience=15,
            random_state=42,
        )
        cnn_clf.fit(self.X_tr_s, self.y_tr_s, X_val=self.X_val_s, y_val=self.y_val_s)

        tf_clf = DeepRegimeClassifier(
            model_name='transformer',
            hidden_dim=32,
            nhead=4,
            dim_feedforward=64,
            num_layers=1,
            dropout=0.2,
            n_epochs=50,
            patience=15,
            random_state=42,
        )
        tf_clf.fit(self.X_tr_s, self.y_tr_s, X_val=self.X_val_s, y_val=self.y_val_s)

        # Baseline MACrossover metrics
        base_ret = float(self.base_ma_metrics["Total returns"])
        base_sharpe = float(self.base_ma_metrics["Sharpe Ratio"])
        self.assertAlmostEqual(base_ret, 0.1372, places=3)
        self.assertAlmostEqual(base_sharpe, 1.27, delta=0.05)

        # Evaluate CNN1D
        cnn_preds = cnn_clf.predict(self.X_te_s, threshold=0.22, smooth_window=3)
        base = MACrossover(self.test_df.copy(), 20, 50, full_data=self.df_raw)
        hyb_cnn = MLHybridStrategy(self.test_df.copy(), base, cnn_preds)
        bt_cnn = Backtest(self.test_df.copy(), hyb_cnn, starting_cash=100000.0)
        bt_cnn.run()
        m_cnn = bt_cnn.cal_metrics()

        cnn_ret = float(m_cnn["Total returns"])
        cnn_sharpe = float(m_cnn["Sharpe Ratio"])
        cnn_dd = float(m_cnn["Max drawdown"])

        self.assertGreaterEqual(cnn_ret, 0.15, f"CNN Return {cnn_ret:.4f} < 0.15")
        self.assertGreaterEqual(cnn_sharpe, 1.40, f"CNN Sharpe {cnn_sharpe:.4f} < 1.40")
        self.assertGreaterEqual(cnn_dd, -0.08, f"CNN MaxDD {cnn_dd:.4f} < -0.08")
        self.assertGreater(cnn_ret, base_ret, f"CNN Return {cnn_ret:.4f} <= base {base_ret:.4f}")
        self.assertGreater(cnn_sharpe, base_sharpe, f"CNN Sharpe {cnn_sharpe:.4f} <= base {base_sharpe:.4f}")

        # Evaluate Transformer
        tf_preds = tf_clf.predict(self.X_te_s, threshold=0.22, smooth_window=3)
        base_tf = MACrossover(self.test_df.copy(), 20, 50, full_data=self.df_raw)
        hyb_tf = MLHybridStrategy(self.test_df.copy(), base_tf, tf_preds)
        bt_tf = Backtest(self.test_df.copy(), hyb_tf, starting_cash=100000.0)
        bt_tf.run()
        m_tf = bt_tf.cal_metrics()

        tf_ret = float(m_tf["Total returns"])
        tf_sharpe = float(m_tf["Sharpe Ratio"])
        tf_dd = float(m_tf["Max drawdown"])

        self.assertGreaterEqual(tf_ret, 0.15, f"Transformer Return {tf_ret:.4f} < 0.15")
        self.assertGreaterEqual(tf_sharpe, 1.40, f"Transformer Sharpe {tf_sharpe:.4f} < 1.40")
        self.assertGreaterEqual(tf_dd, -0.08, f"Transformer MaxDD {tf_dd:.4f} < -0.08")
        self.assertGreater(tf_ret, base_ret, f"Transformer Return {tf_ret:.4f} <= base {base_ret:.4f}")
        self.assertGreater(tf_sharpe, base_sharpe, f"Transformer Sharpe {tf_sharpe:.4f} <= base {base_sharpe:.4f}")

        # Threshold sensitivity sweep
        for thresh in [0.18, 0.22, 0.25]:
            preds_sweep = cnn_clf.predict(self.X_te_s, threshold=thresh, smooth_window=3)
            base_s = MACrossover(self.test_df.copy(), 20, 50, full_data=self.df_raw)
            hyb_s = MLHybridStrategy(self.test_df.copy(), base_s, preds_sweep)
            bt_s = Backtest(self.test_df.copy(), hyb_s, starting_cash=100000.0)
            bt_s.run()
            m_s = bt_s.cal_metrics()
            ret_s = float(m_s["Total returns"])
            self.assertGreater(ret_s, 0.05, f"Sensitivity return {ret_s:.4f} too low at thresh={thresh}")
            self.assertFalse(math.isnan(ret_s), f"NaN return at thresh={thresh}")


if __name__ == '__main__':
    unittest.main()
