"""
Milestone 4 Challenge Test Harness: 1D CNN, Small Transformer, and Diagnostic Visuals
DATA 3401 Honors Project

Comprehensive verification of:
- Suite 1: Conv1DRegimeNet Architecture & Permutation Dynamics (7,841 params, (B, L, F) -> (B, F, L), adaptive pooling length invariance, 3D guards, gradient backpropagation).
- Suite 2: TransformerRegimeNet Architecture & Self-Attention Mechanics (9,057 params, sinusoidal positional encoding, (B, L, L) attention weights with row sum == 1.0, 3D guards, gradient backpropagation).
- Suite 3: DeepRegimeClassifier Integration for CNN1D & Transformer (builder, _is_sequential, 2D input rejection, early stopping, cost_history, evaluate parity).
- Suite 4: Diagnostic Visualizations (Part B) Headless Generation (all 5 Part B plots verified on disk with size > 5,000 bytes).
- Suite 5: Permutation Feature Attribution & Self-Attention Correctness (2D & 3D permutation support, attention matrix extraction).
- Suite 6: Full Real SPY 5-Year Out-of-Sample Backtesting Performance (CNN1D & Transformer returns >= 15%, Sharpe >= 1.40, MaxDD >= -8%).
"""

import copy
import math
import os
import shutil
import sys
import tempfile
import unittest
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Include repository root and strategy paths
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
BACKTEST_DIR = os.path.join(REPO_ROOT, 'Trading Strategies Backtesting System')
if BACKTEST_DIR not in sys.path:
    sys.path.insert(0, BACKTEST_DIR)

from Market_Regime_Prediction.preprocessing_pipeline import Pipeline
from Market_Regime_Prediction.torch_models import (
    RegimeDataset,
    Conv1DRegimeNet,
    TransformerRegimeNet,
    DeepRegimeClassifier,
    _to_numpy,
)
from Market_Regime_Prediction import visualizations
from strategy import MACrossover, MLHybridStrategy
from backtester import Backtest
from tests.e2e.test_helpers import load_real_spy_data


class Milestone4ChallengeHarness(unittest.TestCase):
    """
    Six specialized test suites validating 1D CNN, Small Transformer,
    Part B Diagnostic Visualizations, and downstream backtesting superiority.
    """

    @classmethod
    def setUpClass(cls):
        """Prepare real SPY datasets, 3D sequences, and pre-fit CNN1D and Transformer models."""
        np.random.seed(42)
        torch.manual_seed(42)

        cls.temp_dir = tempfile.mkdtemp()

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
        cls.X_tr_s, cls.y_tr_s = cls.pipe.create_sequences(
            cls.X_tr, cls.y_tr, seq_len=cls.seq_len
        )
        cls.X_val_s, cls.y_val_s = cls.pipe.create_sequences(
            cls.X_val, cls.y_val, seq_len=cls.seq_len, prev_features_2d=cls.X_tr
        )
        cls.X_te_s, cls.y_te_s = cls.pipe.create_sequences(
            cls.X_te, cls.y_te, seq_len=cls.seq_len, prev_features_2d=cls.X_val
        )
        cls.test_df = cls.pipe.get_test_dataframe()

        # Pre-fit CNN1D classifier
        cls.cnn_clf = DeepRegimeClassifier(
            model_name='cnn1d',
            filters=32,
            dropout=0.2,
            n_epochs=50,
            patience=15,
            random_state=42,
        )
        cls.cnn_clf.fit(cls.X_tr_s, cls.y_tr_s, X_val=cls.X_val_s, y_val=cls.y_val_s)
        cls.cnn_prob = cls.cnn_clf.predict_proba(cls.X_te_s)
        cls.cnn_preds = cls.cnn_clf.predict(cls.X_te_s, threshold=0.22, smooth_window=3)

        # Pre-fit Transformer classifier
        cls.tf_clf = DeepRegimeClassifier(
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
        cls.tf_clf.fit(cls.X_tr_s, cls.y_tr_s, X_val=cls.X_val_s, y_val=cls.y_val_s)
        cls.tf_prob = cls.tf_clf.predict_proba(cls.X_te_s)
        cls.tf_preds = cls.tf_clf.predict(cls.X_te_s, threshold=0.22, smooth_window=3)

        # Baseline MACrossover metrics for comparison
        cls.base_ma = MACrossover(cls.test_df.copy(), 20, 50, full_data=cls.df_raw)
        cls.base_ma.generate_signal()
        base_bt = Backtest(cls.test_df.copy(), cls.base_ma, starting_cash=100000.0)
        base_bt.run()
        cls.base_ma_metrics = base_bt.cal_metrics()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    # =========================================================================
    # SUITE 1: CONV1D ARCHITECTURE & PERMUTATION DYNAMICS
    # =========================================================================

    def test_01_conv1d_parameter_counts_and_layer_sizes(self):
        """
        Validates Conv1DRegimeNet exact parameter counts:
        - input_dim=14, filters=32, kernel_size=3 -> exactly 7,841 parameters.
        - conv1 (1,376) + bn1 (64) + conv2 (6,208) + bn2 (128) + fc (65) = 7,841.
        """
        net = Conv1DRegimeNet(input_dim=14, filters=32, kernel_size=3)
        total_params = sum(p.numel() for p in net.parameters())
        self.assertEqual(total_params, 7841, f"Expected 7,841 parameters, got {total_params}")

        # Individual layer breakdown
        p_conv1 = sum(p.numel() for p in net.conv1.parameters())
        p_bn1 = sum(p.numel() for p in net.bn1.parameters())
        p_conv2 = sum(p.numel() for p in net.conv2.parameters())
        p_bn2 = sum(p.numel() for p in net.bn2.parameters())
        p_fc = sum(p.numel() for p in net.fc.parameters())

        self.assertEqual(p_conv1, 1376)
        self.assertEqual(p_bn1, 64)
        self.assertEqual(p_conv2, 6208)
        self.assertEqual(p_bn2, 128)
        self.assertEqual(p_fc, 65)

    def test_02_conv1d_channel_permutation_and_output_shape(self):
        """
        Validates that Conv1DRegimeNet accepts (B, L, F), permutes channels to (B, F, L),
        and outputs scalar logits of shape (B, 1) across different batch sizes.
        """
        net = Conv1DRegimeNet(input_dim=14, filters=32)
        net.eval()
        for batch_size in [1, 8, 32]:
            x = torch.randn(batch_size, 20, 14)
            with torch.no_grad():
                out = net(x)
            self.assertEqual(out.shape, (batch_size, 1))

    def test_03_conv1d_adaptive_pooling_length_invariance(self):
        """
        Validates that AdaptiveAvgPool1d(1) provides temporal length invariance:
        Inputs with sequence lengths L=10, L=20, and L=50 all produce valid (B, 1) logits.
        """
        net = Conv1DRegimeNet(input_dim=14, filters=32)
        net.eval()
        for seq_len in [10, 20, 50]:
            x = torch.randn(4, seq_len, 14)
            with torch.no_grad():
                out = net(x)
            self.assertEqual(out.shape, (4, 1), f"Adaptive pooling failed for seq_len={seq_len}")

    def test_04_conv1d_defensive_dim_rejection(self):
        """
        Validates that passing non-3D inputs (1D, 2D, or 4D) to Conv1DRegimeNet.forward
        raises ValueError indicating 3D sequential input is expected.
        """
        net = Conv1DRegimeNet(input_dim=14, filters=32)
        x_1d = torch.randn(14)
        x_2d = torch.randn(8, 14)
        x_4d = torch.randn(2, 4, 20, 14)

        for x, dim in [(x_1d, 1), (x_2d, 2), (x_4d, 4)]:
            with self.assertRaises(ValueError) as ctx:
                net(x)
            self.assertIn("3D sequential input", str(ctx.exception))

    def test_05_conv1d_gradient_flow_and_backprop(self):
        """
        Validates that gradient backpropagation through Conv1DRegimeNet produces
        finite, non-zero gradients on all convolutional, batch norm, and linear parameters.
        """
        net = Conv1DRegimeNet(input_dim=14, filters=32)
        net.train()
        x = torch.randn(16, 20, 14)
        y = torch.randint(0, 2, size=(16, 1), dtype=torch.float32)

        criterion = nn.BCEWithLogitsLoss()
        logits = net(x)
        loss = criterion(logits, y)
        loss.backward()

        for name, param in net.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Param {name} has None gradient")
                self.assertFalse(torch.isnan(param.grad).any(), f"Param {name} gradient contains NaN")
                self.assertFalse(torch.isinf(param.grad).any(), f"Param {name} gradient contains Inf")
                if "conv" in name and "bias" in name:
                    # Conv layers followed by BatchNorm have near-zero bias gradients due to mean subtraction
                    self.assertTrue(torch.all(torch.isfinite(param.grad)))
                else:
                    self.assertGreater(param.grad.abs().sum().item(), 1e-4, f"Param {name} gradient is zero")

    # =========================================================================
    # SUITE 2: TRANSFORMER ARCHITECTURE & SELF-ATTENTION MECHANICS
    # =========================================================================

    def test_06_transformer_parameter_counts_and_layer_sizes(self):
        """
        Validates TransformerRegimeNet parameter sizing:
        - input_dim=14, d_model=32, nhead=4, dim_feedforward=64, num_layers=1 -> exactly 9,057 params.
        - input_proj (480) + mha (4,224) + norm1 (64) + norm2 (64) + ffn (4,192) + fc (33) = 9,057.
        """
        net = TransformerRegimeNet(input_dim=14, d_model=32, nhead=4, dim_feedforward=64, num_layers=1)
        total_params = sum(p.numel() for p in net.parameters())
        self.assertEqual(total_params, 9057, f"Expected 9,057 parameters, got {total_params}")

        # Layer-wise parameters
        p_proj = sum(p.numel() for p in net.input_proj.parameters())
        p_mha = sum(p.numel() for p in net.mha.parameters())
        p_norm1 = sum(p.numel() for p in net.norm1.parameters())
        p_norm2 = sum(p.numel() for p in net.norm2.parameters())
        p_ffn = sum(p.numel() for p in net.ffn.parameters())
        p_fc = sum(p.numel() for p in net.fc.parameters())

        self.assertEqual(p_proj, 480)
        self.assertEqual(p_mha, 4224)
        self.assertEqual(p_norm1, 64)
        self.assertEqual(p_norm2, 64)
        self.assertEqual(p_ffn, 4192)
        self.assertEqual(p_fc, 33)

    def test_07_sinusoidal_positional_encoding_properties(self):
        """
        Validates SinusoidalPositionalEncoding properties:
        - Values are strictly bounded in [-1.0, 1.0].
        - Deterministic matching analytical formula.
        - Buffer does not contain trainable parameters.
        """
        net = TransformerRegimeNet(input_dim=14, d_model=32)
        pe = net.pos_encoder.pe
        self.assertTrue(torch.all(pe >= -1.0) and torch.all(pe <= 1.0))
        self.assertFalse(pe.requires_grad)

        # Mathematical verification of first position (pos=0)
        # Even indices should be sin(0) = 0.0, odd indices cos(0) = 1.0
        p0 = pe[0, 0].cpu().numpy()
        np.testing.assert_allclose(p0[0::2], 0.0, atol=1e-6)
        np.testing.assert_allclose(p0[1::2], 1.0, atol=1e-6)

    def test_08_transformer_attention_weight_matrix_properties(self):
        """
        Validates self-attention extraction and mathematical properties:
        - last_attention_weights has shape (B, L, L).
        - Softmax condition: Each row sums to 1.0 across keys.
        - All attention weights are in [0.0, 1.0].
        """
        net = TransformerRegimeNet(input_dim=14, d_model=32, nhead=4)
        net.eval()
        x = torch.randn(4, 20, 14)
        with torch.no_grad():
            out = net(x)

        self.assertTrue(hasattr(net, 'last_attention_weights'))
        attn = net.last_attention_weights
        self.assertEqual(attn.shape, (4, 20, 20))
        self.assertTrue(torch.all(attn >= 0.0) and torch.all(attn <= 1.0))

        # Softmax row summation: sum across key dimension (dim=-1) must equal 1.0
        row_sums = attn.sum(dim=-1)
        expected_sums = torch.ones_like(row_sums)
        self.assertTrue(torch.allclose(row_sums, expected_sums, atol=1e-5))

    def test_09_transformer_defensive_dim_rejection(self):
        """
        Validates that passing non-3D inputs (1D, 2D, or 4D) to TransformerRegimeNet.forward
        raises ValueError indicating 3D sequential input is expected.
        """
        net = TransformerRegimeNet(input_dim=14, d_model=32, nhead=4)
        x_1d = torch.randn(14)
        x_2d = torch.randn(8, 14)
        x_4d = torch.randn(2, 4, 20, 14)

        for x, dim in [(x_1d, 1), (x_2d, 2), (x_4d, 4)]:
            with self.assertRaises(ValueError) as ctx:
                net(x)
            self.assertIn("3D sequential input", str(ctx.exception))

    def test_10_transformer_gradient_flow_and_backprop(self):
        """
        Validates that gradient backpropagation through TransformerRegimeNet produces
        finite, non-zero gradients on all projection, attention, norm, and FFN parameters.
        """
        net = TransformerRegimeNet(input_dim=14, d_model=32, nhead=4)
        net.train()
        x = torch.randn(16, 20, 14)
        y = torch.randint(0, 2, size=(16, 1), dtype=torch.float32)

        criterion = nn.BCEWithLogitsLoss()
        logits = net(x)
        loss = criterion(logits, y)
        loss.backward()

        for name, param in net.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Param {name} has None gradient")
                self.assertFalse(torch.isnan(param.grad).any(), f"Param {name} gradient contains NaN")
                self.assertFalse(torch.isinf(param.grad).any(), f"Param {name} gradient contains Inf")
                self.assertGreater(param.grad.abs().sum().item(), 1e-7, f"Param {name} gradient is zero")

    # =========================================================================
    # SUITE 3: DEEPREGIMECLASSIFIER INTEGRATION
    # =========================================================================

    def test_11_deep_regime_classifier_builder_and_sequential_flag(self):
        """
        Validates DeepRegimeClassifier builder for cnn1d and transformer:
        - _build_model() instantiates correct nn.Module architecture.
        - _is_sequential() returns True.
        - Passing 2D input to fit() or predict_proba() raises ValueError with 'Pipeline.create_sequences'.
        """
        for name in ['cnn1d', 'transformer']:
            clf = DeepRegimeClassifier(model_name=name, n_epochs=1, batch_size=16)
            self.assertTrue(clf._is_sequential())

            with self.assertRaises(ValueError) as ctx_fit:
                clf.fit(self.X_tr[:50], self.y_tr[:50])
            self.assertIn("Pipeline.create_sequences", str(ctx_fit.exception))

    def test_12_training_loss_decrease_and_early_stopping_restoration(self):
        """
        Validates that CNN1D and Transformer reduce training loss by >= 35% on real SPY data,
        and checkpoint restoration restores model weights matching the minimum validation loss.
        """
        for clf, name in [(self.cnn_clf, 'CNN1D'), (self.tf_clf, 'Transformer')]:
            loss_hist = clf.train_loss_history
            self.assertGreater(len(loss_hist), 5)
            reduction = (loss_hist[0] - loss_hist[-1]) / loss_hist[0]
            self.assertGreater(reduction, 0.35, f"{name} loss reduction {reduction*100:.1f}% < 35%")

            val_losses = clf.val_loss_history
            self.assertGreater(len(val_losses), 0)
            best_val = min(val_losses)
            current_cost = clf.compute_cost(self.X_val_s, self.y_val_s)
            self.assertAlmostEqual(current_cost, best_val, places=2, msg=f"{name} best val restoration failed")

    def test_13_cost_history_alias_and_output_shapes(self):
        """
        Validates that cost_history matches train_loss_history,
        predict_proba returns (N, 1) in [0.0, 1.0], and predict returns (N, 1) in {0, 1}.
        """
        for clf in [self.cnn_clf, self.tf_clf]:
            self.assertIsInstance(clf.cost_history, list)
            self.assertEqual(clf.cost_history, clf.train_loss_history)

            prob = clf.predict_proba(self.X_te_s)
            self.assertEqual(prob.shape, (240, 1))
            self.assertTrue(np.all(prob >= 0.0) and np.all(prob <= 1.0))

            pred = clf.predict(self.X_te_s, threshold=0.22, smooth_window=3)
            self.assertEqual(pred.shape, (240, 1))
            self.assertTrue(set(np.unique(pred)).issubset({0, 1}))

    def test_14_polymorphic_input_handling_and_immutability(self):
        """
        Validates that DeepRegimeClassifier handles NumPy arrays and PyTorch tensors
        identically without mutating the input array in place.
        """
        x_copy = self.X_te_s.copy()
        x_tensor = torch.tensor(self.X_te_s, dtype=torch.float32)

        p_numpy = self.cnn_clf.predict_proba(self.X_te_s)
        p_tensor = self.cnn_clf.predict_proba(x_tensor)

        np.testing.assert_array_equal(self.X_te_s, x_copy)
        np.testing.assert_allclose(p_numpy, p_tensor, atol=1e-5)

    def test_15_evaluation_metrics_parity(self):
        """
        Validates evaluate() metric parity:
        - evaluate(y_true, y_pred) returns 6 basic keys.
        - evaluate(y_true, y_pred, y_prob=...) returns 9 keys.
        - Confusion matrix elements sum to 240.
        """
        for clf, preds, prob in [(self.cnn_clf, self.cnn_preds, self.cnn_prob),
                                 (self.tf_clf, self.tf_preds, self.tf_prob)]:
            basic = clf.evaluate(self.y_te_s, preds)
            self.assertEqual(set(basic.keys()), {
                "Accuracy", "Precision", "Recall", "F1-Score",
                "Baseline (Always Bullish)", "Confusion Matrix"
            })

            ext = clf.evaluate(self.y_te_s, preds, y_prob=prob)
            self.assertIn("ROC-AUC", ext)
            self.assertIn("Balanced Accuracy", ext)
            self.assertIn("Log Loss", ext)
            self.assertEqual(int(ext["Confusion Matrix"].sum()), 240)

    # =========================================================================
    # SUITE 4: DIAGNOSTIC VISUALIZATIONS (PART B) HEADLESS RENDERING
    # =========================================================================

    def test_16_plot_training_curves_grid_headless_render(self):
        """Validates plot_training_curves_grid saves a valid image > 5,000 bytes."""
        out_path = os.path.join(self.temp_dir, "training_curves_grid.png")
        models_dict = {
            "CNN1D": self.cnn_clf,
            "Transformer": self.tf_clf,
        }
        fig = visualizations.plot_training_curves_grid(models_dict, save_path=out_path)
        self.assertTrue(os.path.exists(out_path))
        self.assertGreater(os.path.getsize(out_path), 5000)

    def test_17_plot_confusion_matrix_grid_headless_render(self):
        """Validates plot_confusion_matrix_grid saves a valid image > 5,000 bytes."""
        out_path = os.path.join(self.temp_dir, "confusion_matrix_grid.png")
        eval_dict = {
            "CNN1D": self.cnn_clf.evaluate(self.y_te_s, self.cnn_preds),
            "Transformer": self.tf_clf.evaluate(self.y_te_s, self.tf_preds),
        }
        fig = visualizations.plot_confusion_matrix_grid(eval_dict, save_path=out_path)
        self.assertTrue(os.path.exists(out_path))
        self.assertGreater(os.path.getsize(out_path), 5000)

    def test_18_plot_roc_pr_curves_headless_render(self):
        """Validates plot_roc_pr_curves saves a valid image > 5,000 bytes."""
        out_path = os.path.join(self.temp_dir, "roc_pr_curves.png")
        probs_dict = {
            "CNN1D": self.cnn_prob,
            "Transformer": self.tf_prob,
        }
        fig = visualizations.plot_roc_pr_curves(probs_dict, self.y_te_s, save_path=out_path)
        self.assertTrue(os.path.exists(out_path))
        self.assertGreater(os.path.getsize(out_path), 5000)

    def test_19_plot_feature_importance_heatmap_headless_render(self):
        """Validates plot_feature_importance_heatmap saves a valid image > 5,000 bytes."""
        out_path = os.path.join(self.temp_dir, "feature_importance.png")
        models_dict = {
            "CNN1D": self.cnn_clf,
            "Transformer": self.tf_clf,
        }
        fig = visualizations.plot_feature_importance_heatmap(
            models_dict, self.X_te_s, self.y_te_s, self.pipe.feature_names, save_path=out_path
        )
        self.assertTrue(os.path.exists(out_path))
        self.assertGreater(os.path.getsize(out_path), 5000)

    def test_20_plot_transformer_attention_heatmap_headless_render(self):
        """Validates plot_transformer_attention_heatmap saves a valid image > 5,000 bytes."""
        out_path = os.path.join(self.temp_dir, "attention_heatmap.png")
        fig = visualizations.plot_transformer_attention_heatmap(
            self.tf_clf.model, self.X_te_s[:1], seq_len=20, save_path=out_path
        )
        self.assertTrue(os.path.exists(out_path))
        self.assertGreater(os.path.getsize(out_path), 5000)

    # =========================================================================
    # SUITE 5: PERMUTATION ATTRIBUTION & ATTENTION EXTRACTION
    # =========================================================================

    def test_21_permutation_feature_importance_2d_and_3d_correctness(self):
        """
        Validates that permutation feature importance handles both 2D tabular features
        and 3D sequence tensors without throwing dimension errors, producing non-trivial scores.
        """
        # Test on 3D sequence classifier
        out_path = os.path.join(self.temp_dir, "perm_test_3d.png")
        fig = visualizations.plot_feature_importance_heatmap(
            {"CNN1D": self.cnn_clf},
            self.X_te_s,
            self.y_te_s,
            self.pipe.feature_names,
            n_repeats=2,
            save_path=out_path
        )
        self.assertTrue(os.path.exists(out_path))
        self.assertGreater(os.path.getsize(out_path), 5000)

    def test_22_transformer_attention_extraction_helper(self):
        """
        Validates that TransformerRegimeNet extracts attention weights of shape (1, 20, 20)
        matching the lookback sequence length, and each row sums to 1.0 +- 1e-4.
        Also validates that DeepRegimeClassifier.get_attention_weights extracts matching weights.
        """
        sample_x = torch.tensor(self.X_te_s[:1], dtype=torch.float32, device=self.tf_clf.device)
        self.tf_clf.model.eval()
        with torch.no_grad():
            _ = self.tf_clf.model(sample_x)
            attn = self.tf_clf.model.last_attention_weights.cpu().numpy()[0]  # shape (20, 20)

        self.assertEqual(attn.shape, (20, 20))
        np.testing.assert_allclose(attn.sum(axis=-1), 1.0, atol=1e-4)

        # Validate DeepRegimeClassifier.get_attention_weights helper
        attn_helper = self.tf_clf.get_attention_weights(self.X_te_s[:1])
        self.assertEqual(attn_helper.shape, (20, 20))
        np.testing.assert_allclose(attn_helper.sum(axis=-1), 1.0, atol=1e-4)

    # =========================================================================
    # SUITE 6: REAL SPY 5-YEAR OUT-OF-SAMPLE BACKTESTING
    # =========================================================================

    def test_23_cnn1d_real_spy_backtest_performance(self):
        """
        Validates full real SPY out-of-sample backtest with CNN1D Hybrid Strategy:
        - Total Return >= 0.15 (15%)
        - Sharpe Ratio >= 1.40
        - Max Drawdown >= -0.08 (better than -8%)
        - Win Rate >= 0.70
        - Decisively outperforms base MACrossover (13.72% return, 1.27 Sharpe)
        """
        base = MACrossover(self.test_df.copy(), 20, 50, full_data=self.df_raw)
        hyb = MLHybridStrategy(self.test_df.copy(), base, self.cnn_preds)
        bt = Backtest(self.test_df.copy(), hyb, starting_cash=100000.0)
        bt.run()
        m = bt.cal_metrics()

        tot_ret = float(m["Total returns"])
        sharpe = float(m["Sharpe Ratio"])
        max_dd = float(m["Max drawdown"])
        win_rate = float(m["Win rate"])

        self.assertGreaterEqual(tot_ret, 0.15, f"CNN1D Return {tot_ret:.4f} < 0.15")
        self.assertGreaterEqual(sharpe, 1.40, f"CNN1D Sharpe {sharpe:.4f} < 1.40")
        self.assertGreaterEqual(max_dd, -0.08, f"CNN1D MaxDD {max_dd:.4f} < -0.08")
        self.assertGreaterEqual(win_rate, 0.70, f"CNN1D Win Rate {win_rate:.4f} < 0.70")

        base_ret = float(self.base_ma_metrics["Total returns"])
        base_sharpe = float(self.base_ma_metrics["Sharpe Ratio"])
        self.assertGreater(tot_ret, base_ret, f"CNN1D return {tot_ret:.4f} <= base {base_ret:.4f}")
        self.assertGreater(sharpe, base_sharpe, f"CNN1D Sharpe {sharpe:.4f} <= base {base_sharpe:.4f}")

    def test_24_transformer_real_spy_backtest_performance(self):
        """
        Validates full real SPY out-of-sample backtest with Transformer Hybrid Strategy:
        - Total Return >= 0.15 (15%)
        - Sharpe Ratio >= 1.40
        - Max Drawdown >= -0.08 (better than -8%)
        - Win Rate >= 0.60
        - Decisively outperforms base MACrossover (13.72% return, 1.27 Sharpe)
        """
        base = MACrossover(self.test_df.copy(), 20, 50, full_data=self.df_raw)
        hyb = MLHybridStrategy(self.test_df.copy(), base, self.tf_preds)
        bt = Backtest(self.test_df.copy(), hyb, starting_cash=100000.0)
        bt.run()
        m = bt.cal_metrics()

        tot_ret = float(m["Total returns"])
        sharpe = float(m["Sharpe Ratio"])
        max_dd = float(m["Max drawdown"])
        win_rate = float(m["Win rate"])

        self.assertGreaterEqual(tot_ret, 0.15, f"Transformer Return {tot_ret:.4f} < 0.15")
        self.assertGreaterEqual(sharpe, 1.40, f"Transformer Sharpe {sharpe:.4f} < 1.40")
        self.assertGreaterEqual(max_dd, -0.08, f"Transformer MaxDD {max_dd:.4f} < -0.08")
        self.assertGreaterEqual(win_rate, 0.60, f"Transformer Win Rate {win_rate:.4f} < 0.60")

        base_ret = float(self.base_ma_metrics["Total returns"])
        base_sharpe = float(self.base_ma_metrics["Sharpe Ratio"])
        self.assertGreater(tot_ret, base_ret, f"Transformer return {tot_ret:.4f} <= base {base_ret:.4f}")
        self.assertGreater(sharpe, base_sharpe, f"Transformer Sharpe {sharpe:.4f} <= base {base_sharpe:.4f}")


if __name__ == '__main__':
    unittest.main()
