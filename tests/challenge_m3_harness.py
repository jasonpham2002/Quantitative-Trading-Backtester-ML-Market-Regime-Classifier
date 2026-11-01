"""
Milestone 3 Challenge Test Harness: Recurrent Models (LSTM & GRU)
DATA 3401 Honors Project

Comprehensive verification of:
- Suite 1: Recurrent Architecture & Parameter Parity (LSTM 6,177 vs GRU 4,641 parameters, 75% cell ratio, dropout conditioning).
- Suite 2: Temporal Memory & Sequence Sensitivity (BPTT memory retention, order non-commutativity).
- Suite 3: Training Dynamics & Early Stopping (loss decrease, checkpoint restoration to min val loss).
- Suite 4: Gradient Flow & Numerical Robustness (finite non-zero gradients on all recurrent gates, bounded gradient norms).
- Suite 5: Benchmark Evaluation Metrics (Acc >= 0.82, Prec >= 0.85, Rec >= 0.90, F1 >= 0.88, ROC-AUC >= 0.85, Log Loss <= 0.46).
- Suite 6: Full Real SPY 5-Year E2E Backtester Integration (LSTM/GRU return >= 15%, Sharpe >= 1.40, MaxDD >= -8%).
"""

import copy
import os
import sys
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
    RecurrentRegimeNet,
    DeepRegimeClassifier,
    _to_numpy,
)
from strategy import MACrossover, MLHybridStrategy
from backtester import Backtest
from tests.e2e.test_helpers import load_real_spy_data


class Milestone3RecurrentChallengeHarness(unittest.TestCase):
    """
    Six specialized test suites validating architectural authenticity,
    temporal learning dynamics, evaluation metrics, and backtest superiority.
    """

    @classmethod
    def setUpClass(cls):
        """Prepare real SPY datasets, 3D sequences, and pre-fit recurrent models."""
        np.random.seed(42)
        torch.manual_seed(42)

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

        # Fit baseline LSTM classifier
        cls.lstm_clf = DeepRegimeClassifier(
            model_name='lstm',
            hidden_dim=32,
            num_layers=1,
            dropout=0.2,
            n_epochs=50,
            patience=15,
            random_state=42,
        )
        cls.lstm_clf.fit(cls.X_tr_s, cls.y_tr_s, X_val=cls.X_val_s, y_val=cls.y_val_s)
        cls.lstm_prob = cls.lstm_clf.predict_proba(cls.X_te_s)
        cls.lstm_preds = cls.lstm_clf.predict(cls.X_te_s, threshold=0.22, smooth_window=3)

        # Fit baseline GRU classifier
        cls.gru_clf = DeepRegimeClassifier(
            model_name='gru',
            hidden_dim=32,
            num_layers=1,
            dropout=0.2,
            n_epochs=50,
            patience=15,
            random_state=42,
        )
        cls.gru_clf.fit(cls.X_tr_s, cls.y_tr_s, X_val=cls.X_val_s, y_val=cls.y_val_s)
        cls.gru_prob = cls.gru_clf.predict_proba(cls.X_te_s)
        cls.gru_preds = cls.gru_clf.predict(cls.X_te_s, threshold=0.22, smooth_window=3)

        # Baseline MACrossover for comparison
        cls.base_ma = MACrossover(cls.test_df.copy(), 20, 50, full_data=cls.df_raw)
        cls.base_ma.generate_signal()
        base_bt = Backtest(cls.test_df.copy(), cls.base_ma, starting_cash=100000.0)
        base_bt.run()
        cls.base_ma_metrics = base_bt.cal_metrics()

    # =========================================================================
    # SUITE 1: RECURRENT ARCHITECTURE & PARAMETER PARITY
    # =========================================================================

    def test_01_lstm_gru_parameter_counts_and_gate_ratio(self):
        """
        Validates exact parameter sizing:
        - LSTM (14 input, 32 hidden, 1 layer) = 6,177 parameters (4 gates).
        - GRU (14 input, 32 hidden, 1 layer) = 4,641 parameters (3 gates).
        - GRU cell parameters are exactly 75.0% of LSTM cell parameters.
        """
        lstm_net = RecurrentRegimeNet(input_dim=14, hidden_dim=32, num_layers=1, cell_type='lstm')
        gru_net = RecurrentRegimeNet(input_dim=14, hidden_dim=32, num_layers=1, cell_type='gru')

        lstm_total = sum(p.numel() for p in lstm_net.parameters())
        gru_total = sum(p.numel() for p in gru_net.parameters())

        lstm_cell = sum(p.numel() for p in lstm_net.rnn.parameters())
        gru_cell = sum(p.numel() for p in gru_net.rnn.parameters())

        self.assertEqual(lstm_total, 6177, f"Expected 6,177 LSTM parameters, got {lstm_total}")
        self.assertEqual(gru_total, 4641, f"Expected 4,641 GRU parameters, got {gru_total}")
        self.assertEqual(lstm_cell, 6144, f"Expected 6,144 LSTM cell weights, got {lstm_cell}")
        self.assertEqual(gru_cell, 4608, f"Expected 4,608 GRU cell weights, got {gru_cell}")

        # Exact 3/4 gate parameter ratio (3 gates for GRU vs 4 gates for LSTM)
        gate_ratio = gru_cell / lstm_cell
        self.assertAlmostEqual(gate_ratio, 0.75, places=4, msg="GRU/LSTM cell parameter ratio must be 0.75")

    def test_02_dropout_conditioning(self):
        """
        Validates dropout handling:
        - Single-layer rnn_dropout must be 0.0 to prevent PyTorch UserWarning.
        - Multi-layer (num_layers=2) rnn_dropout must match the specified dropout rate.
        """
        with warnings.catch_warnings(record=True) as recorded_warnings:
            warnings.simplefilter("always")
            net_l1 = RecurrentRegimeNet(input_dim=14, hidden_dim=32, num_layers=1, dropout=0.2)
            self.assertEqual(net_l1.rnn.dropout, 0.0)
            # PyTorch emits no warning when num_layers=1 and dropout=0.0
            rnn_warnings = [w for w in recorded_warnings if "dropout option adds dropout after all but last" in str(w.message)]
            self.assertEqual(len(rnn_warnings), 0)

        net_l2 = RecurrentRegimeNet(input_dim=14, hidden_dim=32, num_layers=2, dropout=0.25)
        self.assertEqual(net_l2.rnn.dropout, 0.25)

    def test_03_hidden_state_extraction_parity(self):
        """
        Validates that slicing out[:, -1, :] strictly equals hn[-1] for both
        LSTM and GRU with zero numerical divergence.
        """
        x = torch.randn(8, 20, 14)

        # LSTM parity
        lstm = RecurrentRegimeNet(input_dim=14, hidden_dim=32, num_layers=1, cell_type='lstm')
        lstm.eval()
        with torch.no_grad():
            out_l, (hn_l, cn_l) = lstm.rnn(x)
            diff_l = (out_l[:, -1, :] - hn_l[-1]).abs().max().item()
            self.assertEqual(diff_l, 0.0, "LSTM out[:, -1, :] must match hn[-1] with 0.0 difference")
            logits_l = lstm(x)
            self.assertEqual(logits_l.shape, (8, 1))

        # GRU parity
        gru = RecurrentRegimeNet(input_dim=14, hidden_dim=32, num_layers=1, cell_type='gru')
        gru.eval()
        with torch.no_grad():
            out_g, hn_g = gru.rnn(x)
            diff_g = (out_g[:, -1, :] - hn_g[-1]).abs().max().item()
            self.assertEqual(diff_g, 0.0, "GRU out[:, -1, :] must match hn[-1] with 0.0 difference")
            logits_g = gru(x)
            self.assertEqual(logits_g.shape, (8, 1))

    def test_04_defensive_dim_rejection_in_forward(self):
        """
        Passing non-3D input to RecurrentRegimeNet.forward must raise ValueError
        with an informative message indicating 3D sequential input is expected.
        """
        net = RecurrentRegimeNet(input_dim=14, hidden_dim=32, num_layers=1)
        x_2d = torch.randn(8, 14)
        x_4d = torch.randn(2, 4, 20, 14)

        with self.assertRaises(ValueError) as ctx_2d:
            net(x_2d)
        self.assertIn("3D sequential input", str(ctx_2d.exception))

        with self.assertRaises(ValueError) as ctx_4d:
            net(x_4d)
        self.assertIn("3D sequential input", str(ctx_4d.exception))

    # =========================================================================
    # SUITE 2: TEMPORAL MEMORY & SEQUENCE SENSITIVITY
    # =========================================================================

    def test_05_temporal_memory_gradient_saliency_and_decay(self):
        """
        Validates authentic backpropagation through time (BPTT):
        - Output logits have non-zero gradient saliency with respect to input at t=0 (t-19 days).
        - Perturbing the earliest feature produces a non-zero shift in prediction.
        """
        model = self.lstm_clf.model
        model.eval()

        sample_x = torch.tensor(self.X_te_s[0:1], dtype=torch.float32, device=self.lstm_clf.device, requires_grad=True)
        logit = model(sample_x)
        logit.backward()

        grad = sample_x.grad.abs().cpu().numpy()[0]  # shape (20, 14)
        t0_grad = np.sum(grad[0])   # t - 19 (earliest lookback bar)
        t_final_grad = np.sum(grad[-1])  # t - 0 (latest lookback bar)

        self.assertGreater(t0_grad, 1e-6, "Input at t-19 must receive non-zero BPTT gradient")
        self.assertGreater(t_final_grad, 1e-6, "Input at t-0 must receive non-zero BPTT gradient")

        # Perturbation test: perturb feature 0 at t=0
        x_perturbed = sample_x.detach().clone()
        x_perturbed[0, 0, 0] += 2.0
        with torch.no_grad():
            logit_orig = model(sample_x).item()
            logit_pert = model(x_perturbed).item()
        delta = abs(logit_orig - logit_pert)
        self.assertGreater(delta, 1e-5, "Perturbing t-19 must produce non-zero logit delta")

    def test_06_temporal_order_non_commutativity(self):
        """
        Validates temporal order sensitivity:
        Reversing or shuffling the time dimension significantly alters predictions,
        proving the model captures temporal trajectory rather than a static bag of features.
        """
        X_orig = self.X_te_s
        X_reversed = X_orig[:, ::-1, :]

        p_orig_lstm = self.lstm_clf.predict_proba(X_orig)
        p_rev_lstm = self.lstm_clf.predict_proba(X_reversed)
        divergence_lstm = float(np.mean(np.abs(p_orig_lstm - p_rev_lstm)))

        p_orig_gru = self.gru_clf.predict_proba(X_orig)
        p_rev_gru = self.gru_clf.predict_proba(X_reversed)
        divergence_gru = float(np.mean(np.abs(p_orig_gru - p_rev_gru)))

        self.assertGreater(divergence_lstm, 0.05, f"LSTM mean reversal divergence {divergence_lstm:.4f} must exceed 0.05")
        self.assertGreater(divergence_gru, 0.05, f"GRU mean reversal divergence {divergence_gru:.4f} must exceed 0.05")

    # =========================================================================
    # SUITE 3: TRAINING DYNAMICS & EARLY STOPPING
    # =========================================================================

    def test_07_training_loss_decrease_and_convergence(self):
        """
        Validates that recurrent models optimize smoothly and reduce training loss
        by at least 40% on real SPY sequences.
        """
        loss_hist = self.lstm_clf.train_loss_history
        self.assertGreater(len(loss_hist), 5)
        initial_loss = loss_hist[0]
        final_loss = loss_hist[-1]
        reduction = (initial_loss - final_loss) / initial_loss

        self.assertGreater(reduction, 0.40, f"Expected >= 40% loss reduction, got {reduction * 100:.2f}%")

    def test_08_early_stopping_checkpoint_restoration(self):
        """
        Validates early stopping:
        - Validation loss achieves < 0.25 on SPY validation split.
        - Checkpoint restoration restores model weights that evaluate to minimum validation loss.
        """
        val_losses = self.lstm_clf.val_loss_history
        self.assertGreater(len(val_losses), 0)
        best_val = min(val_losses)
        self.assertLess(best_val, 0.25, f"Best validation loss {best_val:.4f} should be < 0.25")

        # Verify that loaded model achieves the best validation loss
        current_val_cost = self.lstm_clf.compute_cost(self.X_val_s, self.y_val_s)
        self.assertAlmostEqual(
            current_val_cost,
            best_val,
            places=3,
            msg="Restored model weights must match best validation loss checkpoint",
        )

    def test_09_cost_history_alias(self):
        """cost_history must exist as a list and match train_loss_history."""
        self.assertIsInstance(self.lstm_clf.cost_history, list)
        self.assertEqual(self.lstm_clf.cost_history, self.lstm_clf.train_loss_history)
        self.assertEqual(self.gru_clf.cost_history, self.gru_clf.train_loss_history)

    # =========================================================================
    # SUITE 4: GRADIENT FLOW & NUMERICAL ROBUSTNESS
    # =========================================================================

    def test_10_finite_non_zero_gradients_across_all_gates(self):
        """
        Validates that all recurrent gates receive finite, non-zero gradients
        during compute_gradient.
        """
        for clf, name in [(self.lstm_clf, 'LSTM'), (self.gru_clf, 'GRU')]:
            grads = clf.compute_gradient(self.X_tr_s[:32], self.y_tr_s[:32])
            self.assertIsInstance(grads, list)
            self.assertGreater(len(grads), 0)
            for idx, grad in enumerate(grads):
                self.assertFalse(np.isnan(grad).any(), f"{name} param {idx} gradient contains NaN")
                self.assertFalse(np.isinf(grad).any(), f"{name} param {idx} gradient contains Inf")
                self.assertGreater(
                    np.abs(grad).sum(),
                    1e-7,
                    f"{name} param {idx} gradient is completely zero",
                )

    def test_11_bounded_gradient_norms_under_stress(self):
        """
        Under extreme input perturbation (50x standard deviation synthetic noise),
        gradient norms must remain bounded (< 5.0) with zero NaN or Inf occurrences.
        """
        np.random.seed(99)
        noisy_x = self.X_tr_s[:32].copy() + np.random.randn(*self.X_tr_s[:32].shape) * 50.0

        for clf, name in [(self.lstm_clf, 'LSTM'), (self.gru_clf, 'GRU')]:
            grads = clf.compute_gradient(noisy_x, self.y_tr_s[:32])
            total_norm_sq = sum(np.sum(g ** 2) for g in grads)
            total_norm = np.sqrt(total_norm_sq)
            self.assertFalse(np.isnan(total_norm), f"{name} gradient norm is NaN under stress")
            self.assertLess(total_norm, 5.0, f"{name} gradient norm {total_norm:.4f} exploded under stress")

    def test_12_polymorphic_input_handling_and_no_mutation(self):
        """
        Validates that DeepRegimeClassifier handles NumPy arrays and PyTorch tensors
        without mutating inputs in place.
        """
        x_copy = self.X_te_s.copy()
        x_tensor = torch.tensor(self.X_te_s, dtype=torch.float32)

        p_from_numpy = self.lstm_clf.predict_proba(self.X_te_s)
        p_from_tensor = self.lstm_clf.predict_proba(x_tensor)

        np.testing.assert_array_equal(self.X_te_s, x_copy)
        np.testing.assert_allclose(p_from_numpy, p_from_tensor, atol=1e-5)

    def test_13_sequential_model_rejects_2d_input(self):
        """
        Validates that sequential models (lstm, gru) reject 2D inputs in fit() and
        predict_proba() with a ValueError mentioning Pipeline.create_sequences.
        """
        clf = DeepRegimeClassifier(model_name='lstm', n_epochs=1, batch_size=16)
        x_2d = self.X_tr[:50]
        y_2d = self.y_tr[:50]

        with self.assertRaises(ValueError) as ctx_fit:
            clf.fit(x_2d, y_2d)
        self.assertIn("Pipeline.create_sequences", str(ctx_fit.exception))

        # After fitting on valid 3D data, predict_proba on 2D must also raise ValueError
        clf.fit(self.X_tr_s[:32], self.y_tr_s[:32])
        with self.assertRaises(ValueError) as ctx_pred:
            clf.predict_proba(x_2d)
        self.assertIn("Pipeline.create_sequences", str(ctx_pred.exception))

    # =========================================================================
    # SUITE 5: BENCHMARK EVALUATION METRICS
    # =========================================================================

    def test_14_evaluate_metric_parity_and_backward_compatibility(self):
        """
        Validates evaluation metric dictionary:
        - evaluate(y_true, y_pred) without y_prob returns exactly the 6 original keys.
        - evaluate(y_true, y_pred, y_prob=y_prob) adds ROC-AUC, Balanced Accuracy, and Log Loss.
        """
        y_true = self.y_te_s
        y_pred = self.lstm_preds
        y_prob = self.lstm_prob

        # Backward compatibility check
        metrics_basic = self.lstm_clf.evaluate(y_true, y_pred)
        expected_basic_keys = {
            "Accuracy",
            "Precision",
            "Recall",
            "F1-Score",
            "Baseline (Always Bullish)",
            "Confusion Matrix",
        }
        self.assertEqual(set(metrics_basic.keys()), expected_basic_keys)

        # Extended metrics check with probabilities
        metrics_ext = self.lstm_clf.evaluate(y_true, y_pred, y_prob=y_prob)
        expected_ext_keys = expected_basic_keys | {"ROC-AUC", "Balanced Accuracy", "Log Loss"}
        self.assertEqual(set(metrics_ext.keys()), expected_ext_keys)

    def test_15_pure_numpy_roc_auc_balanced_acc_log_loss(self):
        """
        Validates mathematical properties of the pure NumPy implementations:
        - ROC-AUC: perfect ranking = 1.0, inverted ranking = 0.0, identical prob = 0.5.
        - Balanced Accuracy = 0.5 * (recall + specificity).
        - Log Loss = binary cross-entropy.
        """
        y_t = np.array([0, 0, 1, 1])

        # ROC-AUC edge cases
        self.assertAlmostEqual(self.lstm_clf.roc_auc_score(y_t, [0.1, 0.2, 0.8, 0.9]), 1.0)
        self.assertAlmostEqual(self.lstm_clf.roc_auc_score(y_t, [0.9, 0.8, 0.2, 0.1]), 0.0)
        self.assertAlmostEqual(self.lstm_clf.roc_auc_score(y_t, [0.5, 0.5, 0.5, 0.5]), 0.5)

        # Balanced Accuracy
        y_p = np.array([0, 1, 1, 1])
        # TN=1, FP=1 (spec=0.5); FN=0, TP=2 (rec=1.0) -> bal_acc = 0.75
        self.assertAlmostEqual(self.lstm_clf.balanced_accuracy_score(y_t, y_p), 0.75)

        # Log Loss
        p = np.array([0.1, 0.1, 0.9, 0.9])
        expected_ll = -np.mean([np.log(0.9), np.log(0.9), np.log(0.9), np.log(0.9)])
        self.assertAlmostEqual(self.lstm_clf.log_loss_score(y_t, p), float(expected_ll), places=5)

    def test_16_spy_benchmark_metrics_thresholds(self):
        """
        Validates out-of-sample benchmark thresholds on the 240-day SPY test set:
        - Accuracy >= 0.82
        - Precision >= 0.85
        - Recall >= 0.90
        - F1-Score >= 0.88
        - ROC-AUC >= 0.85
        - Log Loss <= 0.46
        - Confusion Matrix sum == 240
        """
        for clf, preds, prob, name in [
            (self.lstm_clf, self.lstm_preds, self.lstm_prob, "LSTM"),
            (self.gru_clf, self.gru_preds, self.gru_prob, "GRU"),
        ]:
            m = clf.evaluate(self.y_te_s, preds, y_prob=prob)
            self.assertGreaterEqual(m["Accuracy"], 0.82, f"{name} Accuracy {m['Accuracy']:.4f} < 0.82")
            self.assertGreaterEqual(m["Precision"], 0.85, f"{name} Precision {m['Precision']:.4f} < 0.85")
            self.assertGreaterEqual(m["Recall"], 0.90, f"{name} Recall {m['Recall']:.4f} < 0.90")
            self.assertGreaterEqual(m["F1-Score"], 0.88, f"{name} F1-Score {m['F1-Score']:.4f} < 0.88")
            self.assertGreaterEqual(m["ROC-AUC"], 0.85, f"{name} ROC-AUC {m['ROC-AUC']:.4f} < 0.85")
            self.assertLessEqual(m["Log Loss"], 0.46, f"{name} Log Loss {m['Log Loss']:.4f} > 0.46")

            cm = m["Confusion Matrix"]
            self.assertEqual(int(cm.sum()), 240, f"{name} Confusion matrix elements must sum to 240")

    # =========================================================================
    # SUITE 6: FULL REAL SPY 5-YEAR E2E BACKTESTER INTEGRATION
    # =========================================================================

    def test_17_ml_hybrid_strategy_probability_guard_and_warmup_warning(self):
        """
        Validates MLHybridStrategy defensive guards:
        - Passing continuous probabilities triggers UserWarning and auto-thresholds at 0.5.
        - Passing truncated predictions shorter than base strategy triggers UserWarning.
        """
        base = MACrossover(self.test_df.copy(), 20, 50, full_data=self.df_raw)

        # 1. Continuous probability warning
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            strat = MLHybridStrategy(self.test_df.copy(), base, self.lstm_prob)
            prob_warnings = [w for w in recorded if "contains continuous probabilities" in str(w.message)]
            self.assertEqual(len(prob_warnings), 1)
            # Must be auto-thresholded to binary integers
            self.assertTrue(np.all(np.isin(strat.ml_predictions, [0, 1])))

        # 2. Missing warmup length warning
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            truncated_preds = self.lstm_preds[:100]
            strat_trunc = MLHybridStrategy(self.test_df.copy(), base, truncated_preds)
            trunc_warnings = [w for w in recorded if "less than base strategy length" in str(w.message)]
            self.assertEqual(len(trunc_warnings), 1)

    def test_18_lstm_real_spy_backtest_performance(self):
        """
        Validates full real SPY out-of-sample backtest with LSTM Hybrid Strategy:
        - Total Return >= 0.15 (15%)
        - Sharpe Ratio >= 1.40
        - Max Drawdown >= -0.08 (better than -8%)
        - 3 <= Total Trades <= 25
        - Win Rate >= 0.70
        - Decisively outperforms base MACrossover (13.72% return, 1.27 Sharpe)
        """
        base = MACrossover(self.test_df.copy(), 20, 50, full_data=self.df_raw)
        hyb = MLHybridStrategy(self.test_df.copy(), base, self.lstm_preds)
        bt = Backtest(self.test_df.copy(), hyb, starting_cash=100000.0)
        bt.run()
        m = bt.cal_metrics()

        tot_ret = float(m["Total returns"])
        sharpe = float(m["Sharpe Ratio"])
        max_dd = float(m["Max drawdown"])
        win_rate = float(m["Win rate"])
        trades = int(len(bt.trade_history))

        self.assertGreaterEqual(tot_ret, 0.15, f"LSTM Return {tot_ret:.4f} < 0.15")
        self.assertGreaterEqual(sharpe, 1.40, f"LSTM Sharpe {sharpe:.4f} < 1.40")
        self.assertGreaterEqual(max_dd, -0.08, f"LSTM MaxDD {max_dd:.4f} < -0.08")
        self.assertGreaterEqual(win_rate, 0.70, f"LSTM Win Rate {win_rate:.4f} < 0.70")
        self.assertTrue(3 <= trades <= 25, f"LSTM trade count {trades} outside [3, 25]")

        # Superiority over base MACrossover
        base_ret = float(self.base_ma_metrics["Total returns"])
        base_sharpe = float(self.base_ma_metrics["Sharpe Ratio"])
        self.assertGreater(tot_ret, base_ret, f"LSTM return {tot_ret:.4f} <= base {base_ret:.4f}")
        self.assertGreater(sharpe, base_sharpe, f"LSTM Sharpe {sharpe:.4f} <= base {base_sharpe:.4f}")

    def test_19_gru_real_spy_backtest_performance(self):
        """
        Validates full real SPY out-of-sample backtest with GRU Hybrid Strategy:
        - Total Return >= 0.15 (15%)
        - Sharpe Ratio >= 1.40
        - Max Drawdown >= -0.08 (better than -8%)
        - 3 <= Total Trades <= 25
        - Win Rate >= 0.70
        - Decisively outperforms base MACrossover (13.72% return, 1.27 Sharpe)
        """
        base = MACrossover(self.test_df.copy(), 20, 50, full_data=self.df_raw)
        hyb = MLHybridStrategy(self.test_df.copy(), base, self.gru_preds)
        bt = Backtest(self.test_df.copy(), hyb, starting_cash=100000.0)
        bt.run()
        m = bt.cal_metrics()

        tot_ret = float(m["Total returns"])
        sharpe = float(m["Sharpe Ratio"])
        max_dd = float(m["Max drawdown"])
        win_rate = float(m["Win rate"])
        trades = int(len(bt.trade_history))

        self.assertGreaterEqual(tot_ret, 0.15, f"GRU Return {tot_ret:.4f} < 0.15")
        self.assertGreaterEqual(sharpe, 1.40, f"GRU Sharpe {sharpe:.4f} < 1.40")
        self.assertGreaterEqual(max_dd, -0.08, f"GRU MaxDD {max_dd:.4f} < -0.08")
        self.assertGreaterEqual(win_rate, 0.70, f"GRU Win Rate {win_rate:.4f} < 0.70")
        self.assertTrue(3 <= trades <= 25, f"GRU trade count {trades} outside [3, 25]")

        # Superiority over base MACrossover
        base_ret = float(self.base_ma_metrics["Total returns"])
        base_sharpe = float(self.base_ma_metrics["Sharpe Ratio"])
        self.assertGreater(tot_ret, base_ret, f"GRU return {tot_ret:.4f} <= base {base_ret:.4f}")
        self.assertGreater(sharpe, base_sharpe, f"GRU Sharpe {sharpe:.4f} <= base {base_sharpe:.4f}")


if __name__ == '__main__':
    unittest.main()
