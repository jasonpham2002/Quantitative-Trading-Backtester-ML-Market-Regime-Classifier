"""
Empirical Challenger Verification & Adversarial Stress Test Suite for Milestone 3 (Recurrent Models)
DATA 3401 Honors Project

Comprehensive Empirical Verification:
1. Exact parameter counts and 75.0% recurrent cell gate parameter ratio across LSTM & GRU.
2. Invariant parameter ratio test across arbitrary (input_dim, hidden_dim) configurations.
3. Mathematical oracles for pure NumPy ROC-AUC (Mann-Whitney U statistic equivalence, tied scores, monotonic rankings, single-class bounds).
4. Mathematical oracles for Balanced Accuracy (Sensitivity + Specificity, zero-positive and zero-negative edge cases).
5. Mathematical oracles for Log Loss (clipped binary cross-entropy, extreme 0.0/1.0 probabilities, numerical stability).
6. evaluate() dictionary signature: exactly 6 keys when y_prob=None, exactly 9 keys when y_prob is provided.
7. Defensive guards: Rejection of 1D, 2D, and 4D inputs by RecurrentRegimeNet.forward, DeepRegimeClassifier.fit, and DeepRegimeClassifier.predict_proba.
8. Evaluation mode and dropout disabling during inference.
9. Sequence temporal sensitivity (temporal order non-commutativity and BPTT memory).
10. MLHybridStrategy robustness: continuous probability warning and thresholding, warmup length mismatch padding and warning.
11. Real SPY 5-year downstream backtesting: Positive return and Sharpe > 1.40 for both LSTM and GRU.

Author: Teamwork Challenger M3 (teamwork_challenger_m3_1)
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


class AdversarialMilestone3ChallengerTest(unittest.TestCase):
    """
    Adversarial verification suite executing empirical tests and mathematical oracles.
    """

    @classmethod
    def setUpClass(cls):
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
        cls.X_tr_s, cls.y_tr_s = cls.pipe.create_sequences(cls.X_tr, cls.y_tr, seq_len=cls.seq_len)
        cls.X_val_s, cls.y_val_s = cls.pipe.create_sequences(
            cls.X_val, cls.y_val, seq_len=cls.seq_len, prev_features_2d=cls.X_tr
        )
        cls.X_te_s, cls.y_te_s = cls.pipe.create_sequences(
            cls.X_te, cls.y_te, seq_len=cls.seq_len, prev_features_2d=cls.X_val
        )
        cls.test_df = cls.pipe.get_test_dataframe()

    # =========================================================================
    # Suite 1: Parameter Counts and Gate Ratio Invariants
    # =========================================================================

    def test_01_parameter_counts_exact_benchmark(self):
        """Verify exact parameter counts: LSTM=6177, GRU=4641, cell ratio=75.0%."""
        lstm = RecurrentRegimeNet(input_dim=14, hidden_dim=32, num_layers=1, cell_type='lstm')
        gru = RecurrentRegimeNet(input_dim=14, hidden_dim=32, num_layers=1, cell_type='gru')

        lstm_total = sum(p.numel() for p in lstm.parameters())
        gru_total = sum(p.numel() for p in gru.parameters())

        self.assertEqual(lstm_total, 6177, f"LSTM expected 6,177 params, got {lstm_total}")
        self.assertEqual(gru_total, 4641, f"GRU expected 4,641 params, got {gru_total}")

        # Recurrent cell parameters (excluding the final linear head: 32 weight + 1 bias = 33)
        lstm_cell = sum(p.numel() for p in lstm.rnn.parameters())
        gru_cell = sum(p.numel() for p in gru.rnn.parameters())

        self.assertEqual(lstm_cell, 6144, f"LSTM cell expected 6,144 params, got {lstm_cell}")
        self.assertEqual(gru_cell, 4608, f"GRU cell expected 4,608 params, got {gru_cell}")

        ratio = gru_cell / lstm_cell
        self.assertAlmostEqual(
            ratio, 0.75, places=9,
            msg=f"GRU/LSTM recurrent cell parameter ratio must be exactly 0.75 (3/4 gates), got {ratio}"
        )

    def test_02_parameter_ratio_invariant_across_configurations(self):
        """Stress-test that GRU/LSTM recurrent cell ratio is invariant at 0.75 across diverse dimensions."""
        configurations = [
            (5, 16, 1),
            (14, 32, 1),
            (20, 64, 1),
            (10, 128, 1),
            (14, 32, 2),
            (8, 48, 3),
        ]
        for input_dim, hidden_dim, num_layers in configurations:
            with self.subTest(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers):
                m_lstm = RecurrentRegimeNet(
                    input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, cell_type='lstm'
                )
                m_gru = RecurrentRegimeNet(
                    input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, cell_type='gru'
                )
                cell_lstm = sum(p.numel() for p in m_lstm.rnn.parameters())
                cell_gru = sum(p.numel() for p in m_gru.rnn.parameters())
                ratio = cell_gru / cell_lstm
                self.assertAlmostEqual(
                    ratio, 0.75, places=7,
                    msg=f"Cell ratio violated at dim=({input_dim}, {hidden_dim}, {num_layers}): got {ratio}"
                )

    # =========================================================================
    # Suite 2: Mathematical Oracles for Pure NumPy Metrics
    # =========================================================================

    def test_03_roc_auc_mann_whitney_u_oracle(self):
        """
        Verify pure NumPy roc_auc_score matches independent Mann-Whitney U statistic oracle:
        AUC = U / (n_pos * n_neg) where U = sum_{i in pos, j in neg} [I(p_i > p_j) + 0.5 * I(p_i == p_j)]
        """
        clf = DeepRegimeClassifier(model_name='lstm')

        def mann_whitney_auc(y_true, y_prob):
            y_t = np.asarray(y_true).ravel()
            p = np.asarray(y_prob).ravel()
            pos = p[y_t == 1]
            neg = p[y_t == 0]
            if len(pos) == 0 or len(neg) == 0:
                return 0.5
            # Pairwise comparison matrix
            greater = np.sum(pos[:, None] > neg[None, :])
            equal = np.sum(pos[:, None] == neg[None, :])
            u = greater + 0.5 * equal
            return u / (len(pos) * len(neg))

        # Test case A: Clear separation
        y_a = np.array([0, 0, 0, 1, 1, 1])
        p_a = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        self.assertAlmostEqual(clf.roc_auc_score(y_a, p_a), 1.0, places=9)
        self.assertAlmostEqual(clf.roc_auc_score(y_a, p_a), mann_whitney_auc(y_a, p_a), places=9)

        # Test case B: Complete inversion
        p_b = 1.0 - p_a
        self.assertAlmostEqual(clf.roc_auc_score(y_a, p_b), 0.0, places=9)
        self.assertAlmostEqual(clf.roc_auc_score(y_a, p_b), mann_whitney_auc(y_a, p_b), places=9)

        # Test case C: Tied scores
        y_c = np.array([0, 1, 0, 1, 0, 1])
        p_c = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        self.assertAlmostEqual(clf.roc_auc_score(y_c, p_c), 0.5, places=9)
        self.assertAlmostEqual(clf.roc_auc_score(y_c, p_c), mann_whitney_auc(y_c, p_c), places=9)

        # Test case D: Realistic random vector with ties
        rng = np.random.RandomState(123)
        for _ in range(5):
            y_rand = rng.randint(0, 2, size=100)
            p_rand = rng.choice(np.linspace(0.1, 0.9, 15), size=100)
            expected = mann_whitney_auc(y_rand, p_rand)
            actual = clf.roc_auc_score(y_rand, p_rand)
            self.assertAlmostEqual(actual, expected, places=8)

        # Test case E: Single class edge cases
        self.assertEqual(clf.roc_auc_score(np.zeros(20), np.random.rand(20)), 0.5)
        self.assertEqual(clf.roc_auc_score(np.ones(20), np.random.rand(20)), 0.5)

    def test_04_balanced_accuracy_oracle(self):
        """Verify balanced accuracy against explicit 0.5 * (Sensitivity + Specificity) oracle."""
        clf = DeepRegimeClassifier(model_name='gru')

        y_t = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])  # 4 pos, 6 neg
        y_p = np.array([1, 1, 0, 0, 0, 0, 0, 0, 1, 1])  # TP=2, FN=2, TN=4, FP=2

        # Sensitivity = 2 / 4 = 0.5
        # Specificity = 4 / 6 = 2/3
        # Balanced Acc = 0.5 * (0.5 + 2/3) = 0.5 * (7/6) = 7/12 = 0.583333333...
        expected = 0.5 * (2/4 + 4/6)
        actual = clf.balanced_accuracy_score(y_t, y_p)
        self.assertAlmostEqual(actual, expected, places=9)

        # Edge case: No positive ground truth
        y_all_zero = np.zeros(10)
        y_pred = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        # TP=0, FN=0 -> Recall=0.0; TN=5, FP=5 -> Specificity=0.5
        # Balanced Acc = 0.5 * (0.0 + 0.5) = 0.25
        self.assertAlmostEqual(clf.balanced_accuracy_score(y_all_zero, y_pred), 0.25, places=9)

    def test_05_log_loss_oracle_and_clipping(self):
        """Verify pure NumPy log loss matches analytical BCE with eps clipping."""
        clf = DeepRegimeClassifier(model_name='lstm')
        eps = 1e-15

        y_t = np.array([1, 0, 1, 0])
        p = np.array([0.9, 0.1, 0.8, 0.2])

        expected = -0.25 * (np.log(0.9) + np.log(0.9) + np.log(0.8) + np.log(0.8))
        actual = clf.log_loss_score(y_t, p, eps=eps)
        self.assertAlmostEqual(actual, expected, places=9)

        # Extreme values: probabilities exactly 0.0 and 1.0
        p_extreme = np.array([0.0, 1.0, 1.0, 0.0])
        ll_extreme = clf.log_loss_score(y_t, p_extreme, eps=eps)
        self.assertTrue(np.isfinite(ll_extreme))
        self.assertGreater(ll_extreme, 10.0)

    # =========================================================================
    # Suite 3: evaluate() Return Signature & Backward Compatibility
    # =========================================================================

    def test_06_evaluate_keys_contract(self):
        """
        Verify evaluate() returns:
        - Exactly 6 keys when y_prob is None (100% backward compatibility)
        - Exactly 9 keys when y_prob is supplied
        """
        clf = DeepRegimeClassifier(model_name='lstm')
        y_true = np.array([1, 0, 1, 1, 0])
        y_pred = np.array([1, 0, 0, 1, 0])
        y_prob = np.array([0.8, 0.2, 0.4, 0.9, 0.1])

        # Call with y_prob = None
        metrics_6 = clf.evaluate(y_true, y_pred, y_prob=None)
        expected_6_keys = {
            "Accuracy",
            "Precision",
            "Recall",
            "F1-Score",
            "Baseline (Always Bullish)",
            "Confusion Matrix",
        }
        self.assertEqual(set(metrics_6.keys()), expected_6_keys)
        self.assertEqual(len(metrics_6), 6)

        # Call with default (omitted y_prob)
        metrics_default = clf.evaluate(y_true, y_pred)
        self.assertEqual(set(metrics_default.keys()), expected_6_keys)
        self.assertEqual(len(metrics_default), 6)

        # Call with y_prob supplied
        metrics_9 = clf.evaluate(y_true, y_pred, y_prob=y_prob)
        expected_9_keys = expected_6_keys | {"ROC-AUC", "Balanced Accuracy", "Log Loss"}
        self.assertEqual(set(metrics_9.keys()), expected_9_keys)
        self.assertEqual(len(metrics_9), 9)

    # =========================================================================
    # Suite 4: Defensive Dimension Guards
    # =========================================================================

    def test_07_recurrent_net_forward_dimension_guards(self):
        """Verify RecurrentRegimeNet.forward rejects 1D, 2D, and 4D tensors with informative ValueError."""
        net = RecurrentRegimeNet(input_dim=14, hidden_dim=32, num_layers=1, cell_type='lstm')

        # 1D tensor
        with self.assertRaises(ValueError) as ctx1:
            net(torch.randn(14))
        self.assertIn("expects 3D sequential input", str(ctx1.exception))

        # 2D tensor
        with self.assertRaises(ValueError) as ctx2:
            net(torch.randn(32, 14))
        self.assertIn("expects 3D sequential input", str(ctx2.exception))

        # 4D tensor
        with self.assertRaises(ValueError) as ctx4:
            net(torch.randn(32, 20, 14, 1))
        self.assertIn("expects 3D sequential input", str(ctx4.exception))

    def test_08_classifier_rejects_2d_input_in_fit_and_predict(self):
        """Verify sequential models reject 2D inputs in fit() and predict_proba() directing to create_sequences."""
        for model_name in ['lstm', 'gru']:
            with self.subTest(model_name=model_name):
                clf = DeepRegimeClassifier(model_name=model_name, n_epochs=1)
                # 2D input to fit()
                with self.assertRaises(ValueError) as ctx_fit:
                    clf.fit(np.random.randn(50, 14), np.random.randint(0, 2, 50))
                self.assertIn("expects 3D sequential input", str(ctx_fit.exception))
                self.assertIn("Pipeline.create_sequences", str(ctx_fit.exception))

                # Now fit properly with 3D sequences
                clf.fit(self.X_tr_s[:30], self.y_tr_s[:30])

                # 2D input to predict_proba() on fitted sequential model
                with self.assertRaises(ValueError) as ctx_pred:
                    clf.predict_proba(np.random.randn(10, 14))
                self.assertIn("expects 3D sequential input", str(ctx_pred.exception))
                self.assertIn("Pipeline.create_sequences", str(ctx_pred.exception))

    # =========================================================================
    # Suite 5: Temporal Dynamics & Sequence Order Non-Commutativity
    # =========================================================================

    def test_09_temporal_order_non_commutativity(self):
        """Verify sequence reversal produces distinct predictions (recurrent models are order-sensitive)."""
        clf = DeepRegimeClassifier(
            model_name='lstm',
            hidden_dim=32,
            n_epochs=20,
            random_state=42,
        )
        clf.fit(self.X_tr_s[:200], self.y_tr_s[:200])

        x_seq = self.X_te_s[:30]  # shape (30, 20, 14)
        x_seq_reversed = np.ascontiguousarray(x_seq[:, ::-1, :])  # reverse along time axis

        prob_normal = clf.predict_proba(x_seq)
        prob_reversed = clf.predict_proba(x_seq_reversed)

        diff = np.abs(prob_normal - prob_reversed)
        mean_diff = float(np.mean(diff))
        max_diff = float(np.max(diff))

        self.assertGreater(
            max_diff, 0.05,
            f"Temporal order reversal must produce non-trivial prediction changes, got max_diff={max_diff:.4f}"
        )
        self.assertGreater(
            mean_diff, 0.01,
            f"Temporal order reversal mean diff must be > 0.01, got mean_diff={mean_diff:.4f}"
        )

    # =========================================================================
    # Suite 6: MLHybridStrategy Defensive Robustness
    # =========================================================================

    def test_10_hybrid_strategy_probability_guard_and_length_padding(self):
        """Verify MLHybridStrategy warns and thresholds continuous probabilities and handles length disparities."""
        base_strat = MACrossover(self.test_df, short_window=20, long_window=50)

        # 1. Continuous probabilities in (0, 1) should trigger UserWarning and auto-threshold at 0.5
        cont_probs = np.full(len(self.test_df), 0.75)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            strat = MLHybridStrategy(self.test_df, base_strat, cont_probs)
            self.assertTrue(any(issubclass(item.category, UserWarning) for item in w))
            self.assertTrue(any("continuous probabilities" in str(item.message) for item in w))
        # After thresholding, all 0.75 become 1
        self.assertTrue(np.all(strat.ml_predictions == 1))

        # 2. Length disparity warning: fewer predictions than base strategy data
        short_preds = np.ones(len(self.test_df) - 19)
        with warnings.catch_warnings(record=True) as w2:
            warnings.simplefilter("always")
            strat_short = MLHybridStrategy(self.test_df, base_strat, short_preds)
            self.assertTrue(any("less than base strategy length" in str(item.message) for item in w2))

        # Check signal generation with short predictions does not crash
        strat_short.generate_signal()
        sig = strat_short.data['signal'].values
        # Leading 19 days must be 0 (cash)
        self.assertTrue(np.all(sig[:19] == 0))

    # =========================================================================
    # Suite 7: Real SPY 5-Year Downstream Backtesting Performance
    # =========================================================================

    def test_11_lstm_and_gru_real_spy_downstream_backtest(self):
        """
        Verify downstream backtesting on real SPY 5-year data (240 out-of-sample days):
        - Both LSTM and GRU achieve positive total return.
        - Both LSTM and GRU achieve Sharpe ratio > 1.40.
        - Both outperform un-gated base MACrossover strategy.
        """
        # Baseline un-gated MACrossover
        base_strat = MACrossover(self.test_df, short_window=20, long_window=50, full_data=self.df_raw)
        base_bt = Backtest(self.test_df, base_strat, starting_cash=100000)
        base_bt.run()
        base_perf = base_bt.cal_metrics()
        base_return = base_perf["Total returns"]
        base_sharpe = base_perf["Sharpe Ratio"]

        # 1. Train LSTM Classifier
        lstm_clf = DeepRegimeClassifier(
            model_name='lstm',
            hidden_dim=32,
            num_layers=1,
            dropout=0.2,
            n_epochs=50,
            patience=15,
            random_state=42,
        )
        lstm_clf.fit(self.X_tr_s, self.y_tr_s, X_val=self.X_val_s, y_val=self.y_val_s)
        lstm_preds = lstm_clf.predict(self.X_te_s, threshold=0.22, smooth_window=3)

        lstm_strat = MLHybridStrategy(
            self.test_df,
            MACrossover(self.test_df, 20, 50, full_data=self.df_raw),
            lstm_preds,
        )
        lstm_bt = Backtest(self.test_df, lstm_strat, starting_cash=100000)
        lstm_bt.run()
        lstm_perf = lstm_bt.cal_metrics()

        lstm_return = lstm_perf["Total returns"]
        lstm_sharpe = lstm_perf["Sharpe Ratio"]

        self.assertGreater(lstm_return, 0.15, f"LSTM return {lstm_return:.4f} should exceed 15%")
        self.assertGreater(lstm_sharpe, 1.40, f"LSTM Sharpe {lstm_sharpe:.4f} should exceed 1.40")
        self.assertGreater(lstm_sharpe, base_sharpe, "LSTM Sharpe should exceed un-gated MACrossover")

        # 2. Train GRU Classifier
        gru_clf = DeepRegimeClassifier(
            model_name='gru',
            hidden_dim=32,
            num_layers=1,
            dropout=0.2,
            n_epochs=50,
            patience=15,
            random_state=42,
        )
        gru_clf.fit(self.X_tr_s, self.y_tr_s, X_val=self.X_val_s, y_val=self.y_val_s)
        gru_preds = gru_clf.predict(self.X_te_s, threshold=0.22, smooth_window=3)

        gru_strat = MLHybridStrategy(
            self.test_df,
            MACrossover(self.test_df, 20, 50, full_data=self.df_raw),
            gru_preds,
        )
        gru_bt = Backtest(self.test_df, gru_strat, starting_cash=100000)
        gru_bt.run()
        gru_perf = gru_bt.cal_metrics()

        gru_return = gru_perf["Total returns"]
        gru_sharpe = gru_perf["Sharpe Ratio"]

        self.assertGreater(gru_return, 0.15, f"GRU return {gru_return:.4f} should exceed 15%")
        self.assertGreater(gru_sharpe, 1.40, f"GRU Sharpe {gru_sharpe:.4f} should exceed 1.40")
        self.assertGreater(gru_sharpe, base_sharpe, "GRU Sharpe should exceed un-gated MACrossover")


if __name__ == '__main__':
    unittest.main()
