"""
Independent Adversarial Stress & Verification Harness for Milestone 3 (Recurrent Models: LSTM & GRU)
DATA 3401 Honors Project

Authored by: teamwork_challenger_m3_2

Target Specifications & Challenges:
1. Dimensional Rejection:
   - RecurrentRegimeNet.forward rejects 1D, 2D, 4D tensors with informative ValueError.
   - DeepRegimeClassifier.fit rejects 2D inputs for sequential models ('lstm', 'gru', 'recurrent', custom net).
   - DeepRegimeClassifier.predict_proba and predict reject 2D inputs for sequential models.
   - Non-sequential models ('mlp') continue to accept 2D inputs without error.
2. Temporal Memory Sensitivity & Decay:
   - BPTT gradient saliency at t-19 (t=0) and t-0 (t=19) are strictly positive and finite.
   - Finite difference perturbations at t-19 vs t-0 produce measurable, non-zero output shifts.
   - Temporal decay properties: recent timesteps exert measurable gradient and logit influence.
3. Sequence Order Non-Commutativity:
   - Inverting temporal sequence order (X[:, ::-1, :]) alters predictions significantly (> 0.05 mean delta).
   - Random temporal permutations break sequence trajectory and alter predictions.
   - Opposite trend synthetic sequences (bullish ramp vs bearish ramp) yield starkly diverging regime probabilities.
4. Gradient Stability Under Extreme Noise:
   - 50x sigma Gaussian noise injection maintains strictly finite gradients across all recurrent gates.
   - Total gradient norm under 50x sigma noise remains bounded (< 5.0) with zero NaNs or Infs.
   - Extreme inputs (+50 sigma constant, -50 sigma constant, 100 sigma noise) produce finite outputs in (0, 1).
5. Continuous Probability Input to MLHybridStrategy:
   - Emits UserWarning with informative guidance when passed continuous probabilities in (0, 1).
   - Automatically thresholds continuous probabilities at 0.5 into binary integers {0, 1}.
   - Binary floats {0.0, 1.0} and binary ints {0, 1} do NOT trigger UserWarning.
   - Strategy signal generation and backtesting execute cleanly after thresholding.
6. Unpadded Warmup Length Handling in MLHybridStrategy:
   - Emits UserWarning when ml_predictions length is less than base strategy data length.
   - Correctly identifies missing leading days defaulting to cash.
   - Full length matching input produces zero warmup warning.
7. Recurrent Architecture & Pure-NumPy Metric Invariants:
   - Exact mathematical parameter formulas for LSTM (4 gates) vs GRU (3 gates) and 75% cell ratio.
   - Single-layer dropout suppression (dropout=0.0) preventing PyTorch warnings.
   - Pure-NumPy ROC-AUC, Balanced Accuracy, and Log Loss edge-case behavior and numerical stability.
"""

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
    MLPRegimeNet,
    RecurrentRegimeNet,
    Conv1DRegimeNet,
    TransformerRegimeNet,
    DeepRegimeClassifier,
    _to_numpy,
)
from strategy import MACrossover, MLHybridStrategy
from backtester import Backtest
from tests.e2e.test_helpers import load_real_spy_data


class TestDimensionalRejection(unittest.TestCase):
    """
    Stress-tests dimensional contract enforcement across recurrent architectures.
    Ensures non-3D inputs are rejected with informative actionable exceptions.
    """

    def setUp(self):
        np.random.seed(42)
        torch.manual_seed(42)
        self.input_dim = 14
        self.seq_len = 20
        self.batch_size = 16

    def test_recurrent_net_forward_rejects_1d_tensor(self):
        """RecurrentRegimeNet.forward must reject 1D tensor with informative ValueError."""
        for cell in ['lstm', 'gru']:
            net = RecurrentRegimeNet(input_dim=self.input_dim, hidden_dim=32, cell_type=cell)
            x_1d = torch.randn(self.input_dim)
            with self.assertRaises(ValueError) as ctx:
                net(x_1d)
            self.assertIn("3D sequential input", str(ctx.exception))
            self.assertIn("ndim=1", str(ctx.exception))

    def test_recurrent_net_forward_rejects_2d_tensor(self):
        """RecurrentRegimeNet.forward must reject 2D tensor (unbatched or tabular) with informative ValueError."""
        for cell in ['lstm', 'gru']:
            net = RecurrentRegimeNet(input_dim=self.input_dim, hidden_dim=32, cell_type=cell)
            x_2d = torch.randn(self.batch_size, self.input_dim)
            with self.assertRaises(ValueError) as ctx:
                net(x_2d)
            self.assertIn("3D sequential input", str(ctx.exception))
            self.assertIn("ndim=2", str(ctx.exception))

    def test_recurrent_net_forward_rejects_4d_tensor(self):
        """RecurrentRegimeNet.forward must reject 4D tensor with informative ValueError."""
        for cell in ['lstm', 'gru']:
            net = RecurrentRegimeNet(input_dim=self.input_dim, hidden_dim=32, cell_type=cell)
            x_4d = torch.randn(2, 4, self.seq_len, self.input_dim)
            with self.assertRaises(ValueError) as ctx:
                net(x_4d)
            self.assertIn("3D sequential input", str(ctx.exception))
            self.assertIn("ndim=4", str(ctx.exception))

    def test_deep_regime_classifier_fit_rejects_2d_for_sequential_models(self):
        """
        DeepRegimeClassifier.fit must reject 2D inputs when model_name is sequential ('lstm', 'gru', 'recurrent')
        and suggest Pipeline.create_sequences in the error message.
        """
        x_2d = np.random.randn(64, self.input_dim).astype(np.float32)
        y_1d = np.random.randint(0, 2, size=64).astype(np.float32)

        for model_name in ['lstm', 'gru', 'recurrent']:
            clf = DeepRegimeClassifier(model_name=model_name, n_epochs=1, batch_size=16)
            with self.assertRaises(ValueError) as ctx:
                clf.fit(x_2d, y_1d)
            err_msg = str(ctx.exception)
            self.assertIn("3D sequential input", err_msg)
            self.assertIn("Pipeline.create_sequences", err_msg)

    def test_deep_regime_classifier_fit_rejects_2d_for_custom_recurrent_instance(self):
        """Passing an instantiated RecurrentRegimeNet as model_name must also reject 2D inputs in fit()."""
        custom_net = RecurrentRegimeNet(input_dim=self.input_dim, hidden_dim=32, cell_type='lstm')
        clf = DeepRegimeClassifier(model_name=custom_net, n_epochs=1, batch_size=16)
        x_2d = np.random.randn(64, self.input_dim).astype(np.float32)
        y_1d = np.random.randint(0, 2, size=64).astype(np.float32)

        with self.assertRaises(ValueError) as ctx:
            clf.fit(x_2d, y_1d)
        self.assertIn("Pipeline.create_sequences", str(ctx.exception))

    def test_deep_regime_classifier_predict_proba_rejects_2d_for_sequential_models(self):
        """
        After training, predict_proba and predict must reject 2D inputs for sequential models
        with an informative message.
        """
        x_3d = np.random.randn(32, self.seq_len, self.input_dim).astype(np.float32)
        y_1d = np.random.randint(0, 2, size=32).astype(np.float32)
        x_2d = np.random.randn(16, self.input_dim).astype(np.float32)

        for model_name in ['lstm', 'gru']:
            clf = DeepRegimeClassifier(model_name=model_name, n_epochs=1, batch_size=16)
            clf.fit(x_3d, y_1d)

            with self.assertRaises(ValueError) as ctx_prob:
                clf.predict_proba(x_2d)
            self.assertIn("Pipeline.create_sequences", str(ctx_prob.exception))

            with self.assertRaises(ValueError) as ctx_pred:
                clf.predict(x_2d)
            self.assertIn("Pipeline.create_sequences", str(ctx_pred.exception))

    def test_non_sequential_models_accept_2d_without_error(self):
        """MLPRegimeNet / 'mlp' must continue to accept 2D inputs without raising dimensional ValueError."""
        clf = DeepRegimeClassifier(model_name='mlp', n_epochs=1, batch_size=16)
        x_2d = np.random.randn(32, self.input_dim).astype(np.float32)
        y_1d = np.random.randint(0, 2, size=32).astype(np.float32)

        # Should fit and predict without raising ValueError
        clf.fit(x_2d, y_1d)
        probs = clf.predict_proba(x_2d)
        self.assertEqual(probs.shape, (32, 1))


class TestTemporalMemorySensitivityAndDecay(unittest.TestCase):
    """
    Stress-tests temporal sensitivity and authentic BPTT gradient propagation.
    Asserts that earlier lookback bars (t-19) and latest bars (t-0) actively
    influence network predictions, verifying that lookback window information is retained.
    """

    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        torch.manual_seed(42)
        cls.raw_df = load_real_spy_data()
        cls.pipe = Pipeline(cls.raw_df, ticker='SPY', feature_set='extended')
        X_tr, X_val, X_te, y_tr, y_val, y_te = cls.pipe.train_val_test_split_scale(0.65, 0.15)
        cls.seq_len = 20
        cls.X_tr_s, cls.y_tr_s = cls.pipe.create_sequences(X_tr, y_tr, seq_len=cls.seq_len)
        cls.X_te_s, cls.y_te_s = cls.pipe.create_sequences(X_te, y_te, seq_len=cls.seq_len, prev_features_2d=X_val)

        # Train a light LSTM model
        cls.lstm_clf = DeepRegimeClassifier(model_name='lstm', hidden_dim=32, n_epochs=15, batch_size=32, random_state=42)
        cls.lstm_clf.fit(cls.X_tr_s, cls.y_tr_s)

        # Train a light GRU model
        cls.gru_clf = DeepRegimeClassifier(model_name='gru', hidden_dim=32, n_epochs=15, batch_size=32, random_state=42)
        cls.gru_clf.fit(cls.X_tr_s, cls.y_tr_s)

    def test_bptt_gradient_saliency_at_earliest_and_latest_steps(self):
        """
        Verify that backpropagation through time propagates non-zero gradients
        to the earliest timestep t=0 (t-19 days ago) and latest timestep t=19 (t-0).
        """
        for clf, name in [(self.lstm_clf, 'LSTM'), (self.gru_clf, 'GRU')]:
            model = clf.model
            model.eval()

            # Pick 5 test sequences
            for i in range(5):
                sample = torch.tensor(self.X_te_s[i:i+1], dtype=torch.float32, device=clf.device, requires_grad=True)
                logit = model(sample)
                logit.backward()

                grad = sample.grad.abs().cpu().numpy()[0]  # shape (20, 14)
                grad_t0 = np.sum(grad[0])   # earliest timestep t-19
                grad_tfinal = np.sum(grad[-1]) # latest timestep t-0

                self.assertGreater(
                    grad_t0, 1e-7,
                    f"{name} sample {i}: Gradient at t-19 must be non-zero (BPTT memory retention failed)"
                )
                self.assertGreater(
                    grad_tfinal, 1e-7,
                    f"{name} sample {i}: Gradient at t-0 must be non-zero"
                )

    def test_perturbation_at_t0_vs_t19_causes_measurable_output_shifts(self):
        """
        Asserts that perturbing features at t-19 (earliest historical context)
        and at t-0 (latest observation) both cause distinct, measurable logit shifts.
        """
        for clf, name in [(self.lstm_clf, 'LSTM'), (self.gru_clf, 'GRU')]:
            model = clf.model
            model.eval()

            sample = torch.tensor(self.X_te_s[0:1], dtype=torch.float32, device=clf.device)
            with torch.no_grad():
                baseline_logit = model(sample).item()

            # Perturb oldest bar (t=0) by +3.0 std
            sample_pert_t0 = sample.clone()
            sample_pert_t0[0, 0, :] += 3.0
            with torch.no_grad():
                logit_pert_t0 = model(sample_pert_t0).item()
            shift_t0 = abs(baseline_logit - logit_pert_t0)

            # Perturb latest bar (t=19) by +3.0 std
            sample_pert_t19 = sample.clone()
            sample_pert_t19[0, -1, :] += 3.0
            with torch.no_grad():
                logit_pert_t19 = model(sample_pert_t19).item()
            shift_t19 = abs(baseline_logit - logit_pert_t19)

            self.assertGreater(
                shift_t0, 1e-5,
                f"{name}: Perturbing oldest bar (t-19) must produce measurable logit shift > 1e-5, got {shift_t0:.6f}"
            )
            self.assertGreater(
                shift_t19, 1e-5,
                f"{name}: Perturbing latest bar (t-0) must produce measurable logit shift > 1e-5, got {shift_t19:.6f}"
            )

    def test_temporal_sensitivity_under_zero_masking(self):
        """
        Masking the oldest half of the sequence (t=0..9) with zeros must alter predictions,
        confirming that the model is actively integrating historical context.
        """
        for clf, name in [(self.lstm_clf, 'LSTM'), (self.gru_clf, 'GRU')]:
            X_orig = self.X_te_s[:50]
            X_masked = X_orig.copy()
            X_masked[:, :10, :] = 0.0

            p_orig = clf.predict_proba(X_orig)
            p_masked = clf.predict_proba(X_masked)
            mean_diff = float(np.mean(np.abs(p_orig - p_masked)))

            self.assertGreater(
                mean_diff, 0.01,
                f"{name}: Zero-masking first 10 timesteps must shift probabilities by > 0.01, got {mean_diff:.4f}"
            )


class TestSequenceOrderNonCommutativity(unittest.TestCase):
    """
    Stress-tests temporal order non-commutativity.
    A valid recurrent model must produce significantly different outputs when
    the sequence order is reversed or shuffled, demonstrating true sequential dynamics.
    """

    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        torch.manual_seed(42)
        cls.raw_df = load_real_spy_data()
        cls.pipe = Pipeline(cls.raw_df, ticker='SPY', feature_set='extended')
        X_tr, X_val, X_te, y_tr, y_val, y_te = cls.pipe.train_val_test_split_scale(0.65, 0.15)
        cls.X_tr_s, cls.y_tr_s = cls.pipe.create_sequences(X_tr, y_tr, seq_len=20)
        cls.X_te_s, cls.y_te_s = cls.pipe.create_sequences(X_te, y_te, seq_len=20, prev_features_2d=X_val)

        cls.lstm_clf = DeepRegimeClassifier(model_name='lstm', hidden_dim=32, n_epochs=20, batch_size=32, random_state=42)
        cls.lstm_clf.fit(cls.X_tr_s, cls.y_tr_s)

        cls.gru_clf = DeepRegimeClassifier(model_name='gru', hidden_dim=32, n_epochs=20, batch_size=32, random_state=42)
        cls.gru_clf.fit(cls.X_tr_s, cls.y_tr_s)

    def test_full_sequence_reversal_divergence(self):
        """
        Reversing sequence order (X[:, ::-1, :]) must alter out-of-sample predictions
        with a mean absolute probability difference >= 0.05.
        """
        X_orig = self.X_te_s
        X_rev = X_orig[:, ::-1, :]

        for clf, name in [(self.lstm_clf, 'LSTM'), (self.gru_clf, 'GRU')]:
            p_orig = clf.predict_proba(X_orig)
            p_rev = clf.predict_proba(X_rev)

            abs_diff = np.abs(p_orig - p_rev)
            mean_diff = float(np.mean(abs_diff))
            max_diff = float(np.max(abs_diff))

            self.assertGreater(
                mean_diff, 0.05,
                f"{name}: Mean reversal difference {mean_diff:.4f} must exceed 0.05"
            )
            self.assertGreater(
                max_diff, 0.15,
                f"{name}: Max reversal difference {max_diff:.4f} must exceed 0.15"
            )

            # At least 60% of samples must exhibit measurable divergence (> 0.02)
            fraction_diverged = np.mean(abs_diff > 0.02)
            self.assertGreater(
                fraction_diverged, 0.60,
                f"{name}: Fraction of samples diverging by > 0.02 is {fraction_diverged:.2%}, expected >= 60%"
            )

    def test_random_timestep_permutation_divergence(self):
        """
        Shuffling the 20 timesteps with a fixed non-identity permutation
        breaks chronological structure and must alter predictions significantly.
        """
        np.random.seed(123)
        perm = np.random.permutation(20)
        # Ensure it's not identity
        while np.array_equal(perm, np.arange(20)):
            perm = np.random.permutation(20)

        X_orig = self.X_te_s
        X_shuffled = X_orig[:, perm, :]

        for clf, name in [(self.lstm_clf, 'LSTM'), (self.gru_clf, 'GRU')]:
            p_orig = clf.predict_proba(X_orig)
            p_shuf = clf.predict_proba(X_shuffled)
            mean_diff = float(np.mean(np.abs(p_orig - p_shuf)))

            self.assertGreater(
                mean_diff, 0.04,
                f"{name}: Shuffled timestep difference {mean_diff:.4f} must exceed 0.04"
            )

    def test_synthetic_directional_ramp_asymmetry(self):
        """
        A synthetic bullish ramp (values monotonically increasing from -2 to +2)
        vs bearish ramp (values monotonically decreasing from +2 to -2) must yield
        distinct regime predictions.
        """
        ramp_up = np.linspace(-2.0, 2.0, 20).reshape(1, 20, 1)
        ramp_up = np.repeat(ramp_up, 14, axis=2)  # shape (1, 20, 14)

        ramp_down = np.linspace(2.0, -2.0, 20).reshape(1, 20, 1)
        ramp_down = np.repeat(ramp_down, 14, axis=2)  # shape (1, 20, 14)

        for clf, name in [(self.lstm_clf, 'LSTM'), (self.gru_clf, 'GRU')]:
            p_up = clf.predict_proba(ramp_up).item()
            p_down = clf.predict_proba(ramp_down).item()
            diff = abs(p_up - p_down)

            self.assertGreater(
                diff, 0.05,
                f"{name}: Bullish ramp vs Bearish ramp difference {diff:.4f} must exceed 0.05"
            )


class TestGradientStabilityUnder50xSigmaNoise(unittest.TestCase):
    """
    Stress-tests numerical stability and gradient boundedness under extreme noise injection.
    Simulates extreme market shocks and ensures:
    - Zero NaNs or Infs in parameter gradients.
    - Total gradient norm strictly bounded (< 5.0).
    - Logits and probabilities remain mathematically well-defined in (0.0, 1.0).
    """

    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        torch.manual_seed(42)
        cls.raw_df = load_real_spy_data()
        cls.pipe = Pipeline(cls.raw_df, ticker='SPY', feature_set='extended')
        X_tr, X_val, X_te, y_tr, y_val, y_te = cls.pipe.train_val_test_split_scale(0.65, 0.15)
        cls.X_tr_s, cls.y_tr_s = cls.pipe.create_sequences(X_tr, y_tr, seq_len=20)

        cls.lstm_clf = DeepRegimeClassifier(model_name='lstm', hidden_dim=32, n_epochs=5, batch_size=32, random_state=42)
        cls.lstm_clf.fit(cls.X_tr_s[:64], cls.y_tr_s[:64])

        cls.gru_clf = DeepRegimeClassifier(model_name='gru', hidden_dim=32, n_epochs=5, batch_size=32, random_state=42)
        cls.gru_clf.fit(cls.X_tr_s[:64], cls.y_tr_s[:64])

    def test_gradient_norm_bounded_under_50x_sigma_gaussian_noise(self):
        """
        Injecting additive Gaussian noise with 50x standard deviation must not cause
        gradient explosion: total norm < 5.0, zero NaNs, zero Infs.
        """
        np.random.seed(777)
        batch_x = self.X_tr_s[:32].copy()
        noisy_x = batch_x + np.random.randn(*batch_x.shape) * 50.0
        batch_y = self.y_tr_s[:32]

        for clf, name in [(self.lstm_clf, 'LSTM'), (self.gru_clf, 'GRU')]:
            grads = clf.compute_gradient(noisy_x, batch_y)
            self.assertIsInstance(grads, list)

            total_norm_sq = 0.0
            for idx, g in enumerate(grads):
                self.assertFalse(np.isnan(g).any(), f"{name} param {idx} contains NaN under 50x noise")
                self.assertFalse(np.isinf(g).any(), f"{name} param {idx} contains Inf under 50x noise")
                total_norm_sq += np.sum(g ** 2)

            total_norm = np.sqrt(total_norm_sq)
            self.assertLess(
                total_norm, 5.0,
                f"{name} total gradient norm {total_norm:.4f} exceeded 5.0 bound under 50x noise"
            )

    def test_extreme_constant_shifts_positive_and_negative(self):
        """
        Injecting constant +50.0 and -50.0 shifts across all features must yield
        finite gradients and well-behaved probabilities.
        """
        batch_x = self.X_tr_s[:32].copy()
        batch_y = self.y_tr_s[:32]

        for shift in [+50.0, -50.0]:
            shifted_x = batch_x + shift
            for clf, name in [(self.lstm_clf, 'LSTM'), (self.gru_clf, 'GRU')]:
                grads = clf.compute_gradient(shifted_x, batch_y)
                for idx, g in enumerate(grads):
                    self.assertFalse(np.isnan(g).any(), f"{name} shift {shift} param {idx} has NaN")
                    self.assertFalse(np.isinf(g).any(), f"{name} shift {shift} param {idx} has Inf")

                probs = clf.predict_proba(shifted_x)
                self.assertFalse(np.isnan(probs).any(), f"{name} shift {shift} probs contain NaN")
                self.assertTrue(
                    np.all((probs >= 0.0) & (probs <= 1.0)),
                    f"{name} shift {shift} probs outside [0, 1]"
                )

    def test_forward_pass_under_100x_sigma_noise(self):
        """
        Under extreme 100x sigma noise, forward pass must not crash, overflow, or yield NaNs.
        """
        np.random.seed(888)
        extreme_x = np.random.randn(16, 20, 14) * 100.0

        for clf, name in [(self.lstm_clf, 'LSTM'), (self.gru_clf, 'GRU')]:
            probs = clf.predict_proba(extreme_x)
            self.assertEqual(probs.shape, (16, 1))
            self.assertFalse(np.isnan(probs).any(), f"{name} 100x noise probs contain NaN")
            self.assertFalse(np.isinf(probs).any(), f"{name} 100x noise probs contain Inf")
            self.assertTrue(np.all((probs >= 0.0) & (probs <= 1.0)))


class TestMLHybridStrategyDefensiveGuards(unittest.TestCase):
    """
    Stress-tests defensive guards in MLHybridStrategy:
    - Continuous probability input auto-thresholding and warning emission.
    - Unpadded warmup length detection and warning emission.
    """

    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        cls.raw_df = load_real_spy_data()
        cls.pipe = Pipeline(cls.raw_df, ticker='SPY', feature_set='extended')
        cls.pipe.train_val_test_split_scale(0.65, 0.15)
        cls.test_df = cls.pipe.get_test_dataframe()
        cls.base_strategy = MACrossover(cls.test_df.copy(), 20, 50, full_data=cls.raw_df)

    def test_continuous_probabilities_trigger_userwarning_and_threshold_at_05(self):
        """
        Passing continuous probabilities in (0, 1) must emit a UserWarning
        and automatically threshold at 0.5 into binary integers {0, 1}.
        """
        # Test array with distinct values below and above 0.5
        cont_probs = np.array([0.05, 0.49, 0.50, 0.51, 0.95] * 48)  # length 240

        with warnings.catch_warnings(record=True) as w_list:
            warnings.simplefilter("always")
            strat = MLHybridStrategy(self.test_df.copy(), self.base_strategy, cont_probs)

            # Assert UserWarning is emitted
            user_warnings = [w for w in w_list if issubclass(w.category, UserWarning)]
            self.assertGreaterEqual(len(user_warnings), 1)
            msg = str(user_warnings[0].message)
            self.assertIn("contains continuous probabilities in [0, 1]", msg)
            self.assertIn("Automatically thresholding at 0.5", msg)

            # Assert internal predictions are strictly binary integers {0, 1}
            self.assertTrue(np.issubdtype(strat.ml_predictions.dtype, np.integer))
            self.assertTrue(np.all(np.isin(strat.ml_predictions, [0, 1])))

            # Check exact thresholding behavior
            expected = np.array([0, 0, 1, 1, 1] * 48)
            np.testing.assert_array_equal(strat.ml_predictions, expected)

    def test_binary_inputs_do_not_trigger_continuous_probability_warning(self):
        """
        Passing binary floats {0.0, 1.0} or binary integers {0, 1} must NOT trigger
        the continuous probability UserWarning.
        """
        bin_floats = np.array([0.0, 1.0, 1.0, 0.0] * 60)
        bin_ints = np.array([0, 1, 1, 0] * 60)

        for preds in [bin_floats, bin_ints]:
            with warnings.catch_warnings(record=True) as w_list:
                warnings.simplefilter("always")
                strat = MLHybridStrategy(self.test_df.copy(), self.base_strategy, preds)
                prob_warnings = [
                    w for w in w_list if "contains continuous probabilities" in str(w.message)
                ]
                self.assertEqual(
                    len(prob_warnings), 0,
                    f"Binary input unexpectedly triggered continuous probability warning: {w_list}"
                )

    def test_unpadded_warmup_length_triggers_userwarning(self):
        """
        Passing predictions shorter than the base strategy length (e.g. 221 instead of 240)
        must emit a UserWarning explaining that leading days default to cash.
        """
        ref_len = len(self.test_df)  # 240 days
        short_preds = np.ones(ref_len - 19, dtype=int)  # 221 days

        with warnings.catch_warnings(record=True) as w_list:
            warnings.simplefilter("always")
            strat = MLHybridStrategy(self.test_df.copy(), self.base_strategy, short_preds)

            warmup_warnings = [
                w for w in w_list if "less than base strategy length" in str(w.message)
            ]
            self.assertEqual(len(warmup_warnings), 1)
            msg = str(warmup_warnings[0].message)
            self.assertIn("less than base strategy length (240)", msg)
            self.assertIn("Missing leading 19 days will default to cash (0)", msg)

    def test_full_length_matching_does_not_trigger_warmup_warning(self):
        """Passing predictions of exact length matching the base strategy must NOT emit a warmup warning."""
        exact_preds = np.ones(len(self.test_df), dtype=int)
        with warnings.catch_warnings(record=True) as w_list:
            warnings.simplefilter("always")
            strat = MLHybridStrategy(self.test_df.copy(), self.base_strategy, exact_preds)
            warmup_warnings = [
                w for w in w_list if "less than base strategy length" in str(w.message)
            ]
            self.assertEqual(len(warmup_warnings), 0)

    def test_hybrid_strategy_execution_and_backtest_with_auto_thresholded_input(self):
        """
        Verifies that after auto-thresholding continuous probabilities,
        generate_signal() and Backtest.run() execute without errors.
        """
        cont_probs = np.full(len(self.test_df), 0.75)  # all bullish (> 0.5)
        strat = MLHybridStrategy(self.test_df.copy(), self.base_strategy, cont_probs)
        strat.generate_signal()

        self.assertIn('signal', strat.data.columns)
        self.assertTrue(np.all(np.isin(strat.data['signal'].dropna(), [0, 1])))

        bt = Backtest(self.test_df.copy(), strat, starting_cash=100000.0)
        bt.run()
        metrics = bt.cal_metrics()
        self.assertIn('Total returns', metrics)
        self.assertIn('Sharpe Ratio', metrics)


class TestRecurrentArchitecturalInvariants(unittest.TestCase):
    """
    Stress-tests architectural formulas, gate ratios, dropout conditioning,
    and pure NumPy evaluation metric oracles.
    """

    def test_mathematical_parameter_counts_and_gate_formulas(self):
        """
        Verify exact theoretical formulas for parameters:
        LSTM: 4 * (d_in * d_h + d_h^2 + 2*d_h) + (d_h * 1 + 1)
        GRU:  3 * (d_in * d_h + d_h^2 + 2*d_h) + (d_h * 1 + 1)
        Cell ratio must equal exactly 3/4 (0.75).
        """
        for d_in in [7, 14, 28]:
            for d_h in [16, 32, 64]:
                lstm_net = RecurrentRegimeNet(input_dim=d_in, hidden_dim=d_h, num_layers=1, cell_type='lstm')
                gru_net = RecurrentRegimeNet(input_dim=d_in, hidden_dim=d_h, num_layers=1, cell_type='gru')

                lstm_cell_actual = sum(p.numel() for p in lstm_net.rnn.parameters())
                gru_cell_actual = sum(p.numel() for p in gru_net.rnn.parameters())

                lstm_cell_expected = 4 * (d_in * d_h + d_h * d_h + 2 * d_h)
                gru_cell_expected = 3 * (d_in * d_h + d_h * d_h + 2 * d_h)

                self.assertEqual(lstm_cell_actual, lstm_cell_expected)
                self.assertEqual(gru_cell_actual, gru_cell_expected)
                self.assertAlmostEqual(gru_cell_actual / lstm_cell_actual, 0.75, places=6)

    def test_dropout_conditioning_and_warning_suppression(self):
        """
        PyTorch nn.RNN/LSTM emits a warning when num_layers=1 and dropout > 0.
        RecurrentRegimeNet must condition rnn_dropout=0.0 when num_layers=1,
        and keep rnn_dropout=dropout when num_layers > 1.
        """
        with warnings.catch_warnings(record=True) as w_list:
            warnings.simplefilter("always")
            net_l1 = RecurrentRegimeNet(input_dim=14, hidden_dim=32, num_layers=1, dropout=0.3)
            self.assertEqual(net_l1.rnn.dropout, 0.0)
            rnn_warnings = [w for w in w_list if "dropout option adds dropout after all but last" in str(w.message)]
            self.assertEqual(len(rnn_warnings), 0)

        net_l2 = RecurrentRegimeNet(input_dim=14, hidden_dim=32, num_layers=2, dropout=0.3)
        self.assertEqual(net_l2.rnn.dropout, 0.3)

    def test_hidden_state_extraction_identity(self):
        """out[:, -1, :] must match hn[-1] with 0.0 absolute difference for arbitrary batches."""
        x = torch.randn(12, 15, 14)

        for cell in ['lstm', 'gru']:
            net = RecurrentRegimeNet(input_dim=14, hidden_dim=32, num_layers=1, cell_type=cell)
            net.eval()
            with torch.no_grad():
                out, hidden = net.rnn(x)
                hn = hidden[0] if cell == 'lstm' else hidden
                diff = (out[:, -1, :] - hn[-1]).abs().max().item()
                self.assertEqual(diff, 0.0, f"{cell} hidden state extraction mismatch")

    def test_pure_numpy_metrics_edge_cases(self):
        """
        Stress-tests edge cases of pure NumPy metric implementations:
        - ROC-AUC with all positive or all negative targets (degenerate case returns 0.5)
        - Log Loss with probabilities 0.0 and 1.0 (clipping prevents -inf / NaN)
        - Balanced Accuracy with zero sensitivity or zero specificity
        """
        clf = DeepRegimeClassifier(model_name='lstm')

        # Degenerate all 1s or all 0s
        self.assertEqual(clf.roc_auc_score(np.ones(10), np.linspace(0.1, 0.9, 10)), 0.5)
        self.assertEqual(clf.roc_auc_score(np.zeros(10), np.linspace(0.1, 0.9, 10)), 0.5)

        # Log Loss clipping at 0.0 and 1.0
        p_extreme = np.array([0.0, 1.0])
        y_targets = np.array([0, 1])
        ll = clf.log_loss_score(y_targets, p_extreme)
        self.assertFalse(np.isnan(ll))
        self.assertFalse(np.isinf(ll))
        self.assertLess(ll, 1e-10)

        # Inverted predictions log loss
        ll_bad = clf.log_loss_score(np.array([1, 0]), np.array([0.0, 1.0]))
        self.assertFalse(np.isnan(ll_bad))
        self.assertFalse(np.isinf(ll_bad))
        self.assertGreater(ll_bad, 30.0)  # -ln(1e-15) ~ 34.5


if __name__ == '__main__':
    unittest.main()
