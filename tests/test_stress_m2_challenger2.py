"""
Empirical Adversarial Stress Test Suite for Milestone 2:
- PyTorch RegimeDataset & DeepRegimeClassifier Polymorphic Input Handling
- Defensive Hardening Checks in preprocessing_pipeline.py
- Extreme Class Imbalances (99% Bullish / 99% Bearish) & BCEWithLogitsLoss Numeric Stability

Author: Teamwork Challenger 2 Agent
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


class TestPolymorphicInputs(unittest.TestCase):
    """
    Stress-tests RegimeDataset and DeepRegimeClassifier across polymorphic input types:
    2D/3D, NumPy, Pandas DataFrame/Series, PyTorch Tensors, non-contiguous/strided arrays,
    and lists.
    """

    def setUp(self):
        np.random.seed(42)
        torch.manual_seed(42)
        self.n_samples = 64
        self.n_features = 14
        self.seq_len = 10

        self.X_2d_np = np.random.randn(self.n_samples, self.n_features).astype(np.float64)
        self.y_1d_np = np.random.randint(0, 2, size=self.n_samples)
        self.y_2d_np = self.y_1d_np.reshape(-1, 1)

        self.X_3d_np = np.random.randn(self.n_samples, self.seq_len, self.n_features).astype(np.float32)

    def test_1_1_regimedataset_numpy_dtypes(self):
        """RegimeDataset seamlessly converts float64, float32, int32, int64 to float32 tensors."""
        for dtype in [np.float64, np.float32, np.int32, np.int64]:
            X = self.X_2d_np.astype(dtype)
            y = self.y_1d_np.astype(dtype)
            ds = RegimeDataset(X, y)
            self.assertEqual(len(ds), self.n_samples)
            x_item, y_item = ds[0]
            self.assertEqual(x_item.dtype, torch.float32)
            self.assertEqual(y_item.dtype, torch.float32)
            self.assertEqual(x_item.shape, (self.n_features,))
            self.assertEqual(y_item.shape, (1,))

    def test_1_2_regimedataset_pandas_dataframe_series(self):
        """RegimeDataset wraps pandas DataFrame features and Series targets."""
        df_X = pd.DataFrame(self.X_2d_np, columns=[f"feat_{i}" for i in range(self.n_features)])
        s_y = pd.Series(self.y_1d_np, name="regime")

        ds = RegimeDataset(df_X, s_y)
        self.assertEqual(len(ds), self.n_samples)
        x_item, y_item = ds[5]
        self.assertEqual(x_item.dtype, torch.float32)
        self.assertEqual(y_item.dtype, torch.float32)
        np.testing.assert_allclose(x_item.numpy(), df_X.iloc[5].values, atol=1e-5)
        self.assertEqual(y_item.item(), float(s_y.iloc[5]))

    def test_1_3_regimedataset_python_lists(self):
        """RegimeDataset accepts raw nested Python lists for X and y."""
        X_list = self.X_2d_np.tolist()
        y_list = self.y_1d_np.tolist()

        ds = RegimeDataset(X_list, y_list)
        self.assertEqual(len(ds), self.n_samples)
        x_item, y_item = ds[0]
        self.assertEqual(x_item.dtype, torch.float32)
        self.assertEqual(y_item.dtype, torch.float32)
        np.testing.assert_allclose(x_item.numpy(), self.X_2d_np[0], atol=1e-5)

    def test_1_4_regimedataset_torch_tensors(self):
        """RegimeDataset accepts torch tensors with requires_grad=True and detaches/clones safely."""
        t_X = torch.tensor(self.X_2d_np, dtype=torch.float64, requires_grad=True)
        t_y = torch.tensor(self.y_2d_np, dtype=torch.int64)

        ds = RegimeDataset(t_X, t_y)
        self.assertFalse(ds.X.requires_grad)
        self.assertEqual(ds.X.dtype, torch.float32)
        self.assertEqual(ds.y.dtype, torch.float32)
        self.assertEqual(ds.y.shape, (self.n_samples, 1))

    def test_1_5_regimedataset_3d_sequences(self):
        """RegimeDataset wraps 3D sequence arrays (N, L, F) and returns (L, F) items."""
        ds = RegimeDataset(self.X_3d_np, self.y_2d_np)
        self.assertEqual(len(ds), self.n_samples)
        x_item, y_item = ds[0]
        self.assertEqual(x_item.shape, (self.seq_len, self.n_features))
        self.assertEqual(y_item.shape, (1,))

    def test_1_6_regimedataset_non_contiguous_arrays(self):
        """RegimeDataset safely handles non-contiguous, transposed, and Fortran-ordered arrays."""
        # Case A: Step-sliced array (strided non-contiguous)
        X_strided = self.X_2d_np[::2]
        y_strided = self.y_2d_np[::2]
        self.assertFalse(X_strided.flags.c_contiguous)
        ds_strided = RegimeDataset(X_strided, y_strided)
        self.assertEqual(len(ds_strided), len(X_strided))
        self.assertTrue(ds_strided.X.is_contiguous())

        # Case B: Fortran-ordered array (verify RegimeDataset successfully wraps and DataLoader collates)
        X_fortran = np.asfortranarray(self.X_2d_np)
        self.assertFalse(X_fortran.flags.c_contiguous)
        self.assertTrue(X_fortran.flags.f_contiguous)
        ds_fortran = RegimeDataset(X_fortran, self.y_2d_np)
        self.assertEqual(len(ds_fortran), self.n_samples)
        x_item, _ = ds_fortran[0]
        self.assertEqual(x_item.shape, (self.n_features,))
        # DataLoader collation produces contiguous batch
        loader_fortran = torch.utils.data.DataLoader(ds_fortran, batch_size=16)
        batch_xf, _ = next(iter(loader_fortran))
        self.assertTrue(batch_xf.is_contiguous())

        # Case C: Transposed tensor
        t_transposed = torch.randn(self.n_features, self.n_samples).t()
        self.assertFalse(t_transposed.is_contiguous())
        ds_transposed = RegimeDataset(t_transposed, self.y_2d_np)
        self.assertEqual(len(ds_transposed), self.n_samples)

    def test_1_7_regimedataset_memory_isolation(self):
        """Mutating original NumPy array does not corrupt RegimeDataset tensors."""
        X_orig = self.X_2d_np.copy()
        ds = RegimeDataset(X_orig, self.y_2d_np)
        orig_val = ds.X[0, 0].item()

        X_orig[0, 0] = 999999.0
        self.assertEqual(ds.X[0, 0].item(), orig_val)

    def test_1_8_regimedataset_none_target(self):
        """RegimeDataset handles y=None for unlabelled test sets / inference."""
        ds = RegimeDataset(self.X_2d_np)
        self.assertEqual(len(ds), self.n_samples)
        item = ds[0]
        self.assertEqual(item.shape, (self.n_features,))
        self.assertIsNone(ds.y)

    def test_1_9_regimedataset_length_mismatch_raises(self):
        """RegimeDataset raises ValueError when len(X) != len(y)."""
        with self.assertRaises(ValueError):
            RegimeDataset(self.X_2d_np, self.y_1d_np[:10])

    def test_1_10_deepregimeclassifier_fit_polymorphic(self):
        """DeepRegimeClassifier.fit trains successfully on DataFrame, Series, and strided inputs."""
        df_X = pd.DataFrame(self.X_2d_np)
        s_y = pd.Series(self.y_1d_np)

        clf = DeepRegimeClassifier(model_name='mlp', n_epochs=5, batch_size=16, random_state=42)
        clf.fit(df_X, s_y)
        self.assertTrue(clf.is_fitted)
        self.assertEqual(len(clf.train_loss_history), 5)

        # Non-contiguous input
        X_sub = self.X_2d_np[::2]
        y_sub = self.y_1d_np[::2]
        clf2 = DeepRegimeClassifier(model_name='mlp', n_epochs=3, batch_size=8, random_state=42)
        clf2.fit(X_sub, y_sub)
        self.assertTrue(clf2.is_fitted)

    def test_1_11_deepregimeclassifier_predict_polymorphic(self):
        """predict_proba and predict accept DataFrame, Tensor, and non-contiguous arrays."""
        clf = DeepRegimeClassifier(model_name='mlp', n_epochs=3, random_state=42)
        clf.fit(self.X_2d_np, self.y_2d_np)

        # Test with DataFrame
        df_X = pd.DataFrame(self.X_2d_np)
        probs_df = clf.predict_proba(df_X)
        self.assertEqual(probs_df.shape, (self.n_samples, 1))
        preds_df = clf.predict(df_X)
        self.assertEqual(preds_df.shape, (self.n_samples, 1))

        # Test with PyTorch Tensor
        t_X = torch.tensor(self.X_2d_np, dtype=torch.float32)
        probs_t = clf.predict_proba(t_X)
        self.assertEqual(probs_t.shape, (self.n_samples, 1))

        # Test with Non-contiguous strided array
        X_stride = self.X_2d_np[::3]
        probs_s = clf.predict_proba(X_stride)
        self.assertEqual(probs_s.shape, (len(X_stride), 1))

        # Values should match identically between numpy and dataframe
        np.testing.assert_allclose(probs_df, clf.predict_proba(self.X_2d_np), atol=1e-5)

    def test_1_12_deepregimeclassifier_cost_and_gradient_polymorphic(self):
        """compute_cost and compute_gradient accept polymorphic inputs."""
        clf = DeepRegimeClassifier(model_name='mlp', n_epochs=2, random_state=42)
        clf.fit(self.X_2d_np, self.y_2d_np)

        df_X = pd.DataFrame(self.X_2d_np)
        s_y = pd.Series(self.y_1d_np)

        cost = clf.compute_cost(df_X, s_y)
        self.assertIsInstance(cost, float)
        self.assertGreater(cost, 0.0)

        grads = clf.compute_gradient(df_X, s_y)
        self.assertIsInstance(grads, list)
        self.assertGreater(len(grads), 0)
        for g in grads:
            self.assertFalse(np.isnan(g).any())

    def test_1_13_deepregimeclassifier_3d_sequential_models(self):
        """Sequential models (lstm, gru, cnn1d, transformer) accept 3D inputs and polymorphic types."""
        models = ['lstm', 'gru', 'cnn1d', 'transformer']
        for m_name in models:
            clf = DeepRegimeClassifier(model_name=m_name, n_epochs=2, batch_size=16, random_state=42)
            clf.fit(self.X_3d_np, self.y_2d_np)
            self.assertTrue(clf.is_fitted, f"Failed fitting {m_name}")

            probs = clf.predict_proba(self.X_3d_np)
            self.assertEqual(probs.shape, (self.n_samples, 1), f"Prob shape failed for {m_name}")
            self.assertTrue(np.all(probs >= 0.0) and np.all(probs <= 1.0))

            preds = clf.predict(self.X_3d_np)
            self.assertEqual(preds.shape, (self.n_samples, 1), f"Pred shape failed for {m_name}")
            self.assertTrue(set(np.unique(preds)).issubset({0, 1}))


class TestDefensivePipelineChecks(unittest.TestCase):
    """
    Stress-tests the 3 defensive hardening checks in preprocessing_pipeline.py:
    1. Replacement and removal of [np.inf, -np.inf] before dropna.
    2. Strict validation of train_ratio and val_ratio in train_val_test_split_scale.
    3. Strict validation of seq_len in create_sequences.
    """

    def setUp(self):
        np.random.seed(42)
        self.dates = pd.bdate_range('2023-01-01', periods=150)
        self.base_df = pd.DataFrame({
            'Open': 100.0 + np.cumsum(np.random.randn(150)),
            'High': 102.0 + np.cumsum(np.random.randn(150)),
            'Low': 98.0 + np.cumsum(np.random.randn(150)),
            'Close': 100.0 + np.cumsum(np.random.randn(150)),
            'Volume': 1e6 + np.random.randint(0, 1000, 150),
        }, index=self.dates)

    def test_2_1_inf_replacement_in_clean_and_prepare(self):
        """clean_and_prepare replaces +inf and -inf with nan and drops them without error."""
        df_inf = self.base_df.copy()
        df_inf.loc[self.dates[70], 'Close'] = np.inf
        df_inf.loc[self.dates[75], 'Open'] = -np.inf
        df_inf.loc[self.dates[80], 'Volume'] = np.inf

        pipe = Pipeline(df_inf, feature_set='extended')
        clean = pipe.clean_and_prepare()

        self.assertFalse(np.isinf(clean.values).any(), "clean_df contains infinite values!")
        self.assertFalse(np.isnan(clean.values).any(), "clean_df contains NaN values!")
        self.assertNotIn(self.dates[70], clean.index)
        self.assertNotIn(self.dates[75], clean.index)
        self.assertNotIn(self.dates[80], clean.index)

    def test_2_2_inf_in_features_computation(self):
        """Indicators that evaluate to inf are cleanly purged in clean_and_prepare."""
        df_zero_open = self.base_df.copy()
        # Open = 0 would make candle_body = (Close - Open) / Open -> Inf
        df_zero_open.loc[self.dates[60], 'Open'] = 0.0

        pipe = Pipeline(df_zero_open, feature_set='extended')
        clean = pipe.clean_and_prepare()

        self.assertFalse(np.isinf(clean[pipe.feature_names].values).any())
        self.assertNotIn(self.dates[60], clean.index)

    def test_2_3_split_ratio_validation_train_ratio_zero_or_negative(self):
        """train_ratio <= 0 raises ValueError."""
        pipe = Pipeline(self.base_df, feature_set='basic')
        for bad_tr in [0.0, -0.1, -0.65]:
            with self.assertRaises(ValueError):
                pipe.train_val_test_split_scale(train_ratio=bad_tr, val_ratio=0.15)

    def test_2_4_split_ratio_validation_val_ratio_zero_or_negative(self):
        """val_ratio <= 0 raises ValueError."""
        pipe = Pipeline(self.base_df, feature_set='basic')
        for bad_vr in [0.0, -0.05, -0.15]:
            with self.assertRaises(ValueError):
                pipe.train_val_test_split_scale(train_ratio=0.65, val_ratio=bad_vr)

    def test_2_5_split_ratio_validation_sum_equal_or_greater_than_one(self):
        """train_ratio + val_ratio >= 1.0 raises ValueError."""
        pipe = Pipeline(self.base_df, feature_set='basic')
        # Exactly 1.0 (leaving 0% for test)
        with self.assertRaises(ValueError):
            pipe.train_val_test_split_scale(train_ratio=0.8, val_ratio=0.2)

        # Strictly greater than 1.0
        with self.assertRaises(ValueError):
            pipe.train_val_test_split_scale(train_ratio=0.7, val_ratio=0.4)

        with self.assertRaises(ValueError):
            pipe.train_val_test_split_scale(train_ratio=1.0, val_ratio=0.1)

    def test_2_6_split_ratio_valid_boundaries(self):
        """Valid split ratios execute cleanly and partition rows strictly."""
        pipe = Pipeline(self.base_df, feature_set='basic')
        X_tr, X_val, X_te, y_tr, y_val, y_te = pipe.train_val_test_split_scale(train_ratio=0.65, val_ratio=0.15)

        total_rows = len(pipe.clean_df)
        self.assertEqual(len(X_tr) + len(X_val) + len(X_te), total_rows)
        self.assertEqual(len(y_tr) + len(y_val) + len(y_te), total_rows)
        self.assertGreater(len(X_te), 0)

    def test_2_7_create_sequences_seq_len_zero_or_negative(self):
        """seq_len <= 0 raises ValueError."""
        features = np.random.randn(50, 4)
        for bad_seq in [0, -1, -20]:
            with self.assertRaises(ValueError):
                Pipeline.create_sequences(features, seq_len=bad_seq)

    def test_2_8_create_sequences_seq_len_booleans(self):
        """seq_len=True and seq_len=False raise ValueError (cannot be bool)."""
        features = np.random.randn(50, 4)
        with self.assertRaises(ValueError):
            Pipeline.create_sequences(features, seq_len=True)
        with self.assertRaises(ValueError):
            Pipeline.create_sequences(features, seq_len=False)

    def test_2_9_create_sequences_seq_len_non_int_types(self):
        """Float, string, None seq_len raises ValueError."""
        features = np.random.randn(50, 4)
        for bad_type in [20.0, 1.5, "20", None]:
            with self.assertRaises(ValueError):
                Pipeline.create_sequences(features, seq_len=bad_type)

    def test_2_10_create_sequences_valid_seq_len_boundary(self):
        """seq_len=1 (minimum allowable window) succeeds."""
        features = np.random.randn(50, 4)
        targets = np.random.randint(0, 2, size=(50, 1))
        X_seq, y_seq = Pipeline.create_sequences(features, targets, seq_len=1)
        self.assertEqual(X_seq.shape, (50, 1, 4))
        self.assertEqual(y_seq.shape, (50, 1))


class TestImbalancedClassesAndStability(unittest.TestCase):
    """
    Stress-tests DeepRegimeClassifier under heavily imbalanced targets:
    - 99% Bullish / 1% Bearish
    - 99% Bearish / 1% Bullish
    - 100% Bullish (0 Bearish)
    - 100% Bearish (0 Bullish)
    - BCEWithLogitsLoss pos_weight behavior and numeric stability.
    """

    def setUp(self):
        np.random.seed(42)
        torch.manual_seed(42)
        self.n_samples = 200
        self.n_features = 14
        self.X = np.random.randn(self.n_samples, self.n_features)

    def test_3_1_extreme_bullish_imbalance_99_percent(self):
        """
        99% Bullish (198 bullish, 2 bearish):
        pos_weight = 2 / 198 ~= 0.0101.
        Loss remains finite, gradients are finite, predictions and probabilities are valid.
        """
        y_imbalanced = np.ones((self.n_samples, 1), dtype=int)
        y_imbalanced[:2] = 0  # 2 bearish, 198 bullish

        clf = DeepRegimeClassifier(
            model_name='mlp',
            n_epochs=20,
            batch_size=32,
            class_weight='balanced',
            random_state=42
        )
        clf.fit(self.X, y_imbalanced)

        self.assertTrue(clf.is_fitted)
        for loss in clf.train_loss_history:
            self.assertFalse(np.isnan(loss), "Encountered NaN loss in 99% bullish training")
            self.assertFalse(np.isinf(loss), "Encountered Inf loss in 99% bullish training")
            self.assertGreater(loss, 0.0)

        probs = clf.predict_proba(self.X)
        self.assertFalse(np.isnan(probs).any())
        self.assertTrue(np.all(probs >= 0.0) and np.all(probs <= 1.0))

        preds = clf.predict(self.X)
        self.assertFalse(np.isnan(preds).any())
        self.assertTrue(set(np.unique(preds)).issubset({0, 1}))

        eval_metrics = clf.evaluate(y_imbalanced, preds)
        self.assertIn("Accuracy", eval_metrics)
        self.assertIn("F1-Score", eval_metrics)
        self.assertFalse(np.isnan(eval_metrics["F1-Score"]))
        self.assertFalse(np.isnan(eval_metrics["Precision"]))
        self.assertFalse(np.isnan(eval_metrics["Recall"]))

    def test_3_2_extreme_bearish_imbalance_99_percent(self):
        """
        99% Bearish (2 bullish, 198 bearish):
        pos_weight = 198 / 2 = 99.0.
        Loss remains finite without explosion, gradients remain bounded.
        """
        y_imbalanced = np.zeros((self.n_samples, 1), dtype=int)
        y_imbalanced[:2] = 1  # 2 bullish, 198 bearish

        clf = DeepRegimeClassifier(
            model_name='mlp',
            n_epochs=20,
            batch_size=32,
            class_weight='balanced',
            random_state=42
        )
        clf.fit(self.X, y_imbalanced)

        self.assertTrue(clf.is_fitted)
        for loss in clf.train_loss_history:
            self.assertFalse(np.isnan(loss), "Encountered NaN loss in 99% bearish training")
            self.assertFalse(np.isinf(loss), "Encountered Inf loss in 99% bearish training")
            self.assertGreater(loss, 0.0)

        probs = clf.predict_proba(self.X)
        self.assertFalse(np.isnan(probs).any())
        self.assertTrue(np.all(probs >= 0.0) and np.all(probs <= 1.0))

        preds = clf.predict(self.X)
        self.assertTrue(set(np.unique(preds)).issubset({0, 1}))

        eval_metrics = clf.evaluate(y_imbalanced, preds)
        self.assertFalse(np.isnan(eval_metrics["F1-Score"]))

    def test_3_3_monolithic_100_percent_bullish(self):
        """
        100% Bullish (all 1s, zero 0s):
        pos_weight = 0.0 / 200 = 0.0.
        Does not crash, loss remains finite.
        """
        y_mono = np.ones((self.n_samples, 1), dtype=int)

        clf = DeepRegimeClassifier(
            model_name='mlp',
            n_epochs=10,
            batch_size=32,
            class_weight='balanced',
            random_state=42
        )
        clf.fit(self.X, y_mono)

        self.assertTrue(clf.is_fitted)
        for loss in clf.train_loss_history:
            self.assertFalse(np.isnan(loss))
            self.assertFalse(np.isinf(loss))

        probs = clf.predict_proba(self.X)
        self.assertFalse(np.isnan(probs).any())
        self.assertTrue(np.all(probs >= 0.0) and np.all(probs <= 1.0))

    def test_3_4_monolithic_100_percent_bearish(self):
        """
        100% Bearish (all 0s, zero 1s):
        pos_weight = 200.0 / max(0, 1.0) = 200.0.
        Does not crash, loss remains finite.
        """
        y_mono = np.zeros((self.n_samples, 1), dtype=int)

        clf = DeepRegimeClassifier(
            model_name='mlp',
            n_epochs=10,
            batch_size=32,
            class_weight='balanced',
            random_state=42
        )
        clf.fit(self.X, y_mono)

        self.assertTrue(clf.is_fitted)
        for loss in clf.train_loss_history:
            self.assertFalse(np.isnan(loss))
            self.assertFalse(np.isinf(loss))

        probs = clf.predict_proba(self.X)
        self.assertFalse(np.isnan(probs).any())
        self.assertTrue(np.all(probs >= 0.0) and np.all(probs <= 1.0))

    def test_3_5_extreme_feature_inputs_numerical_stability(self):
        """Extreme feature magnitudes (+/- 1000) do not cause NaN or Inf in forward/sigmoid."""
        X_extreme = np.full((50, self.n_features), 1000.0)
        X_extreme[25:] = -1000.0
        y_dummy = np.random.randint(0, 2, size=(50, 1))

        clf = DeepRegimeClassifier(model_name='mlp', n_epochs=5, random_state=42)
        clf.fit(X_extreme, y_dummy)

        probs = clf.predict_proba(X_extreme)
        self.assertFalse(np.isnan(probs).any())
        self.assertFalse(np.isinf(probs).any())
        self.assertTrue(np.all(probs >= 0.0) and np.all(probs <= 1.0))

        # Test sigmoid_ clipping
        clipped_high = clf.sigmoid_(np.array([1000.0, 10000.0]))
        clipped_low = clf.sigmoid_(np.array([-1000.0, -10000.0]))
        np.testing.assert_allclose(clipped_high, 1.0, atol=1e-7)
        np.testing.assert_allclose(clipped_low, 0.0, atol=1e-7)

    def test_3_6_sequential_recurrent_and_transformer_imbalance_stability(self):
        """Recurrent (LSTM/GRU) and Transformer architectures maintain stability under 95% imbalance."""
        X_seq = np.random.randn(80, 10, self.n_features).astype(np.float32)
        y_seq = np.ones((80, 1), dtype=int)
        y_seq[:4] = 0  # 95% bullish

        for m_name in ['lstm', 'gru', 'cnn1d', 'transformer']:
            clf = DeepRegimeClassifier(model_name=m_name, n_epochs=3, batch_size=16, random_state=42)
            clf.fit(X_seq, y_seq)
            for loss in clf.train_loss_history:
                self.assertFalse(np.isnan(loss), f"{m_name} produced NaN loss under imbalance")
            probs = clf.predict_proba(X_seq)
            self.assertFalse(np.isnan(probs).any(), f"{m_name} produced NaN prob under imbalance")
            self.assertEqual(probs.shape, (80, 1))


if __name__ == '__main__':
    unittest.main(verbosity=2)
