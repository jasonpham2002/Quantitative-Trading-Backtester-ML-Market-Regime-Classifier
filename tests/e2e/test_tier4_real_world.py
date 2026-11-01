"""
Tier 4: Real-World Application Scenarios E2E Tests
Covers full end-to-end execution on the actual 5-year SPY market dataset,
240-day out-of-sample benchmark replication, performance metrics verification,
headless visualization generation to disk, and notebook structural integrity.
"""

import unittest
import json
import os
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from tests.e2e.test_helpers import (
    HAS_PREPROCESSING,
    HAS_EXTENDED_PIPELINE,
    HAS_TRAIN_VAL_TEST,
    HAS_CREATE_SEQUENCES,
    HAS_LOGISTIC_REGRESSION,
    HAS_BACKTESTER,
    HAS_VISUALIZATIONS,
    load_real_spy_data,
    SPY_DATA_PATH,
    BACKTESTER_DIR,
    MockDeepRegimeClassifier,
)

if HAS_PREPROCESSING:
    from preprocessing_pipeline import Pipeline

if HAS_LOGISTIC_REGRESSION:
    from LogisticRegression import LogisticRegression

if HAS_BACKTESTER:
    from strategy import MACrossover, RSIMeanReversion, MLHybridStrategy
    from backtester import Backtest

if HAS_VISUALIZATIONS:
    import visualizations


class TestTier4RealSPYDataset(unittest.TestCase):
    """Tier 4: Real-World Group 1 — 5-Year SPY Dataset Ingestion & Alignment (>=4 tests)."""

    def setUp(self):
        self.df_raw = load_real_spy_data()

    def test_spy_dataset_structure_and_columns(self):
        """Test 4.1: SPY_5_years loads with 1254 raw rows and MultiIndex OHLCV columns."""
        self.assertEqual(len(self.df_raw), 1254)
        self.assertTrue(isinstance(self.df_raw.columns, pd.MultiIndex))
        price_levels = set(self.df_raw.columns.get_level_values(0))
        for col in ['Close', 'High', 'Low', 'Open', 'Volume']:
            self.assertIn(col, price_levels)

    def test_spy_clean_dataframe_drops_54_rows(self):
        """Test 4.2: SPY data cleaning drops 49 initial rows + 5 tail rows -> exactly 1200 rows."""
        pipe = Pipeline(self.df_raw, ticker='SPY')
        clean_df = pipe.clean_and_prepare()
        self.assertEqual(len(clean_df), 1200)
        self.assertEqual(clean_df.index[0].strftime('%Y-%m-%d'), '2021-11-11')
        self.assertEqual(clean_df.index[-1].strftime('%Y-%m-%d'), '2026-08-25')

    def test_spy_out_of_sample_test_window_240_days(self):
        """Test 4.3: 80/20 split produces exactly 240 out-of-sample test trading days."""
        pipe = Pipeline(self.df_raw, ticker='SPY')
        _, _, _, _ = pipe.train_test_split_scale(0.8)
        test_df = pipe.get_test_dataframe()
        self.assertEqual(len(test_df), 240)
        self.assertEqual(test_df.index[0].strftime('%Y-%m-%d'), '2025-09-11')
        self.assertEqual(test_df.index[-1].strftime('%Y-%m-%d'), '2026-08-25')

    @unittest.skipUnless(HAS_TRAIN_VAL_TEST, "train_val_test_split_scale required")
    def test_spy_65_15_20_split_matches_existing_chapter4_test_dates(self):
        """Test 4.4: 65/15/20 split test dates match existing 80/20 test dates 100% verbatim."""
        p_basic = Pipeline(self.df_raw, ticker='SPY')
        p_basic.train_test_split_scale(0.8)
        test_dates_8020 = p_basic.get_test_dataframe().index

        p_ext = Pipeline(self.df_raw, ticker='SPY', feature_set='extended' if HAS_EXTENDED_PIPELINE else 'basic')
        p_ext.train_val_test_split_scale(0.65, 0.15)
        test_dates_651520 = p_ext.test_df.index

        self.assertEqual(len(test_dates_651520), 240)
        self.assertTrue(test_dates_8020.equals(test_dates_651520))


class TestTier4RealWorldBacktestBenchmarks(unittest.TestCase):
    """Tier 4: Real-World Group 2 — Out-of-Sample Benchmark Execution on SPY (>=3 tests)."""

    def setUp(self):
        self.df_raw = load_real_spy_data()
        pipe = Pipeline(self.df_raw, ticker='SPY')
        pipe.clean_and_prepare()
        pipe.train_test_split_scale(0.8)
        self.test_df = pipe.get_test_dataframe()

    @unittest.skipUnless(HAS_BACKTESTER, "Backtester required")
    def test_dual_ma_crossover_benchmark_on_real_spy(self):
        """Test 4.5: Dual MA (20/50) benchmark achieves expected performance on real SPY test set."""
        base_ma = MACrossover(self.test_df.copy(), 20, 50, full_data=self.df_raw)
        bt = Backtest(self.test_df.copy(), base_ma, starting_cash=100000.0)
        bt.run()
        m = bt.cal_metrics()

        # Expected return is ~13.72%
        self.assertGreater(m["Total returns"], 0.08)
        self.assertLess(m["Total returns"], 0.22)
        # Expected Sharpe is ~1.27
        self.assertGreater(m["Sharpe Ratio"], 0.9)
        # Expected Max Drawdown is ~-5.05%
        self.assertGreater(m["Max drawdown"], -0.10)
        self.assertLess(m["Max drawdown"], 0.0)

    @unittest.skipUnless(HAS_BACKTESTER, "Backtester required")
    def test_buy_and_hold_benchmark_on_real_spy(self):
        """Test 4.6: Buy & Hold benchmark matches SPY close-to-close return (~17.76%)."""
        base_ma = MACrossover(self.test_df.copy(), 20, 50, full_data=self.df_raw)
        bt = Backtest(self.test_df.copy(), base_ma, starting_cash=100000.0)
        bt.run()
        m = bt.cal_metrics()
        expected_bnh = (self.test_df['Close'].iloc[-1] - self.test_df['Close'].iloc[0]) / self.test_df['Close'].iloc[0]
        self.assertAlmostEqual(m["Baseline score (buy-hold)"], expected_bnh, places=4)

    @unittest.skipUnless(HAS_BACKTESTER and HAS_LOGISTIC_REGRESSION, "Backtester and LR required")
    def test_real_spy_logistic_regression_hybrid_backtest(self):
        """Test 4.7: Full Chapter 4 LogisticRegression MLHybridStrategy backtest on real SPY."""
        pipe = Pipeline(self.df_raw, ticker='SPY')
        X_tr, X_te, y_tr, y_te = pipe.train_test_split_scale(0.8)
        test_df = pipe.get_test_dataframe()

        clf = LogisticRegression(learning_rate=0.05, n_epochs=3000, class_weight='balanced')
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te, threshold=0.22, smooth_window=3)

        base_ma = MACrossover(test_df.copy(), 20, 50, full_data=self.df_raw)
        strat_hyb = MLHybridStrategy(test_df.copy(), base_ma, y_pred)
        bt = Backtest(test_df.copy(), strat_hyb, starting_cash=100000.0)
        bt.run()
        m = bt.cal_metrics()

        # Hybrid strategy achieves Sharpe > 1.2 and positive return
        self.assertGreater(m["Total returns"], 0.10)
        self.assertGreater(m["Sharpe Ratio"], 1.20)
        self.assertGreater(m["Max drawdown"], -0.08)


class TestTier4HeadlessVisualizations(unittest.TestCase):
    """Tier 4: Real-World Group 3 — Headless Matplotlib Figure Generation (>=3 tests)."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @unittest.skipUnless(HAS_VISUALIZATIONS and hasattr(visualizations, 'plot_feature_correlation_heatmap'), "plot_feature_correlation_heatmap not available")
    def test_eda_correlation_heatmap_headless_render(self):
        """Test 4.8: plot_feature_correlation_heatmap saves valid image without GUI."""
        out_path = os.path.join(self.temp_dir, "heatmap.png")
        df_raw = load_real_spy_data()
        pipe = Pipeline(df_raw, ticker='SPY')
        clean_df = pipe.clean_and_prepare()
        visualizations.plot_feature_correlation_heatmap(clean_df, pipe.feature_names, save_path=out_path)
        self.assertTrue(os.path.exists(out_path))
        self.assertGreater(os.path.getsize(out_path), 5000)

    @unittest.skipUnless(HAS_VISUALIZATIONS and hasattr(visualizations, 'plot_volatility_regime_timeline'), "plot_volatility_regime_timeline not available")
    def test_volatility_timeline_headless_render(self):
        """Test 4.9: plot_volatility_regime_timeline saves valid image without GUI."""
        out_path = os.path.join(self.temp_dir, "timeline.png")
        df_raw = load_real_spy_data()
        pipe = Pipeline(df_raw, ticker='SPY')
        clean_df = pipe.clean_and_prepare()
        visualizations.plot_volatility_regime_timeline(clean_df, save_path=out_path)
        self.assertTrue(os.path.exists(out_path))
        self.assertGreater(os.path.getsize(out_path), 5000)

    @unittest.skipUnless(HAS_VISUALIZATIONS and hasattr(visualizations, 'plot_regime_agreement_barcode'), "plot_regime_agreement_barcode (M5) not yet implemented")
    def test_regime_barcode_heatmap_headless_render(self):
        """Test 4.10: plot_regime_agreement_barcode saves valid image without GUI."""
        out_path = os.path.join(self.temp_dir, "barcode.png")
        dates = pd.bdate_range('2025-01-01', periods=240)
        close = np.linspace(500, 600, 240)
        preds_dict = {
            "LR": np.ones(240),
            "MLP": np.ones(240),
        }
        y_true = np.ones(240)
        visualizations.plot_regime_agreement_barcode(dates, close, preds_dict, y_true, save_path=out_path)
        self.assertTrue(os.path.exists(out_path))
        self.assertGreater(os.path.getsize(out_path), 5000)

    @unittest.skipUnless(HAS_VISUALIZATIONS and hasattr(visualizations, 'plot_risk_return_frontier'), "plot_risk_return_frontier (M5) not yet implemented")
    def test_risk_return_frontier_headless_render(self):
        """Test 4.11: plot_risk_return_frontier saves valid image without GUI."""
        out_path = os.path.join(self.temp_dir, "frontier.png")
        metrics_dict = {
            "Buy & Hold": {"Total returns": 0.1776, "Max drawdown": -0.05, "Sharpe Ratio": 1.2},
            "Hybrid": {"Total returns": 0.1696, "Max drawdown": -0.044, "Sharpe Ratio": 1.54},
        }
        visualizations.plot_risk_return_frontier(metrics_dict, save_path=out_path)
        self.assertTrue(os.path.exists(out_path))
        self.assertGreater(os.path.getsize(out_path), 5000)


class TestTier4NotebookIntegrity(unittest.TestCase):
    """Tier 4: Real-World Group 4 — TradingNotebook.ipynb Structure & Syntax Integrity (>=2 tests)."""

    def setUp(self):
        self.nb_path = BACKTESTER_DIR / "TradingNotebook.ipynb"

    def test_notebook_file_exists_and_valid_json(self):
        """Test 4.11: TradingNotebook.ipynb exists, is valid JSON, and has nbformat >= 4."""
        self.assertTrue(self.nb_path.exists())
        with open(self.nb_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
        self.assertIn("cells", nb)
        self.assertGreaterEqual(nb.get("nbformat", 0), 4)
        self.assertGreater(len(nb["cells"]), 10)

    def test_notebook_python_cells_have_valid_syntax(self):
        """Test 4.12: All code cells in TradingNotebook.ipynb compile to valid Python syntax."""
        with open(self.nb_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)

        code_cells = [cell for cell in nb["cells"] if cell.get("cell_type") == "code"]
        self.assertGreater(len(code_cells), 0)

        for i, cell in enumerate(code_cells):
            source = "".join(cell.get("source", []))
            # Filter out IPython magic commands (! or % prefix) before syntax checking
            filtered_lines = [
                line for line in source.splitlines()
                if not line.strip().startswith(('!', '%'))
            ]
            clean_source = "\n".join(filtered_lines)
            try:
                compile(clean_source, f"Cell_{i}", "exec")
            except SyntaxError as e:
                self.fail(f"Syntax error in notebook code cell {i}: {e}")


if __name__ == '__main__':
    unittest.main()
