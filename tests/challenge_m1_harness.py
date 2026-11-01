"""
Challenger 1 Empirical Verification Harness for Milestone 1
Adversarial stress-testing of preprocessing_pipeline.py:
1. Mathematical precision vs NumPy oracles across all 14 features.
2. Dimensionless & scale invariance stress test.
3. Zero-leakage invariance under future perturbations & data tampering.
4. 1-to-1 temporal sequence alignment for out-of-sample days (t=0..239).
"""

import os
import sys
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('Market_Regime_Prediction'))

import numpy as np
import pandas as pd
from Market_Regime_Prediction.preprocessing_pipeline import Pipeline

def run_tests():
    print("=" * 70)
    print("CHALLENGER 1: EMPIRICAL STRESS TEST SUITE FOR MILESTONE 1")
    print("=" * 70)

    # Load SPY real data
    csv_path = 'Trading Strategies Backtesting System/SPY_5_years'
    df_raw = pd.read_csv(csv_path, header=[0, 1], index_col=0, parse_dates=True)
    print(f"Loaded SPY raw dataset: shape {df_raw.shape}")

    failures = []

    # -------------------------------------------------------------------------
    # SUITE 1: MATHEMATICAL PRECISION AGAINST INDEPENDENT NUMPY ORACLES
    # -------------------------------------------------------------------------
    print("\n[SUITE 1] Mathematical Precision Against Independent NumPy Oracles...")
    pipe = Pipeline(df_raw, ticker='SPY', feature_set='extended')
    feat_df = pipe.generate_features()
    clean_df = pipe.clean_and_prepare()

    # Raw extracted columns
    spy_df = df_raw.xs('SPY', axis=1, level=1).copy().sort_index()
    c = spy_df['Close'].values
    o = spy_df['Open'].values
    h = spy_df['High'].values
    l = spy_df['Low'].values
    v = spy_df['Volume'].values

    # Oracle 1: candle_body = (Close - Open) / Open
    oracle_candle_body = (c - o) / o
    diff = np.nanmax(np.abs(feat_df['candle_body'].values - oracle_candle_body))
    print(f"  - candle_body max diff: {diff:.2e}")
    if diff > 1e-12:
        failures.append(f"candle_body failed precision test: diff={diff}")

    # Oracle 2: intraday_range = (High - Low) / Close
    oracle_intraday_range = (h - l) / c
    diff = np.nanmax(np.abs(feat_df['intraday_range'].values - oracle_intraday_range))
    print(f"  - intraday_range max diff: {diff:.2e}")
    if diff > 1e-12:
        failures.append(f"intraday_range failed precision test: diff={diff}")

    # Oracle 3: volume_ratio = Volume / SMA_20(Volume)
    # Independent NumPy convolution
    kernel_20 = np.ones(20) / 20.0
    v_sma20 = np.convolve(v, kernel_20, mode='full')[:len(v)]
    v_sma20[:19] = np.nan
    oracle_volume_ratio = v / (v_sma20 + 1e-10)
    diff = np.nanmax(np.abs(feat_df['volume_ratio'].values[19:] - oracle_volume_ratio[19:]))
    print(f"  - volume_ratio max diff (post-warmup): {diff:.2e}")
    if diff > 1e-10:
        failures.append(f"volume_ratio failed precision test: diff={diff}")

    # Oracle 4: ret_1d = (Close[t] - Close[t-1]) / Close[t-1]
    oracle_ret_1d = np.empty_like(c)
    oracle_ret_1d[0] = np.nan
    oracle_ret_1d[1:] = (c[1:] - c[:-1]) / c[:-1]
    diff = np.nanmax(np.abs(feat_df['ret_1d'].values[1:] - oracle_ret_1d[1:]))
    print(f"  - ret_1d max diff: {diff:.2e}")
    if diff > 1e-12:
        failures.append(f"ret_1d failed precision test: diff={diff}")

    # Oracle 5: ret_5d = (Close[t] - Close[t-5]) / Close[t-5]
    oracle_ret_5d = np.empty_like(c)
    oracle_ret_5d[:5] = np.nan
    oracle_ret_5d[5:] = (c[5:] - c[:-5]) / c[:-5]
    diff = np.nanmax(np.abs(feat_df['ret_5d'].values[5:] - oracle_ret_5d[5:]))
    print(f"  - ret_5d max diff: {diff:.2e}")
    if diff > 1e-12:
        failures.append(f"ret_5d failed precision test: diff={diff}")

    # Oracle 6: ret_10d = (Close[t] - Close[t-10]) / Close[t-10]
    oracle_ret_10d = np.empty_like(c)
    oracle_ret_10d[:10] = np.nan
    oracle_ret_10d[10:] = (c[10:] - c[:-10]) / c[:-10]
    diff = np.nanmax(np.abs(feat_df['ret_10d'].values[10:] - oracle_ret_10d[10:]))
    print(f"  - ret_10d max diff: {diff:.2e}")
    if diff > 1e-12:
        failures.append(f"ret_10d failed precision test: diff={diff}")

    # Oracle 7: vol_10d = rolling 10-day std of daily returns (ddof=1)
    ret_series = pd.Series(oracle_ret_1d)
    oracle_vol_10d = np.array([
        np.std(oracle_ret_1d[i-9:i+1], ddof=1) if i >= 10 else np.nan
        for i in range(len(c))
    ])
    diff = np.nanmax(np.abs(feat_df['vol_10d'].values[10:] - oracle_vol_10d[10:]))
    print(f"  - vol_10d max diff: {diff:.2e}")
    if diff > 1e-12:
        failures.append(f"vol_10d failed precision test: diff={diff}")

    # Oracle 8: vol_20d = rolling 20-day std of daily returns (ddof=1)
    oracle_vol_20d = np.array([
        np.std(oracle_ret_1d[i-19:i+1], ddof=1) if i >= 20 else np.nan
        for i in range(len(c))
    ])
    diff = np.nanmax(np.abs(feat_df['vol_20d'].values[20:] - oracle_vol_20d[20:]))
    print(f"  - vol_20d max diff: {diff:.2e}")
    if diff > 1e-12:
        failures.append(f"vol_20d failed precision test: diff={diff}")

    # Oracle 9: sma20_ratio = Close / SMA_20(Close)
    c_sma20 = np.convolve(c, kernel_20, mode='full')[:len(c)]
    c_sma20[:19] = np.nan
    oracle_sma20_ratio = c / c_sma20
    diff = np.nanmax(np.abs(feat_df['sma20_ratio'].values[19:] - oracle_sma20_ratio[19:]))
    print(f"  - sma20_ratio max diff: {diff:.2e}")
    if diff > 1e-10:
        failures.append(f"sma20_ratio failed precision test: diff={diff}")

    # Oracle 10: sma50_ratio = Close / SMA_50(Close)
    kernel_50 = np.ones(50) / 50.0
    c_sma50 = np.convolve(c, kernel_50, mode='full')[:len(c)]
    c_sma50[:49] = np.nan
    oracle_sma50_ratio = c / c_sma50
    diff = np.nanmax(np.abs(feat_df['sma50_ratio'].values[49:] - oracle_sma50_ratio[49:]))
    print(f"  - sma50_ratio max diff: {diff:.2e}")
    if diff > 1e-10:
        failures.append(f"sma50_ratio failed precision test: diff={diff}")

    # Oracle 11: rsi_14
    deltas = np.diff(c, prepend=np.nan)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    kernel_14 = np.ones(14) / 14.0
    avg_gain = np.convolve(gains, kernel_14, mode='full')[:len(c)]
    avg_loss = np.convolve(losses, kernel_14, mode='full')[:len(c)]
    avg_gain[:14] = np.nan
    avg_loss[:14] = np.nan
    rs = avg_gain / (avg_loss + 1e-10)
    oracle_rsi_14 = 100.0 - (100.0 / (1.0 + rs))
    diff = np.nanmax(np.abs(feat_df['rsi_14'].values[14:] - oracle_rsi_14[14:]))
    print(f"  - rsi_14 max diff: {diff:.2e}")
    if diff > 1e-10:
        failures.append(f"rsi_14 failed precision test: diff={diff}")

    # Oracle 12: macd_norm = (EMA_12 - EMA_26) / Close
    def compute_ema(series, span):
        alpha = 2.0 / (span + 1.0)
        ema = np.empty_like(series, dtype=float)
        ema[0] = series[0]
        for t in range(1, len(series)):
            ema[t] = alpha * series[t] + (1.0 - alpha) * ema[t-1]
        return ema

    ema_12 = compute_ema(c, 12)
    ema_26 = compute_ema(c, 26)
    macd_raw = ema_12 - ema_26
    oracle_macd_norm = macd_raw / c
    diff = np.nanmax(np.abs(feat_df['macd_norm'].values - oracle_macd_norm))
    print(f"  - macd_norm max diff: {diff:.2e}")
    if diff > 1e-12:
        failures.append(f"macd_norm failed precision test: diff={diff}")

    # Oracle 13: macd_hist_norm = (MACD - EMA_9(MACD)) / Close
    macd_sig = compute_ema(macd_raw, 9)
    oracle_macd_hist_norm = (macd_raw - macd_sig) / c
    diff = np.nanmax(np.abs(feat_df['macd_hist_norm'].values - oracle_macd_hist_norm))
    print(f"  - macd_hist_norm max diff: {diff:.2e}")
    if diff > 1e-12:
        failures.append(f"macd_hist_norm failed precision test: diff={diff}")

    # Oracle 14: vol_regime = vol_10d / (vol_20d + 1e-10)
    oracle_vol_regime = oracle_vol_10d / (oracle_vol_20d + 1e-10)
    diff = np.nanmax(np.abs(feat_df['vol_regime'].values[20:] - oracle_vol_regime[20:]))
    print(f"  - vol_regime max diff: {diff:.2e}")
    if diff > 1e-12:
        failures.append(f"vol_regime failed precision test: diff={diff}")

    # -------------------------------------------------------------------------
    # SUITE 2: DIMENSIONLESS & SCALE INVARIANCE
    # -------------------------------------------------------------------------
    print("\n[SUITE 2] Scale Invariance Stress Test (Dimensionless verification)...")
    scale_factor = 345.678
    scaled_df = spy_df.copy()
    scaled_df['Open'] *= scale_factor
    scaled_df['High'] *= scale_factor
    scaled_df['Low'] *= scale_factor
    scaled_df['Close'] *= scale_factor
    scaled_df['Volume'] *= 42.0  # arbitrary volume multiplier

    pipe_scaled = Pipeline(scaled_df, feature_set='extended')
    feat_scaled_df = pipe_scaled.generate_features()

    for col in pipe.feature_names:
        v_orig = feat_df[col].dropna().values
        v_scaled = feat_scaled_df[col].dropna().values
        max_abs_diff = np.max(np.abs(v_orig - v_scaled))
        print(f"  - {col:16s} scale invariance diff: {max_abs_diff:.2e}")
        # Allow small numerical diff up to 1e-6 due to regularizing epsilon constants (+1e-10)
        if max_abs_diff > 1e-6:
            failures.append(f"Feature {col} is NOT dimensionless / scale-invariant! max_diff={max_abs_diff}")

    # -------------------------------------------------------------------------
    # SUITE 3: ZERO-LEAKAGE INVARIANCE UNDER FUTURE DATA PERTURBATIONS
    # -------------------------------------------------------------------------
    print("\n[SUITE 3] Zero-Leakage Invariance Under Future Data Perturbations...")
    pipe_baseline = Pipeline(df_raw, ticker='SPY', feature_set='extended')
    X_tr_base, X_val_base, X_te_base, y_tr_base, y_val_base, y_te_base = pipe_baseline.train_val_test_split_scale(0.65, 0.15)
    mean_base = pipe_baseline.mean_.copy()
    std_base = pipe_baseline.std_.copy()

    # Adversarial Attack A: Corrupt future test set data
    # Modify the last 100 rows of raw test data in the clean_df directly before split_scale
    clean_tampered = pipe_baseline.clean_df.copy()
    tampered_indices = clean_tampered.index[960 + 50:]
    clean_tampered.loc[tampered_indices, 'Close'] = 99999999.0
    clean_tampered.loc[tampered_indices, pipe_baseline.feature_names] = np.random.randn(len(tampered_indices), len(pipe_baseline.feature_names)) * 1000.0

    # Build a tampered pipeline instance
    pipe_tampered = Pipeline(df_raw, ticker='SPY', feature_set='extended')
    pipe_tampered.clean_df = clean_tampered

    X_tr_tamp, X_val_tamp, X_te_tamp, _, _, _ = pipe_tampered.train_val_test_split_scale(0.65, 0.15)
    mean_tamp = pipe_tampered.mean_
    std_tamp = pipe_tampered.std_

    diff_mean = np.max(np.abs(mean_base - mean_tamp))
    diff_std = np.max(np.abs(std_base - std_tamp))
    diff_X_tr = np.max(np.abs(X_tr_base - X_tr_tamp))
    diff_X_val = np.max(np.abs(X_val_base - X_val_tamp))

    print(f"  - Mean diff after test set corruption: {diff_mean:.2e}")
    print(f"  - Std diff after test set corruption:  {diff_std:.2e}")
    print(f"  - X_tr diff after test corruption:     {diff_X_tr:.2e}")
    print(f"  - X_val diff after test corruption:    {diff_X_val:.2e}")

    if diff_mean != 0.0 or diff_std != 0.0 or diff_X_tr != 0.0 or diff_X_val != 0.0:
        failures.append("Zero-leakage violated: Training statistics changed when test set was corrupted!")

    # Verify that test rows 0..49 (unmodified raw rows) produce identical scaled outputs despite future 50..239 being corrupted
    diff_unmodified_te = np.max(np.abs(X_te_base[:50] - X_te_tamp[:50]))
    print(f"  - Unmodified test rows diff:           {diff_unmodified_te:.2e}")
    if diff_unmodified_te != 0.0:
        failures.append(f"Future leakage detected: Unmodified test rows changed when subsequent test rows were corrupted! diff={diff_unmodified_te}")

    # Adversarial Attack B: Corrupt validation set data
    clean_val_tampered = pipe_baseline.clean_df.copy()
    val_indices = clean_val_tampered.index[780:960]
    clean_val_tampered.loc[val_indices, pipe_baseline.feature_names] = 888888.0
    pipe_val_tampered = Pipeline(df_raw, ticker='SPY', feature_set='extended')
    pipe_val_tampered.clean_df = clean_val_tampered
    pipe_val_tampered.train_val_test_split_scale(0.65, 0.15)

    diff_val_attack_mean = np.max(np.abs(mean_base - pipe_val_tampered.mean_))
    diff_val_attack_std = np.max(np.abs(std_base - pipe_val_tampered.std_))
    print(f"  - Mean diff after val set corruption:  {diff_val_attack_mean:.2e}")
    print(f"  - Std diff after val set corruption:   {diff_val_attack_std:.2e}")
    if diff_val_attack_mean != 0.0 or diff_val_attack_std != 0.0:
        failures.append("Zero-leakage violated: Training statistics changed when validation set was corrupted!")

    # -------------------------------------------------------------------------
    # SUITE 4: 1-TO-1 TEMPORAL SEQUENCE ALIGNMENT (t = 0 .. 239)
    # -------------------------------------------------------------------------
    print("\n[SUITE 4] 1-to-1 Temporal Sequence Alignment (t = 0 .. 239)...")
    X_te_seq, y_te_seq = Pipeline.create_sequences(X_te_base, y_te_base, seq_len=20, prev_features_2d=X_val_base)
    print(f"  - X_te_seq shape: {X_te_seq.shape}")
    print(f"  - y_te_seq shape: {y_te_seq.shape}")

    if X_te_seq.shape != (240, 20, 14):
        failures.append(f"X_te_seq expected shape (240, 20, 14), got {X_te_seq.shape}")
    if y_te_seq.shape != (240, 1):
        failures.append(f"y_te_seq expected shape (240, 1), got {y_te_seq.shape}")

    # Check temporal alignment for every day t in 0..239
    mismatches = 0
    for t in range(240):
        # 1. Timestep -1 must exactly equal X_te_base[t]
        last_step = X_te_seq[t, -1, :]
        expected_last = X_te_base[t, :]
        if not np.allclose(last_step, expected_last, atol=1e-12):
            mismatches += 1
            if mismatches <= 3:
                print(f"    Mismatch at t={t}: diff={np.max(np.abs(last_step - expected_last))}")

        # 2. Target must match y_te_base[t]
        if y_te_seq[t, 0] != y_te_base[t, 0]:
            failures.append(f"Target mismatch at t={t}: y_seq={y_te_seq[t, 0]}, y_base={y_te_base[t, 0]}")

    print(f"  - Timestep -1 alignment mismatches across all 240 days: {mismatches}")
    if mismatches > 0:
        failures.append(f"Sequence window timestep -1 failed alignment for {mismatches} test days!")

    # Check warmup boundary: t=0 window lookback
    # Timestep 0..18 should match X_val_base[-19:]
    t0_warmup = X_te_seq[0, :19, :]
    expected_warmup = X_val_base[-19:, :]
    diff_warmup = np.max(np.abs(t0_warmup - expected_warmup))
    print(f"  - t=0 warmup context max diff vs X_val[-19:]: {diff_warmup:.2e}")
    if diff_warmup > 1e-12:
        failures.append(f"t=0 warmup context failed alignment: diff={diff_warmup}")

    # Check date index alignment
    test_dates = pipe_baseline.get_test_dataframe().index
    print(f"  - Test date range: {test_dates[0].strftime('%Y-%m-%d')} to {test_dates[-1].strftime('%Y-%m-%d')} ({len(test_dates)} dates)")
    if len(test_dates) != 240:
        failures.append(f"Test dataframe length is {len(test_dates)}, expected 240")

    # Check memory independence: modifying X_te_seq must NOT mutate X_te_base or X_val_base
    X_te_seq_copy = X_te_seq.copy()
    X_te_seq[0, 0, 0] += 999.0
    if np.any(X_val_base[-19, 0] == X_te_seq[0, 0, 0]):
        failures.append("Memory hazard: Mutating X_te_seq altered X_val_base (strided array not properly decoupled)!")
    X_te_seq[:] = X_te_seq_copy  # restore

    # -------------------------------------------------------------------------
    # SUITE 5: TARGET HORIZON & STATIONARY LOOKBACK BOUNDS
    # -------------------------------------------------------------------------
    print("\n[SUITE 5] Target Horizon & Boundary Invariance Check...")
    # Target definition check: Target = 1 if Close[t+5] > SMA_50[t] else 0
    clean = pipe_baseline.clean_df
    close_vals = clean['Close'].values
    target_vals = clean['target'].values
    raw_close = df_raw.xs('SPY', axis=1, level=1)['Close'].sort_index()

    # For any clean row t, verify target calculation against raw data
    target_errors = 0
    for idx_name in clean.index[:50]:
        loc_in_raw = raw_close.index.get_loc(idx_name)
        future_loc = loc_in_raw + 5
        sma_locs = range(loc_in_raw - 49, loc_in_raw + 1)
        expected_sma = raw_close.iloc[sma_locs].mean()
        expected_future = raw_close.iloc[future_loc]
        expected_target = 1 if expected_future > expected_sma else 0
        actual_target = clean.loc[idx_name, 'target']
        if actual_target != expected_target:
            target_errors += 1
    print(f"  - Target alignment errors on sample rows: {target_errors}")
    if target_errors > 0:
        failures.append(f"Target calculation mismatch against ground truth: {target_errors} errors")

    # -------------------------------------------------------------------------
    # SUITE 6: ADVERSARIAL EDGE CASES (Constant Price, Zero Vol, Monotonic)
    # -------------------------------------------------------------------------
    print("\n[SUITE 6] Adversarial Edge Cases (Flatline, Zero Volume, Monotonic Runs)...")
    
    # 6.1 Flatline / Zero Volatility
    flat_dates = pd.bdate_range('2022-01-01', periods=150)
    flat_df = pd.DataFrame({
        'Open': 100.0,
        'High': 100.0,
        'Low': 100.0,
        'Close': 100.0,
        'Volume': 1000.0
    }, index=flat_dates)
    pipe_flat = Pipeline(flat_df, feature_set='extended')
    feat_flat = pipe_flat.generate_features()
    clean_flat = pipe_flat.clean_and_prepare()
    X_tr_flat, X_val_flat, X_te_flat, _, _, _ = pipe_flat.train_val_test_split_scale(0.65, 0.15)
    
    # Check for NaNs or Infs
    has_nan_inf = np.isnan(X_tr_flat).any() or np.isinf(X_tr_flat).any()
    print(f"  - Flatline dataset zero-std division test: has_nan_inf={has_nan_inf}")
    if has_nan_inf:
        failures.append("Flatline dataset produced NaN or Inf in scaled features!")

    # 6.2 Zero Volume
    zero_vol_df = flat_df.copy()
    zero_vol_df['Volume'] = 0.0
    pipe_zvol = Pipeline(zero_vol_df, feature_set='extended')
    feat_zvol = pipe_zvol.generate_features()
    has_nan_inf_zvol = np.isnan(feat_zvol['volume_ratio'].dropna()).any() or np.isinf(feat_zvol['volume_ratio'].dropna()).any()
    print(f"  - Zero volume test: has_nan_inf={has_nan_inf_zvol}")
    if has_nan_inf_zvol:
        failures.append("Zero volume produced NaN or Inf in volume_ratio!")

    # 6.3 Pure Monotonic Bull Run (RSI upper bound)
    bull_df = pd.DataFrame({
        'Open': np.linspace(100, 200, 150),
        'High': np.linspace(101, 201, 150),
        'Low': np.linspace(99, 199, 150),
        'Close': np.linspace(100.5, 200.5, 150),
        'Volume': 1000.0
    }, index=flat_dates)
    pipe_bull = Pipeline(bull_df, feature_set='extended')
    feat_bull = pipe_bull.generate_features()
    rsi_bull = feat_bull['rsi_14'].dropna().values
    print(f"  - Pure bull run RSI values (min, max): {rsi_bull.min():.2f}, {rsi_bull.max():.2f}")
    if not np.allclose(rsi_bull, 100.0, atol=1e-5):
        failures.append(f"Monotonic bull run RSI expected 100.0, got min={rsi_bull.min()}")

    # 6.4 Pure Monotonic Bear Drop (RSI lower bound)
    bear_df = pd.DataFrame({
        'Open': np.linspace(200, 100, 150),
        'High': np.linspace(201, 101, 150),
        'Low': np.linspace(199, 99, 150),
        'Close': np.linspace(199.5, 99.5, 150),
        'Volume': 1000.0
    }, index=flat_dates)
    pipe_bear = Pipeline(bear_df, feature_set='extended')
    feat_bear = pipe_bear.generate_features()
    rsi_bear = feat_bear['rsi_14'].dropna().values
    print(f"  - Pure bear drop RSI values (min, max): {rsi_bear.min():.2f}, {rsi_bear.max():.2f}")
    if not np.allclose(rsi_bear, 0.0, atol=1e-5):
        failures.append(f"Monotonic bear drop RSI expected 0.0, got max={rsi_bear.max()}")

    # -------------------------------------------------------------------------
    # SUITE 7: EXHAUSTIVE TEMPORAL CAUSALITY ACROSS ALL 20 TIMESTEPS & 240 DAYS
    # -------------------------------------------------------------------------
    print("\n[SUITE 7] Exhaustive Temporal Causality Across All 20 Timesteps & 240 Days...")
    # In X_te_seq[t, k, :], where t in [0..239] and k in [0..19]:
    # Let combined_scaled = vstack([X_val_base[-19:], X_te_base])
    # The element at index (t, k) MUST correspond strictly to combined_scaled[t + k]
    combined_scaled = np.vstack([X_val_base[-19:], X_te_base])
    full_causality_errors = 0
    for t in range(240):
        for k in range(20):
            expected_vec = combined_scaled[t + k]
            actual_vec = X_te_seq[t, k, :]
            if not np.array_equal(actual_vec, expected_vec):
                full_causality_errors += 1
    print(f"  - Exhaustive 240x20 window-to-timeline matching errors: {full_causality_errors}")
    if full_causality_errors > 0:
        failures.append(f"Temporal causality violated in {full_causality_errors} window cells!")

    # -------------------------------------------------------------------------
    # SUITE 8: BACKWARD COMPATIBILITY VERIFICATION
    # -------------------------------------------------------------------------
    print("\n[SUITE 8] Backward Compatibility Verification (feature_set='basic')...")
    pipe_basic = Pipeline(df_raw, ticker='SPY', feature_set='basic')
    clean_basic = pipe_basic.clean_and_prepare()
    X_tr_b, X_te_b, y_tr_b, y_te_b = pipe_basic.train_test_split_scale(0.8)
    print(f"  - Basic feature names ({len(pipe_basic.feature_names)}): {pipe_basic.feature_names}")
    print(f"  - Basic clean_df shape: {clean_basic.shape}")
    print(f"  - Basic split shapes: X_tr={X_tr_b.shape}, X_te={X_te_b.shape}")

    expected_basic_features = ['ret_5d', 'vol_20d', 'rsi_14', 'sma50_ratio']
    if pipe_basic.feature_names != expected_basic_features:
        failures.append(f"Basic feature set mismatch: got {pipe_basic.feature_names}")
    if X_tr_b.shape != (960, 4) or X_te_b.shape != (240, 4):
        failures.append(f"Basic split shape mismatch: ({X_tr_b.shape}, {X_te_b.shape})")

    # Verify that basic feature values match extended feature values exactly on the same columns
    for f_name in expected_basic_features:
        b_vals = clean_basic[f_name].values
        e_vals = pipe_baseline.clean_df[f_name].values
        max_diff = np.max(np.abs(b_vals - e_vals))
        print(f"  - basic vs extended matching on '{f_name}': max diff = {max_diff:.2e}")
        if max_diff > 1e-14:
            failures.append(f"Discrepancy between basic and extended for {f_name}: diff={max_diff}")

    # Verify that test split index in basic and extended are 100% identical
    basic_test_index = pipe_basic.test_df.index
    extended_test_index = pipe_baseline.test_df.index
    if not basic_test_index.equals(extended_test_index):
        failures.append("Test index mismatch between basic and extended splits!")
    else:
        print("  - Basic and Extended test splits have 100% identical datetime indices.")

    print("\n" + "=" * 70)
    if failures:
        print(f"FAILED: {len(failures)} failures detected!")
        for f in failures:
            print(f"  [X] {f}")
        return False
    else:
        print("PASSED: ALL EMPIRICAL CHALLENGE CHECKS PASSED WITH 0 FAILURES!")
        print("=" * 70)
        return True

if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)

