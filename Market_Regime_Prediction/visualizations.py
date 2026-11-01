"""
Quantitative & Deep Learning Visualization Suite for Market Regime Prediction.

Contains modular Matplotlib plotting utilities for:
- Part A: Exploratory Data Analysis & Feature Dynamics (Correlation Heatmap, Regime Distributions, Timeline)
- Part B: Deep Learning Diagnostics & Model Interpretability (Training Curves Grid, Confusion Matrix Grid, ROC/PR Curves, Feature Importance Heatmap, Transformer Attention Heatmap)
- Part C: Multi-Model Strategy Comparison & Portfolio Analytics (Planned Milestone 5)
"""

import os
import tempfile

# Ensure headless Matplotlib safety and writable cache directory
os.environ.setdefault('MPLCONFIGDIR', os.path.join(tempfile.gettempdir(), 'matplotlib'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


# ==============================================================================
# Part A: Exploratory Data Analysis & Feature Dynamics (EDA)
# ==============================================================================

def plot_feature_correlation_heatmap(df, feature_names, target_col='target', save_path=None, ax=None):
    """
    Computes and plots the Pearson correlation matrix for input features and the forward regime target.
    Exposes multicollinearity clusters and linear correlations with the target.

    Parameters:
        df (pd.DataFrame): DataFrame containing features and target column.
        feature_names (list of str): List of feature column names.
        target_col (str): Target column name (default 'target').
        save_path (str, optional): File path to save the generated figure.
        ax (matplotlib.axes.Axes, optional): Pre-existing axes for plotting.

    Returns:
        matplotlib.figure.Figure: The generated or updated figure object.
    """
    cols = [col for col in feature_names if col in df.columns]
    if target_col in df.columns and target_col not in cols:
        cols.append(target_col)

    corr = df[cols].corr()

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 10))
        created_fig = True
    else:
        fig = ax.figure

    im = ax.imshow(corr.values, cmap='coolwarm', vmin=-1.0, vmax=1.0)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson Correlation")

    n_cols = len(cols)
    ax.set_xticks(range(n_cols))
    ax.set_yticks(range(n_cols))
    ax.set_xticklabels(cols, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(cols, fontsize=9)

    # Annotate correlation coefficient in each cell with dynamic text contrast
    for i in range(n_cols):
        for j in range(n_cols):
            val = corr.iloc[i, j]
            text_color = "white" if abs(val) > 0.55 else "black"
            font_size = 7.5 if n_cols > 12 else 9.0
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=font_size)

    ax.set_title("Feature & Regime Target Correlation Heatmap", fontsize=13, fontweight='bold', pad=12)

    if created_fig:
        fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight', dpi=150)

    return fig


def plot_regime_feature_distributions(df, feature_names, target_col='target', save_path=None):
    """
    Renders a multi-panel grid comparing standardized feature distributions between
    Bullish (target == 1) and Bearish (target == 0) market regimes.

    Parameters:
        df (pd.DataFrame): DataFrame containing features and target column.
        feature_names (list of str): List of feature column names.
        target_col (str): Binary regime label column ('target').
        save_path (str, optional): File path to save the generated figure.

    Returns:
        matplotlib.figure.Figure: The generated figure object.
    """
    valid_features = [f for f in feature_names if f in df.columns]
    n_feats = len(valid_features)
    if n_feats == 0:
        raise ValueError("No valid features found in DataFrame to plot distributions.")
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    n_cols = 4 if n_feats > 4 else n_feats
    n_rows = int(np.ceil(n_feats / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.0 * n_cols, 3.0 * n_rows))
    axes_flat = np.atleast_1d(axes).ravel()

    for idx, feat in enumerate(valid_features):
        ax = axes_flat[idx]
        bullish = df.loc[df[target_col] == 1, feat].dropna()
        bearish = df.loc[df[target_col] == 0, feat].dropna()

        # Combine ranges to align histogram bins
        combined = pd.concat([bullish, bearish])
        if len(combined) > 0:
            bins = np.linspace(np.percentile(combined, 1), np.percentile(combined, 99), 25)
        else:
            bins = 20

        ax.hist(bullish, bins=bins, density=True, alpha=0.55, color='#2ca02c', label='Bullish (1)')
        ax.hist(bearish, bins=bins, density=True, alpha=0.55, color='#d62728', label='Bearish (0)')

        # Mark medians
        if len(bullish) > 0:
            ax.axvline(bullish.median(), color='#1b611b', linestyle='--', linewidth=1.2, alpha=0.8)
        if len(bearish) > 0:
            ax.axvline(bearish.median(), color='#801717', linestyle='--', linewidth=1.2, alpha=0.8)

        ax.set_title(feat, fontsize=10, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.3)
        if idx == 0:
            ax.legend(loc='upper right', fontsize=8)

    # Hide unused subplots
    for j in range(n_feats, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Feature Distributions by Market Regime (Bullish vs. Bearish)", fontsize=14, fontweight='bold', y=0.995)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight', dpi=150)

    return fig


def plot_volatility_regime_timeline(df, train_ratio=0.65, val_ratio=0.15, save_path=None):
    """
    Renders a 3-panel chronological timeline sharing the x-axis:
    1. Asset Close price + 50-day SMA with shaded Train (65%), Val (15%), and Test (20%) splits.
    2. Rolling return volatility comparison (vol_10d vs vol_20d).
    3. Forward-looking market regime target state (Bullish vs Bearish).

    Parameters:
        df (pd.DataFrame): Cleaned DataFrame with DatetimeIndex, containing 'Close',
                           volatility features, and 'target'.
        train_ratio (float): Ratio for training split (default 0.65).
        val_ratio (float): Ratio for validation split (default 0.15).
        save_path (str, optional): File path to save the generated figure.

    Returns:
        matplotlib.figure.Figure: The generated figure object.
    """
    if 'Close' not in df.columns:
        raise ValueError("DataFrame must contain 'Close' price column.")

    n = len(df)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_start_date = df.index[0]
    train_end_date = df.index[train_end - 1]
    val_start_date = df.index[train_end]
    val_end_date = df.index[val_end - 1]
    test_start_date = df.index[val_end]
    test_end_date = df.index[-1]

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(15, 11), sharex=True,
        gridspec_kw={'height_ratios': [2.6, 1.8, 1.2]}
    )

    # Panel 1: Price and SMA with Split Shading
    ax1.plot(df.index, df['Close'], color='#1f77b4', lw=1.5, label='Close Price')
    if 'sma_50' in df.columns:
        ax1.plot(df.index, df['sma_50'], color='#ff7f0e', lw=1.2, linestyle='--', label='50-day SMA')
    else:
        sma_50 = df['Close'].rolling(50, min_periods=1).mean()
        ax1.plot(df.index, sma_50, color='#ff7f0e', lw=1.2, linestyle='--', label='50-day SMA')

    # Chronological Split Spans across all axes
    for ax in (ax1, ax2, ax3):
        ax.axvspan(train_start_date, val_start_date, color='#4682b4', alpha=0.10)
        ax.axvspan(val_start_date, test_start_date, color='#ffa500', alpha=0.12)
        ax.axvspan(test_start_date, test_end_date, color='#2ca02c', alpha=0.12)

    # Labels for split spans on Panel 1
    ax1.plot([], [], color='#4682b4', alpha=0.5, lw=6, label=f'Train Split ({int(train_ratio*100)}%, {train_end}d)')
    ax1.plot([], [], color='#ffa500', alpha=0.5, lw=6, label=f'Val Split ({int(val_ratio*100)}%, {val_end - train_end}d)')
    ax1.plot([], [], color='#2ca02c', alpha=0.5, lw=6, label=f'Test Split ({int((1-train_ratio-val_ratio)*100)}%, {n - val_end}d)')

    ax1.set_ylabel("Price ($)", fontsize=11)
    ax1.set_title("Market History & Chronological Split Timeline", fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, linestyle='--', alpha=0.3)

    # Panel 2: Volatility Dynamics
    has_vol = False
    if 'vol_10d' in df.columns:
        ax2.plot(df.index, df['vol_10d'], color='#e377c2', lw=1.2, alpha=0.85, label='10-day Volatility')
        has_vol = True
    if 'vol_20d' in df.columns:
        ax2.plot(df.index, df['vol_20d'], color='#17becf', lw=1.2, alpha=0.85, label='20-day Volatility')
        has_vol = True
    if not has_vol:
        ret = df['Close'].pct_change()
        ax2.plot(df.index, ret.rolling(10).std(), color='#e377c2', lw=1.2, label='10-day Volatility')
        ax2.plot(df.index, ret.rolling(20).std(), color='#17becf', lw=1.2, label='20-day Volatility')

    ax2.set_ylabel("Rolling Volatility", fontsize=11)
    ax2.set_title("Volatility Clustering & Regime Expansion", fontsize=11, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, linestyle='--', alpha=0.3)

    # Panel 3: Ground-Truth Regime Target
    if 'target' in df.columns:
        bullish_mask = (df['target'] == 1)
        ax3.fill_between(df.index, 0, 1, where=bullish_mask, color='#2ca02c', alpha=0.45, step='mid', label='Bullish Regime (1)')
        ax3.fill_between(df.index, 0, 1, where=~bullish_mask, color='#d62728', alpha=0.45, step='mid', label='Bearish Regime (0)')
        ax3.step(df.index, df['target'], color='black', lw=0.7, where='mid')
        ax3.set_ylim(-0.1, 1.1)
        ax3.set_yticks([0, 1])
        ax3.set_yticklabels(['Bearish (0)', 'Bullish (1)'])
        ax3.set_ylabel("Regime Target", fontsize=11)
        ax3.set_title("Forward Market Regime Target Label (Horizon=5d)", fontsize=11, fontweight='bold')
        ax3.legend(loc='upper left', fontsize=9)
        ax3.grid(True, linestyle='--', alpha=0.3)

    ax3.set_xlabel("Date", fontsize=11)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight', dpi=150)

    return fig


# ==============================================================================
# Pure NumPy Metric Helpers (No Scikit-Learn Dependency)
# ==============================================================================

def _compute_roc_curve(y_true, y_prob):
    """
    Computes False Positive Rate (FPR), True Positive Rate (TPR), and ROC-AUC
    using pure NumPy trapezoidal integration without scikit-learn.
    """
    y_t = np.asarray(y_true).ravel()
    p = np.asarray(y_prob).ravel()
    n_pos = int(np.sum(y_t == 1))
    n_neg = int(np.sum(y_t == 0))
    if n_pos == 0 or n_neg == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), 0.5

    desc_idx = np.argsort(-p, kind='mergesort')
    y_sorted = y_t[desc_idx]
    p_sorted = p[desc_idx]

    distinct_indices = np.where(np.diff(p_sorted))[0]
    threshold_indices = np.r_[distinct_indices, len(y_sorted) - 1]

    tps = np.cumsum(y_sorted == 1)[threshold_indices]
    fps = np.cumsum(y_sorted == 0)[threshold_indices]

    tps = np.r_[0, tps]
    fps = np.r_[0, fps]

    fpr = fps / n_neg
    tpr = tps / n_pos

    auc = float(np.sum(0.5 * (tpr[1:] + tpr[:-1]) * (fpr[1:] - fpr[:-1])))
    return fpr, tpr, auc


def _compute_pr_curve(y_true, y_prob):
    """
    Computes Recall, Precision, and Average Precision (AP)
    using pure NumPy trapezoidal integration without scikit-learn.
    """
    y_t = np.asarray(y_true).ravel()
    p = np.asarray(y_prob).ravel()
    n_pos = int(np.sum(y_t == 1))
    if n_pos == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 0.0]), 0.0

    desc_idx = np.argsort(-p, kind='mergesort')
    y_sorted = y_t[desc_idx]
    p_sorted = p[desc_idx]

    distinct_indices = np.where(np.diff(p_sorted))[0]
    threshold_indices = np.r_[distinct_indices, len(y_sorted) - 1]

    tps = np.cumsum(y_sorted == 1)[threshold_indices]
    fps = np.cumsum(y_sorted == 0)[threshold_indices]

    precision = tps / np.maximum(tps + fps, 1)
    recall = tps / n_pos

    # Standard PR curve display: prepend point at recall=0 with precision=1.0
    precision = np.r_[1.0, precision]
    recall = np.r_[0.0, recall]

    recall_diff = np.diff(recall)
    ap = float(np.sum(precision[1:] * recall_diff))
    return recall, precision, ap


# ==============================================================================
# Part B: Deep Learning Diagnostics & Model Interpretability (Milestone 4)
# ==============================================================================

def plot_training_curves_grid(models_dict, save_path=None):
    """
    Renders a multi-panel grid of Training vs. Validation Loss curves across epochs
    for multiple market regime classifiers (e.g. LogisticRegression, MLP, LSTM, GRU, CNN1D, Transformer).
    Identifies and marks the early-stopping checkpoint (minimum validation loss) with a dashed line
    and star marker to diagnose overfitting and convergence dynamics.

    Parameters:
        models_dict (dict): Dictionary mapping model name (str) to either:
            - A fitted model instance (DeepRegimeClassifier or LogisticRegression)
            - A history dictionary containing 'train'/'train_loss' and optionally 'val'/'val_loss'
            - A list or 1D array of loss values (treated as training loss)
        save_path (str, optional): Destination file path to save the generated image.

    Returns:
        matplotlib.figure.Figure: The generated figure object.
    """
    if not models_dict:
        raise ValueError("models_dict cannot be empty.")

    n_models = len(models_dict)
    n_cols = 3 if n_models >= 5 else (2 if n_models >= 2 else 1)
    n_rows = int(np.ceil(n_models / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.8 * n_cols, 3.6 * n_rows))
    axes_flat = np.atleast_1d(axes).ravel()

    for idx, (name, model_obj) in enumerate(models_dict.items()):
        ax = axes_flat[idx]
        train_loss = []
        val_loss = []

        if hasattr(model_obj, 'train_loss_history') and model_obj.train_loss_history:
            train_loss = list(model_obj.train_loss_history)
        elif hasattr(model_obj, 'cost_history') and model_obj.cost_history:
            train_loss = list(model_obj.cost_history)

        if hasattr(model_obj, 'val_loss_history') and model_obj.val_loss_history:
            val_loss = list(model_obj.val_loss_history)

        if isinstance(model_obj, dict):
            for k in ['train', 'train_loss', 'train_loss_history', 'cost_history', 'loss']:
                if k in model_obj and model_obj[k] is not None:
                    train_loss = list(model_obj[k])
                    break
            for k in ['val', 'val_loss', 'val_loss_history', 'val_cost_history']:
                if k in model_obj and model_obj[k] is not None:
                    val_loss = list(model_obj[k])
                    break
        elif isinstance(model_obj, (list, tuple, np.ndarray)) and not train_loss:
            train_loss = list(model_obj)

        if train_loss:
            epochs = np.arange(1, len(train_loss) + 1)
            ax.plot(epochs, train_loss, color='#1f77b4', lw=1.8, label=f'Train (final: {train_loss[-1]:.4f})')

        if val_loss and len(val_loss) > 0:
            val_epochs = np.arange(1, len(val_loss) + 1)
            ax.plot(val_epochs, val_loss, color='#ff7f0e', lw=1.8, label=f'Val (final: {val_loss[-1]:.4f})')

            best_idx = int(np.argmin(val_loss))
            best_epoch = best_idx + 1
            best_val = val_loss[best_idx]
            ax.axvline(best_epoch, color='#d62728', linestyle='--', lw=1.2, alpha=0.85,
                       label=f'Best Val: {best_val:.4f} (ep {best_epoch})')
            ax.scatter([best_epoch], [best_val], color='#d62728', marker='*', s=140, zorder=5, edgecolor='black', lw=0.6)

        ax.set_title(name, fontsize=11, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=9.5)
        ax.set_ylabel('Binary Cross-Entropy Loss', fontsize=9.5)
        ax.legend(loc='upper right', fontsize=8, framealpha=0.8)
        ax.grid(True, linestyle='--', alpha=0.35)

    for j in range(n_models, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Training & Validation Loss Convergence Across Models", fontsize=13, fontweight='bold', y=0.995)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight', dpi=150)

    return fig


def plot_confusion_matrix_grid(eval_results_dict, save_path=None):
    """
    Renders a multi-panel grid of 2x2 confusion matrices for all evaluated models.
    Each cell is annotated with raw counts, row-normalized percentages, and cell labels (TN, FP, FN, TP)
    alongside overall Accuracy and F1-Score.

    Parameters:
        eval_results_dict (dict): Dictionary mapping model name (str) to either:
            - An evaluation dictionary containing 'Confusion Matrix' (and optionally 'Accuracy', 'F1-Score')
            - A 2x2 numpy array or list [[TN, FP], [FN, TP]]
            - A tuple of (y_true, y_pred)
        save_path (str, optional): Destination file path to save the generated image.

    Returns:
        matplotlib.figure.Figure: The generated figure object.
    """
    if not eval_results_dict:
        raise ValueError("eval_results_dict cannot be empty.")

    n_models = len(eval_results_dict)
    n_cols = 3 if n_models >= 5 else (2 if n_models >= 2 else 1)
    n_rows = int(np.ceil(n_models / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4.0 * n_rows))
    axes_flat = np.atleast_1d(axes).ravel()

    for idx, (name, eval_obj) in enumerate(eval_results_dict.items()):
        ax = axes_flat[idx]
        cm = None
        acc = None
        f1 = None

        if isinstance(eval_obj, dict):
            for k in ["Confusion Matrix", "confusion_matrix", "cm", "CM"]:
                if k in eval_obj and eval_obj[k] is not None:
                    cm = np.asarray(eval_obj[k], dtype=int)
                    break
            acc = eval_obj.get("Accuracy", None)
            f1 = eval_obj.get("F1-Score", None)
        elif isinstance(eval_obj, (tuple, list)) and len(eval_obj) == 2 and not isinstance(eval_obj[0], (list, np.ndarray, tuple)):
            y_t = np.asarray(eval_obj[0]).ravel()
            y_p = np.asarray(eval_obj[1]).ravel()
            tp = int(np.sum((y_t == 1) & (y_p == 1)))
            fp = int(np.sum((y_t == 0) & (y_p == 1)))
            fn = int(np.sum((y_t == 1) & (y_p == 0)))
            tn = int(np.sum((y_t == 0) & (y_p == 0)))
            cm = np.array([[tn, fp], [fn, tp]])
            acc = float(np.mean(y_t == y_p))
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        else:
            cm = np.asarray(eval_obj, dtype=int)

        if cm is None or cm.shape != (2, 2):
            raise ValueError(f"Confusion matrix for '{name}' must have shape (2, 2).")

        if acc is None:
            acc = float(cm[0, 0] + cm[1, 1]) / max(float(cm.sum()), 1.0)
        if f1 is None:
            tp = cm[1, 1]
            fp = cm[0, 1]
            fn = cm[1, 0]
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            f1 = (2 * prec * rec) / max(prec + rec, 1e-12)

        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.zeros_like(cm, dtype=float)
        np.divide(cm.astype(float), np.maximum(row_sums, 1), out=cm_norm, where=(row_sums > 0))

        im = ax.imshow(cm_norm, cmap='Blues', vmin=0.0, vmax=1.0)

        cell_labels = [["TN", "FP"], ["FN", "TP"]]
        for r in range(2):
            for c in range(2):
                count = cm[r, c]
                pct = cm_norm[r, c] * 100.0
                tag = cell_labels[r][c]
                txt = f"{tag}: {count}\n({pct:.1f}%)"
                text_color = "white" if cm_norm[r, c] > 0.55 else "black"
                ax.text(c, r, txt, ha="center", va="center", color=text_color, fontsize=10, fontweight='bold')

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Pred Bearish (0)', 'Pred Bullish (1)'], fontsize=9)
        ax.set_yticklabels(['True Bearish (0)', 'True Bullish (1)'], fontsize=9)
        ax.set_title(f"{name}\nAcc: {acc:.1%} | F1: {f1:.3f}", fontsize=10.5, fontweight='bold', pad=8)

    for j in range(n_models, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Out-of-Sample Confusion Matrix Grid Across Models", fontsize=13, fontweight='bold', y=0.995)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight', dpi=150)

    return fig


def plot_roc_pr_curves(models_probs_dict, y_true, save_path=None):
    """
    Renders a 2-panel figure comparing multi-model diagnostic trade-offs:
    - Left: ROC curves (FPR vs TPR with annotated AUC) across all models + random guess baseline (0.5).
    - Right: Precision-Recall curves (Recall vs Precision with annotated AP) + baseline prevalence line.
    Pure NumPy and Matplotlib calculations without scikit-learn.

    Parameters:
        models_probs_dict (dict): Dictionary mapping model name (str) to predicted probabilities (1D or 2D array).
        y_true (array-like): Ground truth binary regime labels.
        save_path (str, optional): Destination file path to save the generated image.

    Returns:
        matplotlib.figure.Figure: The generated figure object.
    """
    if not models_probs_dict:
        raise ValueError("models_probs_dict cannot be empty.")

    y_t = np.asarray(y_true).ravel()
    prevalence = float(np.mean(y_t == 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left panel baseline (ROC)
    ax1.plot([0, 1], [0, 1], 'k--', lw=1.2, alpha=0.6, label='Random Guess (AUC = 0.500)')

    # Right panel baseline (PR)
    ax2.axhline(prevalence, color='navy', linestyle='--', lw=1.2, alpha=0.6,
                label=f'Prevalence Baseline ({prevalence:.2f})')

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

    for idx, (name, probs) in enumerate(models_probs_dict.items()):
        color = colors[idx % len(colors)]
        p_arr = np.asarray(probs).ravel()

        fpr, tpr, auc = _compute_roc_curve(y_t, p_arr)
        ax1.plot(fpr, tpr, color=color, lw=2.0, label=f"{name} (AUC = {auc:.3f})")

        rec, prec, ap = _compute_pr_curve(y_t, p_arr)
        ax2.plot(rec, prec, color=color, lw=2.0, label=f"{name} (AP = {ap:.3f})")

    # Left Panel Styling
    ax1.set_xlim([-0.02, 1.02])
    ax1.set_ylim([-0.02, 1.05])
    ax1.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=10.5, fontweight='bold')
    ax1.set_ylabel('True Positive Rate (Recall / Sensitivity)', fontsize=10.5, fontweight='bold')
    ax1.set_title('Receiver Operating Characteristic (ROC)', fontsize=12, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=9, framealpha=0.85)
    ax1.grid(True, linestyle='--', alpha=0.35)

    # Right Panel Styling
    ax2.set_xlim([-0.02, 1.02])
    ax2.set_ylim([-0.02, 1.05])
    ax2.set_xlabel('Recall (Sensitivity)', fontsize=10.5, fontweight='bold')
    ax2.set_ylabel('Precision (Positive Predictive Value)', fontsize=10.5, fontweight='bold')
    ax2.set_title('Precision-Recall (PR) Curves', fontsize=12, fontweight='bold')
    ax2.legend(loc='lower left', fontsize=9, framealpha=0.85)
    ax2.grid(True, linestyle='--', alpha=0.35)

    fig.suptitle("Out-of-Sample ROC & Precision-Recall Diagnostic Curves", fontsize=13, fontweight='bold', y=0.98)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight', dpi=150)

    return fig


def _eval_model_metric(model, X, y, metric='f1'):
    """Evaluates a single classification metric for feature importance."""
    y_t = np.asarray(y).ravel()

    if metric == 'prob_shift':
        return model.predict_proba(X).ravel()

    if metric == 'roc_auc':
        p = model.predict_proba(X).ravel()
        return _compute_roc_curve(y_t, p)[2]

    # Binary prediction metrics
    if hasattr(model, 'predict'):
        y_p = model.predict(X).ravel()
    else:
        p = model.predict_proba(X).ravel()
        y_p = (p >= 0.5).astype(int)

    if metric == 'accuracy':
        return float(np.mean(y_t == y_p))

    # default: f1
    tp = int(np.sum((y_t == 1) & (y_p == 1)))
    fp = int(np.sum((y_t == 0) & (y_p == 1)))
    fn = int(np.sum((y_t == 1) & (y_p == 0)))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return float((2.0 * prec * rec) / (prec + rec)) if (prec + rec) > 0 else 0.0


def _resolve_model_inputs(model, X_test):
    """
    Resolves matching 2D tabular or 3D sequence array for a model.
    Handles dict, tuple, or direct ndarray representations.
    """
    is_seq = False
    if hasattr(model, '_is_sequential'):
        is_seq = model._is_sequential()
    elif hasattr(model, 'model_name'):
        is_seq = str(model.model_name).lower() in {'lstm', 'gru', 'recurrent', 'cnn1d', 'cnn', 'conv1d', 'transformer', 'attention'}

    if isinstance(X_test, dict):
        if is_seq:
            for k in ['3d', 'seq', 'sequences', 'windowed']:
                if k in X_test:
                    return X_test[k]
        else:
            for k in ['2d', 'tab', 'tabular']:
                if k in X_test:
                    return X_test[k]
        return next(iter(X_test.values()))

    if isinstance(X_test, (list, tuple)) and len(X_test) == 2:
        x0, x1 = np.asarray(X_test[0]), np.asarray(X_test[1])
        if is_seq:
            return x0 if x0.ndim == 3 else x1
        else:
            return x0 if x0.ndim == 2 else x1

    arr = np.asarray(X_test)
    if is_seq:
        if arr.ndim == 2:
            seq_len = getattr(model, 'seq_len', 20)
            return np.repeat(arr[:, np.newaxis, :], seq_len, axis=1)
        return arr
    else:
        if arr.ndim == 3:
            return arr[:, -1, :]
        return arr


def plot_feature_importance_heatmap(models_dict, X_test, y_test, feature_names,
                                    metric='f1', n_repeats=5, save_path=None, random_state=42):
    """
    Computes Permutation Feature Importance for each model across all features and
    renders a (Models x Features) annotated heatmap.
    Robustly handles both 2D tabular and 3D sequence tensors without crashing.

    Parameters:
        models_dict (dict): Dictionary mapping model name (str) to fitted classifier instances.
        X_test: Test features. Can be:
            - Dict: {'2d': X_tabular, '3d': X_sequential}
            - Tuple: (X_tabular, X_sequential)
            - 3D ndarray (N, L, F): sequential models receive full tensor; tabular receive last step X[:, -1, :]
            - 2D ndarray (N, F): tabular models receive full tensor; sequential models replicate along time
        y_test: Ground truth binary regime labels.
        feature_names (list of str): Names corresponding to feature columns.
        metric (str): 'f1', 'accuracy', 'roc_auc', or 'prob_shift'.
        n_repeats (int): Number of permutation shuffles per feature.
        save_path (str, optional): Destination file path to save the generated image.
        random_state (int): Seed for reproducible permutations.

    Returns:
        matplotlib.figure.Figure: The generated figure object.
    """
    rng = np.random.default_rng(random_state)
    n_models = len(models_dict)
    n_feats = len(feature_names)
    model_names = list(models_dict.keys())

    importance_matrix = np.zeros((n_models, n_feats), dtype=float)

    for m_idx, (m_name, model) in enumerate(models_dict.items()):
        X_m = _resolve_model_inputs(model, X_test)
        n_samples = X_m.shape[0]

        if metric == 'prob_shift':
            base_p = _eval_model_metric(model, X_m, y_test, metric='prob_shift')
            for f_idx in range(n_feats):
                shifts = []
                for _ in range(n_repeats):
                    X_perm = X_m.copy()
                    p_idx = rng.permutation(n_samples)
                    if X_m.ndim == 2:
                        X_perm[:, f_idx] = X_m[p_idx, f_idx]
                    elif X_m.ndim == 3:
                        X_perm[:, :, f_idx] = X_m[p_idx, :, f_idx]
                    p_perm = _eval_model_metric(model, X_perm, y_test, metric='prob_shift')
                    shifts.append(float(np.mean(np.abs(p_perm - base_p))))
                importance_matrix[m_idx, f_idx] = float(np.mean(shifts))
        else:
            base_score = _eval_model_metric(model, X_m, y_test, metric=metric)
            for f_idx in range(n_feats):
                drops = []
                for _ in range(n_repeats):
                    X_perm = X_m.copy()
                    p_idx = rng.permutation(n_samples)
                    if X_m.ndim == 2:
                        X_perm[:, f_idx] = X_m[p_idx, f_idx]
                    elif X_m.ndim == 3:
                        X_perm[:, :, f_idx] = X_m[p_idx, :, f_idx]
                    perm_score = _eval_model_metric(model, X_perm, y_test, metric=metric)
                    drops.append(base_score - perm_score)
                importance_matrix[m_idx, f_idx] = float(np.mean(drops))

    fig, ax = plt.subplots(figsize=(max(11, n_feats * 0.8), max(4, n_models * 0.75 + 1.5)))

    vmax = max(float(np.max(importance_matrix)), 0.05)
    vmin = min(float(np.min(importance_matrix)), 0.0)
    cmap = 'coolwarm' if vmin < -0.01 else 'YlOrRd'

    im = ax.imshow(importance_matrix, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    metric_label = "Probability Shift (|Δp|)" if metric == 'prob_shift' else f"Degradation in {metric.upper()}"
    cbar.set_label(f"Permutation Importance ({metric_label})", fontsize=10)

    ax.set_xticks(range(n_feats))
    ax.set_yticks(range(n_models))
    ax.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=9.5)
    ax.set_yticklabels(model_names, fontsize=10, fontweight='bold')

    for i in range(n_models):
        for j in range(n_feats):
            val = importance_matrix[i, j]
            txt = f"{val:+.3f}" if metric != 'prob_shift' else f"{val:.3f}"
            rel_val = (val - vmin) / max(vmax - vmin, 1e-12)
            color = "white" if (rel_val > 0.65 or (rel_val < 0.25 and cmap == 'coolwarm')) else "black"
            ax.text(j, i, txt, ha="center", va="center", color=color, fontsize=8)

    ax.set_title(f"Permutation Feature Importance Across Models ({metric_label}, {n_repeats} repeats)",
                 fontsize=12, fontweight='bold', pad=12)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight', dpi=150)

    return fig


def plot_transformer_attention_heatmap(transformer_model, X_sample=None, seq_len=20, save_path=None):
    """
    Extracts the (L x L) self-attention matrix from TransformerRegimeNet for a given sample
    or averaged across samples, and renders an annotated heatmap with lookback time ticks (t-19 to t).
    Highlights the bottom row ('t') which drives the market regime classification output.

    Parameters:
        transformer_model: DeepRegimeClassifier(model_name='transformer'), TransformerRegimeNet,
                           or a pre-computed attention matrix (2D/3D ndarray or Tensor).
        X_sample (array-like, optional): 2D (L, F) or 3D (B, L, F) input sequences to feed forward.
        seq_len (int): Default sequence length (default 20).
        save_path (str, optional): Destination file path to save the generated image.

    Returns:
        matplotlib.figure.Figure: The generated figure object.
    """
    attn_matrix = None

    # Case 1: Pre-computed matrix passed directly
    if isinstance(transformer_model, (np.ndarray, torch.Tensor)):
        arr = transformer_model.detach().cpu().numpy() if isinstance(transformer_model, torch.Tensor) else np.asarray(transformer_model)
        if arr.ndim == 3:
            attn_matrix = np.mean(arr, axis=0)
        elif arr.ndim == 2:
            attn_matrix = arr

    # Case 2: DeepRegimeClassifier with get_attention_weights
    if attn_matrix is None and hasattr(transformer_model, 'get_attention_weights'):
        try:
            attn_matrix = transformer_model.get_attention_weights(X_sample, aggregate='mean')
        except Exception:
            pass

    # Case 3: DeepRegimeClassifier or TransformerRegimeNet instance
    if attn_matrix is None:
        net = getattr(transformer_model, 'model', transformer_model)
        if hasattr(transformer_model, 'device'):
            device = transformer_model.device
        elif hasattr(net, 'parameters'):
            try:
                device = next(net.parameters()).device
            except StopIteration:
                device = torch.device('cpu')
        else:
            device = torch.device('cpu')

        if X_sample is not None and hasattr(net, 'eval'):
            net.eval()
            x_arr = np.asarray(X_sample, dtype=np.float32)
            if x_arr.ndim == 2:
                x_arr = x_arr[np.newaxis, :, :]  # (1, L, F)
            x_tensor = torch.as_tensor(x_arr, dtype=torch.float32, device=device)
            with torch.no_grad():
                net(x_tensor)
            if hasattr(net, 'last_attention_weights') and net.last_attention_weights is not None:
                weights = net.last_attention_weights.detach().cpu().numpy()
                attn_matrix = np.mean(weights, axis=0) if weights.ndim == 3 else weights

        elif hasattr(net, 'last_attention_weights') and net.last_attention_weights is not None:
            weights = net.last_attention_weights.detach().cpu().numpy()
            attn_matrix = np.mean(weights, axis=0) if weights.ndim == 3 else weights

    if attn_matrix is None:
        raise ValueError("Could not extract attention matrix. Provide X_sample or pass an already-evaluated TransformerRegimeNet.")

    L = attn_matrix.shape[0]
    ticks = [f"t-{L - 1 - i}" if i < L - 1 else "t" for i in range(L)]

    fig, ax = plt.subplots(figsize=(max(9, L * 0.45), max(8, L * 0.45)))
    im = ax.imshow(attn_matrix, cmap='viridis', aspect='auto', vmin=0.0, vmax=max(float(np.max(attn_matrix)), 0.1))
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Self-Attention Weight (Softmax Normalized)", fontsize=10)

    ax.set_xticks(range(L))
    ax.set_yticks(range(L))
    ax.set_xticklabels(ticks, rotation=45, ha='right', fontsize=8.5)
    ax.set_yticklabels(ticks, fontsize=8.5)

    # Highlight final row (t) which forms regime prediction
    ax.axhline(L - 1 - 0.5, color='crimson', lw=2.0, alpha=0.85)

    # Annotate cells if sequence length is manageable
    if L <= 20:
        max_val = np.max(attn_matrix)
        for i in range(L):
            for j in range(L):
                val = attn_matrix[i, j]
                color = "white" if val > (max_val * 0.55) else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=6.5)

    ax.set_xlabel("Key Lookback Step (Historical Days Attended To)", fontsize=10.5, fontweight='bold', labelpad=8)
    ax.set_ylabel("Query Lookback Step (Output Position in Sequence)", fontsize=10.5, fontweight='bold', labelpad=8)
    ax.set_title(f"Transformer Self-Attention Heatmap (L = {L})\nBottom Row ('t') Drives Market Regime Prediction",
                 fontsize=12, fontweight='bold', pad=12)

    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight', dpi=150)

    return fig


# ==============================================================================
# Part C: Multi-Model Strategy Comparison & Portfolio Analytics (Milestone 5)
# ==============================================================================

def plot_regime_agreement_barcode(test_dates, close_prices, predictions_dict, y_true=None, save_path=None):
    """
    Renders a 2-panel synchronized out-of-sample chart:
    - Top Panel: Out-of-sample asset Close price curve with ground-truth Bullish/Bearish shading (if y_true is provided).
    - Bottom Panel: A horizontal Regime Barcode Heatmap where each row represents a model's binary regime
      predictions (Green = 1 Bullish, Crimson = 0 Bearish) alongside Ground Truth and Model Consensus % annotations.

    Parameters:
        test_dates (array-like or DatetimeIndex): Chronological out-of-sample dates (length N).
        close_prices (array-like or Series): Out-of-sample asset closing prices (length N).
        predictions_dict (dict): Dictionary mapping model name (str) to binary regime predictions (length N).
        y_true (array-like, optional): Ground truth binary regime labels (length N).
        save_path (str, optional): Destination file path to save the generated image.

    Returns:
        matplotlib.figure.Figure: The generated figure object.
    """
    if not predictions_dict:
        raise ValueError("predictions_dict cannot be empty.")

    close_raw = np.asarray(close_prices, dtype=float).ravel()
    n_days = len(close_raw)
    if n_days == 0:
        raise ValueError("close_prices cannot be empty.")

    # Sanitize NaN/Inf in close_prices via linear interpolation / ffill / bfill fallback
    if not np.all(np.isfinite(close_raw)):
        s_close = pd.Series(close_raw).replace([np.inf, -np.inf], np.nan).interpolate(method='linear').ffill().bfill().fillna(100.0)
        close_arr = s_close.to_numpy(dtype=float)
    else:
        close_arr = close_raw

    # Normalize test_dates without misinterpreting integer indices as 1970 nanoseconds or failing on non-date strings
    if len(test_dates) != n_days:
        raise ValueError(f"Length mismatch: test_dates ({len(test_dates)}) vs close_prices ({n_days}).")

    if isinstance(test_dates, pd.DatetimeIndex):
        dates_list = [d.strftime('%Y-%m-%d') for d in test_dates]
    elif isinstance(test_dates, (pd.RangeIndex, range)) or (
         hasattr(test_dates, 'dtype') and np.issubdtype(np.asarray(test_dates).dtype, np.integer)
    ) or (isinstance(test_dates, (list, tuple)) and all(isinstance(x, (int, np.integer)) for x in test_dates)):
        dates_arr = np.asarray(test_dates).ravel()
        dates_list = [str(x) for x in dates_arr]
    else:
        raw_seq = test_dates.to_numpy() if isinstance(test_dates, (pd.Series, pd.Index)) else np.asarray(test_dates)
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                parsed_dt = pd.to_datetime(raw_seq)
            dates_list = [pd.Timestamp(d).strftime('%Y-%m-%d') if pd.notna(d) else str(raw_seq[i]) for i, d in enumerate(parsed_dt)]
        except Exception:
            dates_list = [str(x) for x in raw_seq]

    row_labels = []
    barcode_rows = []
    y_true_arr = None

    if y_true is not None:
        y_raw = np.nan_to_num(np.asarray(y_true, dtype=float).ravel(), nan=0.0, posinf=1.0, neginf=0.0)
        y_true_arr = (y_raw >= 0.5).astype(int)
        if len(y_true_arr) != n_days:
            if len(y_true_arr) > n_days:
                y_true_arr = y_true_arr[-n_days:]
            else:
                pad = np.zeros(n_days - len(y_true_arr), dtype=int)
                y_true_arr = np.concatenate([pad, y_true_arr])
        bull_pct = float(np.mean(y_true_arr == 1)) * 100.0
        row_labels.append(f"Ground Truth ({bull_pct:.0f}% Bull)")
        barcode_rows.append(y_true_arr)

    for model_name, preds in predictions_dict.items():
        p_raw = np.nan_to_num(np.asarray(preds, dtype=float).ravel(), nan=0.0, posinf=1.0, neginf=0.0)
        p_arr = (p_raw >= 0.5).astype(int)
        if len(p_arr) != n_days:
            if len(p_arr) > n_days:
                p_arr = p_arr[-n_days:]
            else:
                pad = np.zeros(n_days - len(p_arr), dtype=int)
                p_arr = np.concatenate([pad, p_arr])
        bull_pct = float(np.mean(p_arr == 1)) * 100.0
        if y_true_arr is not None:
            agree_pct = float(np.mean(p_arr == y_true_arr)) * 100.0
            row_labels.append(f"{model_name} (Acc: {agree_pct:.1f}% | {bull_pct:.0f}% Bull)")
        else:
            row_labels.append(f"{model_name} ({bull_pct:.0f}% Bull)")
        barcode_rows.append(p_arr)

    barcode_matrix = np.vstack(barcode_rows)  # shape: (n_rows, n_days)
    n_rows = barcode_matrix.shape[0]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(15, max(7.5, 4.0 + 0.65 * n_rows)),
        gridspec_kw={'height_ratios': [2.2, max(1.5, 0.45 * n_rows)]}
    )

    x_idx = np.arange(n_days)
    c_min = float(np.min(close_arr))
    c_max = float(np.max(close_arr))
    if abs(c_max - c_min) < 1e-9:
        pad_y = max(abs(c_max) * 0.02, 1.0)
        y_low, y_high = c_min - pad_y, c_max + pad_y
    else:
        pad_y = (c_max - c_min) * 0.06
        y_low, y_high = c_min - pad_y, c_max + pad_y

    # Top Panel: Out-of-Sample Close Price Curve
    ax1.plot(x_idx, close_arr, color='#1f77b4', lw=2.0, label='Out-of-Sample Close Price')
    if y_true_arr is not None:
        bull_mask = (y_true_arr == 1)
        ax1.fill_between(x_idx, y_low, y_high,
                         where=bull_mask, color='#2ca02c', alpha=0.14, step='mid', label='True Bullish Regime (1)')
        ax1.fill_between(x_idx, y_low, y_high,
                         where=~bull_mask, color='#d62728', alpha=0.14, step='mid', label='True Bearish Regime (0)')

    ax1.set_xlim(-0.5, max(n_days - 0.5, 0.5))
    ax1.set_ylim(y_low, y_high)
    ax1.set_ylabel("Asset Close Price ($)", fontsize=11, fontweight='bold')
    ax1.set_title("Out-of-Sample Price Trajectory & Multi-Model Regime Agreement Barcode",
                  fontsize=13, fontweight='bold', pad=10)
    ax1.legend(loc='upper left', fontsize=9.5, framealpha=0.9)
    ax1.grid(True, linestyle='--', alpha=0.35)

    # Format x-ticks with calendar dates
    n_ticks = min(10, n_days)
    tick_positions = np.unique(np.linspace(0, n_days - 1, n_ticks, dtype=int))
    tick_labels = [dates_list[int(pos)] for pos in tick_positions]
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels([])

    # Bottom Panel: Horizontal Regime Barcode Heatmap
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    cmap_barcode = ListedColormap(['#d62728', '#2ca02c'])
    ax2.imshow(barcode_matrix, cmap=cmap_barcode, aspect='auto', vmin=0, vmax=1, interpolation='nearest')

    # Draw horizontal separators between model strips
    for r in range(n_rows - 1):
        ax2.axhline(r + 0.5, color='white', lw=1.8)
    if y_true_arr is not None and n_rows > 1:
        ax2.axhline(0.5, color='black', lw=2.2, linestyle='-')

    ax2.set_yticks(range(n_rows))
    ax2.set_yticklabels(row_labels, fontsize=9.5, fontweight='bold')
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(tick_labels, rotation=30, ha='right', fontsize=9.5)
    ax2.set_xlabel("Out-of-Sample Trading Date", fontsize=11, fontweight='bold', labelpad=8)
    ax2.set_title("Daily Predicted Market Regime Strip (Green = Bullish [1], Red = Bearish [0])",
                  fontsize=11.5, fontweight='bold', pad=8)

    legend_patches = [
        Patch(facecolor='#2ca02c', edgecolor='black', label='Bullish Regime (1 — Active Filter)'),
        Patch(facecolor='#d62728', edgecolor='black', label='Bearish Regime (0 — Cash Protection)')
    ]
    ax2.legend(handles=legend_patches, loc='upper right', fontsize=9, framealpha=0.9)
    ax2.grid(False)

    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight', dpi=150)

    return fig


def plot_risk_return_frontier(metrics_dict, save_path=None):
    """
    Renders a Risk-Return Frontier scatter plot comparing trading strategies and ML hybrid filters:
    - X-Axis: Maximum Drawdown Magnitude (%) [lower left is lower drawdown risk, or plotted as |Max Drawdown (%)|]
    - Y-Axis: Total Cumulative Return (%)
    - Bubble Size & Color: Scaled and colored by annualized Sharpe Ratio, with annotated strategy labels.

    Parameters:
        metrics_dict (dict): Dictionary mapping strategy name (str) to a metrics dictionary containing:
            - 'Total returns' or 'Total Return (%)'
            - 'Max drawdown' or 'Max Drawdown (%)'
            - 'Sharpe Ratio'
        save_path (str, optional): Destination file path to save the generated image.

    Returns:
        matplotlib.figure.Figure: The generated figure object.
    """
    if not metrics_dict:
        raise ValueError("metrics_dict cannot be empty.")

    names = []
    returns_pct = []
    drawdowns_pct = []
    sharpes = []

    for strat_name, m in metrics_dict.items():
        if not isinstance(m, dict):
            raise ValueError(f"Metrics for '{strat_name}' must be a dictionary.")

        # Extract Total Return (%)
        if "Total Return (%)" in m:
            ret_val = float(m["Total Return (%)"])
        elif "Total returns" in m:
            raw_ret = float(m["Total returns"])
            ret_val = raw_ret * 100.0 if np.isfinite(raw_ret) and abs(raw_ret) <= 5.0 else raw_ret
        elif "total_return" in m:
            raw_ret = float(m["total_return"])
            ret_val = raw_ret * 100.0 if np.isfinite(raw_ret) and abs(raw_ret) <= 5.0 else raw_ret
        else:
            raise KeyError(f"Missing 'Total returns' or 'Total Return (%)' in metrics for '{strat_name}'.")

        # Extract Max Drawdown (%) as positive magnitude for clear risk axis
        if "Max Drawdown (%)" in m:
            dd_val = abs(float(m["Max Drawdown (%)"]))
        elif "Max drawdown" in m:
            raw_dd = abs(float(m["Max drawdown"]))
            dd_val = raw_dd * 100.0 if np.isfinite(raw_dd) and raw_dd <= 1.0 else raw_dd
        elif "max_drawdown" in m:
            raw_dd = abs(float(m["max_drawdown"]))
            dd_val = raw_dd * 100.0 if np.isfinite(raw_dd) and raw_dd <= 1.0 else raw_dd
        else:
            raise KeyError(f"Missing 'Max drawdown' or 'Max Drawdown (%)' in metrics for '{strat_name}'.")

        # Extract Sharpe Ratio
        if "Sharpe Ratio" in m:
            sr_val = float(m["Sharpe Ratio"])
        elif "sharpe_ratio" in m:
            sr_val = float(m["sharpe_ratio"])
        elif "Sharpe" in m:
            sr_val = float(m["Sharpe"])
        else:
            sr_val = 0.0

        if not np.isfinite(ret_val):
            ret_val = 0.0
        if not np.isfinite(dd_val):
            dd_val = 0.0
        if not np.isfinite(sr_val):
            sr_val = 0.0

        names.append(str(strat_name))
        returns_pct.append(ret_val)
        drawdowns_pct.append(dd_val)
        sharpes.append(sr_val)

    returns_arr = np.asarray(returns_pct, dtype=float)
    drawdowns_arr = np.asarray(drawdowns_pct, dtype=float)
    sharpes_arr = np.asarray(sharpes, dtype=float)

    # Bubble sizes proportional to Sharpe Ratio (clamped to readable range)
    min_sr = np.min(sharpes_arr)
    max_sr = np.max(sharpes_arr)
    if max_sr > min_sr + 1e-6:
        norm_sr = (sharpes_arr - min_sr) / (max_sr - min_sr)
    else:
        norm_sr = np.ones_like(sharpes_arr) * 0.5
    bubble_sizes = 220.0 + 480.0 * norm_sr

    fig, ax = plt.subplots(figsize=(11.5, 7.5))

    scatter = ax.scatter(
        drawdowns_arr, returns_arr,
        s=bubble_sizes, c=sharpes_arr,
        cmap='viridis', alpha=0.88, edgecolors='black', linewidth=1.5, zorder=4
    )
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Annualized Sharpe Ratio", fontsize=10.5, fontweight='bold')

    # Annotate each strategy point with smart offsets to avoid overlapping text
    x_span = max(float(np.max(drawdowns_arr) - np.min(drawdowns_arr)), 1.0)
    y_span = max(float(np.max(returns_arr) - np.min(returns_arr)), 1.0)
    x_left = min(0.0, float(np.min(drawdowns_arr)) - 0.18 * x_span) if np.min(drawdowns_arr) < 0.2 * x_span else max(0.0, float(np.min(drawdowns_arr)) - 0.18 * x_span)
    ax.set_xlim(x_left, float(np.max(drawdowns_arr)) + 0.25 * x_span)
    ax.set_ylim(float(np.min(returns_arr)) - 0.18 * y_span, float(np.max(returns_arr)) + 0.22 * y_span)

    offsets = [
        (10, 10), (10, -14), (-12, 12), (12, 6), (-12, -14),
        (10, 14), (-14, 8), (12, -10), (8, 16)
    ]
    for idx, name in enumerate(names):
        dx, dy = offsets[idx % len(offsets)]
        ha = 'left' if dx >= 0 else 'right'
        label_text = f"{name}\n(Ret: {returns_arr[idx]:.1f}%, DD: -{drawdowns_arr[idx]:.1f}%, SR: {sharpes_arr[idx]:.2f})"
        ax.annotate(
            label_text,
            xy=(drawdowns_arr[idx], returns_arr[idx]),
            xytext=(dx, dy),
            textcoords='offset points',
            ha=ha, va='center',
            fontsize=8.5, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='gray', alpha=0.85),
            arrowprops=dict(arrowstyle='-', color='gray', lw=0.8, alpha=0.7),
            zorder=5
        )

    # Highlight ideal Northwest quadrant direction (Higher Return, Lower Drawdown)
    ax.set_xlabel("Maximum Drawdown Magnitude (%) [Lower is Safer → Left]", fontsize=11, fontweight='bold', labelpad=8)
    ax.set_ylabel("Out-of-Sample Total Return (%) [Higher is Better → Up]", fontsize=11, fontweight='bold', labelpad=8)
    ax.set_title("Portfolio Risk-Return Frontier Across Technical & ML Hybrid Strategies\n(Bubble Size & Color Scaled by Annualized Sharpe Ratio)",
                 fontsize=12.5, fontweight='bold', pad=12)
    ax.grid(True, linestyle='--', alpha=0.4, zorder=1)

    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight', dpi=150)

    return fig

