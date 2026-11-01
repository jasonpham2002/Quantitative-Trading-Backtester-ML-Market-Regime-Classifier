import copy
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def _to_numpy(arr):
    """Safely converts PyTorch tensors, Pandas objects, or lists to a NumPy ndarray."""
    if arr is None:
        return None
    if isinstance(arr, torch.Tensor) or hasattr(arr, 'detach'):
        return arr.detach().cpu().numpy()
    if isinstance(arr, (pd.DataFrame, pd.Series)) or hasattr(arr, 'values'):
        return arr.values if hasattr(arr, 'values') else np.asarray(arr)
    return np.asarray(arr)


class RegimeDataset(Dataset):
    """
    PyTorch Dataset wrapping 2D tabular feature matrices (N, F) or 3D sequence tensors (N, L, F)
    alongside binary regime target labels (N, 1).
    Seamlessly handles NumPy arrays, Pandas DataFrames/Series, and Torch Tensors,
    converting them to float32 tensors.
    """
    def __init__(self, X, y=None):
        if isinstance(X, torch.Tensor):
            self.X = X.clone().detach().to(dtype=torch.float32).contiguous()
        elif isinstance(X, (pd.DataFrame, pd.Series)) or hasattr(X, 'values'):
            val = X.values if hasattr(X, 'values') else X
            self.X = torch.as_tensor(np.ascontiguousarray(val, dtype=np.float32), dtype=torch.float32).clone().contiguous()
        else:
            self.X = torch.as_tensor(np.ascontiguousarray(X, dtype=np.float32), dtype=torch.float32).clone().contiguous()

        if y is not None:
            if isinstance(y, torch.Tensor):
                self.y = y.clone().detach().to(dtype=torch.float32).reshape(-1, 1).contiguous()
            elif isinstance(y, (pd.DataFrame, pd.Series)) or hasattr(y, 'values'):
                val = y.values if hasattr(y, 'values') else y
                self.y = torch.as_tensor(np.ascontiguousarray(val, dtype=np.float32).reshape(-1, 1), dtype=torch.float32).clone().contiguous()
            else:
                self.y = torch.as_tensor(np.ascontiguousarray(y, dtype=np.float32).reshape(-1, 1), dtype=torch.float32).clone().contiguous()
            if len(self.X) != len(self.y):
                raise ValueError(
                    f"Features length ({len(self.X)}) and labels length ({len(self.y)}) mismatch"
                )
        else:
            self.y = None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


class MLPRegimeNet(nn.Module):
    """
    Feed-forward neural network for tabular market regime classification.
    Takes 2D feature representations (B, F) and produces scalar logits (B, 1).
    Uses LayerNorm and Dropout for regularization across small financial samples.
    """
    def __init__(self, input_dim=14, hidden_dim=32, num_layers=2, dropout=0.2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_p = dropout

        layers = []
        in_dim = input_dim
        curr_hidden = hidden_dim

        for i in range(num_layers):
            layers.append(nn.Linear(in_dim, curr_hidden))
            layers.append(nn.LayerNorm(curr_hidden))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            in_dim = curr_hidden
            if i < num_layers - 1:
                curr_hidden = max(curr_hidden // 2, 1)

        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class RecurrentRegimeNet(nn.Module):
    """
    Recurrent neural network (LSTM or GRU) for sequential temporal regime classification.
    Processes 3D sliding-window sequences (B, L, F) and extracts the final time-step
    hidden state out[:, -1, :] to output a raw scalar logit (B, 1).
    """
    def __init__(self, input_dim=14, hidden_dim=32, num_layers=1, cell_type='lstm', dropout=0.2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.cell_type = str(cell_type).lower()
        self.dropout_p = dropout

        rnn_dropout = dropout if num_layers > 1 else 0.0
        if self.cell_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=rnn_dropout
            )
        elif self.cell_type == 'gru':
            self.rnn = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=rnn_dropout
            )
        else:
            raise ValueError(f"Unsupported cell_type: '{cell_type}'. Choose 'lstm' or 'gru'.")

        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError(
                f"RecurrentRegimeNet expects 3D sequential input of shape (batch_size, seq_len, input_dim), "
                f"but received tensor with ndim={x.ndim} and shape={tuple(x.shape)}."
            )
        out, _ = self.rnn(x)
        last_step = out[:, -1, :]
        last_step = self.dropout(last_step)
        logits = self.fc(last_step)
        return logits


class Conv1DRegimeNet(nn.Module):
    """
    1D Temporal Convolutional network for multi-day chart motif extraction.
    Takes 3D sequence tensors (B, L, F), permutes to channels-first (B, F, L),
    applies 1D convolutions, adaptive average pooling, and outputs logits (B, 1).
    """
    def __init__(self, input_dim=14, filters=32, kernel_size=3, dropout=0.2):
        super().__init__()
        self.input_dim = input_dim
        self.filters = filters

        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(input_dim, filters, kernel_size=kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(filters)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv1d(filters, filters * 2, kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(filters * 2)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(filters * 2, 1)

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError(
                f"Conv1DRegimeNet expects 3D sequential input of shape (batch_size, seq_len, input_dim), "
                f"but received tensor with ndim={x.ndim} and shape={tuple(x.shape)}."
            )
        # Permute from (B, L, F) to (B, F, L) for Conv1d
        x = x.permute(0, 2, 1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence transformer."""
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TransformerRegimeNet(nn.Module):
    """
    Self-Attention Transformer encoder for regime shift detection across lookback sequences.
    Projects features into embedding space, adds sinusoidal positional encoding,
    computes multi-head self-attention (saving attention weights for visualization),
    and projects the final sequence step into a scalar logit (B, 1).
    """
    def __init__(self, input_dim=14, d_model=32, nhead=4, dim_feedforward=64, num_layers=1, dropout=0.2):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.nhead = nhead

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = SinusoidalPositionalEncoding(d_model=d_model, max_len=200)

        self.mha = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            batch_first=True,
            dropout=dropout
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, 1)

        self.last_attention_weights = None

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError(
                f"TransformerRegimeNet expects 3D sequential input of shape (batch_size, seq_len, input_dim), "
                f"but received tensor with ndim={x.ndim} and shape={tuple(x.shape)}."
            )
        x_proj = self.input_proj(x)
        x_emb = self.pos_encoder(x_proj)

        attn_out, attn_weights = self.mha(x_emb, x_emb, x_emb, need_weights=True, average_attn_weights=True)
        self.last_attention_weights = attn_weights.detach()

        x = self.norm1(x_emb + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))

        last_step = x[:, -1, :]
        logits = self.fc(self.dropout(last_step))
        return logits


class DeepRegimeClassifier:
    """
    Unified high-level trainer and predictor for PyTorch market regime classifiers.
    Guarantees 100% API parity with LogisticRegression:
      - fit(X_train, y_train, X_val=None, y_val=None)
      - predict_proba(X) -> (N, 1) float probabilities in [0.0, 1.0]
      - predict(X, threshold=0.22, smooth_window=3) -> (N, 1) binary integers
      - evaluate(y_true, y_pred) -> evaluation dictionary
      - accuracy_test, confusion_matrix, precision_score, recall_score, f1_score, baseline_accuracy
      - self.train_loss_history, self.val_loss_history, and self.cost_history alias
      - Automatic device selection (MPS on Apple Silicon, CUDA, fallback to CPU)
      - Balanced BCEWithLogitsLoss matching LogisticRegression class_weight='balanced'
      - In-memory early stopping with best validation state restoration
    """
    def __init__(
        self,
        model_name='mlp',
        input_dim=None,
        seq_len=20,
        hidden_dim=32,
        num_layers=2,
        dropout=0.2,
        learning_rate=0.001,
        n_epochs=100,
        batch_size=32,
        class_weight='balanced',
        patience=15,
        weight_decay=1e-4,
        device=None,
        nhead=4,
        dim_feedforward=64,
        filters=32,
        cell_type='lstm',
        random_state=42,
    ):
        self.model_name = model_name
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.class_weight = class_weight
        self.patience = patience
        self.weight_decay = weight_decay
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.filters = filters
        self.cell_type = cell_type
        self.random_state = random_state

        # Device selection with safe Apple Silicon probe
        if device is not None:
            self.device = torch.device(device)
        else:
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                try:
                    probe = torch.zeros(1, device='mps')
                    del probe
                    self.device = torch.device('mps')
                except Exception:
                    self.device = torch.device('cpu')
            elif torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')

        self.model = None
        self.is_fitted = False
        self.pos_weight_tensor = None
        self.cost_history = []
        self.train_loss_history = []
        self.val_loss_history = []

    def _build_model(self):
        """Constructs neural network architecture based on model_name string or instance."""
        if isinstance(self.model_name, nn.Module):
            return self.model_name

        name = str(self.model_name).lower()
        if name == 'mlp':
            return MLPRegimeNet(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                dropout=self.dropout,
            )
        elif name in ['lstm', 'gru', 'recurrent']:
            cell = 'gru' if name == 'gru' else self.cell_type
            return RecurrentRegimeNet(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                cell_type=cell,
                dropout=self.dropout,
            )
        elif name in ['cnn1d', 'cnn', 'conv1d']:
            return Conv1DRegimeNet(
                input_dim=self.input_dim,
                filters=self.filters,
                dropout=self.dropout,
            )
        elif name in ['transformer', 'attention']:
            return TransformerRegimeNet(
                input_dim=self.input_dim,
                d_model=self.hidden_dim,
                nhead=self.nhead,
                dim_feedforward=self.dim_feedforward,
                num_layers=self.num_layers,
                dropout=self.dropout,
            )
        else:
            raise ValueError(
                f"Unknown model_name '{self.model_name}'. "
                "Supported: 'mlp', 'lstm', 'gru', 'cnn1d', 'transformer' or nn.Module instance."
            )

    def _is_sequential(self):
        """Returns True if the underlying model expects 3D sequence tensors (B, L, F)."""
        if isinstance(self.model_name, str):
            return self.model_name.lower() in {'lstm', 'gru', 'recurrent', 'cnn1d', 'cnn', 'conv1d', 'transformer', 'attention'}
        if isinstance(self.model_name, (RecurrentRegimeNet, Conv1DRegimeNet, TransformerRegimeNet)):
            return True
        if self.model is not None and isinstance(self.model, (RecurrentRegimeNet, Conv1DRegimeNet, TransformerRegimeNet)):
            return True
        return False

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Trains model weights on feature representations X_train and binary targets y_train.
        Optionally evaluates validation loss each epoch, checkpointing the best weights.
        """
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        X_tr = _to_numpy(X_train)
        y_tr = _to_numpy(y_train)

        if self._is_sequential() and X_tr.ndim == 2:
            raise ValueError(
                f"Sequential model '{self.model_name}' expects 3D sequential input of shape (batch_size, seq_len, input_dim), "
                f"but received 2D input with shape {X_tr.shape}. Please create 3D sequences using Pipeline.create_sequences."
            )

        if self.input_dim is None:
            self.input_dim = X_tr.shape[-1]
        if X_tr.ndim == 3:
            self.seq_len = X_tr.shape[1]

        if self.model is None:
            self.model = self._build_model()
        self.model.to(self.device)

        # Balanced BCEWithLogitsLoss matching LogisticRegression class_weight='balanced'
        y_flat = y_tr.ravel()
        if self.class_weight == 'balanced':
            n0 = float(np.sum(y_flat == 0))
            n1 = float(np.sum(y_flat == 1))
            pos_weight_val = n0 / max(n1, 1.0)
            self.pos_weight_tensor = torch.tensor([pos_weight_val], dtype=torch.float32, device=self.device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight_tensor)
        else:
            self.pos_weight_tensor = None
            criterion = nn.BCEWithLogitsLoss()

        train_ds = RegimeDataset(X_tr, y_tr)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        if X_val is not None and y_val is not None:
            X_v = _to_numpy(X_val)
            y_v = _to_numpy(y_val)
            val_ds = RegimeDataset(X_v, y_v)
            val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)
        else:
            val_loader = None

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        self.train_loss_history = []
        self.val_loss_history = []
        best_val_loss = float('inf')
        best_weights = None
        patience_counter = 0

        for epoch in range(self.n_epochs):
            # Training phase
            self.model.train()
            running_loss = 0.0
            total_samples = 0
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * len(batch_x)
                total_samples += len(batch_x)

            epoch_train_loss = float(running_loss / max(total_samples, 1))
            self.train_loss_history.append(epoch_train_loss)

            # Validation & Early stopping phase
            if val_loader is not None:
                self.model.eval()
                val_running_loss = 0.0
                val_samples = 0
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        batch_x = batch_x.to(self.device)
                        batch_y = batch_y.to(self.device)
                        logits = self.model(batch_x)
                        v_loss = criterion(logits, batch_y)
                        val_running_loss += v_loss.item() * len(batch_x)
                        val_samples += len(batch_x)

                epoch_val_loss = float(val_running_loss / max(val_samples, 1))
                self.val_loss_history.append(epoch_val_loss)

                if epoch_val_loss < best_val_loss - 1e-4:
                    best_val_loss = epoch_val_loss
                    best_weights = copy.deepcopy(self.model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if self.patience is not None and patience_counter >= self.patience:
                        break

        if best_weights is not None:
            self.model.load_state_dict(best_weights)

        self.cost_history = self.train_loss_history
        self.is_fitted = True
        return self

    def predict_proba(self, X):
        """
        Computes predicted probabilities P(Bullish=1 | X).
        Returns a 2D NumPy array of shape (N, 1) with values in [0.0, 1.0].
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")

        X_arr = _to_numpy(X)
        if self._is_sequential() and X_arr.ndim == 2:
            raise ValueError(
                f"Sequential model '{self.model_name}' expects 3D sequential input of shape (batch_size, seq_len, input_dim), "
                f"but received 2D input with shape {X_arr.shape}. Please create 3D sequences using Pipeline.create_sequences."
            )

        self.model.eval()
        ds = RegimeDataset(X_arr)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=False)
        probs_list = []

        with torch.no_grad():
            for batch in loader:
                # loader returns batch_x (or tuple (batch_x, batch_y))
                batch_x = batch[0] if isinstance(batch, (list, tuple)) else batch
                batch_x = batch_x.to(self.device)
                logits = self.model(batch_x)
                probs = torch.sigmoid(logits)
                probs_list.append(probs.detach().cpu().numpy())

        if probs_list:
            return np.vstack(probs_list)
        return np.empty((0, 1), dtype=float)

    def predict(self, X, threshold=0.22, smooth_window=3):
        """
        Returns binary regime predictions {0, 1} based on probability threshold.
        Applies a causal rolling smoothing window to filter single-day probability noise.
        """
        probs = self.predict_proba(X).ravel()
        if smooth_window is not None and smooth_window > 1:
            probs = np.array([
                np.mean(probs[max(0, i - smooth_window + 1):i + 1])
                for i in range(len(probs))
            ])
        return (probs >= threshold).astype(int).reshape(-1, 1)

    def accuracy_test(self, y_true, y_predict):
        """Calculates classification accuracy."""
        y_t = _to_numpy(y_true).ravel()
        y_p = _to_numpy(y_predict).ravel()
        return float(np.mean(y_t == y_p)) if len(y_t) > 0 else 0.0

    def confusion_matrix(self, y_true, y_pred):
        """
        Computes 2x2 confusion matrix:
        [[True Negative,  False Positive],
         [False Negative, True Positive ]]
        """
        y_t = _to_numpy(y_true).ravel()
        y_p = _to_numpy(y_pred).ravel()

        tp = int(np.sum((y_t == 1) & (y_p == 1)))
        fp = int(np.sum((y_t == 0) & (y_p == 1)))
        fn = int(np.sum((y_t == 1) & (y_p == 0)))
        tn = int(np.sum((y_t == 0) & (y_p == 0)))

        return np.array([[tn, fp], [fn, tp]])

    def precision_score(self, y_true, y_pred):
        """Calculates Precision: TP / (TP + FP)."""
        cm = self.confusion_matrix(y_true, y_pred)
        tp = cm[1, 1]
        fp = cm[0, 1]
        return float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0

    def recall_score(self, y_true, y_pred):
        """Calculates Recall (Sensitivity): TP / (TP + FN)."""
        cm = self.confusion_matrix(y_true, y_pred)
        tp = cm[1, 1]
        fn = cm[1, 0]
        return float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    def f1_score(self, y_true, y_pred):
        """Calculates F1-Score: 2 * (Precision * Recall) / (Precision + Recall)."""
        p = self.precision_score(y_true, y_pred)
        r = self.recall_score(y_true, y_pred)
        return float((2.0 * p * r) / (p + r)) if (p + r) > 0 else 0.0

    def baseline_accuracy(self, y_true):
        """
        Calculates naive baseline accuracy of always predicting the majority class (Bullish=1).
        """
        y_t = _to_numpy(y_true).ravel()
        return float(np.mean(y_t == 1)) if len(y_t) > 0 else 0.0

    def roc_auc_score(self, y_true, y_prob):
        """
        Calculates Area Under the Receiver Operating Characteristic Curve (ROC-AUC)
        using pure NumPy trapezoidal integration without scikit-learn dependency.
        """
        y_t = _to_numpy(y_true).ravel()
        p = _to_numpy(y_prob).ravel()
        n_pos = int(np.sum(y_t == 1))
        n_neg = int(np.sum(y_t == 0))
        if n_pos == 0 or n_neg == 0:
            return 0.5
        desc_idx = np.argsort(-p)
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
        auc = np.sum(0.5 * (tpr[1:] + tpr[:-1]) * (fpr[1:] - fpr[:-1]))
        return float(auc)

    def balanced_accuracy_score(self, y_true, y_pred):
        """
        Calculates Balanced Accuracy: 0.5 * (Sensitivity + Specificity)
        where Sensitivity = TP / (TP + FN) and Specificity = TN / (TN + FP).
        """
        cm = self.confusion_matrix(y_true, y_pred)
        tn, fp = cm[0, 0], cm[0, 1]
        fn, tp = cm[1, 0], cm[1, 1]
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        return float(0.5 * (recall + specificity))

    def log_loss_score(self, y_true, y_prob, eps=1e-15):
        """
        Calculates Log Loss (Binary Cross-Entropy) between y_true and predicted probabilities.
        """
        y_t = _to_numpy(y_true).ravel()
        p = _to_numpy(y_prob).ravel()
        if len(y_t) == 0:
            return 0.0
        p_clipped = np.clip(p, eps, 1.0 - eps)
        ll = -np.mean(y_t * np.log(p_clipped) + (1.0 - y_t) * np.log(1.0 - p_clipped))
        return float(ll)

    def evaluate(self, y_true, y_pred, y_prob=None):
        """Returns comprehensive dictionary of all evaluation metrics."""
        metrics = {
            "Accuracy": self.accuracy_test(y_true, y_pred),
            "Precision": self.precision_score(y_true, y_pred),
            "Recall": self.recall_score(y_true, y_pred),
            "F1-Score": self.f1_score(y_true, y_pred),
            "Baseline (Always Bullish)": self.baseline_accuracy(y_true),
            "Confusion Matrix": self.confusion_matrix(y_true, y_pred),
        }
        if y_prob is not None:
            metrics["ROC-AUC"] = self.roc_auc_score(y_true, y_prob)
            metrics["Balanced Accuracy"] = self.balanced_accuracy_score(y_true, y_pred)
            metrics["Log Loss"] = self.log_loss_score(y_true, y_prob)
        return metrics

    def sigmoid_(self, z):
        """Numerically stable sigmoid function matching LogisticRegression."""
        z = np.clip(np.asarray(z, dtype=float), -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def compute_cost(self, X, y):
        """Computes binary cross entropy loss on X, y matching LogisticRegression."""
        if not self.is_fitted or self.model is None:
            return 0.0
        self.model.eval()
        X_t = _to_numpy(X)
        y_t = _to_numpy(y)
        ds = RegimeDataset(X_t, y_t)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=False)
        if self.class_weight == 'balanced' and self.pos_weight_tensor is not None:
            criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight_tensor.to(self.device))
        else:
            criterion = nn.BCEWithLogitsLoss()
        total_loss = 0.0
        total_samples = 0
        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                loss = criterion(self.model(batch_x), batch_y)
                total_loss += loss.item() * len(batch_x)
                total_samples += len(batch_x)
        return float(total_loss / max(total_samples, 1))

    def compute_gradient(self, X, y):
        """Computes parameter gradients on batch (X, y) matching LogisticRegression."""
        if not self.is_fitted or self.model is None:
            return None
        self.model.train()
        X_t = _to_numpy(X)
        y_t = _to_numpy(y)
        ds = RegimeDataset(X_t, y_t)
        loader = DataLoader(ds, batch_size=len(ds), shuffle=False)
        batch_x, batch_y = next(iter(loader))
        batch_x = batch_x.to(self.device)
        batch_y = batch_y.to(self.device)
        self.model.zero_grad()
        if self.class_weight == 'balanced' and self.pos_weight_tensor is not None:
            criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight_tensor.to(self.device))
        else:
            criterion = nn.BCEWithLogitsLoss()
        loss = criterion(self.model(batch_x), batch_y)
        loss.backward()
        grads = [p.grad.detach().cpu().numpy() for p in self.model.parameters() if p.grad is not None]
        return grads

    def gradient_descent(self, X, y):
        """Optimization loop alias matching LogisticRegression."""
        return self.fit(X, y)

    def get_attention_weights(self, X=None, aggregate=None):
        """
        Extracts self-attention weight matrix from a fitted Transformer model.

        Parameters:
            X (array-like, optional): 2D single sequence (L, F) or 3D sequences (N, L, F).
                                      If None, returns cached attention weights from last forward pass.
            aggregate (str, optional): Aggregation method if N > 1 sequences are provided.
                                       'mean': returns average attention matrix across batch (L, L).
                                       None: returns raw batch attention weights (N, L, L).

        Returns:
            np.ndarray: Attention weights array of shape (L, L) for single sequence / mean aggregation,
                        or (N, L, L) for multi-sequence batch.
        """
        is_tf = (
            (isinstance(self.model_name, str) and self.model_name.lower() in {'transformer', 'attention'})
            or isinstance(self.model_name, TransformerRegimeNet)
            or (self.model is not None and hasattr(self.model, 'last_attention_weights'))
        )
        if not is_tf:
            raise ValueError(
                f"Model '{self.model_name}' does not generate attention weights. "
                "Attention extraction is only supported for Transformer ('transformer') architectures."
            )

        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")

        if X is None:
            weights = self.model.last_attention_weights
            if weights is None:
                raise RuntimeError("No cached attention weights found. Run predict_proba(X) or provide X.")
            arr = weights.detach().cpu().numpy()
            return arr[0] if arr.shape[0] == 1 else arr

        X_arr = _to_numpy(X)
        squeeze_batch = False
        if X_arr.ndim == 2:
            X_arr = np.expand_dims(X_arr, axis=0)
            squeeze_batch = True
        elif X_arr.ndim != 3:
            raise ValueError(
                f"get_attention_weights expects 2D (seq_len, input_dim) or 3D (batch_size, seq_len, input_dim) input, "
                f"but received array with ndim={X_arr.ndim} and shape={X_arr.shape}."
            )

        self.model.eval()
        with torch.no_grad():
            x_tensor = torch.as_tensor(X_arr, dtype=torch.float32, device=self.device)
            _ = self.model(x_tensor)
            weights = self.model.last_attention_weights.detach().cpu().numpy()

        if squeeze_batch or weights.shape[0] == 1:
            return weights[0]
        if aggregate == 'mean':
            return np.mean(weights, axis=0)
        return weights
