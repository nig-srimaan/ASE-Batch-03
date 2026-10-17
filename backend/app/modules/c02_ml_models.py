"""
Component 2: Multiple ML Models Module
======================================
Responsibilities:
- Train and manage 3 distinct ML models with clear accuracy–latency–resource trade-offs:
  1. XGBoost (High Accuracy ~92%, Baseline Latency ~80ms, Baseline CPU ~65%)
  2. Random Forest (Balanced Accuracy ~88%, Baseline Latency ~60ms, Baseline CPU ~50%)
  3. Logistic Regression (Low Latency ~20ms, Lower Accuracy ~82%, Baseline CPU ~30%)
- Automatically uses `xgboost` and `scikit-learn` when installed, or falls back to
  pure-NumPy native implementations of Logistic Regression, Random Forest, and
  Gradient Boosted Trees so the codebase runs in any environment without compilation issues.
"""
import os
import json
import pickle
import numpy as np
from typing import Dict, Any, Tuple, Optional

from backend.app.config import config

try:
    from sklearn.linear_model import LogisticRegression as SklearnLR
    from sklearn.ensemble import RandomForestClassifier as SklearnRF
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    from xgboost import XGBClassifier as NativeXGB
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


# ============================================================================
# Pure-NumPy Native ML Estimators (Zero-Dependency Fallback & Deterministic Execution)
# ============================================================================

class NumpyLogisticRegression:
    """Vectorized L2-regularized Logistic Regression trained via batch gradient descent."""
    def __init__(self, lr: float = 0.15, n_iters: int = 350, reg_lambda: float = 0.01):
        self.lr = lr
        self.n_iters = n_iters
        self.reg_lambda = reg_lambda
        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features, dtype=np.float64)
        self.bias = 0.0
        y_float = y.astype(np.float64)

        for _ in range(self.n_iters):
            linear = X @ self.weights + self.bias
            preds = self._sigmoid(linear)
            error = preds - y_float
            dw = (X.T @ error) / n_samples + self.reg_lambda * self.weights
            db = float(np.mean(error))
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # Linear dot product across features
        z = X @ self.weights + self.bias
        p1 = self._sigmoid(z)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(np.int32)


class _SimpleDecisionTree:
    """Fast CART decision tree regressor/classifier node used inside Random Forest & XGBoost."""
    def __init__(self, max_depth: int = 5, min_samples_split: int = 10, max_features: Optional[int] = None, seed: int = 0):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.seed = seed
        self.tree = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        rng = np.random.default_rng(self.seed)
        self.tree = self._build(X, y, depth=0, rng=rng)
        return self

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int, rng: np.random.Generator):
        n_samples, n_features = X.shape
        if depth >= self.max_depth or n_samples < self.min_samples_split or np.std(y) < 1e-7:
            return float(np.mean(y))

        feat_count = self.max_features or n_features
        feat_indices = rng.choice(n_features, size=min(feat_count, n_features), replace=False)

        best_gain = -1.0
        best_feat = None
        best_thresh = None
        parent_var = np.var(y) * n_samples

        for f in feat_indices:
            col = X[:, f]
            percentiles = np.percentile(col, [20, 35, 50, 65, 80])
            for thresh in np.unique(percentiles):
                left_mask = col <= thresh
                n_left = int(np.sum(left_mask))
                n_right = n_samples - n_left
                if n_left < 4 or n_right < 4:
                    continue
                y_left, y_right = y[left_mask], y[~left_mask]
                gain = parent_var - (np.var(y_left) * n_left + np.var(y_right) * n_right)
                if gain > best_gain:
                    best_gain = gain
                    best_feat = int(f)
                    best_thresh = float(thresh)

        if best_feat is None or best_gain <= 1e-6:
            return float(np.mean(y))

        left_mask = X[:, best_feat] <= best_thresh
        return {
            "f": best_feat,
            "t": best_thresh,
            "l": self._build(X[left_mask], y[left_mask], depth + 1, rng),
            "r": self._build(X[~left_mask], y[~left_mask], depth + 1, rng),
        }

    def _predict_row(self, row: np.ndarray, node) -> float:
        while isinstance(node, dict):
            if row[node["f"]] <= node["t"]:
                node = node["l"]
            else:
                node = node["r"]
        return node

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._predict_row(row, self.tree) for row in X], dtype=np.float64)


class NumpyRandomForest:
    """Bootstrap-aggregated ensemble of Decision Trees."""
    def __init__(self, n_estimators: int = 16, max_depth: int = 5, seed: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.seed = seed
        self.trees = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        rng = np.random.default_rng(self.seed)
        n_samples, n_features = X.shape
        max_f = max(4, int(np.sqrt(n_features)) + 2)
        self.trees = []
        for i in range(self.n_estimators):
            boot_idx = rng.choice(n_samples, size=n_samples, replace=True)
            tree = _SimpleDecisionTree(
                max_depth=self.max_depth,
                max_features=max_f,
                seed=self.seed + i * 17
            )
            tree.fit(X[boot_idx], y[boot_idx].astype(np.float64))
            self.trees.append(tree)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        preds = np.mean([t.predict(X) for t in self.trees], axis=0)
        p1 = np.clip(preds, 0.0, 1.0)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(np.int32)


class NumpyXGBoostClassifier:
    """
    Gradient Boosted Decision Trees trained on second-order log-loss gradients
    (the exact mathematical formulation of XGBoost).
    """
    def __init__(self, n_estimators: int = 28, learning_rate: float = 0.22, max_depth: int = 4, seed: int = 101):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.seed = seed
        self.trees = []
        self.base_log_odds: float = 0.0

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))

    def fit(self, X: np.ndarray, y: np.ndarray):
        y_float = y.astype(np.float64)
        pos_ratio = np.clip(np.mean(y_float), 0.01, 0.99)
        self.base_log_odds = float(np.log(pos_ratio / (1.0 - pos_ratio)))
        raw_scores = np.full(len(y_float), self.base_log_odds, dtype=np.float64)

        self.trees = []
        n_features = X.shape[1]
        for i in range(self.n_estimators):
            probs = self._sigmoid(raw_scores)
            # Negative gradient (pseudo-residuals) scaled by Hessian damper
            grad = y_float - probs
            hess = np.maximum(probs * (1.0 - probs), 0.05)
            newton_target = np.clip(grad / hess, -4.0, 4.0)

            tree = _SimpleDecisionTree(
                max_depth=self.max_depth,
                max_features=max(8, int(n_features * 0.75)),
                seed=self.seed + i * 31
            )
            tree.fit(X, newton_target)
            update = tree.predict(X)
            raw_scores += self.learning_rate * update
            self.trees.append(tree)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw_scores = np.full(X.shape[0], self.base_log_odds, dtype=np.float64)
        for tree in self.trees:
            raw_scores += self.learning_rate * tree.predict(X)
        p1 = self._sigmoid(raw_scores)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(np.int32)


class MLModelsModule:
    """
    Component 2: Manages training, serialization, and loading of the 3 candidate ML models:
    - `xgboost`: High accuracy (~92%), higher latency (~80ms at normal load), higher CPU (~65%)
    - `random_forest`: Balanced accuracy (~88%), moderate latency (~60ms at normal load), moderate CPU (~50%)
    - `logistic_regression`: Lower accuracy (~82%), ultra-low latency (~20ms at normal load), low CPU (~30%)
    """

    MODEL_CATALOG = {
        "xgboost": {
            "id": "xgboost",
            "name": "XGBoost",
            "tier": "HEAVY",
            "complexity_rank": 3,
            "baseline_latency_ms": 80.0,
            "baseline_cpu_percent": 65.0,
            "baseline_memory_mb": 420.0,
            "latency_load_sensitivity": 0.42,  # ms per additional req/min above baseline
            "cpu_load_sensitivity": 0.18,
            "description": "High-accuracy gradient boosted trees for deep inspection under normal load.",
        },
        "random_forest": {
            "id": "random_forest",
            "name": "Random Forest",
            "tier": "BALANCED",
            "complexity_rank": 2,
            "baseline_latency_ms": 60.0,
            "baseline_cpu_percent": 50.0,
            "baseline_memory_mb": 265.0,
            "latency_load_sensitivity": 0.18,
            "cpu_load_sensitivity": 0.10,
            "description": "Balanced ensemble model providing solid accuracy (>=88%) with resilient latency.",
        },
        "logistic_regression": {
            "id": "logistic_regression",
            "name": "Logistic Regression",
            "tier": "LIGHT",
            "complexity_rank": 1,
            "baseline_latency_ms": 20.0,
            "baseline_cpu_percent": 30.0,
            "baseline_memory_mb": 95.0,
            "latency_load_sensitivity": 0.05,
            "cpu_load_sensitivity": 0.04,
            "description": "Ultra-fast linear classifier for extreme overload survival (note: ~82% accuracy).",
        },
    }

    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.bundle_path = os.path.join(config.models_dir, "trained_ensemble.pkl")

    def build_and_train_models(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """Trains all 3 candidate models and persists them to `saved_models/trained_ensemble.pkl`."""
        # 1. Logistic Regression (Light)
        if HAS_SKLEARN:
            lr_model = SklearnLR(max_iter=500, random_state=42)
        else:
            lr_model = NumpyLogisticRegression(lr=0.15, n_iters=350, reg_lambda=0.02)
        lr_model.fit(X_train, y_train)

        # 2. Random Forest (Balanced)
        if HAS_SKLEARN:
            rf_model = SklearnRF(n_estimators=35, max_depth=7, random_state=42)
        else:
            rf_model = NumpyRandomForest(n_estimators=28, max_depth=6, seed=42)
        rf_model.fit(X_train, y_train)

        # 3. XGBoost (Heavy / High Accuracy)
        if HAS_XGBOOST:
            xgb_model = NativeXGB(n_estimators=90, max_depth=6, learning_rate=0.15, random_state=42, eval_metric="logloss")
        else:
            xgb_model = NumpyXGBoostClassifier(n_estimators=55, learning_rate=0.18, max_depth=5, seed=101)
        xgb_model.fit(X_train, y_train)

        self.models = {
            "logistic_regression": lr_model,
            "random_forest": rf_model,
            "xgboost": xgb_model,
        }

        with open(self.bundle_path, "wb") as f:
            pickle.dump(self.models, f)

        return self.models

    def load_models(self) -> Dict[str, Any]:
        if self.models:
            return self.models
        if os.path.exists(self.bundle_path):
            with open(self.bundle_path, "rb") as f:
                self.models = pickle.load(f)
            return self.models
        raise FileNotFoundError("Trained models bundle not found. Run evaluation/training pipeline first.")


ml_models_module = MLModelsModule()
