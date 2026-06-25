"""Feature alignment, leakage-safe time-series slicing, and model persistence."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - fallback only when optional wheel is unavailable
    from sklearn.ensemble import GradientBoostingClassifier as XGBClassifier


class DataLeakageException(AssertionError):
    """Raised when training and validation time windows overlap."""


class FeatureEngineeringMatrix:
    """Combines technical and sentiment matrices and creates directional targets.

    The target label is leakage-safe: only past data (up to the decision point)
    contributes to each feature row.  Sentiment is aligned via an as-of backward
    join so headlines never leak from the future into past market rows.

    Parameters
    ----------
    target_horizon:
        Number of periods ahead used to compute the future return.
    positive_return_threshold:
        Minimum future return required to assign class ``1`` (bullish).  Setting
        this to ``0.0`` (the default) reproduces the original binary price-direction
        label.  Use a small positive value (e.g. ``0.002`` for 0.2 %) to require
        the move to exceed expected transaction costs before being counted as a
        tradeable signal.
    """

    def __init__(
        self,
        target_horizon: int = 1,
        positive_return_threshold: float = 0.0,
    ) -> None:
        if target_horizon < 1:
            raise ValueError("target_horizon must be positive")
        self.target_horizon = target_horizon
        self.positive_return_threshold = float(positive_return_threshold)

    @staticmethod
    def _localized_index(frame: pd.DataFrame) -> pd.DataFrame:
        localized = frame.copy()
        index = pd.to_datetime(localized.index, utc=True)
        localized.index = index
        localized = localized[~localized.index.duplicated(keep="last")].sort_index()
        return localized

    def build(self, technical_frame: pd.DataFrame, sentiment_frame: pd.DataFrame) -> pd.DataFrame:
        if "close" not in technical_frame.columns:
            raise ValueError("technical_frame must contain close")
        technical = self._localized_index(technical_frame)
        sentiment = self._localized_index(sentiment_frame)

        # As-of backward join: each technical row receives the most recent
        # sentiment entry at or *before* its timestamp, preventing any future
        # sentiment from leaking into earlier market rows.
        tech_reset = technical.reset_index()
        # Ensure the timestamp column is named consistently after reset_index.
        if "index" in tech_reset.columns and "timestamp" not in tech_reset.columns:
            tech_reset = tech_reset.rename(columns={"index": "timestamp"})

        if sentiment.empty:
            joined = tech_reset.set_index("timestamp")
            joined["sentiment_score"] = 0.0
        else:
            sent_reset = sentiment[["sentiment_score"]].reset_index()
            if "index" in sent_reset.columns and "timestamp" not in sent_reset.columns:
                sent_reset = sent_reset.rename(columns={"index": "timestamp"})
            # Both frames must be sorted for merge_asof.
            tech_reset = tech_reset.sort_values("timestamp")
            sent_reset = sent_reset.sort_values("timestamp")
            joined = pd.merge_asof(tech_reset, sent_reset, on="timestamp", direction="backward")
            joined = joined.set_index("timestamp")

        if joined.empty:
            raise ValueError("technical and sentiment matrices do not overlap")

        joined["sentiment_score"] = joined["sentiment_score"].astype(float).fillna(0.0)

        # Leakage-safe target: future close is shifted *backwards* relative to
        # the current row, then rows without a valid future target are dropped.
        # If close is zero (degenerate bar), future_return becomes NaN which
        # evaluates to False in the comparison, labelling the row class 0.
        future_close = joined["close"].shift(-self.target_horizon)
        safe_close = joined["close"].replace(0.0, np.nan)
        future_return = future_close / safe_close - 1.0
        joined["target_direction"] = (future_return > self.positive_return_threshold).astype(int)
        joined = joined.iloc[:-self.target_horizon]
        joined = joined.replace([np.inf, -np.inf], 0.0).ffill().bfill().fillna(0.0)
        return joined


@dataclass(frozen=True)
class Fold:
    train_features: pd.DataFrame
    train_target: pd.Series
    validation_features: pd.DataFrame
    validation_target: pd.Series
    scaler: StandardScaler


class TimeSeriesDataset:
    """Sequential walk-forward dataset that prevents temporal contamination."""

    def __init__(self, matrix: pd.DataFrame, target_column: str = "target_direction") -> None:
        if target_column not in matrix.columns:
            raise ValueError(f"{target_column} not found")
        self.matrix = matrix.copy().sort_index()
        self.target_column = target_column
        if not isinstance(self.matrix.index, pd.DatetimeIndex):
            self.matrix.index = pd.to_datetime(self.matrix.index, utc=True)
        elif self.matrix.index.tz is None:
            self.matrix.index = self.matrix.index.tz_localize("UTC")
        self.feature_columns = [column for column in self.matrix.columns if column != target_column]

    @staticmethod
    def _assert_no_overlap(train_index: pd.DatetimeIndex, validation_index: pd.DatetimeIndex) -> None:
        if len(train_index) == 0 or len(validation_index) == 0:
            raise ValueError("train and validation windows must be non-empty")
        if train_index.max() >= validation_index.min():
            raise DataLeakageException("training timestamps overlap validation timestamps")

    def walk_forward_splits(self, initial_train_size: int, validation_size: int) -> Iterator[Fold]:
        if initial_train_size <= 0 or validation_size <= 0:
            raise ValueError("split sizes must be positive")
        total = len(self.matrix)
        start = initial_train_size
        while start + validation_size <= total:
            train = self.matrix.iloc[:start]
            validation = self.matrix.iloc[start : start + validation_size]
            self._assert_no_overlap(train.index, validation.index)
            scaler = StandardScaler()
            train_features = pd.DataFrame(
                scaler.fit_transform(train[self.feature_columns]),
                index=train.index,
                columns=self.feature_columns,
            )
            validation_features = pd.DataFrame(
                scaler.transform(validation[self.feature_columns]),
                index=validation.index,
                columns=self.feature_columns,
            )
            yield Fold(
                train_features=train_features,
                train_target=train[self.target_column].astype(int),
                validation_features=validation_features,
                validation_target=validation[self.target_column].astype(int),
                scaler=scaler,
            )
            start += validation_size

    def first_split(self, initial_train_size: int, validation_size: int) -> Fold:
        return next(self.walk_forward_splits(initial_train_size, validation_size))


class WalkForwardEvaluator:
    """Evaluates a predictor model across all walk-forward folds without leakage.

    The scaler and model are fitted exclusively on each training fold and then
    applied to the following validation fold, so no information crosses folds.
    """

    def evaluate(
        self,
        dataset: TimeSeriesDataset,
        model_class: type,
        initial_train_size: int,
        validation_size: int,
        **model_kwargs: Any,
    ) -> Dict[str, Any]:
        """Run walk-forward evaluation and return per-fold and aggregate metrics.

        Returns
        -------
        dict with keys:
            ``fold_results`` – DataFrame with per-fold metrics
            ``mean_accuracy``, ``mean_f1``, ``mean_precision``, ``mean_recall``
            ``mean_positive_rate`` – average fraction of positive (class-1) labels
            ``n_folds`` – number of folds evaluated
        """
        fold_records: List[Dict[str, Any]] = []
        for fold in dataset.walk_forward_splits(initial_train_size, validation_size):
            model = model_class(**model_kwargs)
            model.fit(fold.train_features, fold.train_target)
            predictions = model.predict(fold.validation_features)
            true = fold.validation_target.to_numpy(dtype=int)
            preds = np.asarray(predictions, dtype=int)
            fold_records.append(
                {
                    "accuracy": float(accuracy_score(true, preds)),
                    "precision": float(precision_score(true, preds, zero_division=0)),
                    "recall": float(recall_score(true, preds, zero_division=0)),
                    "f1": float(f1_score(true, preds, zero_division=0)),
                    "positive_rate": float(true.mean()),
                    "train_start": fold.train_features.index.min(),
                    "train_end": fold.train_features.index.max(),
                    "val_start": fold.validation_features.index.min(),
                    "val_end": fold.validation_features.index.max(),
                }
            )
        if not fold_records:
            raise ValueError("No folds were produced — check initial_train_size and validation_size")
        folds_df = pd.DataFrame(fold_records)
        return {
            "fold_results": folds_df,
            "mean_accuracy": float(folds_df["accuracy"].mean()),
            "mean_f1": float(folds_df["f1"].mean()),
            "mean_precision": float(folds_df["precision"].mean()),
            "mean_recall": float(folds_df["recall"].mean()),
            "mean_positive_rate": float(folds_df["positive_rate"].mean()),
            "n_folds": len(fold_records),
        }


class AlphaShieldPredictorModel:
    """XGBoost classifier wrapper with configurable hyperparameters.

    Hyperparameters can be supplied as constructor arguments or loaded from a
    parsed ``settings.yaml`` config dict via :meth:`from_config`.  Sensible
    anti-overfit defaults are provided for every parameter.
    """

    def __init__(
        self,
        random_state: int = 42,
        max_depth: int = 5,
        learning_rate: float = 0.01,
        n_estimators: int = 100,
        subsample: float = 0.8,
    ) -> None:
        kwargs: Dict[str, Any] = {
            "max_depth": int(max_depth),
            "learning_rate": float(learning_rate),
            "n_estimators": int(n_estimators),
            "subsample": float(subsample),
            "random_state": int(random_state),
        }
        if getattr(XGBClassifier, "__module__", "").startswith("xgboost"):
            kwargs.update({"eval_metric": "logloss", "objective": "binary:logistic"})
        self.model = XGBClassifier(**kwargs)
        self.feature_metadata: Dict[str, object] = {}

    @classmethod
    def from_config(cls, config: Dict[str, Any], random_state: int = 42) -> "AlphaShieldPredictorModel":
        """Construct a model from a parsed ``settings.yaml`` config dict."""
        model_cfg = config.get("model", {}) if config else {}
        return cls(
            random_state=random_state,
            max_depth=int(model_cfg.get("max_depth", 5)),
            learning_rate=float(model_cfg.get("learning_rate", 0.01)),
            n_estimators=int(model_cfg.get("n_estimators", 100)),
            subsample=float(model_cfg.get("subsample", 0.8)),
        )

    def fit(self, features: pd.DataFrame, target: pd.Series) -> "AlphaShieldPredictorModel":
        self.feature_metadata = {
            "feature_names": list(features.columns),
            "trained_start": str(features.index.min()),
            "trained_end": str(features.index.max()),
        }
        self.model.fit(features, target.astype(int))
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.model.predict(features), dtype=int)

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        if hasattr(self.model, "predict_proba"):
            return np.asarray(self.model.predict_proba(features), dtype=float)
        labels = self.predict(features)
        return np.column_stack([1 - labels, labels]).astype(float)

    def evaluate_accuracy(self, features: pd.DataFrame, target: pd.Series) -> float:
        return float(accuracy_score(target.astype(int), self.predict(features)))

    def save_checkpoint(self, path: str | Path = "models/checkpoint.pkl") -> Path:
        checkpoint = Path(path)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        with checkpoint.open("wb") as handle:
            pickle.dump({"model": self.model, "feature_metadata": self.feature_metadata}, handle)
        return checkpoint

    @classmethod
    def load_checkpoint(cls, path: str | Path = "models/checkpoint.pkl") -> "AlphaShieldPredictorModel":
        with Path(path).open("rb") as handle:
            payload = pickle.load(handle)
        instance = cls()
        instance.model = payload["model"]
        instance.feature_metadata = payload.get("feature_metadata", {})
        return instance
