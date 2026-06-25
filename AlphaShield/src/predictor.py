"""Feature alignment, leakage-safe time-series slicing, and model persistence."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - fallback only when optional wheel is unavailable
    from sklearn.ensemble import GradientBoostingClassifier as XGBClassifier


class DataLeakageException(AssertionError):
    """Raised when training and validation time windows overlap."""


class FeatureEngineeringMatrix:
    """Combines technical and sentiment matrices and creates directional targets."""

    def __init__(self, target_horizon: int = 1) -> None:
        if target_horizon < 1:
            raise ValueError("target_horizon must be positive")
        self.target_horizon = target_horizon

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
        joined = technical.join(sentiment, how="inner")
        if joined.empty:
            raise ValueError("technical and sentiment matrices do not overlap")
        joined["sentiment_score"] = joined["sentiment_score"].astype(float).fillna(0.0)
        future_close = joined["close"].shift(-self.target_horizon)
        joined["target_direction"] = (future_close > joined["close"]).astype(int)
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


class AlphaShieldPredictorModel:
    """XGBoost classifier wrapper with fixed anti-overfit hyperparameters."""

    def __init__(self, random_state: int = 42) -> None:
        kwargs = {
            "max_depth": 5,
            "learning_rate": 0.01,
            "n_estimators": 100,
            "subsample": 0.8,
            "random_state": random_state,
        }
        if getattr(XGBClassifier, "__module__", "").startswith("xgboost"):
            kwargs.update({"eval_metric": "logloss", "objective": "binary:logistic"})
        self.model = XGBClassifier(**kwargs)
        self.feature_metadata: Dict[str, object] = {}

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
