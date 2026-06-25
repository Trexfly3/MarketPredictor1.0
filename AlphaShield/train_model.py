"""Train AlphaShield with leakage-safe walk-forward validation and tuning.

This script is designed for accuracy-first model training while still respecting
financial time-series constraints:

* no random train/test split for time-series data
* technical indicators are shifted to avoid lookahead leakage
* validation uses expanding walk-forward folds
* hyperparameters are selected by walk-forward cross-validation
* the decision threshold is tuned from validation probabilities
* the final model, scaler, feature list, threshold, and metrics are saved

Expected market-data CSV columns:
    timestamp, open, high, low, close, volume

Optional columns:
    symbol

Optional sentiment CSV columns:
    timestamp, sentiment_score

Examples:
    python train_model.py --data data/historical_bars.csv
    python train_model.py --data data/SPY.csv --sentiment data/sentiment.csv --metric f1
    python train_model.py --data data/bars.csv --target-threshold 0.002 --trials 60
"""

from __future__ import annotations

import argparse
import itertools
import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from src.data_pipeline import TechnicalIndicatorEngine
from src.predictor import AlphaShieldPredictorModel, FeatureEngineeringMatrix, TimeSeriesDataset

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = ROOT / "config" / "settings.yaml"
DEFAULT_MODEL_PATH = ROOT / "models" / "checkpoint.pkl"
DEFAULT_BUNDLE_PATH = ROOT / "models" / "training_bundle.pkl"
DEFAULT_METRICS_PATH = ROOT / "models" / "training_metrics.json"

REQUIRED_MARKET_COLUMNS = {"timestamp", "open", "high", "low", "close", "volume"}
SUPPORTED_METRICS = {"f1", "balanced_accuracy", "accuracy", "precision", "recall", "roc_auc"}


@dataclass(frozen=True)
class TrialResult:
    """Summary for one hyperparameter candidate."""

    params: Dict[str, Any]
    threshold: float
    score: float
    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: Optional[float]
    positive_prediction_rate: float
    positive_label_rate: float
    folds: int


def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    return loaded or {}


def read_market_data(path: Path, symbol: Optional[str] = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Market data file not found: {path}")

    frame = pd.read_csv(path)
    frame.columns = [str(column).strip().lower() for column in frame.columns]

    missing = REQUIRED_MARKET_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Market data is missing required columns: {sorted(missing)}")

    if symbol and "symbol" in frame.columns:
        frame = frame[frame["symbol"].astype(str).str.upper() == symbol.upper()].copy()
        if frame.empty:
            raise ValueError(f"No rows found for symbol {symbol!r}")

    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["timestamp"])

    for column in ["open", "high", "low", "close", "volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["open", "high", "low", "close", "volume"])

    # Keep only valid bars. Bad high/low rows silently poison technical indicators.
    frame = frame[
        (frame["open"] > 0)
        & (frame["high"] > 0)
        & (frame["low"] > 0)
        & (frame["close"] > 0)
        & (frame["volume"] >= 0)
        & (frame["high"] >= frame[["open", "close", "low"]].max(axis=1))
        & (frame["low"] <= frame[["open", "close", "high"]].min(axis=1))
    ].copy()

    if frame.empty:
        raise ValueError("No valid market bars remain after cleaning")

    sort_columns = ["timestamp"]
    if "symbol" in frame.columns:
        sort_columns = ["symbol", "timestamp"]
    frame = frame.sort_values(sort_columns).drop_duplicates(sort_columns, keep="last")
    return frame


def read_sentiment_data(path: Optional[Path], fallback_index: pd.DatetimeIndex) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame({"sentiment_score": 0.0}, index=fallback_index)
    if not path.exists():
        raise FileNotFoundError(f"Sentiment file not found: {path}")

    sentiment = pd.read_csv(path)
    sentiment.columns = [str(column).strip().lower() for column in sentiment.columns]
    required = {"timestamp", "sentiment_score"}
    missing = required - set(sentiment.columns)
    if missing:
        raise ValueError(f"Sentiment data is missing required columns: {sorted(missing)}")

    sentiment["timestamp"] = pd.to_datetime(sentiment["timestamp"], utc=True, errors="coerce")
    sentiment["sentiment_score"] = pd.to_numeric(sentiment["sentiment_score"], errors="coerce")
    sentiment = sentiment.dropna(subset=["timestamp", "sentiment_score"])
    if sentiment.empty:
        return pd.DataFrame({"sentiment_score": 0.0}, index=fallback_index)

    sentiment = sentiment.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    return sentiment.set_index("timestamp")[["sentiment_score"]]


def make_technical_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Create technical features, handling one or many symbols.

    For multi-symbol data, indicators are calculated per symbol before the
    matrices are concatenated. This prevents one ticker's close/volume history
    from leaking into another ticker's indicators.
    """

    engine = TechnicalIndicatorEngine()
    if "symbol" not in raw.columns:
        return engine.transform(raw)

    matrices: List[pd.DataFrame] = []
    for symbol, group in raw.groupby("symbol", sort=True):
        matrix = engine.transform(group)
        matrix["symbol_code"] = pd.factorize(pd.Series([symbol] * len(matrix)))[0][0]
        matrices.append(matrix)
    return pd.concat(matrices).sort_index()


def build_matrix(
    raw: pd.DataFrame,
    sentiment: pd.DataFrame,
    target_horizon: int,
    target_threshold: float,
) -> pd.DataFrame:
    technical = make_technical_features(raw)
    matrix = FeatureEngineeringMatrix(
        target_horizon=target_horizon,
        positive_return_threshold=target_threshold,
    ).build(technical, sentiment)
    matrix = matrix.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
    if matrix["target_direction"].nunique() < 2:
        raise ValueError(
            "The target has only one class after feature engineering. "
            "Use more data, a different timeframe, or a lower --target-threshold."
        )
    return matrix


def split_plan(total_rows: int, initial_train_size: Optional[int], validation_size: Optional[int]) -> Tuple[int, int]:
    if total_rows < 100:
        raise ValueError("At least 100 engineered rows are recommended for walk-forward training")

    initial = initial_train_size or max(int(total_rows * 0.60), 50)
    validation = validation_size or max(int(total_rows * 0.10), 10)

    if initial + validation > total_rows:
        validation = max(total_rows - initial, 1)
    if initial <= 0 or validation <= 0 or initial + validation > total_rows:
        raise ValueError(
            f"Invalid split sizes for {total_rows} rows: "
            f"initial_train_size={initial}, validation_size={validation}"
        )
    return initial, validation


def candidate_grid(config: Dict[str, Any], trials: int) -> List[Dict[str, Any]]:
    configured = config.get("model", {}) if config else {}
    base = {
        "learning_rate": float(configured.get("learning_rate", 0.01)),
        "n_estimators": int(configured.get("n_estimators", 100)),
        "max_depth": int(configured.get("max_depth", 5)),
        "subsample": float(configured.get("subsample", 0.8)),
    }

    learning_rates = sorted({base["learning_rate"], 0.003, 0.005, 0.01, 0.02, 0.05})
    n_estimators = sorted({base["n_estimators"], 100, 200, 400, 700})
    max_depths = sorted({base["max_depth"], 2, 3, 4, 5, 6})
    subsamples = sorted({base["subsample"], 0.65, 0.8, 0.95})

    grid = [
        {
            "learning_rate": learning_rate,
            "n_estimators": n_estimator,
            "max_depth": max_depth,
            "subsample": subsample,
        }
        for learning_rate, n_estimator, max_depth, subsample in itertools.product(
            learning_rates, n_estimators, max_depths, subsamples
        )
    ]

    # Prefer compact, conservative candidates first; they tend to generalize
    # better on noisy market data and make short runs useful.
    grid.sort(key=lambda p: (p["max_depth"], p["learning_rate"], p["n_estimators"], -p["subsample"]))
    return grid[: max(1, trials)]


def predict_positive_probability(model: AlphaShieldPredictorModel, features: pd.DataFrame) -> np.ndarray:
    probabilities = model.predict_proba(features)
    if probabilities.ndim == 2 and probabilities.shape[1] > 1:
        return probabilities[:, 1]
    return probabilities.reshape(-1)


def classification_metrics(true: np.ndarray, probabilities: np.ndarray, threshold: float) -> Dict[str, float | None]:
    predictions = (probabilities >= threshold).astype(int)
    metrics: Dict[str, float | None] = {
        "accuracy": float(accuracy_score(true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(true, predictions)),
        "precision": float(precision_score(true, predictions, zero_division=0)),
        "recall": float(recall_score(true, predictions, zero_division=0)),
        "f1": float(f1_score(true, predictions, zero_division=0)),
        "positive_prediction_rate": float(predictions.mean()),
        "positive_label_rate": float(true.mean()),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(true, probabilities))
    except ValueError:
        metrics["roc_auc"] = None
    return metrics


def choose_threshold(true: np.ndarray, probabilities: np.ndarray, metric: str) -> Tuple[float, Dict[str, float | None]]:
    # Avoid extreme thresholds that make the model never trade or always trade.
    thresholds = np.linspace(0.20, 0.80, 121)
    best_threshold = 0.50
    best_metrics = classification_metrics(true, probabilities, best_threshold)
    best_score = float(best_metrics.get(metric) or 0.0)

    for threshold in thresholds:
        metrics = classification_metrics(true, probabilities, float(threshold))
        score = float(metrics.get(metric) or 0.0)
        if score > best_score:
            best_threshold = float(threshold)
            best_metrics = metrics
            best_score = score
    return best_threshold, best_metrics


def evaluate_candidate(
    dataset: TimeSeriesDataset,
    params: Dict[str, Any],
    initial_train_size: int,
    validation_size: int,
    metric: str,
    random_state: int,
) -> TrialResult:
    all_true: List[np.ndarray] = []
    all_probabilities: List[np.ndarray] = []
    fold_count = 0

    for fold in dataset.walk_forward_splits(initial_train_size, validation_size):
        model = AlphaShieldPredictorModel(random_state=random_state, **params)
        model.fit(fold.train_features, fold.train_target)
        probabilities = predict_positive_probability(model, fold.validation_features)
        all_true.append(fold.validation_target.to_numpy(dtype=int))
        all_probabilities.append(probabilities)
        fold_count += 1

    if fold_count == 0:
        raise ValueError("No walk-forward folds were produced")

    true = np.concatenate(all_true)
    probabilities = np.concatenate(all_probabilities)
    threshold, metrics = choose_threshold(true, probabilities, metric)
    score = float(metrics.get(metric) or 0.0)

    return TrialResult(
        params=params,
        threshold=threshold,
        score=score,
        accuracy=float(metrics["accuracy"] or 0.0),
        balanced_accuracy=float(metrics["balanced_accuracy"] or 0.0),
        precision=float(metrics["precision"] or 0.0),
        recall=float(metrics["recall"] or 0.0),
        f1=float(metrics["f1"] or 0.0),
        roc_auc=None if metrics["roc_auc"] is None else float(metrics["roc_auc"]),
        positive_prediction_rate=float(metrics["positive_prediction_rate"] or 0.0),
        positive_label_rate=float(metrics["positive_label_rate"] or 0.0),
        folds=fold_count,
    )


def tune_model(
    matrix: pd.DataFrame,
    config: Dict[str, Any],
    initial_train_size: int,
    validation_size: int,
    metric: str,
    trials: int,
    random_state: int,
) -> Tuple[TrialResult, List[TrialResult]]:
    dataset = TimeSeriesDataset(matrix)
    results: List[TrialResult] = []

    for index, params in enumerate(candidate_grid(config, trials), start=1):
        result = evaluate_candidate(
            dataset=dataset,
            params=params,
            initial_train_size=initial_train_size,
            validation_size=validation_size,
            metric=metric,
            random_state=random_state,
        )
        results.append(result)
        print(
            f"[{index:03d}/{trials:03d}] {metric}={result.score:.4f} "
            f"f1={result.f1:.4f} bal_acc={result.balanced_accuracy:.4f} "
            f"threshold={result.threshold:.3f} params={result.params}"
        )

    results.sort(key=lambda result: (result.score, result.balanced_accuracy, result.f1), reverse=True)
    return results[0], results


def train_final_model(
    matrix: pd.DataFrame,
    params: Dict[str, Any],
    random_state: int,
) -> Tuple[AlphaShieldPredictorModel, StandardScaler, pd.DataFrame, pd.Series]:
    dataset = TimeSeriesDataset(matrix)
    features = matrix[dataset.feature_columns]
    target = matrix[dataset.target_column].astype(int)

    scaler = StandardScaler()
    scaled_features = pd.DataFrame(
        scaler.fit_transform(features),
        index=features.index,
        columns=dataset.feature_columns,
    )

    model = AlphaShieldPredictorModel(random_state=random_state, **params)
    model.fit(scaled_features, target)
    return model, scaler, scaled_features, target


def save_outputs(
    model: AlphaShieldPredictorModel,
    scaler: StandardScaler,
    matrix: pd.DataFrame,
    best: TrialResult,
    all_results: List[TrialResult],
    args: argparse.Namespace,
    model_path: Path,
    bundle_path: Path,
    metrics_path: Path,
) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    # Compatible with AlphaShieldPredictorModel.load_checkpoint().
    model.save_checkpoint(model_path)

    dataset = TimeSeriesDataset(matrix)
    bundle = {
        "model": model.model,
        "scaler": scaler,
        "feature_columns": dataset.feature_columns,
        "target_column": dataset.target_column,
        "threshold": best.threshold,
        "best_params": best.params,
        "feature_metadata": model.feature_metadata,
        "training_args": vars(args),
    }
    with bundle_path.open("wb") as handle:
        pickle.dump(bundle, handle)

    metrics = {
        "best": asdict(best),
        "top_trials": [asdict(result) for result in all_results[:10]],
        "rows": int(len(matrix)),
        "features": dataset.feature_columns,
        "target_distribution": matrix[dataset.target_column].value_counts(normalize=True).sort_index().to_dict(),
    }
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, default=str)


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Accuracy-first AlphaShield trainer")
    parser.add_argument("--data", type=Path, required=True, help="CSV with timestamp/open/high/low/close/volume columns")
    parser.add_argument("--sentiment", type=Path, default=None, help="Optional CSV with timestamp/sentiment_score columns")
    parser.add_argument("--symbol", type=str, default=None, help="Optional symbol filter when the data CSV contains multiple symbols")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to settings.yaml")
    parser.add_argument("--model-out", type=Path, default=DEFAULT_MODEL_PATH, help="Compatible model checkpoint output path")
    parser.add_argument("--bundle-out", type=Path, default=DEFAULT_BUNDLE_PATH, help="Full model+scaler bundle output path")
    parser.add_argument("--metrics-out", type=Path, default=DEFAULT_METRICS_PATH, help="Training metrics JSON output path")
    parser.add_argument("--target-horizon", type=int, default=None, help="Future periods used to create the label")
    parser.add_argument("--target-threshold", type=float, default=None, help="Minimum future return required for class 1")
    parser.add_argument("--initial-train-size", type=int, default=None, help="Rows in the first expanding training fold")
    parser.add_argument("--validation-size", type=int, default=None, help="Rows in each walk-forward validation fold")
    parser.add_argument("--metric", choices=sorted(SUPPORTED_METRICS), default="f1", help="Metric to optimize")
    parser.add_argument("--trials", type=int, default=30, help="Number of hyperparameter candidates to evaluate")
    parser.add_argument("--random-state", type=int, default=42, help="Reproducible model seed")
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    if args.trials < 1:
        raise ValueError("--trials must be at least 1")

    config = load_config(args.config)
    system_cfg = config.get("system", {}) if config else {}
    target_cfg = config.get("target", {}) if config else {}

    target_horizon = args.target_horizon
    if target_horizon is None:
        target_horizon = int(system_cfg.get("target_variable_horizon_periods", 1))

    target_threshold = args.target_threshold
    if target_threshold is None:
        target_threshold = float(target_cfg.get("positive_return_threshold", 0.0))

    raw = read_market_data(args.data, symbol=args.symbol)
    technical_preview = make_technical_features(raw)
    sentiment = read_sentiment_data(args.sentiment, technical_preview.index)
    matrix = build_matrix(
        raw=raw,
        sentiment=sentiment,
        target_horizon=target_horizon,
        target_threshold=target_threshold,
    )
    initial_train_size, validation_size = split_plan(len(matrix), args.initial_train_size, args.validation_size)

    print(f"Engineered rows: {len(matrix)}")
    print(f"Feature columns: {len(TimeSeriesDataset(matrix).feature_columns)}")
    print(f"Target positive rate: {matrix['target_direction'].mean():.4f}")
    print(f"Walk-forward split: initial_train_size={initial_train_size}, validation_size={validation_size}")
    print(f"Optimizing metric: {args.metric}")

    best, all_results = tune_model(
        matrix=matrix,
        config=config,
        initial_train_size=initial_train_size,
        validation_size=validation_size,
        metric=args.metric,
        trials=args.trials,
        random_state=args.random_state,
    )

    final_model, scaler, _, _ = train_final_model(
        matrix=matrix,
        params=best.params,
        random_state=args.random_state,
    )
    save_outputs(
        model=final_model,
        scaler=scaler,
        matrix=matrix,
        best=best,
        all_results=all_results,
        args=args,
        model_path=args.model_out,
        bundle_path=args.bundle_out,
        metrics_path=args.metrics_out,
    )

    print("\nBest walk-forward result:")
    print(json.dumps(asdict(best), indent=2, default=str))
    print(f"\nSaved compatible checkpoint: {args.model_out}")
    print(f"Saved full training bundle: {args.bundle_out}")
    print(f"Saved metrics: {args.metrics_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
