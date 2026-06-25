from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_pipeline import TechnicalIndicatorEngine
from src.predictor import FeatureEngineeringMatrix, TimeSeriesDataset
from src.sentiment_engine import NewsHeadline, SentimentScorerEngine


@pytest.fixture
def synthetic_market_bundle():
    rng = np.random.default_rng(7)
    rows = 500
    index = pd.date_range("2024-01-01", periods=rows, freq="D", tz="UTC")
    drift = 0.0004
    volatility = 0.015
    shocks = rng.normal(drift, volatility, rows)
    close = 100.0 * np.exp(np.cumsum(shocks))
    open_ = close * (1.0 + rng.normal(0.0, 0.002, rows))
    high = np.maximum(open_, close) * (1.0 + rng.uniform(0.001, 0.01, rows))
    low = np.minimum(open_, close) * (1.0 - rng.uniform(0.001, 0.01, rows))
    volume = rng.integers(100_000, 500_000, rows)
    prices = pd.DataFrame(
        {
            "timestamp": index,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )
    headlines = np.where(
        shocks >= 0,
        "Apple breakthrough profit surge lifts market",
        "Tesla lawsuit deficit warning pressures market",
    )
    sentiment_records = [NewsHeadline(timestamp, headline, "synthetic") for timestamp, headline in zip(index, headlines)]
    sentiment = SentimentScorerEngine().to_frame(sentiment_records)
    return prices, sentiment


def test_zero_lookahead_bias(synthetic_market_bundle):
    prices, sentiment = synthetic_market_bundle
    engine = TechnicalIndicatorEngine()
    baseline = engine.transform(prices)
    modified_prices = prices.copy(deep=True)
    t = 300
    modified_prices.loc[t, "close"] = modified_prices.loc[t, "close"] * 10.0
    modified_prices.loc[t, "high"] = max(modified_prices.loc[t, "high"], modified_prices.loc[t, "close"])
    modified = engine.transform(modified_prices)

    prior_index = baseline.index[:t]
    pd.testing.assert_frame_equal(baseline.loc[prior_index], modified.loc[prior_index])

    builder = FeatureEngineeringMatrix()
    baseline_matrix = builder.build(baseline, sentiment)
    modified_matrix = builder.build(modified, sentiment)
    common_prior = baseline_matrix.index.intersection(modified_matrix.index[modified_matrix.index < baseline.index[t]])
    pd.testing.assert_frame_equal(baseline_matrix.loc[common_prior], modified_matrix.loc[common_prior])

    dataset = TimeSeriesDataset(baseline_matrix)
    fold = dataset.first_split(initial_train_size=250, validation_size=50)
    assert fold.train_features.index.max() < fold.validation_features.index.min()


def test_rsi_boundary(synthetic_market_bundle):
    prices, _ = synthetic_market_bundle
    indicators = TechnicalIndicatorEngine().transform(prices)
    assert indicators["rsi_14"].between(0.0, 100.0).all()
    assert not indicators.select_dtypes(include=[float, int]).isna().any().any()


def test_synthetic_integration_pipeline(synthetic_market_bundle):
    prices, sentiment = synthetic_market_bundle
    indicators = TechnicalIndicatorEngine().transform(prices)
    matrix = FeatureEngineeringMatrix().build(indicators, sentiment)
    dataset = TimeSeriesDataset(matrix)
    fold = dataset.first_split(initial_train_size=300, validation_size=100)
    assert len(fold.train_features) == 300
    assert len(fold.validation_features) == 100
    assert set(fold.train_target.unique()).issubset({0, 1})
    assert np.isfinite(fold.train_features.to_numpy()).all()
    assert np.isfinite(fold.validation_features.to_numpy()).all()
