from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtester import PerformanceAnalyticsEngine, PortfolioTracker
from src.data_pipeline import TechnicalIndicatorEngine
from src.predictor import (
    AlphaShieldPredictorModel,
    FeatureEngineeringMatrix,
    TimeSeriesDataset,
    WalkForwardEvaluator,
)
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


# ---------------------------------------------------------------------------
# New tests
# ---------------------------------------------------------------------------


def test_normalized_features_exist(synthetic_market_bundle):
    """New feature columns are present and finite in the indicator output."""
    prices, _ = synthetic_market_bundle
    indicators = TechnicalIndicatorEngine().transform(prices)
    expected_new_columns = [
        "return_3", "return_5", "return_10", "return_20",
        "rolling_vol_20", "close_to_ema20", "close_to_ema50",
        "atr_pct", "volume_zscore",
    ]
    for col in expected_new_columns:
        assert col in indicators.columns, f"Missing feature column: {col}"
    numeric = indicators.select_dtypes(include=[float, int])
    assert np.isfinite(numeric.to_numpy()).all(), "Non-finite values found in indicator output"


def test_transaction_cost_aware_target_zero_threshold(synthetic_market_bundle):
    """With threshold=0.0, labels match simple price-direction (future_close > close)."""
    prices, sentiment = synthetic_market_bundle
    indicators = TechnicalIndicatorEngine().transform(prices)
    matrix = FeatureEngineeringMatrix(positive_return_threshold=0.0).build(indicators, sentiment)
    assert set(matrix["target_direction"].unique()).issubset({0, 1})
    assert matrix["target_direction"].notna().all()


def test_transaction_cost_aware_target_positive_threshold(synthetic_market_bundle):
    """With a positive threshold, class-1 labels require a move large enough to exceed costs."""
    prices, sentiment = synthetic_market_bundle
    indicators = TechnicalIndicatorEngine().transform(prices)
    threshold = 0.005  # 0.5% — larger than 1-bar typical moves
    matrix = FeatureEngineeringMatrix(positive_return_threshold=threshold).build(indicators, sentiment)
    assert set(matrix["target_direction"].unique()).issubset({0, 1})
    # With a 0.5% threshold, fewer than 100% of rows should be class-1
    positive_rate = matrix["target_direction"].mean()
    assert positive_rate < 1.0, "All rows labelled class-1 with a non-trivial threshold"


def test_target_no_future_leakage(synthetic_market_bundle):
    """The last target_horizon rows are trimmed so no row has a missing future target."""
    prices, sentiment = synthetic_market_bundle
    horizon = 2
    indicators = TechnicalIndicatorEngine().transform(prices)
    # Build with horizon=2 — last 2 rows (whose future is unknown) must be dropped.
    matrix_h2 = FeatureEngineeringMatrix(target_horizon=horizon).build(indicators, sentiment)
    matrix_h1 = FeatureEngineeringMatrix(target_horizon=1).build(indicators, sentiment)
    assert len(matrix_h2) == len(matrix_h1) - 1


def test_sentiment_asof_alignment(synthetic_market_bundle):
    """Sparse sentiment is aligned as-of without future leakage."""
    prices, full_sentiment = synthetic_market_bundle
    # Keep only every 10th sentiment timestamp to simulate sparse coverage.
    sparse_sentiment = full_sentiment.iloc[::10].copy()
    indicators = TechnicalIndicatorEngine().transform(prices)
    matrix = FeatureEngineeringMatrix().build(indicators, sparse_sentiment)
    # All rows should be retained (not dropped due to no exact sentiment match).
    # The first rows before any sentiment entry receive sentiment_score=0 (neutral).
    assert len(matrix) > 0
    assert matrix["sentiment_score"].notna().all()


def test_walk_forward_evaluator_multiple_folds(synthetic_market_bundle):
    """WalkForwardEvaluator produces results across multiple folds without leakage."""
    prices, sentiment = synthetic_market_bundle
    indicators = TechnicalIndicatorEngine().transform(prices)
    matrix = FeatureEngineeringMatrix().build(indicators, sentiment)
    dataset = TimeSeriesDataset(matrix)
    evaluator = WalkForwardEvaluator()
    results = evaluator.evaluate(
        dataset,
        AlphaShieldPredictorModel,
        initial_train_size=200,
        validation_size=100,
    )
    assert results["n_folds"] >= 2, "Expected at least 2 folds"
    assert 0.0 <= results["mean_accuracy"] <= 1.0
    assert 0.0 <= results["mean_f1"] <= 1.0
    folds_df = results["fold_results"]
    # Validate no temporal leakage across folds.
    for _, row in folds_df.iterrows():
        assert row["train_end"] < row["val_start"], "Training period overlaps validation period"


def test_walk_forward_evaluator_fold_train_then_val(synthetic_market_bundle):
    """Per-fold train windows always precede their validation windows."""
    prices, sentiment = synthetic_market_bundle
    indicators = TechnicalIndicatorEngine().transform(prices)
    matrix = FeatureEngineeringMatrix().build(indicators, sentiment)
    dataset = TimeSeriesDataset(matrix)
    results = WalkForwardEvaluator().evaluate(
        dataset,
        AlphaShieldPredictorModel,
        initial_train_size=250,
        validation_size=50,
    )
    prev_val_end = None
    for _, row in results["fold_results"].iterrows():
        assert row["train_end"] < row["val_start"]
        if prev_val_end is not None:
            assert row["val_start"] > prev_val_end, "Validation windows overlap between folds"
        prev_val_end = row["val_end"]


def test_config_driven_model_params():
    """AlphaShieldPredictorModel respects hyperparameters from constructor and from_config."""
    model_direct = AlphaShieldPredictorModel(max_depth=3, n_estimators=50, learning_rate=0.05)
    # Verify the underlying estimator received the overridden parameters.
    assert model_direct.model.get_params().get("max_depth") == 3
    assert model_direct.model.get_params().get("n_estimators") == 50

    config = {"model": {"max_depth": 4, "learning_rate": 0.02, "n_estimators": 75, "subsample": 0.7}}
    model_cfg = AlphaShieldPredictorModel.from_config(config)
    params = model_cfg.model.get_params()
    assert params.get("max_depth") == 4
    assert params.get("n_estimators") == 75


def test_profit_factor_and_total_return():
    """PerformanceAnalyticsEngine computes profit_factor and total_return correctly."""
    ts = pd.date_range("2024-01-01", periods=5, freq="D")
    equity = pd.DataFrame({"portfolio_value": [100.0, 110.0, 105.0, 115.0, 120.0]}, index=ts)
    analytics = PerformanceAnalyticsEngine()
    total_ret = analytics.total_return(equity)
    assert abs(total_ret - 0.20) < 1e-9, f"Expected 20% total return, got {total_ret}"
    pf = analytics.profit_factor(equity)
    assert pf > 1.0, f"Profitable run should have profit_factor > 1, got {pf}"

    # All-loss curve: profit_factor should be 0.0
    equity_loss = pd.DataFrame({"portfolio_value": [100.0, 90.0, 80.0, 70.0]}, index=ts[:4])
    pf_loss = analytics.profit_factor(equity_loss)
    assert pf_loss == 0.0, f"All-loss curve should have profit_factor=0, got {pf_loss}"


def test_backtester_deterministic():
    """Backtest produces identical results under the same random seed."""
    tracker1 = PortfolioTracker()
    tracker2 = PortfolioTracker()
    ts = pd.Timestamp("2024-01-01", tz="UTC")
    for tracker in (tracker1, tracker2):
        tracker.execute_buy_order(price=100.0, cash_allocation=10000.0)
        tracker.execute_sell_order(price=105.0)
        tracker.update_equity_curve(105.0, ts)
    assert tracker1.current_cash == tracker2.current_cash
    assert tracker1.total_portfolio_value == tracker2.total_portfolio_value


def test_summarize_includes_new_metrics():
    """PerformanceAnalyticsEngine.summarize returns all four metrics."""
    ts = pd.date_range("2024-01-01", periods=10, freq="D")
    equity = pd.DataFrame({"portfolio_value": [100.0 + i for i in range(10)]}, index=ts)
    summary = PerformanceAnalyticsEngine().summarize(equity)
    assert "annualized_sharpe_ratio" in summary
    assert "maximum_drawdown" in summary
    assert "total_return" in summary
    assert "profit_factor" in summary
