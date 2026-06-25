# AlphaShield — Market Prediction Framework

> **Disclaimer:** This project is for educational and research purposes only. Market prediction is inherently uncertain and no model output constitutes financial advice. Past simulated performance does not guarantee future results. Use at your own risk.

---

## Overview

AlphaShield is a lightweight Python framework for equity price-direction prediction.  It is designed around three principles:

1. **Leakage safety** — every feature and target is constructed so that only information available at the decision point is used.
2. **Transaction-cost awareness** — target labels can require a minimum future return before a bar is counted as a tradeable signal.
3. **Walk-forward validation** — model evaluation follows a rolling train/validation scheme that mirrors real deployment.

---

## Project Structure

```
AlphaShield/
├── config/
│   └── settings.yaml          # All configurable parameters
├── src/
│   ├── data_pipeline.py       # Market data ingestion & technical indicator engine
│   ├── predictor.py           # Feature matrix, walk-forward dataset, model, evaluator
│   ├── backtester.py          # Trade simulation & performance analytics
│   └── sentiment_engine.py   # News headline ingestion & sentiment scoring
├── tests/
│   └── test_core.py           # Full test suite (14 tests)
├── requirements.txt
└── run_all.py                 # Environment check + test runner
```

---

## Target Definition

The prediction target is a binary label indicating whether the future return over the next `target_variable_horizon_periods` bars exceeds `positive_return_threshold`.

```
future_return = future_close / close - 1
target_direction = 1  if  future_return > positive_return_threshold
                   0  otherwise
```

Setting `positive_return_threshold: 0.0` (the default) reproduces a simple price-direction label (up vs. flat/down).  Setting it to e.g. `0.002` requires the price to move at least 0.2 % upward before being labelled bullish — roughly covering a typical round-trip transaction cost.

The target is **leakage-safe**: the last `target_variable_horizon_periods` rows (whose future close is unknown) are always trimmed from the matrix.

---

## Feature Engineering

`TechnicalIndicatorEngine` computes the following features.  All derived features are shifted forward by **one period** before being added to the feature matrix, so only information known at the start of bar *t* is used when predicting bar *t*.

| Column | Description |
|--------|-------------|
| `open`, `high`, `low`, `close`, `volume` | Raw OHLCV (available for downstream backtesting) |
| `ema_20`, `ema_50` | Exponential moving averages (20- and 50-period) |
| `rsi_14` | 14-period RSI, clipped to [0, 100] |
| `atr_14` | 14-period Average True Range |
| `return_1` | 1-bar percentage return |
| `return_3`, `return_5`, `return_10`, `return_20` | Multi-window percentage returns |
| `rolling_vol_20` | 20-period rolling volatility (std of 1-bar returns) |
| `close_to_ema20` | `close / ema_20 − 1` (distance from fast EMA) |
| `close_to_ema50` | `close / ema_50 − 1` (distance from slow EMA) |
| `atr_pct` | `atr_14 / close` (ATR as a fraction of price) |
| `volume_zscore` | Rolling 20-period z-score of volume |
| `sentiment_score` | As-of aligned sentiment score in [−1, 1] |

Raw price levels (`open`, `high`, `low`, `close`, `volume`) are included in the output DataFrame for downstream backtesting compatibility but are **not** shifted and therefore not part of the feature scaler input in `TimeSeriesDataset` (which excludes the `target_direction` column only — models should be trained to exclude raw OHLCV columns or rely on the normalised alternatives above).

---

## Sentiment Alignment

Sentiment headlines are aligned to market bars using a **backward as-of join** (`pd.merge_asof` with `direction="backward"`).  Each bar receives the most recent sentiment score published *at or before* its timestamp, so no future news can influence past rows.  If no sentiment has been published yet, a neutral score of `0.0` is used.

---

## Walk-Forward Validation

`TimeSeriesDataset.walk_forward_splits(initial_train_size, validation_size)` yields non-overlapping folds in chronological order.  Within each fold:

- `StandardScaler` is **fit only on the training window** and applied to validation.
- Training and validation windows are checked for temporal overlap; a `DataLeakageException` is raised if they overlap.

`WalkForwardEvaluator.evaluate(...)` runs a full model train/predict cycle across all folds and returns:

```python
{
    "fold_results":      pd.DataFrame,   # per-fold accuracy, F1, precision, recall, positive_rate
    "mean_accuracy":     float,
    "mean_f1":           float,
    "mean_precision":    float,
    "mean_recall":       float,
    "mean_positive_rate": float,
    "n_folds":           int,
}
```

---

## Configuration (`settings.yaml`)

```yaml
system:
  data_ingestion_lookback_limit: 200
  target_variable_horizon_periods: 1

target:
  positive_return_threshold: 0.0   # 0.0 = direction; 0.002 = require 0.2% move

backtest:
  initial_cash_balance: 100000.0
  slippage_spread_percentage_minimum: 0.0001
  slippage_spread_percentage_maximum: 0.0005
  execution_fee_scalar: 0.0010

model:
  learning_rate: 0.01
  n_estimators: 100
  max_depth: 5
  subsample: 0.8
```

Load and pass model config with:

```python
import yaml
from AlphaShield.src.predictor import AlphaShieldPredictorModel

config = yaml.safe_load(open("AlphaShield/config/settings.yaml"))
model = AlphaShieldPredictorModel.from_config(config)
```

---

## Running the Tests

```bash
cd AlphaShield
pip install -r requirements.txt
python -m pytest -v
```

Expected output: **14 passed**.

---

## Performance Metrics

`PerformanceAnalyticsEngine.summarize(equity_curve)` returns:

| Metric | Description |
|--------|-------------|
| `annualized_sharpe_ratio` | √252 × mean(returns) / std(returns) |
| `maximum_drawdown` | Peak-to-trough decline as a fraction |
| `total_return` | (final equity − initial equity) / initial equity |
| `profit_factor` | Gross gains / gross losses from period returns (> 1 is profitable) |

---

## Limitations

- The model is an XGBoost classifier trained on historical bars.  Financial markets are non-stationary, and any model trained on past data may not generalise to future regimes.
- Simulated slippage and fees are simplified approximations.  Real execution costs depend on liquidity, bid-ask spread, and market impact.
- Sentiment scoring is a simple lexicon-based approach.  It does not capture nuance, context, or market-moving events reliably.
- This framework does not constitute a complete trading system and should not be used for live trading without extensive additional validation.
