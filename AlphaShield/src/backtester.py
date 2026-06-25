"""Event-driven execution simulation, portfolio accounting, and performance analytics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ExecutionResult:
    side: str
    requested_price: float
    executed_price: float
    shares: float
    gross_value: float
    fee: float
    timestamp: pd.Timestamp


class OrderExecutionSimulator:
    """Applies slippage and fee drag to every simulated transaction."""

    def __init__(
        self,
        fee_scalar: float = 0.0010,
        slippage_min: float = 0.0001,
        slippage_max: float = 0.0005,
        random_seed: Optional[int] = 42,
    ) -> None:
        if not 0 <= slippage_min <= slippage_max:
            raise ValueError("slippage bounds are invalid")
        if fee_scalar < 0:
            raise ValueError("fee_scalar must be non-negative")
        self.fee_scalar = fee_scalar
        self.slippage_min = slippage_min
        self.slippage_max = slippage_max
        self.rng = np.random.default_rng(random_seed)

    def execute(self, side: str, price: float, shares: float, timestamp: Optional[pd.Timestamp] = None) -> ExecutionResult:
        if side not in {"BUY", "SELL"}:
            raise ValueError("side must be BUY or SELL")
        if price <= 0 or shares <= 0:
            raise ValueError("price and shares must be positive")
        slippage = float(self.rng.uniform(self.slippage_min, self.slippage_max))
        executed_price = price * (1 + slippage) if side == "BUY" else price * (1 - slippage)
        gross_value = executed_price * shares
        fee = gross_value * self.fee_scalar
        return ExecutionResult(side, float(price), float(executed_price), float(shares), float(gross_value), float(fee), timestamp or pd.Timestamp.utcnow())


class PortfolioTracker:
    """Tracks cash, shares, valuation, trade ledger, and equity curve."""

    def __init__(self, initial_cash: float = 100000.0, executor: Optional[OrderExecutionSimulator] = None) -> None:
        if initial_cash <= 0:
            raise ValueError("initial_cash must be positive")
        self.current_cash = float(initial_cash)
        self.active_shares_held = 0.0
        self.total_portfolio_value = float(initial_cash)
        self.executor = executor or OrderExecutionSimulator()
        self._trade_rows: List[Dict[str, float | str | pd.Timestamp]] = []
        self._equity_rows: List[Dict[str, float | pd.Timestamp]] = []

    @property
    def trade_ledger(self) -> pd.DataFrame:
        return pd.DataFrame(self._trade_rows)

    @property
    def equity_curve(self) -> pd.DataFrame:
        if not self._equity_rows:
            return pd.DataFrame(columns=["timestamp", "portfolio_value"]).set_index("timestamp")
        return pd.DataFrame(self._equity_rows).set_index("timestamp").sort_index()

    def execute_buy_order(self, price: float, cash_allocation: float) -> None:
        if cash_allocation <= 0:
            raise ValueError("cash_allocation must be positive")
        spendable_cash = min(float(cash_allocation), self.current_cash)
        conservative_shares = spendable_cash / (price * (1 + self.executor.slippage_max) * (1 + self.executor.fee_scalar))
        if conservative_shares <= 0:
            return
        result = self.executor.execute("BUY", price, conservative_shares)
        total_cost = result.gross_value + result.fee
        if total_cost > self.current_cash:
            affordable = self.current_cash / (result.executed_price * (1 + self.executor.fee_scalar))
            result = self.executor.execute("BUY", price, affordable)
            total_cost = result.gross_value + result.fee
        self.current_cash -= total_cost
        self.active_shares_held += result.shares
        self._record_trade(result)
        self.total_portfolio_value = self.current_cash + self.active_shares_held * result.executed_price

    def execute_sell_order(self, price: float) -> None:
        if self.active_shares_held <= 0:
            return
        result = self.executor.execute("SELL", price, self.active_shares_held)
        net_proceeds = result.gross_value - result.fee
        self.current_cash += net_proceeds
        self.active_shares_held = 0.0
        self._record_trade(result)
        self.total_portfolio_value = self.current_cash

    def update_equity_curve(self, current_price: float, timestamp: pd.Timestamp) -> None:
        if current_price <= 0:
            raise ValueError("current_price must be positive")
        self.total_portfolio_value = self.current_cash + self.active_shares_held * float(current_price)
        self._equity_rows.append({"timestamp": pd.Timestamp(timestamp), "portfolio_value": self.total_portfolio_value})

    def _record_trade(self, result: ExecutionResult) -> None:
        self._trade_rows.append(
            {
                "timestamp": result.timestamp,
                "side": result.side,
                "requested_price": result.requested_price,
                "executed_price": result.executed_price,
                "shares": result.shares,
                "gross_value": result.gross_value,
                "fee": result.fee,
            }
        )


class PerformanceAnalyticsEngine:
    """Computes strategy viability metrics from an equity curve."""

    @staticmethod
    def annualized_sharpe_ratio(equity_curve: pd.DataFrame) -> float:
        if "portfolio_value" not in equity_curve.columns or len(equity_curve) < 2:
            return 0.0
        returns = equity_curve["portfolio_value"].astype(float).pct_change().dropna()
        std = returns.std(ddof=1)
        if returns.empty or std == 0 or np.isnan(std):
            return 0.0
        return float(np.sqrt(252.0) * returns.mean() / std)

    @staticmethod
    def maximum_drawdown(equity_curve: pd.DataFrame) -> float:
        if "portfolio_value" not in equity_curve.columns or equity_curve.empty:
            return 0.0
        values = equity_curve["portfolio_value"].astype(float)
        running_peak = values.cummax()
        drawdown = (running_peak - values) / running_peak
        return float(drawdown.max())

    def summarize(self, equity_curve: pd.DataFrame) -> Dict[str, float]:
        return {
            "annualized_sharpe_ratio": self.annualized_sharpe_ratio(equity_curve),
            "maximum_drawdown": self.maximum_drawdown(equity_curve),
        }
