from __future__ import annotations
import numpy as np
import pandas as pd
from .optimizer import MarkowitzOptimizer
from .analytics import Analytics


class Backtester:
    def __init__(self, prices: pd.DataFrame,
                 benchmark_prices: pd.Series,
                 config) -> None:
        self.prices = prices
        self.bench = benchmark_prices
        self.cfg = config

    def run(self, strategy: str = "max_sharpe") -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Walk-forward rolling historical validation backtest."""
        freq = self.cfg.rebalance_freq
        if freq == "M":
            freq = "ME"
        rebal_dates = (self.prices.resample(freq)
                       .first().index)
        rebal_dates = rebal_dates[rebal_dates >= self.prices.index[0] +
                                  pd.Timedelta(days=self.cfg.train_window_days)]

        portfolio_value = pd.Series(index=self.prices.index, dtype=float)
        portfolio_value.iloc[0] = 1.0
        weights_history = []

        for i, date in enumerate(rebal_dates):
            train_start = date - pd.Timedelta(days=self.cfg.train_window_days)
            train = self.prices.loc[train_start:date].iloc[:-1]
            if len(train) < 60:
                continue

            rets = train.pct_change().dropna()
            a = Analytics(rets, self.cfg.risk_free_rate,
                          self.cfg.trading_days)
            mu = a.mean_returns()
            cov_df, _ = a.covariance("ledoit_wolf")

            opt = MarkowitzOptimizer(
                mu, cov_df, self.cfg.sectors, self.cfg.sector_caps,
                risk_free_rate=self.cfg.risk_free_rate,
                long_only=self.cfg.long_only,
            )

            if strategy == "max_sharpe":
                w = opt.max_sharpe()
            elif strategy == "min_variance":
                w = opt.min_variance()
            else:
                w = np.ones(len(self.cfg.tickers)) / len(self.cfg.tickers)
            weights_history.append((date, w))

            end = (rebal_dates[i + 1]
                   if i + 1 < len(rebal_dates)
                   else self.prices.index[-1])
            hold_rets = self.prices.loc[date:end].pct_change().dropna()
            if hold_rets.empty:
                continue
            port_rets = hold_rets.values @ w
            if i == 0:
                portfolio_value.loc[hold_rets.index] = np.cumprod(1 + port_rets)
            else:
                last_val = portfolio_value.loc[:date].iloc[-1]
                portfolio_value.loc[hold_rets.index] = (
                    last_val * np.cumprod(1 + port_rets)
                )

        portfolio_value = portfolio_value.ffill().dropna()
        bench_ret = self.bench.pct_change().dropna()
        benchmark_value = (1 + bench_ret).cumprod()
        benchmark_value = benchmark_value.reindex(portfolio_value.index).ffill()
        benchmark_value.iloc[0] = 1.0
        
        return pd.DataFrame({
            "portfolio": portfolio_value,
            "benchmark": benchmark_value,
        }), pd.DataFrame(weights_history,
                         columns=["date", "weights"]).set_index("date")

    @staticmethod
    def performance(equity: pd.Series, rf: float = 0.04,
                    periods: int = 252) -> dict:
        rets = equity.pct_change().dropna()
        years = len(rets) / periods
        cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1
        vol = rets.std() * np.sqrt(periods)
        sharpe = (cagr - rf) / vol if vol > 0 else 0.0
        drawdown = equity / equity.cummax() - 1
        mdd = drawdown.min()
        return {
            "CAGR": cagr, "Vol": vol, "Sharpe": sharpe,
            "Max Drawdown": mdd, "Total Return":
                equity.iloc[-1] / equity.iloc[0] - 1,
        }