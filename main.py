
import os
import numpy as np
import pandas as pd
from config import EngineConfig
from src import (DataLoader, Analytics, MarkowitzOptimizer,
                 RiskEngine, Backtester, Visualizer)


def main():
    cfg = EngineConfig()
    os.makedirs("outputs", exist_ok=True)

    # ---------------- 1. Data ----------------
    print("→ Loading market data...")
    loader = DataLoader(cfg.tickers, cfg.benchmark, cfg.start, cfg.end)
    prices = loader.fetch()
    asset_prices, bench_prices = loader.split(prices)
    asset_rets = loader.to_returns(asset_prices, log=False)
    bench_rets = loader.to_returns(bench_prices, log=False)

    # ---------------- 2. Analytics ----------------
    print("→ Computing analytics...")
    a = Analytics(asset_rets, cfg.risk_free_rate, cfg.trading_days)
    mu = a.mean_returns()
    cov_df, cov = a.covariance("ledoit_wolf")
    betas = a.beta(bench_rets)
    rolling_sharpe = a.rolling_sharpe(60)
    rolling_beta = a.rolling_beta(bench_rets, 60)

    print("\nAnnualized Sharpe ratios:\n", a.sharpe().round(3))
    print("\nBetas vs S&P 500:\n", betas.round(3))

    # ---------------- 3. Optimization ----------------
    print("→ Solving Markowitz optimization...")
    opt = MarkowitzOptimizer(mu, cov_df, cfg.sectors, cfg.sector_caps,
                             risk_free_rate=cfg.risk_free_rate,
                             long_only=cfg.long_only)
    w_ms = opt.max_sharpe()
    w_mv = opt.min_variance()
    frontier = opt.efficient_frontier(cfg.frontier_points)

    print("\nMax-Sharpe weights:")
    for t, w in zip(cfg.tickers, w_ms):
        if w > 1e-3:
            print(f"  {t:6s}  {w:6.2%}")
    print(f"  Expected return: {(w_ms @ mu):.2%}")
    print(f"  Volatility:       {np.sqrt(w_ms @ cov @ w_ms):.2%}")
    print(f"  Sharpe:           {((w_ms @ mu) - cfg.risk_free_rate) / np.sqrt(w_ms @ cov @ w_ms):.3f}")

    # ---------------- 4. Visualizations (analytics) ----------------
    print("→ Rendering charts...")
    viz = Visualizer()
    viz.efficient_frontier(frontier, w_ms, mu.values, cov,
                           cfg.risk_free_rate, cfg.tickers,
                           "outputs/efficient_frontier.png")
    viz.correlation_heatmap(asset_rets, "outputs/correlation_heatmap.png")
    viz.weights_pie(w_ms, cfg.tickers, "outputs/portfolio_weights.png",
                    title="Max-Sharpe Portfolio Weights")
    viz.rolling_metrics(rolling_sharpe, rolling_beta,
                        "outputs/rolling_metrics.png")

    # ---------------- 5. Risk ----------------
    print("→ Computing risk metrics...")
    risk = RiskEngine(asset_rets, w_ms, cfg.var_confidence,
                      cfg.var_horizon_days, cfg.trading_days)
    metrics = {
        "VaR (Historical, 95%)":   risk.var_historical(),
        "VaR (Parametric Normal)": risk.var_parametric_normal(),
        "VaR (Parametric t, df=5)":risk.var_parametric_t(df=5),
        "VaR (Monte Carlo)":       risk.var_monte_carlo(cfg.mc_simulations),
        "CVaR (Historical, 95%)":  risk.cvar_historical(),
    }
    print("\nRisk Metrics (daily):")
    for k, v in metrics.items():
        print(f"  {k:28s} {v:.3%}")

    stressed = risk.stress_test({
        "COVID Crash (Mar 2020)": -0.04,
        "Rate Shock (-100bps)":   -0.02,
        "Flash Crash":            -0.06,
    })
    print("\nStressed VaR (95%):\n", stressed.round(4))

    viz.var_distribution(
        pd.Series(risk.port_rets, index=asset_rets.index),
        metrics["VaR (Historical, 95%)"],
        metrics["CVaR (Historical, 95%)"],
        "outputs/var_distribution.png",
    )

    # ---------------- 6. Backtest ----------------
    print("→ Running walk-forward backtest...")
    bt = Backtester(asset_prices, bench_prices, cfg)
    equity, weights_hist = bt.run("max_sharpe")

    perf = {
        s: Backtester.performance(equity[s], cfg.risk_free_rate,
                                  cfg.trading_days)
        for col, s in enumerate(equity.columns)
    }
    perf_df = pd.DataFrame(perf).T.round(4)
    print("\nBacktest Performance:\n", perf_df)

    viz.backtest(equity, "outputs/backtest_results.png")
    viz.drawdown(equity, "outputs/drawdown.png")

    print("\n✓ Done. Outputs in ./outputs/")


if __name__ == "__main__":
    main()