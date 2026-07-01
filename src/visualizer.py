from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


sns.set_style("whitegrid")


class Visualizer:
    @staticmethod
    def efficient_frontier(frontier_df: pd.DataFrame,
                           max_sharpe_w: np.ndarray, mu, cov, rf: float,
                           tickers: list, path: str):
        fig, ax = plt.subplots(figsize=(11, 7))
        sc = ax.scatter(frontier_df["vol"], frontier_df["return"],
                        c=frontier_df["sharpe"], cmap="viridis",
                        s=30, label="Efficient Frontier")
        plt.colorbar(sc, ax=ax, label="Sharpe Ratio")
        vols = np.sqrt(np.diag(cov))
        ax.scatter(vols, mu, color="red", s=80, marker="X", label="Assets")
        for i, t in enumerate(tickers):
            ax.annotate(t, (vols[i], mu[i]), fontsize=9,
                        xytext=(5, 5), textcoords="offset points")
        port_ret = max_sharpe_w @ mu
        port_vol = np.sqrt(max_sharpe_w @ cov @ max_sharpe_w)
        ax.scatter(port_vol, port_ret, color="gold", edgecolor="black",
                   s=250, marker="*", label="Max Sharpe", zorder=5)
        cal_x = np.linspace(0, port_vol * 1.5, 50)
        slope = (port_ret - rf) / port_vol
        ax.plot(cal_x, rf + slope * cal_x, "--", color="orange",
                label="Capital Allocation Line")
        ax.set_xlabel("Annualized Volatility"); ax.set_ylabel("Expected Return")
        ax.set_title("Markowitz Efficient Frontier")
        ax.legend()
        fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)

    @staticmethod
    def correlation_heatmap(returns: pd.DataFrame, path: str):
        fig, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(returns.corr(), annot=True, fmt=".2f",
                    cmap="coolwarm", center=0, ax=ax)
        ax.set_title("Asset Return Correlation")
        fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)

    @staticmethod
    def weights_pie(weights: np.ndarray, tickers: list, path: str,
                    title: str = "Optimal Portfolio Weights"):
        fig, ax = plt.subplots(figsize=(9, 9))
        non_zero = [(t, w) for t, w in zip(tickers, weights) if w > 1e-3]
        if not non_zero:
            return
        labels, sizes = zip(*non_zero)
        ax.pie(sizes, labels=labels, autopct="%1.1f%%",
               startangle=90, colors=sns.color_palette("Set2"))
        ax.set_title(title)
        fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)

    @staticmethod
    def rolling_metrics(rolling_sharpe: pd.DataFrame, rolling_beta: pd.DataFrame,
                        path: str):
        fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
        rolling_sharpe.plot(ax=axes[0], colormap="viridis", linewidth=1.2)
        axes[0].set_title("Rolling 60-Day Sharpe Ratio")
        axes[0].set_ylabel("Sharpe"); axes[0].axhline(0, color="black", linewidth=0.5)
        rolling_beta.plot(ax=axes[1], colormap="plasma", linewidth=1.2)
        axes[1].set_title("Rolling 60-Day Beta vs S&P 500")
        axes[1].set_ylabel("Beta"); axes[1].axhline(1, color="red", linestyle="--", linewidth=0.8)
        fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)

    @staticmethod
    def drawdown(equity: pd.DataFrame, path: str):
        fig, ax = plt.subplots(figsize=(12, 5))
        for col in equity.columns:
            dd = equity[col] / equity[col].cummax() - 1
            ax.plot(dd.index, dd.values, label=col, linewidth=1.5)
        ax.set_title("Drawdown")
        ax.set_ylabel("Drawdown"); ax.legend()
        fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)

    @staticmethod
    def var_distribution(returns: pd.Series, var_val: float, cvar_val: float,
                         path: str):
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.hist(returns, bins=80, color="steelblue", alpha=0.7,
                density=True, edgecolor="black", linewidth=0.4)
        ax.axvline(-var_val, color="red", linestyle="--", linewidth=2,
                   label=f"VaR (95%) = {var_val:.2%}")
        ax.axvline(-cvar_val, color="darkred", linestyle="--", linewidth=2,
                   label=f"CVaR (95%) = {cvar_val:.2%}")
        ax.set_xlabel("Daily Return"); ax.set_ylabel("Density")
        ax.set_title("Portfolio Return Distribution & Tail Risk")
        ax.legend()
        fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)

    @staticmethod
    def backtest(equity: pd.DataFrame, path: str):
        fig, ax = plt.subplots(figsize=(12, 6))
        for col in equity.columns:
            ax.plot(equity.index, equity[col], label=col, linewidth=2)
        ax.set_yscale("log")
        ax.set_title("Backtest: Strategy vs Benchmark (log scale)")
        ax.set_ylabel("Cumulative Growth of $1"); ax.legend()
        fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)