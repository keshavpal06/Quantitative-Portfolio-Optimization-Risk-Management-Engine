from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, List, Optional, Tuple


class MarkowitzOptimizer:
    def __init__(self, mu: pd.Series, cov: pd.DataFrame,
                 sectors: Dict[str, str],
                 sector_caps: Dict[str, float],
                 risk_free_rate: float = 0.04,
                 long_only: bool = True) -> None:
        self.tickers = list(mu.index)
        self.n = len(self.tickers)
        self.mu = mu.values
        self.cov = cov.values
        self.sectors = sectors
        self.sector_caps = sector_caps
        self.rf = risk_free_rate
        self.long_only = long_only

    # ---------- Helpers ----------
    def _bounds(self):
        if self.long_only:
            return [(0.0, 1.0)] * self.n
        return [(-1.0, 1.0)] * self.n

    def _sector_constraints(self) -> List[dict]:
        """For each sector with a cap, build a linear inequality A @ w <= b."""
        cons = [{
            "type": "eq",
            "fun": lambda w: np.sum(w) - 1.0,
            "jac": lambda w: np.ones_like(w),
        }]
        for sector, cap in self.sector_caps.items():
            idx = [i for i, t in enumerate(self.tickers)
                   if self.sectors.get(t) == sector]
            if not idx:
                continue
            A = np.zeros(self.n); A[idx] = 1.0
            cons.append({
                "type": "ineq",
                "fun": (lambda A=A: lambda w: cap - A @ w)(),
                "jac": (lambda A=A: lambda w: -A)(),
            })
        return cons

    def _portfolio_stats(self, w: np.ndarray
                         ) -> Tuple[float, float, float]:
        ret = float(w @ self.mu)
        vol = float(np.sqrt(w @ self.cov @ w))
        sharpe = (ret - self.rf) / vol if vol > 0 else 0.0
        return ret, vol, sharpe

    # ---------- Special portfolios ----------
    def min_variance(self) -> np.ndarray:
        w0 = np.ones(self.n) / self.n
        res = minimize(
            lambda w: w @ self.cov @ w,
            w0, jac=lambda w: 2 * self.cov @ w,
            method="SLSQP",
            bounds=self._bounds(),
            constraints=self._sector_constraints(),
            options={"ftol": 1e-12, "maxiter": 1000},
        )
        if not res.success:
            raise RuntimeError(f"Min-var failed: {res.message}")
        return res.x

    def max_sharpe(self) -> np.ndarray:
        w0 = np.ones(self.n) / self.n
        def neg_sharpe(w):
            r, v, _ = self._portfolio_stats(w)
            return -(r - self.rf) / v if v > 0 else 1e6
        def neg_sharpe_jac(w):
            r, v, s = self._portfolio_stats(w)
            if v < 1e-8:
                return np.zeros_like(w)
            dr = self.mu
            dv = (self.cov @ w) / v
            return -(dr * v - (r - self.rf) * dv) / (v * v)
        res = minimize(
            neg_sharpe, w0, jac=neg_sharpe_jac,
            method="SLSQP",
            bounds=self._bounds(),
            constraints=self._sector_constraints(),
            options={"ftol": 1e-12, "maxiter": 1000},
        )
        if not res.success:
            raise RuntimeError(f"Max-Sharpe failed: {res.message}")
        return res.x

    def target_return(self, target: float) -> np.ndarray:
        w0 = np.ones(self.n) / self.n
        cons = self._sector_constraints() + [{
            "type": "eq",
            "fun": (lambda t=target: lambda w: w @ self.mu - t)(),
            "jac": lambda w: self.mu,
        }]
        res = minimize(
            lambda w: w @ self.cov @ w,
            w0, jac=lambda w: 2 * self.cov @ w,
            method="SLSQP",
            bounds=self._bounds(),
            constraints=cons,
            options={"ftol": 1e-12, "maxiter": 1000},
        )
        if not res.success:
            return w0
        return res.x

    # ---------- Efficient frontier ----------
    def efficient_frontier(self, n_points: int = 50
                           ) -> pd.DataFrame:
        """Sweep target returns between min-var and max-asset expected return."""
        lo = self._portfolio_stats(self.min_variance())[0]
        hi = float(self.mu.max())
        targets = np.linspace(lo, hi, n_points)
        rows = []
        for t in targets:
            w = self.target_return(t)
            r, v, s = self._portfolio_stats(w)
            rows.append({"target": t, "return": r, "vol": v,
                         "sharpe": s, "weights": w})
        return pd.DataFrame(rows)