from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple


class RiskEngine:
    def __init__(self, returns: pd.DataFrame, weights: np.ndarray,
                 confidence: float = 0.95,
                 horizon_days: int = 1,
                 trading_days: int = 252) -> None:
        self.returns = returns
        self.w = np.asarray(weights)
        self.alpha = confidence
        self.h = horizon_days
        self.days = trading_days
        self.port_rets = returns.values @ self.w

    # ---------- Distributional assumptions ----------
    def _params(self) -> Tuple[float, float]:
        mu = self.port_rets.mean()
        sigma = self.port_rets.std(ddof=1)
        return mu, sigma

    # ---------- VaR flavors ----------
    def var_historical(self) -> float:
        """Non-parametric: empirical quantile of losses."""
        return -np.quantile(self.port_rets, 1 - self.alpha)

    def var_parametric_normal(self) -> float:
        mu, sigma = self._params()
        z = stats.norm.ppf(1 - self.alpha)
        return -(mu * self.h - z * sigma * np.sqrt(self.h))

    def var_parametric_t(self, df: int = 5) -> float:
        mu, sigma = self._params()
        t_q = stats.t.ppf(1 - self.alpha, df=df)
        return -(mu * self.h - t_q * sigma * np.sqrt(self.h) *
                 np.sqrt((df - 2) / df))

    def var_monte_carlo(self, n_sims: int = 10_000,
                        dist: str = "normal") -> float:
        mu, sigma = self._params()
        if dist == "t":
            sims = stats.t.rvs(df=5, loc=mu, scale=sigma,
                               size=(n_sims, self.h)).sum(axis=1)
        else:
            sims = np.random.normal(mu, sigma, (n_sims, self.h)).sum(axis=1)
        return -np.quantile(sims, 1 - self.alpha)

    # ---------- CVaR (Expected Shortfall) ----------
    def cvar_historical(self) -> float:
        cutoff = np.quantile(self.port_rets, 1 - self.alpha)
        tail = self.port_rets[self.port_rets <= cutoff]
        return -tail.mean()

    # ---------- Marginal / component risk ----------
    def component_var(self) -> pd.Series:
        """Decompose total VaR by asset using Euler allocation."""
        mu, sigma = self._params()
        z = stats.norm.ppf(1 - self.alpha)
        cov_w = self.returns.cov().values @ self.w
        cvar = self.w * cov_w / (sigma + 1e-12) * z
        return pd.Series(cvar, index=self.returns.columns)

    # ---------- Stress tests ----------
    def stress_test(self, scenarios: Dict[str, float]) -> pd.Series:
        """Apply shock scenarios to daily mean return and recompute loss."""
        results = {}
        for name, shock in scenarios.items():
            shocked = self.port_rets + shock
            results[name] = -np.quantile(shocked, 1 - self.alpha)
        return pd.Series(results, name="stressed_VaR")