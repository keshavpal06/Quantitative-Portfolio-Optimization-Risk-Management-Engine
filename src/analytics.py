from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.covariance import LedoitWolf


class Analytics:
    def __init__(self, returns: pd.DataFrame, risk_free_rate: float,
                 trading_days: int = 252) -> None:
        self.returns = returns
        self.rf = risk_free_rate
        self.days = trading_days

    # ---------- Covariance ----------
    def covariance(self, method: str = "ledoit_wolf"
                   ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Annualized covariance matrix.
        method: 'sample' or 'ledoit_wolf' (shrinks toward scaled identity).
        """
        if method == "ledoit_wolf":
            lw = LedoitWolf().fit(self.returns.values)
            cov = lw.covariance_ * self.days
        else:
            cov = self.returns.cov().values * self.days
        return pd.DataFrame(cov, index=self.returns.columns,
                            columns=self.returns.columns), cov

    def rolling_covariance(self, window: int = 60
                           ) -> Tuple[np.ndarray, pd.Index]:
        """3D array (T, N, N) of rolling annualized covariances."""
        rols = self.returns.rolling(window).cov().dropna()
        dates = rols.index.get_level_values(0).unique()
        n = len(self.returns.columns)
        out = np.empty((len(dates), n, n))
        for i, d in enumerate(dates):
            out[i] = rols.xs(d).values * self.days
        return out, dates

    # ---------- Expected returns ----------
    def mean_returns(self, annualized: bool = True) -> pd.Series:
        mu = self.returns.mean()
        return mu * self.days if annualized else mu

    # ---------- Sharpe ratios ----------
    def sharpe(self, annualized: bool = True) -> pd.Series:
        excess = self.returns.mean() - self.rf / self.days
        vol = self.returns.std()
        sr = excess / vol
        return sr * np.sqrt(self.days) if annualized else sr

    def rolling_sharpe(self, window: int = 60) -> pd.DataFrame:
        excess = self.returns - self.rf / self.days
        return (excess.rolling(window).mean() /
                self.returns.rolling(window).std()
                ) * np.sqrt(self.days)

    # ---------- Beta ----------
    def beta(self, benchmark_returns: pd.Series) -> pd.Series:
        """CAPM beta_i = cov(R_i, R_m) / var(R_m)."""
        aligned = self.returns.join(benchmark_returns.rename("mkt"),
                                    how="inner").dropna()
        mkt_var = aligned["mkt"].var()
        betas = {col: aligned[col].cov(aligned["mkt"]) / mkt_var
                 for col in self.returns.columns}
        return pd.Series(betas).sort_values(ascending=False)

    def rolling_beta(self, benchmark_returns: pd.Series,
                     window: int = 60) -> pd.DataFrame:
        aligned = self.returns.join(benchmark_returns.rename("mkt"),
                                    how="inner")
        mkt_var = aligned["mkt"].rolling(window).var()
        betas = {}
        for col in self.returns.columns:
            cov_im = aligned[col].rolling(window).cov(aligned["mkt"])
            betas[col] = cov_im / mkt_var
        return pd.DataFrame(betas).dropna()