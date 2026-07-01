from __future__ import annotations
import numpy as np
import pandas as pd
import yfinance as yf
from typing import List, Tuple


class DataLoader:
    def __init__(self, tickers: List[str], benchmark: str,
                 start: str, end: str) -> None:
        self.tickers = tickers
        self.benchmark = benchmark
        self.start = start
        self.end = end

    def fetch(self) -> pd.DataFrame:
        """Download adjusted close prices for tickers + benchmark."""
        all_tickers = self.tickers + [self.benchmark]
        raw = yf.download(
            all_tickers,
            start=self.start,
            end=self.end,
            auto_adjust=True,        
            progress=False,
            threads=False,
        )["Close"]

        # Drop any ticker with > 5% missing data
        missing_pct = raw.isna().mean()
        valid = missing_pct[missing_pct < 0.05].index.tolist()
        raw = raw[valid].dropna()

        if self.benchmark not in raw.columns:
            raise ValueError(f"Benchmark {self.benchmark} not in downloaded data.")
        return raw

    @staticmethod
    def to_returns(prices: pd.DataFrame, log: bool = False) -> pd.DataFrame:
        """Convert price levels to simple or log returns."""
        if log:
            return np.log(prices / prices.shift(1)).dropna()
        return prices.pct_change().dropna()

    def split(self, prices: pd.DataFrame
              ) -> Tuple[pd.DataFrame, pd.Series]:
        """Return (asset_prices, benchmark_prices)."""
        bench = prices[self.benchmark].copy()
        assets = prices[self.tickers].copy()
        # Ensure all requested tickers actually came back
        missing = set(self.tickers) - set(assets.columns)
        if missing:
            raise ValueError(f"Missing tickers in data: {missing}")
        return assets, bench