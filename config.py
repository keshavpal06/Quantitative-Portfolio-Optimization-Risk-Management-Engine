
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class EngineConfig:
    # Universe
    tickers: List[str] = field(default_factory=lambda: [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
        "META", "JPM", "JNJ", "V", "PG",
    ])
    benchmark: str = "^GSPC"  # S&P 500

    # Sector map (GICS-style, simplified)
    sectors: Dict[str, str] = field(default_factory=lambda: {
        "AAPL": "Tech", "MSFT": "Tech", "GOOGL": "Tech",
        "NVDA": "Tech", "META": "Tech",
        "AMZN": "Consumer Discretionary",
        "JPM": "Financials",
        "JNJ": "Healthcare",
        "V":   "Financials",
        "PG":  "Consumer Staples",
    })

    # Time
    start: str = "2020-01-01"
    end:   str = "2024-12-31"
    trading_days: int = 252

    # Risk-free rate (4% annual)
    risk_free_rate: float = 0.04

    # Sector constraints: max % of portfolio in any sector
    sector_caps: Dict[str, float] = field(default_factory=lambda: {
        "Tech": 0.45,
        "Consumer Discretionary": 0.25,
        "Financials": 0.30,
        "Healthcare": 0.25,
        "Consumer Staples": 0.20,
    })

    # Optimization
    long_only: bool = True
    allow_shorts: bool = False
    frontier_points: int = 50

    # Risk
    var_confidence: float = 0.95
    var_horizon_days: int = 1
    mc_simulations: int = 10_000

    # Backtest
    rebalance_freq: str = "M"      # Monthly
    train_window_days: int = 504   # 2-year rolling training window