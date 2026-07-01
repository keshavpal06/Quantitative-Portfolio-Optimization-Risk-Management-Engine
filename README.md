# QuantForge: Quantitative Portfolio Optimization & Risk Management Engine

A modular, production-grade quantitative finance engine built in Python for asset portfolio optimization, historical/stochastic risk estimation, and walk-forward historical backtesting. This engine leverages robust estimators to navigate estimation error in classical Mean-Variance frameworks.

## 🚀 Core Features

* **Ledoit-Wolf Shrinkage:** Replaces the sample covariance matrix with a shrunk covariance matrix to reduce out-of-sample estimation errors in optimized asset allocation.
* **Markowitz Optimization Framework:** Solves for Maximum Sharpe Ratio and Minimum Variance asset allocations with hard boundary conditions and long-only constraints using `SciPy`.
* **Multi-Model Value at Risk (VaR):** Implements four distinct statistical approaches for asset downside calculation:
    * Historical Simulation
    * Parametric Normal Distribution
    * Parametric Student’s $t$-Distribution (capturing fat tails, $df=5$)
    * Stochastic Monte Carlo Simulation (10,000+ paths)
* **Tail-Risk Metrics:** Computes Conditional Value at Risk (CVaR / Expected Shortfall) and multi-scenario stress-testing models (e.g., historical black-swan shocks like the 2020 COVID Crash).
* **Walk-Forward Backtester:** Evaluates historical strategy returns dynamically against an S&P 500 benchmark using modular trailing estimation and rebalancing constraints.

---

## 🛠️ Project Structure

```text
├── src/
│   ├── data_loader.py    # Fetches, slices, and cleans multi-asset market data
│   ├── analytics.py      # Estimates annualized metrics, shrinkage covariance, and rolling beta
│   ├── optimizer.py      # Executes SciPy convex quadratic optimization solvers
│   ├── risk_engine.py    # Multi-model VaR, CVaR, and component risk decomposition
│   └── backtester.py     # Performs walk-forward rolling simulation pipelines
├── tests/                # Automated parameter and boundary-condition validation tests
├── main.py               # Command-line interface execution script
├── requirements.txt      # Working package dependency list
└── README.md             # Project documentation
