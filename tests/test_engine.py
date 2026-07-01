import numpy as np
import pandas as pd
import pytest
from src.analytics import Analytics
from src.optimizer import MarkowitzOptimizer
from src.risk import RiskEngine


@pytest.fixture
def synthetic_returns():
    np.random.seed(0)
    dates = pd.date_range("2020-01-01", periods=500, freq="B")
    return pd.DataFrame(
        np.random.normal(0.0005, 0.015, (500, 4)),
        index=dates, columns=["A", "B", "C", "D"],
    )


def test_weights_sum_to_one(synthetic_returns):
    a = Analytics(synthetic_returns, 0.04)
    mu = a.mean_returns(); cov, _ = a.covariance("sample")
    opt = MarkowitzOptimizer(mu, cov, {}, {}, 0.04, long_only=True)
    w = opt.max_sharpe()
    assert abs(w.sum() - 1.0) < 1e-4
    assert (w >= -1e-6).all()


def test_min_variance_lower_or_equal_vol(synthetic_returns):
    a = Analytics(synthetic_returns, 0.04)
    mu = a.mean_returns(); cov, _ = a.covariance("sample")
    opt = MarkowitzOptimizer(mu, cov, {}, {}, 0.04, long_only=True)
    w_eq = np.ones(4) / 4
    w_mv = opt.min_variance()
    v_eq = w_eq @ cov.values @ w_eq
    v_mv = w_mv @ cov.values @ w_mv
    assert v_mv <= v_eq + 1e-8


def test_var_positive_and_cvar_greater(synthetic_returns):
    w = np.array([0.4, 0.3, 0.2, 0.1])
    re = RiskEngine(synthetic_returns, w, confidence=0.95, horizon_days=1)
    var = re.var_historical()
    cvar = re.cvar_historical()
    assert var > 0
    assert cvar >= var