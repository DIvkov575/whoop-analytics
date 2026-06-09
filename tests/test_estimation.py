import numpy as np
import pandas as pd
import pytest

from whoop_analytics.analysis.discovery import CausalLink
from whoop_analytics.analysis.estimation import EffectEstimator, EffectResult


@pytest.fixture
def causal_data():
    """X causes Y with known effect size ~0.7."""
    np.random.seed(42)
    n = 200
    x = np.random.randn(n)
    y = 0.7 * x + 0.3 * np.random.randn(n)
    z = np.random.randn(n)

    return pd.DataFrame({"x": x, "y": y, "z": z})


@pytest.fixture
def causal_link():
    return CausalLink(source="x", target="y", lag=0, strength=0.7, p_value=0.001)


def test_estimate_effect_returns_result(causal_data, causal_link):
    estimator = EffectEstimator()
    result = estimator.estimate(
        df=causal_data,
        link=causal_link,
        common_causes=["z"],
    )

    assert isinstance(result, EffectResult)
    assert result.ate is not None
    assert abs(result.ate - 0.7) < 0.15


def test_estimate_runs_refutation(causal_data, causal_link):
    estimator = EffectEstimator()
    result = estimator.estimate(
        df=causal_data,
        link=causal_link,
        common_causes=["z"],
    )

    assert result.refutation_passed is not None
    assert isinstance(result.refutation_p_value, float)


def test_estimate_handles_no_common_causes(causal_data, causal_link):
    estimator = EffectEstimator()
    result = estimator.estimate(
        df=causal_data,
        link=causal_link,
        common_causes=[],
    )

    assert isinstance(result, EffectResult)
    assert result.ate is not None
