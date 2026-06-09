import numpy as np
import pandas as pd
import pytest

from whoop_analytics.analysis.discovery import CausalDiscovery, DiscoveryResult


@pytest.fixture
def synthetic_data():
    """Create synthetic data where X causes Y with 1-day lag."""
    np.random.seed(42)
    n = 150
    x = np.random.randn(n)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = 0.7 * x[t - 1] + 0.3 * np.random.randn()
    z = np.random.randn(n)  # noise variable, no causal role

    dates = pd.date_range("2026-01-01", periods=n, freq="D")
    return pd.DataFrame({"x": x, "y": y, "z": z}, index=dates)


def test_discovery_finds_causal_link(synthetic_data):
    discovery = CausalDiscovery(max_lag=3, significance_level=0.05)
    result = discovery.run(synthetic_data, target="y")

    assert isinstance(result, DiscoveryResult)
    assert len(result.links) > 0

    x_causes = [link for link in result.links if link.source == "x"]
    assert len(x_causes) > 0
    assert x_causes[0].lag == 1


def test_discovery_ignores_noise_variable(synthetic_data):
    discovery = CausalDiscovery(max_lag=3, significance_level=0.05)
    result = discovery.run(synthetic_data, target="y")

    z_causes = [link for link in result.links if link.source == "z"]
    assert len(z_causes) == 0


def test_discovery_result_has_graph(synthetic_data):
    discovery = CausalDiscovery(max_lag=3, significance_level=0.05)
    result = discovery.run(synthetic_data, target="y")

    assert result.variable_names == ["x", "y", "z"]
    assert result.val_matrix is not None
    assert result.val_matrix.shape[0] == 3


def test_discovery_handles_short_data():
    np.random.seed(42)
    dates = pd.date_range("2026-01-01", periods=20, freq="D")
    df = pd.DataFrame({
        "a": np.random.randn(20),
        "b": np.random.randn(20),
    }, index=dates)

    discovery = CausalDiscovery(max_lag=2, significance_level=0.05)
    result = discovery.run(df, target="b")

    assert isinstance(result, DiscoveryResult)
