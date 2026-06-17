from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from meteoraster import Resample


def _daily_single_leadtime_frame(n=10):
    """A production x leadtime frame: daily production dates, a single 0-day leadtime."""
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    cols = pd.MultiIndex.from_tuples([("A", pd.Timedelta("0D"))], names=["series", "leadtime"])
    return pd.DataFrame(np.ones((n, 1)), index=idx, columns=cols)


def test_resample_has_version():
    assert Resample.VERSION


@pytest.mark.parametrize("resampling_type", ["sum", "mean"])
def test_resample_daily_identity(resampling_type):
    """Resampling daily data to a daily step is an identity for sum/mean of ones."""
    df = _daily_single_leadtime_frame(10)

    out = Resample.resample(df, timestep_frequency="1D", resamplingType=resampling_type)

    assert isinstance(out, pd.DataFrame)
    finite = out.to_numpy().ravel()
    finite = finite[np.isfinite(finite)]
    assert finite.size > 0
    assert np.allclose(finite, 1.0, atol=1e-6)
