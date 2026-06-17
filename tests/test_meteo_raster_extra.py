from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from meteoraster import MeteoRaster


def test_trim_removes_nan_edges(sample_meteoraster):
    mr = sample_meteoraster.copy()
    # Blank out the first production date entirely -> trim() should drop it.
    mr.data[0, ...] = np.nan

    mr.trim()

    assert mr.production_datetime.size == 1
    assert mr.data.shape[0] == 1
    assert mr.production_datetime[0] == pd.Timestamp("2023-01-02")


def test_get_missing_structure(sample_meteoraster):
    mr = sample_meteoraster
    missing = mr.get_missing()

    # Transposed: rows = (leadtime, ensemble_member), columns = production dates.
    assert isinstance(missing, pd.DataFrame)
    assert missing.shape == (mr.leadtimes.size * mr.data.shape[1], mr.production_datetime.size)
    assert list(missing.index.names) == ["leadtime", "ensemble_member"]
    # No NaNs in the fixture -> nothing is missing.
    assert (missing.to_numpy() == 0).all()


def test_get_complete_index(sample_meteoraster):
    mr = sample_meteoraster
    ci = mr.get_complete_index()

    assert isinstance(ci, pd.DataFrame)
    assert ci.shape == (mr.production_datetime.size, mr.leadtimes.size)
    assert ci.index.name == "production_datetime"
    assert ci.columns.name == "leadtime"
    assert ci.to_numpy().all()  # fully finite fixture

    # Knock out one pixel across all ensemble members at (prod 0, lead 0).
    mr2 = mr.copy()
    mr2.data[0, :, 0, 0, 0] = np.nan
    ci2 = mr2.get_complete_index(space_completeness=True)
    assert not bool(ci2.iloc[0, 0])  # that (prod, lead) is no longer spatially complete


def test_get_values_from_latlon_by_event(sample_meteoraster):
    mr = sample_meteoraster
    by_prod = mr.get_values_from_latlon(35.0, 0.0)

    by_event = MeteoRaster.get_values_from_latlon_by_event(by_prod)

    assert isinstance(by_event, pd.DataFrame)
    # Same number of (leadtime, ensemble_member) columns, reindexed onto event dates.
    assert by_event.shape[1] == by_prod.shape[1]
    # 2 production dates x leadtimes {1d, 2d} -> event dates {01-02, 01-03, 01-04}
    assert by_event.index.min() == pd.Timestamp("2023-01-02")
    assert by_event.index.max() == pd.Timestamp("2023-01-04")


def test_copy_is_deep(sample_meteoraster):
    mr = sample_meteoraster
    mr2 = mr.copy()
    mr2.data[0, 0, 0, 0, 0] = -999.0
    assert mr.data[0, 0, 0, 0, 0] != -999.0
