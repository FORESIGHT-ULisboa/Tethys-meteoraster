from __future__ import annotations

import pytest


def test_get_cropped_alias_warns_and_delegates(sample_meteoraster):
    mr = sample_meteoraster
    kwargs = dict(from_lat=32, to_lat=38, from_lon=-2, to_lon=7)

    with pytest.warns(DeprecationWarning):
        legacy = mr.getCropped(**kwargs)

    current = mr.get_cropped(**kwargs)
    assert legacy.data.shape == current.data.shape


def test_get_values_from_latlon_alias_warns(sample_meteoraster):
    mr = sample_meteoraster
    with pytest.warns(DeprecationWarning):
        df = mr.getDataFromLatLon(35.0, 0.0)
    assert df is not None
    assert df.shape == mr.get_values_from_latlon(35.0, 0.0).shape


def test_get_values_from_kml_alias_warns(sample_meteoraster, sample_kml):
    mr = sample_meteoraster
    with pytest.warns(DeprecationWarning):
        agg, centroids = mr.getValuesFromKML(kml=str(sample_kml), nameField="zone_id")
    assert "ZoneA" in agg.columns.get_level_values("zone")
