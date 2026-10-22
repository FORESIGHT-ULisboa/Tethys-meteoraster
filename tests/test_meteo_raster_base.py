from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pathlib import Path
import os
import shutil
import tempfile

from meteoraster import MeteoRaster

def test_init_shapes(sample_meteoraster):
    mr = sample_meteoraster
    assert mr.data.shape == (2, 3, 2, 4, 5)
    assert mr.latitudes.shape == (4, 5)
    assert mr.longitudes.shape == (4, 5)
    assert mr.production_datetime.shape == (2,)
    assert mr.leadtimes.shape == (2,)

def test_get_cropped(sample_meteoraster):
    mr = sample_meteoraster
    # Crop spatially
    # Original lats: [40, 36.6, 33.3, 30]
    # Original lons: [-10, -5, 0, 5, 10]
    
    cropped = mr.get_cropped(
        from_lat=32, to_lat=38,
        from_lon=-2, to_lon=7
    )
    
    # Expected lats range: include 36.6 and 33.3. (indices 1, 2)
    # Expected lons range: include 0, 5. (indices 2, 3)
    
    # Check data shape. Time dimensions should be same. Spatial dims changed.
    # New spatial dims: lat size 2, lon size 2.
    assert cropped.data.shape == (2, 3, 2, 2, 2)
    assert cropped.latitudes.shape == (2, 2)
    assert cropped.longitudes.shape == (2, 2)
    
    # Test temporal crop
    cropped_time = mr.get_cropped(
        from_prod_date=pd.Timestamp("2023-01-02")
    )
    assert cropped_time.data.shape == (1, 3, 2, 4, 5)
    assert cropped_time.production_datetime[0] == pd.Timestamp("2023-01-02")

def test_get_values_from_latlon(sample_meteoraster):
    mr = sample_meteoraster
    # Pick a point. (lat=40, lon=-10) is at index (0,0)
    lat = 40.0
    lon = -10.0
    
    # Expected data shape: [production_datetime, ensemble_members] for each leadtime
    # The method returns a dataframe with MultiIndex (prod_date, ensemble) and columns (leadtimes)
    
    df = mr.get_values_from_latlon(lat, lon)
    
    # Rows should be n_prod = 2
    # Columns should be n_lead * n_ens = 2 * 3 = 6
    assert df.shape == (2, 6) 
    
    # Check values match data at (0,0)
    # Data structure: [prod, ens, lead, lat, lon]
    # Check first production date, first ensemble, first leadtime
    expected_value = mr.data[0, 0, 0, 0, 0]

    # Index is prod_date
    # Columns are MultiIndex (leadtime, ensemble_member)
    
    val = df.loc[mr.production_datetime[0], (mr.leadtimes[0], 0)]
    assert np.isclose(val, expected_value)

def test_to_xarray(sample_meteoraster):
    mr = sample_meteoraster
    da = mr.to_xarray()
    
    assert isinstance(da, xr.DataArray)
    # Check dimensions based on code read: ['production_datetime', 'ensemble_member', 'leadtime', 'y', 'x']
    assert da.dims == ('production_datetime', 'ensemble_member', 'leadtime', 'y', 'x')
    assert da.shape == mr.data.shape
    assert da.name == "precip"
    assert da.attrs['units'] == "mm"

def test_save_and_load(sample_meteoraster):
    mr = sample_meteoraster
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    
    try:
        tmp.close() # Close so we can write to it
        mr.save(tmp_path)
        
        # Load it back
        mr_loaded = MeteoRaster.load(tmp_path)
        
        assert isinstance(mr_loaded, MeteoRaster)
        np.testing.assert_allclose(mr.data, mr_loaded.data)
        assert mr.variable == mr_loaded.variable
        assert mr.units == mr_loaded.units
        
    finally:
        if tmp_path.exists():
            os.remove(tmp_path)

def test_is_complete(sample_meteoraster):
    mr = sample_meteoraster.copy()
    assert mr.is_complete() == True
    
    # Introduce NaN
    mr.data[0, 0, 0, 0, 0] = np.nan
    
    # Check completeness
    # Default: full_ensemble=True, space_completeness=False
    # Meaning: All ensembles must be finite for each slice (prod, lead) ?
    # Let's check logic in code:
    # if full_ensemble: finite = finite.all(axis=1) [ens axis]
    # So if one ens member is nan, that pixel is nan. 
    # if space_completeness=False: finite = finite.any(axis=(-2, -1))
    # So if at least one pixel is valid spatialy, it's valid for that (prod, lead)
    # Then .all() over prod and lead.
    
    # Here we have NaN at one pixel.
    # Ensembles at that pixel: indices 1 and 2 are finite. Index 0 is nan.
    # finite.all(axis=1) -> At (0,0,0,0,0), it will be False.
    # But at (0,0,0,0,1) it is True.
    # finite.any(axis=(-2,-1)) -> Will be True because other pixels are fine.
    # So is_complete() should return True because other pixels exist.
    assert mr.is_complete() == True
    
    # Now verify if we require space_completeness=True
    # Then all pixels must be valid.
    # At (0,0) [prod=0, lead=0], pixel (0,0) failed the ensemble check.
    # So valid spatial map has a False at (0,0).
    # finite.all(axis=(-2,-1)) will be False.
    assert mr.is_complete(space_completeness=True) == False
    
    # If we set full_ensemble=False, then one member is enough.
    # Member 1 is finite at (0,0,0,0,0). So any(axis=1) is True.
    assert mr.is_complete(full_ensemble=False, space_completeness=True) == True

def test_join_simple(sample_meteoraster):
    mr1 = sample_meteoraster.copy()
    mr2 = sample_meteoraster.copy()
    
    # Modify mr2 dates to be later
    # original: 2023-01-01, 2023-01-02
    # new: 2023-01-03, 2023-01-04
    mr2.production_datetime = mr1.production_datetime + pd.Timedelta(days=2)
    mr2.data = mr1.data + 10 # Just to distinguish
    
    # Join
    mr1.join(mr2)
    
    assert mr1.production_datetime.size == 4
    assert mr1.data.shape[0] == 4
    np.testing.assert_allclose(mr1.data[2:], mr2.data)

def test_greenwich_fix():
    # Longitudes > 180 should be corrected
    data = np.zeros((1, 1, 1, 2, 2))
    lats = np.array([0, 1])
    lons = np.array([179, 181]) # 181 should become -179
    
    mr = MeteoRaster(
        data=data,
        latitudes=lats,
        longitudes=lons,
        production_datetime=np.array([pd.Timestamp("2020-01-01")]),
        leadtimes=np.array([pd.Timedelta(days=1)]),
    )
    
    # Expected: [-179, 179] (sorted usually? Code says:
    # westIndexes: lons > 180 (index 1: 181) -> -179
    # eastIndexes: lons <= 180 (index 0: 179) -> 179
    # longitudes = concat(west, east) -> [-179, 179]
    # data also reordered.
    
    print(mr.longitudes)
    assert (mr.longitudes < 180).all()
    # Check if sorted? The code concatenates self.longitudes[westIndexes]-360 then self.longitudes[eastIndexes].
    # So if we had [179, 181], west is [181]->[-179], east is [179].
    # Result [-179, 179].
    assert mr.longitudes[0, 0] == -179
    assert mr.longitudes[0, 1] == 179

def test_get_values_from_kml(sample_meteoraster, sample_kml):
    pytest.importorskip('osgeo')
    # Since v3.0 this runs on meteoraster.zonal, which needs no scipy - so the
    # LinearNDInterpolator crash guard this test used to carry is gone with it.

    mr = sample_meteoraster
    # The fixture MR roughly covers Lat [30, 40], Lon [-10, 10]
    # KML polygon covers [-2, 2] x [32, 38] which is inside.
    
    df, centroids = mr.get_values_from_KML(
        kml=str(sample_kml),
        nameField="zone_id",
        getCoverageInfo=False
    )
    
    # Check output structure
    # df rows = production_dates
    assert len(df) == len(mr.production_datetime)
    
    # df columns = MultiIndex (zone, leadtime, ensemble_member)
    assert "ZoneA" in df.columns.get_level_values("zone")
    assert "ZoneB" in df.columns.get_level_values("zone")
    
    # Check that we have values
    assert not df.isna().all().all() # Some values should be present
    assert np.isfinite(df.values).any()

    # Check centroids
    assert "ZoneA" in centroids.index
    # Centroid should be around 0, 35
    assert np.isclose(float(centroids.loc["ZoneA", "x"]), 0.0, atol=1.0)
    assert np.isclose(float(centroids.loc["ZoneA", "y"]), 35.0, atol=1.0)


def test_get_values_from_kml_covers_the_grid_border(grid_raster, grid_kml):
    '''
    The on-paper answer. grid_raster is 10 everywhere except a 1 in the centre cell of
    a 3x3 grid of unit cells, so the area-weighted mean over the whole grid is 9.0.

    The legacy builder clipped its candidates to 0 < i < n-1, so it saw only the centre
    cell and returned 1.0 - plausible-looking, because the weights were renormalised by
    the area it did find. v3.0 runs on meteoraster.zonal and gets 9.0.
    '''

    pytest.importorskip('osgeo')

    agg, centroids = grid_raster.get_values_from_KML(kml=str(grid_kml), nameField='zone_id',
                                                     area_weighting='planar')

    whole = agg.xs('whole', axis=1, level='zone')
    middle = agg.xs('middle', axis=1, level='zone')

    # leadtime k scales the field by k+1, and every member holds the same field
    for leadtime, factor in zip(sorted(set(whole.columns.get_level_values('leadtime'))),
                                (1, 2, 3)):
        assert np.allclose(whole.xs(leadtime, axis=1, level='leadtime').to_numpy(),
                           9.0 * factor)
        assert np.allclose(middle.xs(leadtime, axis=1, level='leadtime').to_numpy(),
                           1.0 * factor)

    assert list(agg.columns.names) == ['zone', 'leadtime', 'ensemble_member']
    # 0-based, matching get_values_from_latlon and get_quantiles_from_KML
    assert sorted(set(agg.columns.get_level_values('ensemble_member'))) == [0, 1]
    assert np.isclose(float(centroids.loc['middle', 'x']), 1.0)
    assert np.isclose(float(centroids.loc['middle', 'y']), 1.0)


def test_get_values_from_kml_reuses_a_coverage(grid_raster, grid_kml):
    pytest.importorskip('osgeo')

    first, _, coverage_info = grid_raster.get_values_from_KML(
        kml=str(grid_kml), nameField='zone_id', getCoverageInfo=True)
    again, _ = grid_raster.get_values_from_KML(kml=str(grid_kml), coverage_info=coverage_info)

    assert np.allclose(first.to_numpy(), again.to_numpy(), equal_nan=True)


def test_get_values_from_kml_ignores_elementwise(grid_raster, grid_kml):
    '''`elementwise` only skipped the missing-data renormalisation, which is now always on.'''

    pytest.importorskip('osgeo')

    with pytest.warns(DeprecationWarning, match='elementwise'):
        grid_raster.get_values_from_KML(kml=str(grid_kml), nameField='zone_id',
                                        elementwise=True)


