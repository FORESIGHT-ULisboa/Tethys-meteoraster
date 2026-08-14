'''
KML zonal aggregation against a real catchment layer.

`catchments.kml` is the Vesdre basin (Belgium) as 26 nested sub-catchments, exported
from QGIS: namespaced, `<MultiGeometry>`-wrapped polygons, 7940 vertices, zones named
by an `id` field. The synthetic KMLs in conftest.py pin the *arithmetic* on a grid whose
weights are computable on paper; this one pins the parts they cannot - a real export's
structure, many zones at once, and zones far smaller than a grid cell.

It is applicable to the fixtures whose grid covers Belgium, at three resolutions, which
is the point: the same 26 zones fall on 1-5 cells at 0.1 deg, 1-4 at 0.25 deg and 1-2 at
1 deg, so most zones end up *inside* a single cell on the coarse grids. The ipma
fixtures are over Portugal and do not overlap it at all.

No assertion here depends on the absolute magnitude of the c3s values - see
resources/PROVENANCE.md, they are ~1000x too large for the mm/month they are labelled.
'''

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from meteoraster import MeteoRaster
from meteoraster import zonal

from conftest import gdal_required, grib_required, require_resource

pytestmark = gdal_required

KML = 'catchments.kml'
NAME_FIELD = 'id'

#: Measured from the file. Zone order is the Placemark order.
ZONE_IDS = ['Chaudfontaine', 'Aval Magne', 'Mosbeux', 'Station Foret (Magne)', 'Vaux',
            'Pepinster', 'Station Verviers', 'Dison', 'Theux', 'Mangombroux', 'SPIXHE',
            'R. Villers', 'Ruyff Baelen', 'Aval Gileppe', 'Polleur', 'Bach',
            'BV Barrage Gileppe', 'Eupen aval conf. Helle', 'BV Barrage Vesdre',
            'ss 22', 'ss 23', 'ss 24', 'BV Soor', 'Belleheid', 'ss 25', 'BV Helle']

#: Bounding box of every ring in the file.
BASIN_LON = (5.6361, 6.2557)
BASIN_LAT = (50.4322, 50.6772)

#: Fixtures whose grid covers the basin, coarsest last.
OVERLAPPING = [
    pytest.param('tethys_NOAA_GFS_0.25_2026.02.02.nct', 4, id='gfs-0.25deg'),
    pytest.param('tethys_c3s_ecmwf51_tprate_2026.07.nct', 2, id='c3s-1deg'),
]


def kml_path():
    return str(require_resource(KML))


def load(name):
    return MeteoRaster.load(require_resource(name), verbose=False)


def zone_level(agg):
    '''
    The zone labels of an aggregate, in column order.

    The columns are a MultiIndex only while a leadtime or ensemble_member level
    survives; with both size-1 they collapse to a plain Index named 'zone'.
    '''

    return list(dict.fromkeys(agg.columns.get_level_values('zone')))


def zone_values(agg, zone):
    '''All values of one zone, whatever the surviving column levels.'''

    if isinstance(agg.columns, pd.MultiIndex):
        return agg.xs(zone, axis=1, level='zone').to_numpy()

    return agg[[zone]].to_numpy()


def covered_cells(raster, coverage, zone):
    '''The raster values of the cells a zone covers, as [.., cell].'''

    index, _ = coverage.cells(coverage.zones.index(zone))
    rows, cols = np.divmod(index, coverage.cols)

    return raster.data[:, :, :, rows, cols]


# --------------------------------------------------------------------------- #
# the layer itself
# --------------------------------------------------------------------------- #

def test_the_layer_reads_as_26_named_zones():
    '''
    Guards the fixture, and the reader against a real QGIS export: namespaced tags and
    polygons wrapped in <MultiGeometry>, neither of which the synthetic KMLs exercise.
    '''

    named = zonal.read_kml_polygons(kml_path(), name_field=NAME_FIELD)

    assert [name for name, _ in named] == ZONE_IDS

    for name, geometry in named:
        assert geometry.IsValid(), name
        assert geometry.Area() > 0, name
        west, east, south, north = geometry.GetEnvelope()
        assert BASIN_LON[0] - 1e-6 <= west and east <= BASIN_LON[1] + 1e-6, name
        assert BASIN_LAT[0] - 1e-6 <= south and north <= BASIN_LAT[1] + 1e-6, name


def test_positions_label_the_zones_without_a_name_field():
    named = zonal.read_kml_polygons(kml_path())

    assert [name for name, _ in named] == [str(i) for i in range(len(ZONE_IDS))]


def test_an_absent_name_field_is_reported():
    with pytest.raises(ValueError, match='not_a_field'):
        zonal.read_kml_polygons(kml_path(), name_field='not_a_field')


# --------------------------------------------------------------------------- #
# means
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('fixture, max_cells', OVERLAPPING)
def test_every_catchment_catches_cells(fixture, max_cells):
    raster = load(fixture)

    coverage = zonal.coverage_from_kml(kml_path(), raster.latitudes, raster.longitudes,
                                      name_field=NAME_FIELD)

    assert coverage.zones == ZONE_IDS
    assert coverage.empty_zones() == []
    counts = [coverage.cells(i)[0].size for i in range(len(coverage))]
    assert min(counts) >= 1
    # a catchment this size cannot spill beyond a handful of cells at this resolution;
    # a coverage that quietly grew its candidate window would show up here
    assert max(counts) <= max_cells


@pytest.mark.parametrize('fixture, max_cells', OVERLAPPING)
def test_the_zonal_mean_stays_within_the_cells_it_averages(fixture, max_cells):
    '''
    A weighted mean with non-negative weights cannot leave the range of its sample.
    Checked per zone against the cells that zone actually covers, so a coverage that
    picked up the wrong cells fails even where the magnitude looks plausible.
    '''

    raster = load(fixture)
    coverage = zonal.coverage_from_kml(kml_path(), raster.latitudes, raster.longitudes,
                                       name_field=NAME_FIELD)
    agg, centroids = raster.get_values_from_KML(kml=kml_path(), nameField=NAME_FIELD)

    assert zone_level(agg) == ZONE_IDS
    assert len(agg) == len(raster.production_datetime)

    for zone in ZONE_IDS:
        cells = covered_cells(raster, coverage, zone)
        got = zone_values(agg, zone)
        finite = np.isfinite(got)
        if not finite.any():
            continue
        assert got[finite].min() >= np.nanmin(cells) - 1e-6, zone
        assert got[finite].max() <= np.nanmax(cells) + 1e-6, zone

    assert BASIN_LON[0] <= centroids['x'].min() and centroids['x'].max() <= BASIN_LON[1]
    assert BASIN_LAT[0] <= centroids['y'].min() and centroids['y'].max() <= BASIN_LAT[1]


@pytest.mark.parametrize('fixture, max_cells', OVERLAPPING)
def test_a_zone_inside_one_cell_returns_that_cell(fixture, max_cells):
    '''
    Most of these catchments are smaller than a cell on the coarse grids. A zone with a
    single covered cell must reproduce that cell exactly - no renormalisation, no
    interpolation.
    '''

    raster = load(fixture)
    coverage = zonal.coverage_from_kml(kml_path(), raster.latitudes, raster.longitudes,
                                       name_field=NAME_FIELD)
    agg, _ = raster.get_values_from_KML(kml=kml_path(), nameField=NAME_FIELD)

    single = [zone for zone in ZONE_IDS
              if coverage.cells(coverage.zones.index(zone))[0].size == 1]
    assert len(single) >= 5, 'expected most catchments to fall inside one cell here'

    for zone in single:
        cells = covered_cells(raster, coverage, zone)
        assert np.allclose(np.sort(zone_values(agg, zone).ravel()),
                           np.sort(cells.ravel()), equal_nan=True), zone


def test_zones_sharing_a_cell_are_indistinguishable_on_a_coarse_grid():
    '''
    Documented consequence, not a defect: on the 1 deg c3s grid several catchments fall
    entirely inside the same cell, so they cannot be told apart. Worth pinning, because
    it is the reason to resample a forecast onto a finer grid before aggregating.
    '''

    raster = load('tethys_c3s_ecmwf51_tprate_2026.07.nct')
    coverage = zonal.coverage_from_kml(kml_path(), raster.latitudes, raster.longitudes,
                                       name_field=NAME_FIELD)
    agg, _ = raster.get_values_from_KML(kml=kml_path(), nameField=NAME_FIELD)

    by_cell = {}
    for zone in ZONE_IDS:
        index, _ = coverage.cells(coverage.zones.index(zone))
        if index.size == 1:
            by_cell.setdefault(int(index[0]), []).append(zone)

    shared = [zones for zones in by_cell.values() if len(zones) > 1]
    assert shared, 'expected at least one cell to hold several whole catchments'

    for zones in shared:
        reference = zone_values(agg, zones[0])
        for zone in zones[1:]:
            assert np.allclose(zone_values(agg, zone), reference, equal_nan=True), zone


# --------------------------------------------------------------------------- #
# quantiles
# --------------------------------------------------------------------------- #

def gfs_first_day():
    '''GFS cropped to the first 24 h - 9 leadtimes rather than 129, so this is quick.'''

    return load('tethys_NOAA_GFS_0.25_2026.02.02.nct').get_cropped(
        to_leadtime=pd.Timedelta(hours=24))


def test_quantiles_are_ordered_and_bracket_the_mean():
    raster = gfs_first_day()
    quantiles, _ = raster.get_quantiles_from_KML(kml=kml_path(), nameField=NAME_FIELD,
                                                quantiles=[0.0, 0.5, 1.0])
    means, _ = raster.get_values_from_KML(kml=kml_path(), nameField=NAME_FIELD)

    assert sorted(set(quantiles.columns.get_level_values('zone'))) == sorted(ZONE_IDS)

    low = quantiles.xs(0.0, axis=1, level='quantile')
    mid = quantiles.xs(0.5, axis=1, level='quantile')
    high = quantiles.xs(1.0, axis=1, level='quantile')

    assert np.nanmax(low.to_numpy() - mid.to_numpy()) <= 1e-6
    assert np.nanmax(mid.to_numpy() - high.to_numpy()) <= 1e-6

    # the area-weighted mean lies between the extremes of the same sample
    for zone in ZONE_IDS:
        floor = zone_values(low, zone)
        ceiling = zone_values(high, zone)
        mean = zone_values(means, zone)
        finite = np.isfinite(mean) & np.isfinite(floor) & np.isfinite(ceiling)
        if not finite.any():
            continue
        assert (mean[finite] >= floor[finite] - 1e-6).all(), zone
        assert (mean[finite] <= ceiling[finite] + 1e-6).all(), zone


def test_a_zone_inside_one_cell_has_degenerate_quantiles():
    '''One cell is one sample value, so every quantile of it is that value.'''

    raster = gfs_first_day()
    coverage = zonal.coverage_from_kml(kml_path(), raster.latitudes, raster.longitudes,
                                       name_field=NAME_FIELD)
    quantiles, _ = raster.get_quantiles_from_KML(kml=kml_path(), nameField=NAME_FIELD,
                                                quantiles=[0.01, 0.5, 0.99])

    single = [zone for zone in ZONE_IDS
              if coverage.cells(coverage.zones.index(zone))[0].size == 1]
    assert single

    for zone in single[:4]:
        block = quantiles.xs(zone, axis=1, level='zone')
        low = block.xs(0.01, axis=1, level='quantile').to_numpy()
        high = block.xs(0.99, axis=1, level='quantile').to_numpy()
        assert np.allclose(low, high, equal_nan=True), zone


# --------------------------------------------------------------------------- #
# reusing a coverage
# --------------------------------------------------------------------------- #

def test_a_coverage_round_trips_and_is_pinned_to_its_grid(tmp_path):
    raster = load('tethys_NOAA_GFS_0.25_2026.02.02.nct')
    coverage = zonal.coverage_from_kml(kml_path(), raster.latitudes, raster.longitudes,
                                       name_field=NAME_FIELD)

    path = tmp_path / 'vesdre.npz'
    coverage.save(path)
    back = zonal.Coverage.load(path)

    assert back.zones == coverage.zones
    assert np.array_equal(back.cell_index, coverage.cell_index)
    assert np.allclose(back.weight, coverage.weight)
    assert back.matches(raster.latitudes, raster.longitudes)

    # the fingerprint hashes the coordinates, so a shifted grid of the same shape misses
    assert not back.matches(raster.latitudes + 0.5, raster.longitudes)


def test_a_reused_coverage_gives_the_same_numbers():
    raster = load('tethys_NOAA_GFS_0.25_2026.02.02.nct')

    fresh, _, coverage_info = raster.get_values_from_KML(
        kml=kml_path(), nameField=NAME_FIELD, getCoverageInfo=True)
    reused, _ = raster.get_values_from_KML(kml=kml_path(), coverage_info=coverage_info)

    assert np.allclose(fresh.to_numpy(), reused.to_numpy(), equal_nan=True)

    # and the same as going through zonal directly
    direct = zonal.apply_coverage(coverage_info, raster.data)
    assert np.allclose(np.sort(direct.ravel()), np.sort(fresh.to_numpy().ravel()),
                       equal_nan=True)


# --------------------------------------------------------------------------- #
# straight off the reader
# --------------------------------------------------------------------------- #

@grib_required
def test_era5land_hourly_aggregates_over_the_catchments(extracted_grib):
    '''
    The end-to-end path the library exists for: GRIB in, catchment time series out, at
    0.1 deg where the zones do span several cells.
    '''

    from meteoraster import utils

    raster = utils.read_ERA5Land_hourly(
        extracted_grib('era5land_tp_belgium.zip'), 'tp',
        file_next_year=extracted_grib('era5land_tp_belgium_next.zip'))

    coverage = zonal.coverage_from_kml(kml_path(), raster.latitudes, raster.longitudes,
                                       name_field=NAME_FIELD)
    assert coverage.empty_zones() == []
    counts = [coverage.cells(i)[0].size for i in range(len(coverage))]
    assert max(counts) >= 3, 'at 0.1 deg the larger catchments must span several cells'

    agg, _ = raster.get_values_from_KML(kml=kml_path(), nameField=NAME_FIELD)

    assert zone_level(agg) == ZONE_IDS
    assert len(agg) == 72                                  # hourly, 2025-11-01..11-03
    assert agg.index[0] == pd.Timestamp('2025-11-01 00:00')

    values = agg.to_numpy()
    assert np.isfinite(values).all(), 'the basin is inland, so no sea cells'
    assert values.min() >= -1e-3                           # mm/hr, float32 packing noise
    assert values.max() <= np.nanmax(raster.data) + 1e-6


@grib_required
def test_era5land_monthly_aggregates_over_the_catchments(extracted_grib):
    '''The global 0.1 deg monthly grid, so this also exercises a full-globe coverage.'''

    from meteoraster import utils

    raster = utils.readERA5Land_monthly(
        extracted_grib('era5land_monthly_tp_and_t2m.zip'), 't2m')

    agg, _ = raster.get_values_from_KML(kml=kml_path(), nameField=NAME_FIELD)

    assert zone_level(agg) == ZONE_IDS
    assert len(agg) == 1
    values = agg.to_numpy()
    assert np.isfinite(values).all()
    # March in eastern Belgium, degC
    assert -20 < values.min() and values.max() < 30
