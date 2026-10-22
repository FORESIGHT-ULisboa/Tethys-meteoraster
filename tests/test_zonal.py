'''
Numeric tests for meteoraster.zonal on a grid whose weights are computable on paper.

The 3x3 fixture grid (see conftest) has cells of area exactly 1, spanning
lon [j-0.5, j+0.5] and lat [2-i-0.5, 2-i+0.5], so every expected weight below is
an exact fraction rather than a tolerance. Those are square-degree weights, so the
on-paper tests pin area_weighting='planar'; the WGS84 weighting is tested against
pyproj at the end.
'''

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from meteoraster import zonal
from conftest import (GRID_LATS, GRID_LONS, RING_WHOLE, RING_MIDDLE, RING_WEST_HALF,
                      RING_ONE_AND_QUARTER, gdal_required, write_kml)

PLANAR = dict(area_weighting='planar')

WHOLE_WKT = 'POLYGON ((-0.5 -0.5, 2.5 -0.5, 2.5 2.5, -0.5 2.5, -0.5 -0.5))'
MIDDLE_WKT = 'POLYGON ((0.5 0.5, 1.5 0.5, 1.5 1.5, 0.5 1.5, 0.5 0.5))'
WEST_HALF_WKT = 'POLYGON ((-0.5 1.5, 0.0 1.5, 0.0 2.5, -0.5 2.5, -0.5 1.5))'
QUARTER_WKT = 'POLYGON ((-0.5 1.5, 0.75 1.5, 0.75 2.5, -0.5 2.5, -0.5 1.5))'


# --------------------------------------------------------------------------- #
# grid geometry (no GDAL needed)
# --------------------------------------------------------------------------- #

def test_cell_corners_cover_the_boundary_ring():
    corner_lat, corner_lon = zonal.cell_corners(GRID_LATS, GRID_LONS)

    assert corner_lat.shape == (4, 4)
    # Half a cell beyond the outermost centres, in both directions. The old
    # interpolator-based version could not extrapolate, so the outer row and
    # column of the grid never received any coverage.
    np.testing.assert_allclose(corner_lon[0], [-0.5, 0.5, 1.5, 2.5])
    np.testing.assert_allclose(corner_lat[:, 0], [2.5, 1.5, 0.5, -0.5])


def test_cell_areas_are_exact():
    corner_lat, corner_lon = zonal.cell_corners(GRID_LATS, GRID_LONS)

    np.testing.assert_allclose(zonal.cell_areas(corner_lat, corner_lon), 1.0)


def test_cell_corners_match_bilinear_on_a_curvilinear_grid():
    # A sheared grid: still affine in the indices, so the corner lattice must be
    # the exact midpoint arithmetic.
    rows, cols = np.meshgrid(np.arange(4.0), np.arange(5.0), indexing='ij')
    latitudes = 10.0 - rows + 0.25 * cols
    longitudes = 2.0 * cols + 0.5 * rows

    corner_lat, corner_lon = zonal.cell_corners(latitudes, longitudes)

    assert corner_lat.shape == (5, 6)
    # The interior corner shared by cells (0,0),(0,1),(1,0),(1,1) is the mean of
    # those four centres.
    assert corner_lat[1, 1] == pytest.approx(latitudes[:2, :2].mean())
    assert corner_lon[1, 1] == pytest.approx(longitudes[:2, :2].mean())


def test_grid_fingerprint_detects_a_shift_at_constant_shape():
    same = zonal.grid_fingerprint(GRID_LATS, GRID_LONS)

    assert zonal.grid_fingerprint(GRID_LATS, GRID_LONS) == same
    # Same shape, different geolocation: the case that otherwise yields plausible
    # numbers computed on the wrong pixels.
    assert zonal.grid_fingerprint(GRID_LATS + 0.01, GRID_LONS) != same


def test_as_grid_rejects_a_degenerate_grid():
    with pytest.raises(ValueError):
        zonal.as_grid(np.array([1.0]), np.array([1.0]))


# --------------------------------------------------------------------------- #
# building a coverage
# --------------------------------------------------------------------------- #

@gdal_required
def test_whole_grid_zone_catches_every_cell_including_the_corners():
    coverage = zonal.coverage_from_geometries(GRID_LATS, GRID_LONS, [WHOLE_WKT], ['all'],
                                              **PLANAR)

    index, weight = coverage.cells(0)

    assert sorted(index.tolist()) == list(range(9))
    np.testing.assert_allclose(weight, 1 / 9)


@gdal_required
def test_fractional_overlap_is_exact():
    coverage = zonal.coverage_from_geometries(
        GRID_LATS, GRID_LONS,
        [MIDDLE_WKT, WEST_HALF_WKT, QUARTER_WKT], ['middle', 'half', 'quarter'],
        normalize='cell_fraction')

    index, weight = coverage.cells(0)
    assert index.tolist() == [4] and weight == pytest.approx(1.0)

    index, weight = coverage.cells(1)
    assert index.tolist() == [0] and weight == pytest.approx(0.5)

    index, weight = coverage.cells(2)
    assert index.tolist() == [0, 1]
    np.testing.assert_allclose(weight, [1.0, 0.25])


@gdal_required
def test_area_normalisation_sums_to_one_per_zone():
    coverage = zonal.coverage_from_geometries(
        GRID_LATS, GRID_LONS, [WHOLE_WKT, QUARTER_WKT], ['all', 'quarter'])

    np.testing.assert_allclose(coverage.total_weight(), [1.0, 1.0])


@gdal_required
def test_normalisation_does_not_change_the_mean():
    values = np.arange(9, dtype=float).reshape(1, 3, 3)
    means = []
    for normalize in zonal.NORMALISATIONS:
        coverage = zonal.coverage_from_geometries(GRID_LATS, GRID_LONS, [QUARTER_WKT],
                                                 ['quarter'], normalize=normalize)
        means.append(zonal.apply_coverage(coverage, values)[0, 0])

    np.testing.assert_allclose(means, means[0])


@gdal_required
def test_geometry_accepts_wkb_and_ogr_objects():
    ogr = pytest.importorskip('osgeo.ogr')
    geometry = ogr.CreateGeometryFromWkt(MIDDLE_WKT)

    from_wkt = zonal.coverage_from_geometries(GRID_LATS, GRID_LONS, [MIDDLE_WKT], ['m'])
    from_wkb = zonal.coverage_from_geometries(GRID_LATS, GRID_LONS,
                                             [bytes(geometry.ExportToWkb())], ['m'])
    from_ogr = zonal.coverage_from_geometries(GRID_LATS, GRID_LONS, [geometry], ['m'])

    for other in (from_wkb, from_ogr):
        np.testing.assert_array_equal(other.cells(0)[0], from_wkt.cells(0)[0])
        np.testing.assert_allclose(other.cells(0)[1], from_wkt.cells(0)[1])


@gdal_required
def test_zone_outside_the_grid_warns_and_aggregates_to_nan():
    with pytest.warns(UserWarning, match='caught no grid cell'):
        coverage = zonal.coverage_from_geometries(
            GRID_LATS, GRID_LONS,
            [WHOLE_WKT, 'POLYGON ((50 50, 51 50, 51 51, 50 51, 50 50))'],
            ['all', 'elsewhere'])

    assert coverage.empty_zones() == ['elsewhere']
    aggregate = zonal.apply_coverage(coverage, np.ones((1, 3, 3)))
    assert np.isfinite(aggregate[0, 0])
    assert np.isnan(aggregate[0, 1])


@gdal_required
def test_zone_outside_the_grid_can_be_made_an_error():
    with pytest.raises(ValueError, match='caught no grid cell'):
        zonal.coverage_from_geometries(
            GRID_LATS, GRID_LONS, ['POLYGON ((50 50, 51 50, 51 51, 50 51, 50 50))'],
            ['elsewhere'], on_empty='raise')


@gdal_required
def test_halo_does_not_change_the_weights():
    plain = zonal.coverage_from_geometries(GRID_LATS, GRID_LONS, [QUARTER_WKT], ['q'], halo=0)
    padded = zonal.coverage_from_geometries(GRID_LATS, GRID_LONS, [QUARTER_WKT], ['q'], halo=3)

    np.testing.assert_array_equal(plain.cells(0)[0], padded.cells(0)[0])
    np.testing.assert_allclose(plain.cells(0)[1], padded.cells(0)[1])


@gdal_required
def test_cos_lat_weighting_rescales_but_keeps_a_uniform_field():
    values = np.full((1, 3, 3), 4.0)
    planar = zonal.coverage_from_geometries(GRID_LATS, GRID_LONS, [WHOLE_WKT], ['all'],
                                            **PLANAR)
    scaled = zonal.coverage_from_geometries(GRID_LATS, GRID_LONS, [WHOLE_WKT], ['all'],
                                           area_weighting='cos_lat')

    assert not np.allclose(planar.cells(0)[1], scaled.cells(0)[1])
    # A constant field averages to the same constant under either weighting.
    assert zonal.apply_coverage(scaled, values)[0, 0] == pytest.approx(4.0)


# --------------------------------------------------------------------------- #
# KML reading
# --------------------------------------------------------------------------- #

@gdal_required
def test_namespaced_kml_is_read_without_rewriting_it(tmp_path):
    path = write_kml(tmp_path, [('whole', RING_WHOLE)], namespaced=True)
    before = path.read_bytes()

    coverage = zonal.coverage_from_kml(path, GRID_LATS, GRID_LONS, name_field='zone_id')

    assert coverage.zones == ['whole']
    assert coverage.cells(0)[0].size == 9
    # The old reader stripped the namespace by rewriting the caller's file.
    assert path.read_bytes() == before


@gdal_required
def test_indented_coordinates_are_parsed(tmp_path):
    ring = '\n        ' + RING_WHOLE.replace(' ', '\n        ') + '\n      '
    path = write_kml(tmp_path, [('whole', ring)])

    coverage = zonal.coverage_from_kml(path, GRID_LATS, GRID_LONS, name_field='zone_id')

    assert coverage.cells(0)[0].size == 9


@gdal_required
def test_coordinates_with_spaces_around_commas_are_parsed(tmp_path):
    path = write_kml(tmp_path, [('whole', RING_WHOLE.replace(',', ' , '))])

    coverage = zonal.coverage_from_kml(path, GRID_LATS, GRID_LONS, name_field='zone_id')

    assert coverage.cells(0)[0].size == 9


@gdal_required
def test_single_zone_layer_keeps_its_zone(tmp_path):
    path = write_kml(tmp_path, [('only', RING_WHOLE)])

    coverage = zonal.coverage_from_kml(path, GRID_LATS, GRID_LONS, name_field='zone_id')

    assert coverage.zones == ['only']
    assert len(coverage) == 1


@gdal_required
def test_mistyped_name_field_raises_instead_of_returning_no_zones(tmp_path):
    path = write_kml(tmp_path, [('whole', RING_WHOLE)])

    with pytest.raises(ValueError, match='zone_ID'):
        zonal.coverage_from_kml(path, GRID_LATS, GRID_LONS, name_field='zone_ID')


@gdal_required
def test_mistyped_name_field_can_be_skipped(tmp_path):
    path = write_kml(tmp_path, [('whole', RING_WHOLE)])

    with pytest.raises(ValueError, match='No usable Placemark'):
        zonal.coverage_from_kml(path, GRID_LATS, GRID_LONS, name_field='zone_ID',
                                on_error='skip')


@gdal_required
def test_duplicate_zone_names_raise(tmp_path):
    path = write_kml(tmp_path, [('same', RING_WHOLE), ('same', RING_MIDDLE)])

    with pytest.raises(ValueError, match='not unique'):
        zonal.coverage_from_kml(path, GRID_LATS, GRID_LONS, name_field='zone_id')


@gdal_required
def test_polygon_with_a_hole(tmp_path):
    # The whole grid minus the centre cell.
    path = tmp_path / 'holed.kml'
    path.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<kml xmlns="http://www.opengis.net/kml/2.2"><Document><Placemark>'
        '<ExtendedData><SchemaData><SimpleData name="zone_id">holed</SimpleData>'
        '</SchemaData></ExtendedData><Polygon>'
        '<outerBoundaryIs><LinearRing><coordinates>' + RING_WHOLE +
        '</coordinates></LinearRing></outerBoundaryIs>'
        '<innerBoundaryIs><LinearRing><coordinates>' + RING_MIDDLE +
        '</coordinates></LinearRing></innerBoundaryIs>'
        '</Polygon></Placemark></Document></kml>\n', encoding='utf-8')

    coverage = zonal.coverage_from_kml(path, GRID_LATS, GRID_LONS, name_field='zone_id',
                                       **PLANAR)

    index, weight = coverage.cells(0)
    assert 4 not in index.tolist()          # the centre cell is the hole
    assert index.size == 8
    np.testing.assert_allclose(weight, 1 / 8)


@gdal_required
def test_name_field_falls_back_to_the_name_element(tmp_path):
    path = tmp_path / 'named.kml'
    path.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<kml xmlns="http://www.opengis.net/kml/2.2"><Document><Placemark>'
        '<name>basin</name><Polygon><outerBoundaryIs><LinearRing><coordinates>'
        + RING_WHOLE +
        '</coordinates></LinearRing></outerBoundaryIs></Polygon>'
        '</Placemark></Document></kml>\n', encoding='utf-8')

    coverage = zonal.coverage_from_kml(path, GRID_LATS, GRID_LONS, name_field='name')

    assert coverage.zones == ['basin']


# --------------------------------------------------------------------------- #
# applying a coverage
# --------------------------------------------------------------------------- #

@gdal_required
def test_area_weighted_mean_uses_the_whole_grid():
    coverage = zonal.coverage_from_geometries(GRID_LATS, GRID_LONS, [WHOLE_WKT], ['all'],
                                              **PLANAR)
    values = np.arange(9, dtype=float).reshape(1, 3, 3)

    # mean(0..8) = 4.0, not the centre cell's 4 by coincidence: see the asymmetric
    # field in test_missing_data_is_renormalised below.
    assert zonal.apply_coverage(coverage, values)[0, 0] == pytest.approx(4.0)


@gdal_required
def test_weighted_mean_of_partial_cells():
    coverage = zonal.coverage_from_geometries(GRID_LATS, GRID_LONS, [QUARTER_WKT], ['q'])
    values = np.arange(9, dtype=float).reshape(1, 3, 3)

    # cell 0 (value 0) with weight 1, cell 1 (value 1) with weight 0.25
    assert zonal.apply_coverage(coverage, values)[0, 0] == pytest.approx(0.25 / 1.25)


@gdal_required
def test_missing_data_is_renormalised_per_zone():
    coverage = zonal.coverage_from_geometries(GRID_LATS, GRID_LONS, [WHOLE_WKT], ['all'],
                                              **PLANAR)
    values = np.arange(9, dtype=float).reshape(1, 3, 3)

    values[0, 0, 0] = np.nan
    # The remaining eight weights renormalise, so the answer is mean(1..8), not
    # mean(0..8) with a zero substituted.
    got = zonal.apply_coverage(coverage, values, min_coverage=0.0)[0, 0]
    assert got == pytest.approx(np.arange(1, 9).mean())


@gdal_required
def test_min_coverage_is_tested_per_zone():
    # The old code took zone 0 as representative and applied its verdict to every
    # zone, so a good zone could be blanked and a fully-missing one kept.
    coverage = zonal.coverage_from_geometries(
        GRID_LATS, GRID_LONS, [MIDDLE_WKT, WHOLE_WKT], ['middle', 'all'])
    values = np.ones((1, 3, 3))
    values[0, 1, 1] = np.nan                 # kills 'middle' entirely, 1/9 of 'all'

    got = zonal.apply_coverage(coverage, values, min_coverage=0.8)

    assert np.isnan(got[0, 0])               # middle has no data at all
    assert got[0, 1] == pytest.approx(1.0)   # all still has 8/9 of its area


@gdal_required
def test_apply_coverage_accepts_any_trailing_grid_axes():
    coverage = zonal.coverage_from_geometries(GRID_LATS, GRID_LONS, [WHOLE_WKT, MIDDLE_WKT],
                                              ['all', 'middle'])
    cube = np.random.default_rng(0).normal(size=(2, 3, 4, 3, 3))

    full = zonal.apply_coverage(coverage, cube)
    assert full.shape == (2, 3, 4, 2)

    # One streamed block must give the same numbers as the corresponding slice.
    block = zonal.apply_coverage(coverage, cube[1, 2])
    np.testing.assert_allclose(block, full[1, 2])


@gdal_required
def test_apply_coverage_preserves_float32():
    coverage = zonal.coverage_from_geometries(GRID_LATS, GRID_LONS, [WHOLE_WKT], ['all'])
    values = np.ones((1, 3, 3), dtype=np.float32)

    assert zonal.apply_coverage(coverage, values).dtype == np.float32


@gdal_required
def test_apply_coverage_rejects_a_mismatched_grid():
    coverage = zonal.coverage_from_geometries(GRID_LATS, GRID_LONS, [WHOLE_WKT], ['all'])

    with pytest.raises(ValueError, match='coverage was built for'):
        zonal.apply_coverage(coverage, np.ones((1, 4, 4)))


# --------------------------------------------------------------------------- #
# persistence
# --------------------------------------------------------------------------- #

@gdal_required
def test_coverage_round_trips_without_pickle(tmp_path):
    coverage = zonal.coverage_from_geometries(GRID_LATS, GRID_LONS, [WHOLE_WKT, MIDDLE_WKT],
                                              ['all', 'middle'])
    path = tmp_path / 'coverage.npz'
    coverage.save(path)

    # allow_pickle=False is what makes the payload safe to read on any install.
    with np.load(str(path), allow_pickle=False) as payload:
        assert 'weight' in payload

    back = zonal.Coverage.load(path)

    assert back.zones == coverage.zones
    assert back.shape == coverage.shape
    np.testing.assert_allclose(back.weight, coverage.weight)
    np.testing.assert_array_equal(back.cell_index, coverage.cell_index)
    assert back.fingerprint == coverage.fingerprint
    assert back.matches(GRID_LATS, GRID_LONS)
    assert not back.matches(GRID_LATS + 0.01, GRID_LONS)


@gdal_required
def test_dense_round_trip_matches_the_legacy_layout():
    coverage = zonal.coverage_from_geometries(GRID_LATS, GRID_LONS, [WHOLE_WKT, MIDDLE_WKT],
                                              ['all', 'middle'])

    dense = coverage.to_dense()
    assert dense.shape == (1, 1, 1, 3, 3, 2)
    assert dense.dtype == np.single

    back = zonal.Coverage.from_dense(dense, zones=coverage.zones)
    np.testing.assert_allclose(back.weight, coverage.weight)
    np.testing.assert_array_equal(back.cell_index, coverage.cell_index)


# --------------------------------------------------------------------------- #
# weighted quantiles (no GDAL needed)
# --------------------------------------------------------------------------- #

QUANTILES = np.array([0.0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0])


@pytest.mark.parametrize('size', [1, 2, 3, 5, 8, 17])
@pytest.mark.parametrize('method', ['hazen', 'linear'])
def test_equal_weights_reduce_to_numpy(size, method):
    rng = np.random.default_rng(size)
    for _ in range(20):
        values = rng.normal(size=size)

        np.testing.assert_allclose(
            zonal.weighted_quantiles(values, np.ones(size), QUANTILES, method=method),
            np.quantile(values, QUANTILES, method=method), atol=1e-12)


def test_equal_weights_are_scale_invariant():
    values = np.random.default_rng(1).normal(size=13)

    np.testing.assert_allclose(
        zonal.weighted_quantiles(values, np.full(13, 3.7), QUANTILES),
        zonal.weighted_quantiles(values, np.ones(13), QUANTILES))


def test_weights_move_the_median():
    values = np.array([0.0, 10.0])

    balanced = zonal.weighted_quantiles(values, np.array([1.0, 1.0]), [0.5])[0]
    heavy_low = zonal.weighted_quantiles(values, np.array([9.0, 1.0]), [0.5])[0]
    heavy_high = zonal.weighted_quantiles(values, np.array([1.0, 9.0]), [0.5])[0]

    assert balanced == pytest.approx(5.0)
    assert heavy_low < balanced < heavy_high


def test_hazen_positions_land_on_replicated_block_centres():
    # This is the relationship with the "unfold each cell round(w/precision) times"
    # trick the implementation replaces: the weighted positions are the centres of
    # each value's block in the replicated sample.
    values = np.array([1.0, 2.0, 3.0, 4.0])
    weights = np.array([1.0, 3.0, 2.0, 1.0])
    centres = (np.cumsum(weights) - weights / 2) / weights.sum()

    np.testing.assert_allclose(zonal.weighted_quantiles(values, weights, centres), values)
    np.testing.assert_allclose(
        np.quantile(np.repeat(values, weights.astype(int)), centres, method='hazen'), values)


def test_endpoints_are_the_extremes():
    rng = np.random.default_rng(2)
    values = rng.normal(size=9)
    weights = rng.uniform(0.1, 1.0, size=9)

    low, high = zonal.weighted_quantiles(values, weights, [0.0, 1.0])

    assert low == pytest.approx(values.min())
    assert high == pytest.approx(values.max())


def test_quantiles_are_monotone_in_q():
    rng = np.random.default_rng(3)
    got = zonal.weighted_quantiles(rng.normal(size=12), rng.uniform(0.1, 1.0, 12),
                                   np.linspace(0, 1, 51))

    assert np.all(np.diff(got) >= -1e-12)


def test_non_finite_values_are_excluded_not_propagated():
    with_nan = zonal.weighted_quantiles(np.array([5.0, np.nan, 1.0, 3.0]), np.ones(4), QUANTILES)
    without = zonal.weighted_quantiles(np.array([5.0, 1.0, 3.0]), np.ones(3), QUANTILES)

    np.testing.assert_allclose(with_nan, without)


def test_zero_weight_is_the_same_as_dropping_the_value():
    weighted = zonal.weighted_quantiles(np.array([5.0, 1.0, 99.0, 3.0]),
                                        np.array([1.0, 1.0, 0.0, 1.0]), QUANTILES)
    dropped = zonal.weighted_quantiles(np.array([5.0, 1.0, 3.0]), np.ones(3), QUANTILES)

    np.testing.assert_allclose(weighted, dropped)


def test_degenerate_samples():
    assert np.isnan(zonal.weighted_quantiles(np.full(4, np.nan), np.ones(4), [0.5])[0])
    np.testing.assert_allclose(
        zonal.weighted_quantiles(np.array([np.nan, 7.0, np.nan]), np.ones(3), QUANTILES), 7.0)


def test_batching_matches_a_per_slice_loop():
    rng = np.random.default_rng(4)
    values = rng.normal(size=(4, 3, 11))
    weights = rng.uniform(0.1, 1.0, size=11)
    values[0, 0, :] = np.nan            # an entirely missing slice
    values[1, 2, 3:] = np.nan           # a partly missing slice

    batch = zonal.weighted_quantiles(values, weights, QUANTILES)
    loop = np.stack([[zonal.weighted_quantiles(values[i, j], weights, QUANTILES)
                      for j in range(3)] for i in range(4)])

    np.testing.assert_allclose(batch, loop, equal_nan=True)


def test_weighted_quantiles_rejects_a_bad_method_and_range():
    with pytest.raises(ValueError):
        zonal.weighted_quantiles(np.arange(3.0), np.ones(3), [0.5], method='nope')
    with pytest.raises(ValueError):
        zonal.weighted_quantiles(np.arange(3.0), np.ones(3), [1.5])


# --------------------------------------------------------------------------- #
# quantiles over a coverage
# --------------------------------------------------------------------------- #

@gdal_required
def test_quantiles_from_coverage_is_area_weighted():
    coverage = zonal.coverage_from_geometries(GRID_LATS, GRID_LONS, [WHOLE_WKT], ['all'],
                                              normalize='cell_fraction')
    values = np.arange(9, dtype=float).reshape(1, 3, 3)

    got = zonal.quantiles_from_coverage(coverage, values, [0.0, 0.5, 1.0])

    assert got.shape == (1, 3, 1)
    # Equal weights over the whole grid, so this is the plain quantile.
    np.testing.assert_allclose(got[0, :, 0],
                               np.quantile(np.arange(9.0), [0.0, 0.5, 1.0], method='hazen'))


@gdal_required
def test_quantiles_from_coverage_pools_a_leading_axis():
    coverage = zonal.coverage_from_geometries(GRID_LATS, GRID_LONS, [WHOLE_WKT], ['all'],
                                              normalize='cell_fraction')
    values = np.stack([np.full((3, 3), 1.0), np.full((3, 3), 5.0)])   # [lead, y, x]

    unpooled = zonal.quantiles_from_coverage(coverage, values, [1.0])
    pooled = zonal.quantiles_from_coverage(coverage, values, [1.0], pool_axes=(0,))

    np.testing.assert_allclose(unpooled[:, 0, 0], [1.0, 5.0])
    assert pooled.shape == (1, 1)
    assert pooled[0, 0] == pytest.approx(5.0)


@gdal_required
def test_quantiles_from_coverage_respects_min_coverage():
    coverage = zonal.coverage_from_geometries(GRID_LATS, GRID_LONS, [MIDDLE_WKT], ['middle'],
                                              normalize='cell_fraction')
    values = np.ones((1, 3, 3))
    values[0, 1, 1] = np.nan

    got = zonal.quantiles_from_coverage(coverage, values, [0.5], min_coverage=0.8)

    assert np.isnan(got[0, 0, 0])


# --------------------------------------------------------------------------- #
# WGS84 area weighting, grid fraction, reuse checks
# --------------------------------------------------------------------------- #

def _geod_rectangle_km2(west, east, south, north):
    '''pyproj area of a lon/lat rectangle, parallels densified so geodesic == straight.'''
    pyproj = pytest.importorskip('pyproj')
    t = np.linspace(0.0, 1.0, 400)
    lons = np.concatenate([west + (east - west) * t, np.full(t.size, east),
                           east - (east - west) * t, np.full(t.size, west)])
    lats = np.concatenate([np.full(t.size, south), south + (north - south) * t,
                           np.full(t.size, north), north - (north - south) * t])
    area, _ = pyproj.Geod(ellps='WGS84').polygon_area_perimeter(lons, lats)
    return abs(area) / 1e6


@pytest.mark.parametrize('south', [-89.0, -45.0, 0.0, 30.0, 60.0, 75.0, 88.0])
def test_ellipsoidal_ring_area_matches_pyproj(south):
    expected = _geod_rectangle_km2(10.0, 12.0, south, south + 1)

    area = zonal.ring_area(np.array([10.0, 12.0, 12.0, 10.0]),
                           np.array([south, south, south + 1, south + 1]))

    assert area == pytest.approx(expected, rel=1e-6)


def test_ellipsoidal_cell_areas_are_true_areas():
    lats = np.array([60.5, 60.0, 59.5])
    lons = np.array([0.0, 0.5, 1.0])
    corner_lat, corner_lon = zonal.cell_corners(lats, lons)

    areas = zonal.cell_areas(corner_lat, corner_lon, 'ellipsoidal')

    assert areas[0, 0] == pytest.approx(_geod_rectangle_km2(-0.25, 0.25, 60.25, 60.75), rel=1e-6)
    assert areas[2, 0] == pytest.approx(_geod_rectangle_km2(-0.25, 0.25, 59.25, 59.75), rel=1e-6)


@gdal_required
def test_ellipsoidal_is_the_default_and_planar_reproduces_v30():
    lats = np.arange(70.0, 49.0, -1.0)
    lons = np.arange(0.0, 6.0, 1.0)
    wkt = 'POLYGON ((0.3 50.3, 5.3 50.3, 5.3 69.3, 0.3 69.3, 0.3 50.3))'
    field = np.broadcast_to(np.meshgrid(lons, lats)[1], (1,) + (lats.size, lons.size))

    default = zonal.coverage_from_geometries(lats, lons, [wkt], ['z'])
    planar = zonal.coverage_from_geometries(lats, lons, [wkt], ['z'], area_weighting='planar')

    assert default.area_weighting == 'ellipsoidal'
    # planar is the plain-degree centroid; true areas pull it south (cos lat)
    planar_mean = zonal.apply_coverage(planar, field)[0, 0]
    true_mean = zonal.apply_coverage(default, field)[0, 0]
    assert planar_mean == pytest.approx(59.8, abs=1e-6)
    assert true_mean < planar_mean - 0.5


@gdal_required
def test_grid_fraction_reports_a_zone_partly_outside_the_grid():
    half_out = 'POLYGON ((1.5 -0.5, 3.5 -0.5, 3.5 2.5, 1.5 2.5, 1.5 -0.5))'

    with pytest.warns(UserWarning, match='partly outside the grid'):
        coverage = zonal.coverage_from_geometries(GRID_LATS, GRID_LONS, [WHOLE_WKT, half_out],
                                                  ['all', 'half'])

    np.testing.assert_allclose(coverage.grid_fraction, [1.0, 0.5], rtol=1e-3)
    assert coverage.centroids.loc['half', 'grid_fraction'] == pytest.approx(0.5, rel=1e-3)

    with pytest.raises(ValueError, match='partly outside the grid'):
        zonal.coverage_from_geometries(GRID_LATS, GRID_LONS, [half_out], ['half'],
                                       on_partial='raise')


@gdal_required
def test_a_zone_on_the_grid_border_is_fully_inside():
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        coverage = zonal.coverage_from_geometries(GRID_LATS, GRID_LONS, [WHOLE_WKT, QUARTER_WKT],
                                                  ['all', 'quarter'])

    np.testing.assert_allclose(coverage.grid_fraction, 1.0)


@gdal_required
def test_an_antimeridian_polygon_warns():
    lats = np.array([56.0, 55.0, 54.0])
    lons = np.array([178.0, 179.0, 180.0])
    with pytest.warns(UserWarning, match='antimeridian'):
        zonal.coverage_from_geometries(
            lats, lons, ['POLYGON ((179 54.5, -179 54.5, -179 55.5, 179 55.5, 179 54.5))'],
            ['dateline'], on_partial='ignore')


@gdal_required
def test_grid_fraction_round_trips_and_old_files_still_load(tmp_path):
    coverage = zonal.coverage_from_geometries(GRID_LATS, GRID_LONS, [WHOLE_WKT], ['all'])
    coverage.save(tmp_path / 'new.npz')
    back = zonal.Coverage.load(tmp_path / 'new.npz')
    np.testing.assert_allclose(back.grid_fraction, coverage.grid_fraction)
    assert 'grid_fraction' in back.centroids.columns

    # a v3.0 file has no grid_fraction key
    coverage.grid_fraction = None
    coverage.save(tmp_path / 'old.npz')
    old = zonal.Coverage.load(tmp_path / 'old.npz')
    assert old.grid_fraction is None
    np.testing.assert_allclose(old.weight, coverage.weight)

