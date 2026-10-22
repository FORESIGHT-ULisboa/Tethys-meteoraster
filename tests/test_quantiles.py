'''
Tests for get_quantiles_from_KML, which was non-functional before v2.6 - the
underlying __groupByQuantile carried an unconditional
`raise Exception('This must be reviewed.')` with the rest of the method dead
behind it.

The fixture field is 10 everywhere except a 1 in the centre cell of the 3x3 grid,
scaled by (leadtime + 1), so every expected value below is exact.
'''

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import RING_MIDDLE, RING_WHOLE, gdal_required, write_kml

pytestmark = gdal_required


@pytest.fixture
def two_zone_kml(tmp_path):
    return write_kml(tmp_path, [('whole', RING_WHOLE), ('middle', RING_MIDDLE)])


def test_returns_a_frame_instead_of_raising(grid_raster, two_zone_kml):
    aggregate, centroids = grid_raster.get_quantiles_from_KML(
        two_zone_kml, nameField='zone_id', quantiles=[0.0, 0.5, 1.0])

    assert isinstance(aggregate, pd.DataFrame)
    assert len(aggregate) == grid_raster.production_datetime.size
    assert sorted(centroids.index) == ['middle', 'whole']


def test_zone_and_quantile_levels_always_survive(grid_raster, two_zone_kml):
    aggregate, _ = grid_raster.get_quantiles_from_KML(
        two_zone_kml, nameField='zone_id', quantiles=[0.5])

    names = list(aggregate.columns.names)
    assert 'zone' in names
    # quantile survives even though only one was asked for.
    assert 'quantile' in names


def test_spatial_extremes_are_exact(grid_raster, two_zone_kml):
    aggregate, _ = grid_raster.get_quantiles_from_KML(
        two_zone_kml, nameField='zone_id', quantiles=[0.0, 1.0])

    first = (aggregate.xs(0, level='ensemble_member', axis=1)
                      .xs(pd.Timedelta(hours=0), level='leadtime', axis=1))

    # Whole grid at leadtime 0: values are 10 except the centre 1.
    assert first[('whole', 0.0)].iloc[0] == pytest.approx(1.0)
    assert first[('whole', 1.0)].iloc[0] == pytest.approx(10.0)
    # The middle zone is the single centre cell, so both extremes are 1.
    assert first[('middle', 0.0)].iloc[0] == pytest.approx(1.0)
    assert first[('middle', 1.0)].iloc[0] == pytest.approx(1.0)


def test_leadtime_scaling_shows_up_per_leadtime(grid_raster, two_zone_kml):
    aggregate, _ = grid_raster.get_quantiles_from_KML(
        two_zone_kml, nameField='zone_id', quantiles=[1.0])

    member = aggregate.xs(0, level='ensemble_member', axis=1)
    for leadtime in range(3):
        column = member.xs(pd.Timedelta(hours=leadtime), level='leadtime', axis=1)
        assert column[('whole', 1.0)].iloc[0] == pytest.approx(10.0 * (leadtime + 1))


def test_median_is_area_weighted_over_the_zone(grid_raster, two_zone_kml):
    aggregate, _ = grid_raster.get_quantiles_from_KML(
        two_zone_kml, nameField='zone_id', quantiles=[0.5])

    first = (aggregate.xs(0, level='ensemble_member', axis=1)
                      .xs(pd.Timedelta(hours=0), level='leadtime', axis=1))

    # Eight cells at 10 and one at 1, equal weights: the weighted median is 10.
    assert first[('whole', 0.5)].iloc[0] == pytest.approx(10.0)


def test_pooling_leadtimes_into_daily_blocks(grid_raster, two_zone_kml):
    '''
    The hourly-to-daily case: for one production, all the values in a zone across
    a block of leadtimes are pooled and the quantiles come from that amalgam.
    '''

    aggregate, _ = grid_raster.get_quantiles_from_KML(
        two_zone_kml, nameField='zone_id', quantiles=[0.0, 1.0],
        pool=('space', 'leadtime'), resampling='1D')

    # All three leadtimes fall in one day, so the leadtime level drops out.
    assert 'leadtime' not in list(aggregate.columns.names)

    member = aggregate.xs(0, level='ensemble_member', axis=1)
    # Pooled over leadtimes 0..2, the field spans 1*1 .. 10*3.
    assert member[('whole', 0.0)].iloc[0] == pytest.approx(1.0)
    assert member[('whole', 1.0)].iloc[0] == pytest.approx(30.0)


def test_pooling_members_drops_the_member_level(grid_raster, two_zone_kml):
    aggregate, _ = grid_raster.get_quantiles_from_KML(
        two_zone_kml, nameField='zone_id', quantiles=[0.5],
        pool=('space', 'members'))

    assert 'ensemble_member' not in list(aggregate.columns.names)
    assert 'leadtime' in list(aggregate.columns.names)


def test_pooling_productions_collapses_the_rows(grid_raster, two_zone_kml):
    aggregate, _ = grid_raster.get_quantiles_from_KML(
        two_zone_kml, nameField='zone_id', quantiles=[0.5],
        pool=('space', 'production'), resampling='7D')

    # Both productions fall in one weekly block.
    assert len(aggregate) == 1


def test_unpooled_resampling_leaves_rows_alone(grid_raster, two_zone_kml):
    aggregate, _ = grid_raster.get_quantiles_from_KML(
        two_zone_kml, nameField='zone_id', quantiles=[0.5], resampling='7D')

    # resampling only bites on a pooled axis, and none is pooled here.
    assert len(aggregate) == grid_raster.production_datetime.size


def test_missing_data_below_min_coverage_is_a_gap(grid_raster, two_zone_kml):
    raster = grid_raster.copy()
    raster.data[0, 0, 0, 1, 1] = np.nan          # the whole of the 'middle' zone

    aggregate, _ = raster.get_quantiles_from_KML(
        two_zone_kml, nameField='zone_id', quantiles=[0.5], min_coverage=0.8)

    first = (aggregate.xs(0, level='ensemble_member', axis=1)
                      .xs(pd.Timedelta(hours=0), level='leadtime', axis=1))

    assert np.isnan(first[('middle', 0.5)].iloc[0])
    # The whole-grid zone still has 8/9 of its area, so it survives.
    assert np.isfinite(first[('whole', 0.5)].iloc[0])


def test_coverage_info_can_be_reused(grid_raster, two_zone_kml):
    first, _, coverage_info = grid_raster.get_quantiles_from_KML(
        two_zone_kml, nameField='zone_id', quantiles=[0.5], getCoverageInfo=True)

    second, _ = grid_raster.get_quantiles_from_KML(
        two_zone_kml, coverage_info=coverage_info, quantiles=[0.5])

    pd.testing.assert_frame_equal(first, second)


def test_legacy_dense_coverage_info_is_accepted(grid_raster, two_zone_kml):
    from meteoraster import zonal

    coverage = zonal.coverage_from_kml(two_zone_kml, grid_raster.latitudes,
                                       grid_raster.longitudes, name_field='zone_id',
                                       normalize='cell_fraction')
    legacy = (coverage.to_dense(), None, coverage.centroids)

    aggregate, _ = grid_raster.get_quantiles_from_KML(
        two_zone_kml, coverage_info=legacy, quantiles=[0.0, 1.0])

    member = (aggregate.xs(0, level='ensemble_member', axis=1)
                       .xs(pd.Timedelta(hours=0), level='leadtime', axis=1))
    assert member[('whole', 1.0)].iloc[0] == pytest.approx(10.0)


def test_both_quantile_methods_run(grid_raster, two_zone_kml):
    for method in ('hazen', 'linear'):
        aggregate, _ = grid_raster.get_quantiles_from_KML(
            two_zone_kml, nameField='zone_id', quantiles=[0.0, 1.0], method=method)
        member = (aggregate.xs(0, level='ensemble_member', axis=1)
                           .xs(pd.Timedelta(hours=0), level='leadtime', axis=1))
        # The extremes are method-independent.
        assert member[('whole', 0.0)].iloc[0] == pytest.approx(1.0)
        assert member[('whole', 1.0)].iloc[0] == pytest.approx(10.0)


def test_precision_is_deprecated_and_ignored(grid_raster, two_zone_kml):
    with pytest.warns(DeprecationWarning, match='precision'):
        with_precision, _ = grid_raster.get_quantiles_from_KML(
            two_zone_kml, nameField='zone_id', quantiles=[0.5], precision=0.1)

    without, _ = grid_raster.get_quantiles_from_KML(
        two_zone_kml, nameField='zone_id', quantiles=[0.5])

    pd.testing.assert_frame_equal(with_precision, without)


def test_pool_must_include_space(grid_raster, two_zone_kml):
    with pytest.raises(ValueError, match="must include 'space'"):
        grid_raster.get_quantiles_from_KML(two_zone_kml, nameField='zone_id',
                                          pool=('leadtime',))


def test_unknown_pool_axis_is_named(grid_raster, two_zone_kml):
    with pytest.raises(ValueError, match='Unknown pool axes: zones'):
        grid_raster.get_quantiles_from_KML(two_zone_kml, nameField='zone_id',
                                          pool=('space', 'zones'))


def test_mistyped_name_field_is_refused(grid_raster, two_zone_kml):
    with pytest.raises(ValueError, match='zone_ID'):
        grid_raster.get_quantiles_from_KML(two_zone_kml, nameField='zone_ID')


def test_dateoffset_leadtimes_cannot_be_pooled_with_a_rule(two_zone_kml):
    from meteoraster import MeteoRaster

    field = np.array([[10.0, 10.0, 10.0], [10.0, 1.0, 10.0], [10.0, 10.0, 10.0]])
    raster = MeteoRaster(
        data=np.stack([field, field * 2])[None, None].astype(np.float32),
        latitudes=np.array([2.0, 1.0, 0.0]), longitudes=np.array([0.0, 1.0, 2.0]),
        production_datetime=pd.to_datetime(['2024-01-01']).values,
        leadtimes=np.array([pd.DateOffset(months=1), pd.DateOffset(months=2)]),
        units='mm', variable='tp', verbose=False)

    with pytest.raises(TypeError, match='no fixed length'):
        raster.get_quantiles_from_KML(two_zone_kml, nameField='zone_id',
                                      quantiles=[0.5], pool=('space', 'leadtime'),
                                      resampling='1D')

    # Pooling the whole axis needs no rule and must work.
    aggregate, _ = raster.get_quantiles_from_KML(
        two_zone_kml, nameField='zone_id', quantiles=[1.0],
        pool=('space', 'leadtime'), resampling=None)
    assert aggregate[('whole', 1.0)].iloc[0] == pytest.approx(20.0)


def test_a_coverage_for_another_grid_is_rejected(grid_raster, two_zone_kml):
    _, _, coverage = grid_raster.get_quantiles_from_KML(kml=str(two_zone_kml), nameField='zone_id',
                                                        getCoverageInfo=True)
    shifted = grid_raster.copy()
    shifted.latitudes = shifted.latitudes + 0.25

    with pytest.raises(ValueError, match='different grid'):
        shifted.get_quantiles_from_KML(kml=str(two_zone_kml), coverage_info=coverage)
    with pytest.raises(ValueError, match='different grid'):
        shifted.get_values_from_KML(kml=str(two_zone_kml), coverage_info=coverage)
