'''
The two surviving readers in meteoraster.utils, against real ERA5-Land GRIB.

Every reader in this module was dead code before v3.0 - they built their dict with
'productionDates' while MeteoRaster.__init__ reads 'production_datetime' - so these
tests exist to keep them alive.

cfgrib is NOT imported at module scope: it raises RuntimeError (not ImportError) when
the ecCodes binary is unreachable, which would fail collection rather than skip. The
grib_required marker in conftest.py probes it safely.

Fixture layout, as CDS delivers a month of ERA5-Land hourly (dataDate / steps present):

    2025-10-31   step 24        last hour of the previous month
    2025-11-01   steps 1-24     full day
    2025-11-02   steps 1-24     full day
    2025-11-03   steps 1-23     last hour withheld into era5land_tp_belgium_next.zip

For accumulations the increment over [t, t+1h) belongs at t, so step s maps to
date + (s-1)h; for instantaneous fields cfgrib's valid_time = date + step, so step s
maps to date + s. Both files therefore trim to the same 72 hourly stamps,
2025-11-01 00:00 .. 2025-11-03 23:00.
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

from conftest import grib_required

pytestmark = grib_required

FIRST_HOUR = pd.Timestamp('2025-11-01 00:00')
LAST_HOUR = pd.Timestamp('2025-11-03 23:00')
FULL_DAYS = ('2025-11-01', '2025-11-02')


@pytest.fixture
def utils():
    from meteoraster import utils

    return utils


# --------------------------------------------------------------------------- #
# readERA5Land_monthly
# --------------------------------------------------------------------------- #

@pytest.fixture(scope='session')
def monthly_grib(extracted_grib):
    return extracted_grib('era5land_monthly_tp_and_t2m.zip')


@pytest.mark.parametrize('variable, units', [('tp', 'mm/month'), ('t2m', 'C')])
def test_monthly_reads_a_single_month(utils, monthly_grib, variable, units):
    '''
    The file holds one message per variable, with different stepTypes ('avgas' for tp,
    'avgid' for 2t) - so each variable needs a shortName filter - and a single time
    step, which collapses the time dimension.
    '''

    raster = utils.readERA5Land_monthly(monthly_grib, variable)

    assert raster.data.shape == (1, 1, 1, 1801, 3600)
    assert raster.units == units
    assert raster.variable == variable
    assert pd.DatetimeIndex(np.atleast_1d(raster.production_datetime)).tolist() == \
        [pd.Timestamp('2026-03-01')]
    assert len(raster.leadtimes) == 1
    assert raster.leadtimes[0] == pd.DateOffset(days=0)

    # the geometry contract applied by __init__
    assert raster.latitudes.shape == raster.data.shape[-2:]
    assert raster.latitudes[0, 0] > raster.latitudes[-1, 0]
    assert raster.longitudes.min() >= -180 and raster.longitudes.max() <= 180


def test_monthly_tp_is_a_monthly_total(utils, monthly_grib):
    '''
    tp is delivered as a mean daily accumulation in m, so a month's total is
    value * 1000 * days_in_month. Checked against the raw message rather than against
    a plausibility band, because a missing factor is exactly what a band misses.
    '''

    xr = pytest.importorskip('xarray')

    raster = utils.readERA5Land_monthly(monthly_grib, 'tp')

    with xr.open_dataset(monthly_grib, engine='cfgrib', indexpath='',
                         backend_kwargs={'filter_by_keys': {'shortName': 'tp'}}) as source:
        raw = np.asarray(source['tp'].data, dtype=float)

    days = pd.Timestamp('2026-03-01').days_in_month
    assert days == 31
    assert np.allclose(raster.data[0, 0, 0], raw * 1000 * days, equal_nan=True)


def test_monthly_units_are_physical(utils, monthly_grib):
    precipitation = utils.readERA5Land_monthly(monthly_grib, 'tp')
    temperature = utils.readERA5Land_monthly(monthly_grib, 't2m')

    values = precipitation.data[np.isfinite(precipitation.data)]
    assert values.min() >= 0
    # mm/month. The wettest cells on Earth reach a few thousand; a missing
    # days_in_month factor would cap this near 100 (i.e. mm/day).
    assert 200 < values.max() < 8000

    values = temperature.data[np.isfinite(temperature.data)]
    assert -80 < values.min() and values.max() < 60      # degC, not Kelvin


def test_monthly_rejects_an_unknown_variable(utils, monthly_grib):
    with pytest.raises(Exception, match='must be defined'):
        utils.readERA5Land_monthly(monthly_grib, 'not_a_variable')


# --------------------------------------------------------------------------- #
# read_ERA5Land_hourly
# --------------------------------------------------------------------------- #

@pytest.fixture(scope='session')
def hourly_tp_grib(extracted_grib):
    return extracted_grib('era5land_tp_belgium.zip')


@pytest.fixture(scope='session')
def hourly_tp_next_grib(extracted_grib):
    return extracted_grib('era5land_tp_belgium_next.zip')


@pytest.fixture(scope='session')
def hourly_t2m_grib(extracted_grib):
    return extracted_grib('era5land_t2m_belgium.zip')


def _stamps(raster):
    return pd.DatetimeIndex(np.asarray(raster.production_datetime).ravel())


def test_hourly_instantaneous_stamps_are_valid_times(utils, hourly_t2m_grib):
    '''
    cfgrib's valid_time is date + step, so step 24 of 2025-10-31 is 2025-11-01 00:00.
    Stamping instantaneous fields at date + (step - step[0]) puts every value an hour
    early, which would make the first surviving stamp 2025-10-31 23:00.
    '''

    raster = utils.read_ERA5Land_hourly(hourly_t2m_grib, 't2m')

    stamps = _stamps(raster)
    assert raster.units == 'C'
    assert raster.data.shape == (72, 1, 1, 71, 141)
    assert stamps[0] == FIRST_HOUR
    assert stamps[-1] == LAST_HOUR
    assert (np.diff(stamps.values) == np.timedelta64(1, 'h')).all()

    values = raster.data[np.isfinite(raster.data)]
    assert -50 < values.min() and values.max() < 50       # degC, not Kelvin


def test_hourly_cumulative_is_de_accumulated(utils, hourly_tp_grib, hourly_tp_next_grib):
    raster = utils.read_ERA5Land_hourly(hourly_tp_grib, 'tp',
                                        file_next_year=hourly_tp_next_grib)

    stamps = _stamps(raster)
    assert raster.units == 'mm/hr'
    assert raster.data.shape == (72, 1, 1, 71, 141)
    assert stamps[0] == FIRST_HOUR
    assert stamps[-1] == LAST_HOUR
    assert (np.diff(stamps.values) == np.timedelta64(1, 'h')).all()

    # A de-accumulation that differences the wrong axis, or in the wrong direction,
    # produces negative rainfall. The float32 packing of the source accumulations
    # leaves increments of order 1e-5 mm, so the floor is that, not zero.
    assert np.nanmin(raster.data) >= -1e-3

    # ERA5-Land is land-only, so the NaNs are its sea mask - the same cells at every
    # hour. An hour-dependent NaN pattern is what a mishandled period boundary leaves
    # behind, so pin the mask rather than demanding everything be finite.
    missing = np.isnan(raster.data[:, 0, 0])
    assert (missing == missing[0]).all()
    assert 0 < missing[0].mean() < 0.5


def test_hourly_increments_re_accumulate_to_the_source(utils, hourly_tp_grib,
                                                       hourly_tp_next_grib):
    '''
    The numeric core: cumulatively summing a day's 24 returned increments must give
    back that day's step 1..24 accumulations from the GRIB, in mm.
    '''

    xr = pytest.importorskip('xarray')

    raster = utils.read_ERA5Land_hourly(hourly_tp_grib, 'tp',
                                        file_next_year=hourly_tp_next_grib)
    stamps = _stamps(raster)

    with xr.open_dataset(hourly_tp_grib, engine='cfgrib', indexpath='') as source:
        accumulated = source['tp'].data * 1000                # m -> mm
        dates = pd.DatetimeIndex(np.atleast_1d(source.time.data))

    for day in FULL_DAYS:
        position = dates.get_loc(pd.Timestamp(day))
        expected = accumulated[position]                      # [step, y, x], steps 1..24

        selection = stamps.floor('1D') == pd.Timestamp(day)
        assert selection.sum() == 24
        increments = raster.data[selection, 0, 0]             # [hour, y, x]

        assert np.allclose(np.cumsum(increments, axis=0), expected,
                           rtol=1e-4, atol=1e-4, equal_nan=True)


def test_hourly_borrows_the_last_hour_from_the_next_period(utils, hourly_tp_grib,
                                                          hourly_tp_next_grib):
    '''
    The last day's step 24 is delivered with the next period, whose first date is this
    period's last day carrying step 24 only. Without that file the hour is NaN and
    trim() drops it.
    '''

    with pytest.warns(UserWarning, match='following'):
        borrowed = utils.read_ERA5Land_hourly(hourly_tp_grib, 'tp',
                                              file_next_year=hourly_tp_next_grib)

    alone = utils.read_ERA5Land_hourly(hourly_tp_grib, 'tp')

    assert _stamps(borrowed)[-1] == LAST_HOUR
    assert _stamps(alone)[-1] == LAST_HOUR - pd.Timedelta(hours=1)
    assert alone.data.shape[0] == borrowed.data.shape[0] - 1

    # everything they share is identical
    assert np.allclose(alone.data, borrowed.data[:-1], equal_nan=True)


def test_hourly_drops_the_trailing_hour_of_the_previous_period(utils, hourly_tp_grib,
                                                              hourly_tp_next_grib):
    '''
    The first date carries only its last step. The guard used to be hard-coded to
    December 31, so it never fired on a month boundary.
    '''

    with pytest.warns(UserWarning, match='previous'):
        utils.read_ERA5Land_hourly(hourly_tp_grib, 'tp',
                                   file_next_year=hourly_tp_next_grib)


def test_the_deleted_readers_are_gone(utils):
    for name in ('readERA5_monthly', 'read_GFS', 'read_C3S', 'readCORDEX_monthly'):
        assert not hasattr(utils, name), '%s should have been removed in v3.0' % name
