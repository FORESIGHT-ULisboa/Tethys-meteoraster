'''
The documented contracts, exercised against real converted Tethys forecasts.

Three fixtures, each for what only it can show:

    c3s   1 production x 51 members x 6 pd.DateOffset(months) leadtimes, mm/month
          -> the monthly adaptation path and the ensemble axis
    ipma  1 production, leadtime spacing changing 1h -> 3h -> 6h, mm and C
          -> the variable-step resampling v3.0 added
    GFS   7 productions x uniform 3h leadtimes out to 16 days, mm/3h and C
          -> the forecast path, where productions are never mixed

`Resample`'s output covers `horizon // timestep` whole steps, so a target step that
does not divide the horizon truncates the tail. Mass conservation is therefore checked
against the fine 1 h result aggregated in blocks - an identity that holds regardless of
truncation - rather than against the input total.
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

from meteoraster import MeteoRaster, Resample
from meteoraster import io as mrio

from conftest import require_resource

C3S = 'tethys_c3s_ecmwf51_tprate_2026.07.nct'
IPMA_TP = 'tethys_ipma_tp_20260814.nct'
IPMA_T2M = 'tethys_ipma_t2m_20260806.nct'
GFS_TP = 'tethys_NOAA_GFS_0.25_2026.02.02.nct'
GFS_T2M = 'tethys_NOAA_GFS_0.25_2026.05.18.nct'

#: A land point inside the ipma domain (central Portugal) and one inside the GFS one.
IPMA_POINT = dict(lat=39.0, lon=-8.0)
GFS_POINT = dict(lat=50.0, lon=5.0)


def load(name):
    return MeteoRaster.load(require_resource(name), verbose=False)


def blocks_of(fine, step_hours):
    '''
    The 1 h result summed in blocks of step_hours, truncated the way Resample does.
    fine is a 1-D array of delay columns for one production stamp.
    '''

    count = fine.size // step_hours

    return fine[:count * step_hours].reshape(count, step_hours).sum(axis=1)


# --------------------------------------------------------------------------- #
# c3s: monthly adaptation, DateOffset leadtimes, the ensemble axis
# --------------------------------------------------------------------------- #

@pytest.fixture(scope='module')
def c3s():
    return load(C3S)


def test_c3s_header_keeps_calendar_leadtimes(c3s):
    '''
    Monthly leadtimes used to come back as bare integers on some xarray versions;
    io pins the decoding.
    '''

    header = mrio.read_header(require_resource(C3S))

    assert header.units == 'mm/month'
    assert header.shape == c3s.data.shape
    assert [lead.months for lead in header.leadtimes] == [1, 2, 3, 4, 5, 6]
    assert all(isinstance(lead, pd.DateOffset) for lead in header.leadtimes)


def test_c3s_round_trip_preserves_calendar_leadtimes(c3s, tmp_path):
    path = tmp_path / 'c3s.nct'
    c3s.save(path)

    back = MeteoRaster.load(path, verbose=False)

    assert [lead.months for lead in back.leadtimes] == [1, 2, 3, 4, 5, 6]
    assert np.allclose(back.data, c3s.data, equal_nan=True)
    assert back.units == c3s.units


def test_c3s_adjust_leadtimes_rebases_the_months(c3s):
    '''
    adjust_leadtimes measures the horizon by applying the last leadtime to the first
    production; reading `.dt.days` off it raised AttributeError on a raster whose
    leadtimes were already DateOffsets - that is, on exactly this data.
    '''

    adjusted = c3s.copy()
    adjusted.adjust_leadtimes(period='months')

    assert adjusted.data.shape == c3s.data.shape
    assert [lead.months for lead in adjusted.leadtimes] == [0, 1, 2, 3, 4, 5]
    # a one-to-one remapping, so nothing is averaged away
    assert np.isfinite(adjusted.data).mean() == np.isfinite(c3s.data).mean()


def test_c3s_members_reduce_while_streaming(c3s):
    blocks = list(mrio.iter_productions(require_resource(C3S), reduce_members='mean'))

    assert len(blocks) == 1
    stamp, block = blocks[0]
    assert block.shape == c3s.data.shape[2:]
    assert np.allclose(block, np.nanmean(c3s.data[0], axis=0), equal_nan=True)


def test_c3s_is_complete(c3s):
    assert c3s.is_complete(full_ensemble=True)
    assert mrio.read_header(require_resource(C3S)).complete


def test_resample_rejects_calendar_leadtimes(c3s):
    '''
    A month has no fixed duration, so Resample refuses it by name. Without the guard
    the caller only saw pandas' "Invalid type for timedelta scalar".
    '''

    frame = c3s.get_values_from_latlon(**IPMA_POINT)

    with pytest.raises(Exception, match='Calendar leadtimes'):
        Resample.resample(frame, '1D', 'sum')


def test_resample_timestep_is_a_different_path_and_takes_calendar_rules(c3s):
    '''
    MeteoRaster.resample_timestep aggregates along *production* time with pandas rules,
    so 'MS' is valid there and rejected in Resample. Easy to conflate.
    '''

    monthly = c3s.copy()
    monthly.resample_timestep('MS')

    assert monthly.data.shape == c3s.data.shape
    assert pd.DatetimeIndex(monthly.production_datetime).tolist() == \
        [pd.Timestamp('2026-07-01')]


# --------------------------------------------------------------------------- #
# ipma: leadtime spacing that changes within one forecast
# --------------------------------------------------------------------------- #

def leadtime_spans(raster):
    lead = pd.TimedeltaIndex(np.asarray(raster.leadtimes)).values

    return np.append(np.diff(lead), np.diff(lead)[-1])


def test_ipma_leadtimes_really_do_change_spacing():
    '''Guards the fixture itself: subsetting must not have flattened the spacing.'''

    spans = leadtime_spans(load(IPMA_TP))

    assert sorted(pd.Timedelta(s).total_seconds() / 3600
                  for s in np.unique(spans)) == [1.0, 3.0, 6.0]


@pytest.mark.parametrize('step_hours', [3, 6])
def test_ipma_sum_conserves_mass_across_the_spacing_change(step_hours):
    '''
    tp is an accumulation per leadtime column, so each column's total is spread over
    its own span. Weighting every column by one scalar step - what v2.x did - would
    scale the 3 h and 6 h columns by the wrong share.
    '''

    frame = load(IPMA_TP).get_values_from_latlon(**IPMA_POINT)

    fine = Resample.resample(frame, '1h', 'sum').to_numpy()[0]
    coarse = Resample.resample(frame, '%uh' % step_hours, 'sum').to_numpy()[0]

    # up-sampling to the gcd grid splits without loss
    assert np.isclose(np.nansum(fine), np.nansum(frame.to_numpy()))
    assert np.allclose(coarse, blocks_of(fine, step_hours))


def test_ipma_mean_is_step_invariant():
    '''t2m is intensive: resampling must not move its magnitude out of range.'''

    frame = load(IPMA_T2M).get_values_from_latlon(**IPMA_POINT)
    source = frame.to_numpy()
    low, high = np.nanmin(source), np.nanmax(source)

    for step in ('1h', '3h', '6h', '1D'):
        values = Resample.resample(frame, step, 'mean').to_numpy()
        finite = values[np.isfinite(values)]
        assert finite.min() >= low - 1e-6, step
        assert finite.max() <= high + 1e-6, step


def test_ipma_the_wrong_resampling_type_silently_changes_units():
    '''
    'sum' on a temperature adds degrees together. Nothing raises - which is why the
    units contract is worth a test rather than only prose.
    '''

    frame = load(IPMA_T2M).get_values_from_latlon(**IPMA_POINT)

    averaged = Resample.resample(frame, '1D', 'mean').to_numpy()
    added = Resample.resample(frame, '1D', 'sum').to_numpy()

    assert np.nanmax(averaged) < 60                      # still degC
    assert np.nanmax(added) > 5 * np.nanmax(averaged)     # 24 hourly values deep


# --------------------------------------------------------------------------- #
# GFS: several productions, so distinct forecasts must stay distinct
# --------------------------------------------------------------------------- #

@pytest.fixture(scope='module')
def gfs_precipitation():
    return load(GFS_TP).get_values_from_latlon(**GFS_POINT)


def test_gfs_is_a_multi_production_forecast(gfs_precipitation):
    assert gfs_precipitation.shape[0] == 7
    spans = np.unique(np.diff(pd.DatetimeIndex(gfs_precipitation.index).values))
    assert spans.size == 1 and spans[0] == np.timedelta64(1, 'D')


@pytest.mark.parametrize('step_hours', [3, 6, 24])
def test_gfs_sum_conserves_mass_per_production(gfs_precipitation, step_hours):
    fine = Resample.resample(gfs_precipitation, '1h', 'sum').to_numpy()[0]
    coarse = Resample.resample(gfs_precipitation, '%uh' % step_hours, 'sum').to_numpy()[0]

    assert np.isclose(np.nansum(fine), np.nansum(gfs_precipitation.to_numpy()[0]))
    assert np.allclose(coarse, blocks_of(fine, step_hours))


def test_gfs_mean_is_step_invariant():
    frame = load(GFS_T2M).get_values_from_latlon(**GFS_POINT)
    source = frame.to_numpy()

    for step in ('1h', '6h', '1D'):
        values = Resample.resample(frame, step, 'mean').to_numpy()
        finite = values[np.isfinite(values)]
        assert finite.min() >= np.nanmin(source) - 1e-6, step
        assert finite.max() <= np.nanmax(source) + 1e-6, step


def test_gfs_a_stamp_between_issues_shifts_the_leadtimes(gfs_precipitation):
    '''
    A production stamp between two issues reuses the last forecast, its leadtimes
    shifted by the elapsed time. So the 06:00 output at delay 0 must equal the 00:00
    output at delay 6 h - same window, same source forecast.
    '''

    out = Resample.resample(gfs_precipitation, '6h', 'sum')
    issue = pd.Timestamp(gfs_precipitation.index[0])

    at_issue = out.loc[issue].to_numpy()
    six_hours_later = out.loc[issue + pd.Timedelta('6h')].to_numpy()

    assert np.allclose(at_issue[1:], six_hours_later[:-1], equal_nan=True)


def test_gfs_productions_are_not_mixed(gfs_precipitation):
    '''
    Only leadtimes are aggregated: losing one forecast must not touch the outputs that
    read an earlier one.
    '''

    holed = gfs_precipitation.copy()
    holed.iloc[1] = np.nan                               # the second issue
    issues = pd.DatetimeIndex(holed.index)

    reference = Resample.resample(gfs_precipitation, '6h', 'sum')
    out = Resample.resample(holed, '6h', 'sum')

    before = out.index < issues[1]
    assert np.allclose(out[before].to_numpy(), reference[before].to_numpy(),
                       equal_nan=True)

    during = (out.index >= issues[1]) & (out.index < issues[2])
    assert np.isnan(out[during].to_numpy()).all()

    after = (out.index >= issues[2]) & (out.index < issues[3])
    assert np.allclose(out[after].to_numpy(), reference[after].to_numpy(),
                       equal_nan=True)
