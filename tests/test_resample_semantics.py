"""Semantics of Resample: what up/down-sampling does to each kind of quantity.

The contract under test (see the Resample docstring):
  'sum'     extensive  - a total per step; splits on up-sampling, adds on down-sampling
  'mean'    intensive  - an interval average; replicates on up-sampling, averages down
  'linear'  intensive  - an instantaneous value; interpolates / samples at the stamp
  'max'     intensive  - an extreme; replicates on up-sampling, maxima down

Single-leadtime frames (time series, aggregated along production time) and
multi-leadtime frames (forecasts, aggregated along leadtime only) must agree on all
four.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from meteoraster import Resample

TYPES = ("sum", "mean", "linear", "max")


def series_frame(values, freq="1D", start="2023-01-01", series=("A",)):
    """A production x leadtime frame with a single 0-day leadtime: a plain time series."""
    values = np.asarray(values, dtype=float)
    idx = pd.date_range(start, periods=values.size, freq=freq)
    cols = pd.MultiIndex.from_tuples([(s, pd.Timedelta("0D")) for s in series], names=["series", "leadtime"])
    return pd.DataFrame(np.tile(values[:, None], (1, len(series))), index=idx, columns=cols)


def forecast_frame(values, prod="6h", lead="3h", start="2023-01-01", series=("A",)):
    """A production x leadtime frame with several leadtimes: a sequence of forecasts."""
    values = np.asarray(values, dtype=float)
    idx = pd.date_range(start, periods=values.shape[0], freq=prod)
    lts = pd.timedelta_range("0h", periods=values.shape[1], freq=lead)
    cols = pd.MultiIndex.from_product([list(series), lts], names=["series", "leadtime"])
    return pd.DataFrame(np.tile(values, (1, len(series))), index=idx, columns=cols)


def row(out, i=0):
    return out.iloc[i].to_numpy()


# --------------------------------------------------------------- 1-D: time series

def test_series_identity_is_a_no_op():
    frame = series_frame([1.0, 2.0, 3.0, 4.0])
    for resampling_type in TYPES:
        out = Resample.resample(frame, "1D", resampling_type)
        assert np.allclose(out.to_numpy().ravel(), [1.0, 2.0, 3.0, 4.0]), resampling_type


def test_series_downsampling():
    frame = series_frame([1.0, 2.0, 3.0, 4.0])
    expected = {"sum": [3.0, 7.0], "mean": [1.5, 3.5], "linear": [1.0, 3.0], "max": [2.0, 4.0]}
    for resampling_type, want in expected.items():
        out = Resample.resample(frame, "2D", resampling_type)
        assert np.allclose(out.to_numpy().ravel(), want), resampling_type


def test_series_upsampling():
    """A daily 1.0 becomes four 6-hourly 0.25 as a total, but stays 1.0 as an average."""
    frame = series_frame(np.ones(4))
    expected = {"sum": 0.25, "mean": 1.0, "max": 1.0}
    for resampling_type, want in expected.items():
        out = Resample.resample(frame, "6h", resampling_type)
        values = out.to_numpy().ravel()
        assert np.allclose(values[np.isfinite(values)], want), resampling_type


def test_series_linear_interpolates_between_stamps():
    frame = series_frame([1.0, 2.0, 3.0])
    out = Resample.resample(frame, "6h", "linear")
    assert np.allclose(out.to_numpy().ravel()[:5], [1.0, 1.25, 1.5, 1.75, 2.0])


# ------------------------------------------------------------- 2-D: forecast frames

def test_forecast_identity_is_a_no_op():
    frame = forecast_frame(np.arange(1.0, 17.0).reshape(2, 8))
    for resampling_type in TYPES:
        out = Resample.resample(frame, "3h", resampling_type)
        assert np.allclose(row(out), np.arange(1.0, 9.0)), resampling_type


def test_forecast_downsampling_along_leadtime():
    frame = forecast_frame(np.arange(1.0, 17.0).reshape(2, 8))
    expected = {"sum": [3.0, 7.0, 11.0, 15.0], "mean": [1.5, 3.5, 5.5, 7.5],
                "linear": [1.0, 3.0, 5.0, 7.0], "max": [2.0, 4.0, 6.0, 8.0]}
    for resampling_type, want in expected.items():
        out = Resample.resample(frame, "6h", resampling_type)
        assert np.allclose(row(out), want), resampling_type


def test_forecast_upsampling_along_leadtime():
    frame = forecast_frame(np.arange(1.0, 17.0).reshape(2, 8))
    expected = {"sum": [1 / 3, 1 / 3, 1 / 3, 2 / 3], "mean": [1.0, 1.0, 1.0, 2.0],
                "linear": [1.0, 4 / 3, 5 / 3, 2.0], "max": [1.0, 1.0, 1.0, 2.0]}
    for resampling_type, want in expected.items():
        out = Resample.resample(frame, "1h", resampling_type)
        assert np.allclose(row(out)[:4], want), resampling_type


def test_forecast_production_times_are_never_mixed():
    """Down-sampling production only subsamples forecasts: issue times stay distinct."""
    frame = forecast_frame(np.arange(1.0, 33.0).reshape(4, 8))
    out = Resample.resample(frame, "1D", "sum")
    # a whole day of leadtimes from the 00:00 issue only
    assert np.allclose(row(out), [np.arange(1.0, 9.0).sum()])


def test_forecast_intermediate_production_shifts_the_leadtimes():
    """A stamp between two issues reuses the last forecast, shifted."""
    frame = forecast_frame(np.arange(1.0, 17.0).reshape(2, 8))
    out = Resample.resample(frame, "3h", "sum")
    shifted = row(out, 1)
    assert np.allclose(shifted[:7], np.arange(2.0, 9.0))
    assert np.isnan(shifted[7]), "past the horizon of the shifted forecast"


# ------------------------------------------------- the two paths must agree

@pytest.mark.parametrize("build", ["series", "forecast"])
@pytest.mark.parametrize("resampling_type,factor", [("sum", 0.25), ("mean", 1.0), ("max", 1.0)])
def test_upsampling_a_constant_agrees_across_paths(build, resampling_type, factor):
    """Splitting by 4 applies to totals only; averages and extremes are step-invariant."""
    if build == "series":
        frame = series_frame(np.full(4, 8.0))
        step = "6h"
    else:
        frame = forecast_frame(np.full((2, 8), 8.0), prod="6h", lead="1D")
        step = "6h"
    out = Resample.resample(frame, step, resampling_type)
    values = out.to_numpy().ravel()
    assert np.allclose(values[np.isfinite(values)], 8.0 * factor)


@pytest.mark.parametrize("resampling_type,want", [("sum", 32.0), ("mean", 8.0), ("max", 8.0)])
def test_downsampling_a_constant_agrees_across_paths(resampling_type, want):
    series = Resample.resample(series_frame(np.full(8, 8.0)), "4D", resampling_type)
    forecast = Resample.resample(forecast_frame(np.full((2, 8), 8.0), prod="1D", lead="1D"),
                                 "4D", resampling_type)
    for out in (series, forecast):
        values = out.to_numpy().ravel()
        assert np.allclose(values[np.isfinite(values)], want)


# --------------------------------------------------------------- units / conservation

def test_sum_conserves_mass_when_upsampling():
    frame = series_frame(np.ones(8))
    out = Resample.resample(frame, "6h", "sum", date_to=frame.index[-1] + pd.Timedelta("18h"))
    assert np.isclose(np.nansum(out.to_numpy()), 8.0)


def test_sum_conserves_mass_when_downsampling():
    frame = series_frame(np.arange(1.0, 9.0))
    out = Resample.resample(frame, "2D", "sum")
    assert np.isclose(np.nansum(out.to_numpy()), np.arange(1.0, 9.0).sum())


@pytest.mark.parametrize("step", ["6h", "1D", "2D"])
@pytest.mark.parametrize("resampling_type", ["mean", "linear", "max"])
def test_intensive_types_are_step_invariant(step, resampling_type):
    """A constant field keeps its magnitude - and therefore its unit - at any step."""
    frame = series_frame(np.full(8, 15.0))
    values = Resample.resample(frame, step, resampling_type).to_numpy().ravel()
    assert np.allclose(values[np.isfinite(values)], 15.0)


@pytest.mark.parametrize("step,ratio", [("6h", 0.25), ("12h", 0.5), ("1D", 1.0), ("2D", 2.0)])
def test_sum_scales_exactly_with_the_step(step, ratio):
    """An accumulation per step rescales by the step ratio: mm/day vs mm/6h."""
    frame = series_frame(np.full(8, 12.0))
    values = Resample.resample(frame, step, "sum").to_numpy().ravel()
    assert np.allclose(values[np.isfinite(values)], 12.0 * ratio)


# -------------------------------------------------------- unevenly spaced leadtimes

#: Hourly out to 6 h, then 3-hourly, then 6-hourly - how operational forecasts are
#: actually delivered. Each column describes [leadtime, leadtime + span), and the last
#: reuses the preceding span, so the horizon is 36 h + 6 h = 42 h.
UNEVEN_LEADTIMES = pd.to_timedelta(['%uh' % h for h in
                                    (0, 1, 2, 3, 4, 5, 6, 9, 12, 15, 18, 24, 30, 36)])
UNEVEN_SPANS = np.array([1., 1., 1., 1., 1., 1., 3., 3., 3., 3., 6., 6., 6., 6.])
UNEVEN_HORIZON = 42


def uneven_frame(values, start="2023-01-01", prod="1D", series=("A",)):
    """A forecast whose leadtime columns are not evenly spaced."""
    values = np.asarray(values, dtype=float).reshape(-1, len(UNEVEN_LEADTIMES))
    idx = pd.date_range(start, periods=values.shape[0], freq=prod)
    cols = pd.MultiIndex.from_product([list(series), UNEVEN_LEADTIMES],
                                      names=["series", "leadtime"])
    return pd.DataFrame(np.tile(values, (1, len(series))), index=idx, columns=cols)


def test_uneven_leadtimes_are_accepted():
    """v2.x rejected these outright; the kernel assumed one scalar leadtime step."""
    out = Resample.resample(uneven_frame(np.arange(1.0, 15.0)), "6h", "sum")
    # the true horizon is 42 h, not 14 x the first span
    assert out.shape[1] == UNEVEN_HORIZON // 6
    assert np.isfinite(row(out)).all()


@pytest.mark.parametrize("step", ["1h", "2h", "3h", "6h"])
def test_uneven_sum_conserves_mass(step):
    """
    Each column's total is spread over its own span, so the grand total survives any
    target step that divides the 42 h horizon. A single scalar span would multiply the
    3 h and 6 h columns by the wrong share.
    """
    values = np.arange(1.0, 15.0)
    out = Resample.resample(uneven_frame(values), step, "sum")
    assert np.isclose(np.nansum(out.to_numpy()), values.sum())


@pytest.mark.parametrize("step", ["1h", "2h", "3h", "6h"])
@pytest.mark.parametrize("resampling_type", ["mean", "linear", "max"])
def test_uneven_intensive_types_are_step_invariant(step, resampling_type):
    out = Resample.resample(uneven_frame(np.full(14, 15.0)), step, resampling_type)
    values = out.to_numpy().ravel()
    assert np.allclose(values[np.isfinite(values)], 15.0)


def test_uneven_mean_is_weighted_by_the_local_span():
    """
    A 6 h window in the hourly region averages six columns; the same window in the
    3-hourly region averages two.
    """
    values = np.arange(1.0, 15.0)
    out = row(Resample.resample(uneven_frame(values), "6h", "mean"))

    assert np.isclose(out[0], values[:6].mean())          # [0h, 6h): columns 0..5
    assert np.isclose(out[1], (values[6] + values[7]) / 2)   # [6h, 12h): 6h and 9h
    assert np.isclose(out[2], (values[8] + values[9]) / 2)   # [12h, 18h): 12h and 15h
    assert np.isclose(out[3], values[10])                 # [18h, 24h): the 18h column alone


def test_uneven_linear_interpolates_over_the_local_span():
    values = np.arange(1.0, 15.0)
    out = row(Resample.resample(uneven_frame(values), "1h", "linear"))

    # inside the hourly region the nodes are hit exactly
    assert np.allclose(out[:7], values[:7])
    # 7 h and 8 h fall between the 6 h and 9 h nodes, a 3 h span
    assert np.isclose(out[7], values[6] + (values[7] - values[6]) / 3)
    assert np.isclose(out[8], values[6] + 2 * (values[7] - values[6]) / 3)
    # 19 h falls between the 18 h and 24 h nodes, a 6 h span
    assert np.isclose(out[19], values[10] + (values[11] - values[10]) / 6)


def test_uneven_missing_input_nans_only_the_windows_it_touches():
    frame = uneven_frame(np.arange(1.0, 15.0))
    frame.iloc[0, 7] = np.nan                             # the 9 h column, span 3 h
    out = row(Resample.resample(frame, "3h", "sum"))

    # 9 h to 12 h is the only 3 h window the 9 h column feeds
    assert np.isnan(out[3])
    assert np.isfinite(np.delete(out, 3)).all()


# ------------------------------------------------------------------ missing data

def test_a_missing_input_nans_only_the_windows_it_touches():
    frame = forecast_frame(np.arange(1.0, 33.0).reshape(4, 8), prod="6h", lead="3h")
    frame.iloc[0, 1] = np.nan
    for resampling_type in ("sum", "mean", "max"):
        out = Resample.resample(frame, "6h", resampling_type)
        assert np.isnan(row(out)[0]), resampling_type
        assert np.isfinite(row(out)[1:]).all(), resampling_type


def test_a_missing_production_row_nans_only_its_own_forecast():
    frame = forecast_frame(np.arange(1.0, 33.0).reshape(4, 8))
    frame.iloc[2, :] = np.nan
    out = Resample.resample(frame, "6h", "sum")
    complete = np.isfinite(out.to_numpy()).all(axis=1)
    assert complete.sum() == out.shape[0] - 1


# ------------------------------------------------------------------ regression guards

def test_two_production_rows_are_enough():
    out = Resample.resample(forecast_frame(np.arange(1.0, 17.0).reshape(2, 8)), "3h", "sum")
    assert out.shape[0] > 0


def test_a_single_forecast_resamples():
    """
    One issue is enough for a forecast: only leadtimes are aggregated, so there is no
    production spacing to infer. v3.0 accepts it; before, it raised.
    """
    frame = forecast_frame(np.arange(1.0, 9.0).reshape(1, 8))

    assert np.allclose(row(Resample.resample(frame, "3h", "sum")), np.arange(1.0, 9.0))
    # 3h -> 6h halves the number of columns and adds pairs
    assert np.allclose(row(Resample.resample(frame, "6h", "sum")), [3.0, 7.0, 11.0, 15.0])
    assert np.allclose(row(Resample.resample(frame, "6h", "mean")), [1.5, 3.5, 5.5, 7.5])


def test_a_single_production_series_is_rejected_clearly():
    """A one-row time series has no production step and nothing to infer it from."""
    frame = series_frame([1.0])
    with pytest.raises(Exception, match="two production timestamps"):
        Resample.resample(frame, "3h", "sum")


def test_leadtime_columns_need_not_be_sorted():
    frame = forecast_frame(np.arange(1.0, 33.0).reshape(4, 8))
    shuffled = frame.iloc[:, [5, 2, 0, 7, 3, 1, 6, 4]]
    assert np.allclose(Resample.resample(shuffled, "6h", "sum").to_numpy(),
                       Resample.resample(frame, "6h", "sum").to_numpy(), equal_nan=True)


def test_date_from_need_not_align_with_midnight():
    """An off-grid date_from used to silently produce an all-NaN frame."""
    frame = series_frame(np.arange(1.0, 9.0))
    out = Resample.resample(frame, "1D", "sum", date_from=pd.Timestamp("2023-01-01 06:00"))
    assert np.isfinite(out.to_numpy()).any()


def test_a_step_that_is_not_a_multiple_of_the_leadtime():
    """90 min against 1 h leadtimes: the exact gcd is 30 min, not a truncated ratio."""
    frame = forecast_frame(np.arange(1.0, 65.0).reshape(8, 8), prod="6h", lead="1h")
    out = Resample.resample(frame, "90min", "sum")
    # [0, 90min) = all of lead 0h (1.0) plus half of lead 1h (2.0)
    assert np.isclose(row(out)[0], 1.0 + 0.5 * 2.0)


def test_calendar_production_steps_raise_instead_of_exploding():
    """A monthly index used to be read as 1 millisecond and exhaust memory."""
    idx = pd.date_range("2023-01-01", periods=6, freq="MS")
    cols = pd.MultiIndex.from_tuples([("A", pd.Timedelta("0D"))], names=["series", "leadtime"])
    frame = pd.DataFrame(np.arange(1.0, 7.0)[:, None], index=idx, columns=cols)
    with pytest.raises(Exception, match="not evenly spaced"):
        Resample.resample(frame, "1D", "sum")


def test_a_calendar_timestep_is_rejected():
    with pytest.raises(Exception, match="calendar step"):
        Resample.resample(series_frame(np.ones(8)), "MS", "sum")


def test_column_order_is_preserved():
    frame = forecast_frame(np.arange(1.0, 17.0).reshape(2, 8), series=("B", "A"))
    out = Resample.resample(frame, "6h", "sum")
    assert [key[0] for key in out.columns][:1] == ["B"]
    assert list(dict.fromkeys(key[0] for key in out.columns)) == ["B", "A"]


def test_headers_are_resampled_independently():
    idx = pd.date_range("2023-01-01", periods=4, freq="6h")
    lts = pd.timedelta_range("0h", periods=4, freq="3h")
    values = np.arange(1.0, 17.0).reshape(4, 4)
    frame = pd.concat({"A": pd.DataFrame(values, idx, lts),
                       "B": pd.DataFrame(values + 1000.0, idx, lts)}, axis=1)
    frame.columns.names = ["series", "leadtime"]
    out = Resample.resample(frame, "3h", "mean")
    assert np.allclose(np.nanmax(np.abs(out["B"].to_numpy() - out["A"].to_numpy() - 1000.0)), 0.0)


def test_a_frame_without_header_levels_works():
    frame = pd.DataFrame(np.ones((5, 3)), pd.date_range("2023-01-01", periods=5, freq="6h"),
                         pd.timedelta_range("0h", periods=3, freq="3h"))
    frame.columns.name = "leadtime"
    assert Resample.resample(frame, "6h", "sum").shape[0] == 5


def test_a_horizon_shorter_than_the_step_is_rejected():
    frame = forecast_frame(np.arange(1.0, 9.0).reshape(4, 2), prod="1D", lead="3h")
    with pytest.raises(Exception, match="shorter than the requested time step"):
        Resample.resample(frame, "1D", "sum")


def test_a_tz_aware_index_resamples_exactly_like_the_same_data_in_utc():
    """DST is handled as spacing, so the local frame and its UTC twin must agree.

    The step is a fixed duration (pd.Timedelta('1D') is 24 h, not a calendar day), so
    after a switch the grid no longer lands on local midnight - deliberately, and
    identically on pandas 2 and 3, where date_range disagrees about this.
    """
    idx = pd.date_range("2023-03-24", periods=6, freq="1D", tz="Europe/Lisbon")
    lts = pd.timedelta_range("0h", periods=4, freq="6h")
    cols = pd.MultiIndex.from_product([["A"], lts], names=["series", "leadtime"])
    frame = pd.DataFrame(np.arange(1.0, 25.0).reshape(6, 4), idx, cols)

    local = Resample.resample(frame, "1D", "sum")
    utc = Resample.resample(frame.tz_convert("UTC"), "1D", "sum")

    assert local.index.tz is not None
    assert list(local.index.tz_convert("UTC")) == list(utc.index)
    assert np.allclose(local.to_numpy(), utc.to_numpy(), equal_nan=True)
    # before the switch the grid still coincides with the production stamps
    assert np.allclose(local.to_numpy().ravel()[:3], [10.0, 26.0, 42.0])


def test_a_naive_and_a_utc_index_agree():
    lts = pd.timedelta_range("0h", periods=4, freq="6h")
    cols = pd.MultiIndex.from_product([["A"], lts], names=["series", "leadtime"])
    values = np.arange(1.0, 25.0).reshape(6, 4)
    naive = pd.DataFrame(values, pd.date_range("2023-01-01", periods=6, freq="1D"), cols)
    aware = pd.DataFrame(values, pd.date_range("2023-01-01", periods=6, freq="1D", tz="UTC"), cols)
    assert np.allclose(Resample.resample(naive, "1D", "sum").to_numpy(),
                       Resample.resample(aware, "1D", "sum").to_numpy(), equal_nan=True)


def test_mixing_a_naive_bound_with_a_tz_aware_index_is_rejected():
    idx = pd.date_range("2023-01-01", periods=4, freq="1D", tz="UTC")
    cols = pd.MultiIndex.from_tuples([("A", pd.Timedelta("0D"))], names=["series", "leadtime"])
    frame = pd.DataFrame(np.ones((4, 1)), idx, cols)
    with pytest.raises(Exception, match="timezone aware"):
        Resample.resample(frame, "1D", "sum", date_from=pd.Timestamp("2023-01-01"))


def test_an_unknown_resampling_type_is_rejected():
    with pytest.raises(Exception, match="not implemented"):
        Resample.resample(series_frame(np.ones(4)), "1D", "median")


def test_missing_leadtime_level_is_rejected():
    frame = pd.DataFrame(np.ones((4, 1)), pd.date_range("2023-01-01", periods=4, freq="1D"),
                         pd.MultiIndex.from_tuples([("A", 0)], names=["series", "horizon"]))
    with pytest.raises(Exception, match="leadtime"):
        Resample.resample(frame, "1D", "sum")


def test_no_deprecated_pandas_idioms():
    """The module must stay clean on pandas >= 2.1 (axis=1 groupby/resample are gone in 3.0)."""
    import warnings

    frame = forecast_frame(np.arange(1.0, 33.0).reshape(4, 8))
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        warnings.simplefilter("error", DeprecationWarning)
        for resampling_type in TYPES:
            Resample.resample(frame, "6h", resampling_type)
            Resample.resample(series_frame(np.arange(1.0, 9.0)), "2D", resampling_type)
