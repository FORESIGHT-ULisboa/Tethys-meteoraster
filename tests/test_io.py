'''
Tests for meteoraster.io - the portable read path.

These are the guards against a .nct written by one install reading back
differently on another. The drift is real and was measured: the same file decodes
`leadtime` as timedelta64[us] on xarray 2026.7 and timedelta64[ns] on 2023.12, and
engine='h5netcdf' raises ValueError("unrecognized engine") on an install without
h5netcdf even when netcdf4 could read the file.
'''

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from meteoraster import MeteoRaster
from meteoraster import io as mrio


@pytest.fixture
def saved(tmp_path, sample_meteoraster):
    path = tmp_path / 'sample.nct'
    sample_meteoraster.save(path)

    return path, sample_meteoraster


@pytest.fixture
def monthly_saved(tmp_path):
    '''
    A raster whose leadtimes are calendar-month DateOffsets, stored as bare ints
    plus a leadtime_format marker. If the marker is lost they read back as ints.
    '''

    data = np.arange(2 * 1 * 3 * 2 * 2, dtype=np.float32).reshape(2, 1, 3, 2, 2)
    raster = MeteoRaster(
        data=data,
        latitudes=np.array([10.0, 9.0]), longitudes=np.array([0.0, 1.0]),
        production_datetime=pd.to_datetime(['2024-01-01', '2024-02-01']).values,
        leadtimes=np.array([pd.DateOffset(months=m) for m in (1, 2, 3)]),
        units='mm', variable='tp', verbose=False)
    path = tmp_path / 'monthly.nct'
    raster.save(path)

    return path, raster


# --------------------------------------------------------------------------- #
# engines
# --------------------------------------------------------------------------- #

def test_available_engines_is_a_subset():
    engines = mrio.available_engines()

    assert set(engines) <= set(mrio.READ_ENGINES)
    assert engines, 'no NetCDF read engine at all'


def test_every_engine_that_can_read_the_file_agrees(saved):
    '''
    The pinned decoding must not depend on which engine happened to open the file.

    Not every available engine can read every format - scipy only reads NETCDF3
    classic, so it is expected to fail on the NETCDF4/HDF5 files save() writes.
    That is exactly why open_raster falls back rather than trusting one name.
    '''

    path, original = saved

    succeeded = []
    for engine in mrio.available_engines():
        try:
            with mrio.open_raster(path, engine=engine) as ds:
                header = mrio.header_from_dataset(ds)
        except Exception:
            continue
        succeeded.append(engine)
        assert header.shape == original.data.shape
        assert str(np.asarray(header.leadtimes).dtype) == 'timedelta64[ns]'
        assert str(header.production_datetime.dtype) == 'datetime64[ns]'
        np.testing.assert_allclose(header.latitudes, original.latitudes)

    assert succeeded, 'no engine could read a file we just wrote'


def test_open_raster_reports_every_engine_it_tried(tmp_path):
    broken = tmp_path / 'broken.nct'
    broken.write_bytes(b'this is not a netcdf file')

    with pytest.raises(OSError) as raised:
        with mrio.open_raster(broken):
            pass

    message = str(raised.value)
    assert 'Could not open' in message
    for engine in mrio.available_engines():
        assert engine in message


def test_open_raster_reports_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        with mrio.open_raster(tmp_path / 'absent.nct'):
            pass


# --------------------------------------------------------------------------- #
# header
# --------------------------------------------------------------------------- #

def test_read_header_matches_the_raster(saved):
    path, original = saved

    header = mrio.read_header(path)

    assert header.variable == original.variable
    assert header.units == original.units
    assert header.shape == original.data.shape
    assert header.dtype == original.data.dtype
    assert (header.n_productions, header.n_members, header.n_leadtimes) == original.data.shape[:3]
    assert (header.rows, header.cols) == original.data.shape[-2:]
    np.testing.assert_allclose(header.latitudes, original.latitudes)
    np.testing.assert_allclose(header.longitudes, original.longitudes)
    assert header.complete is True


def test_header_latitudes_are_north_first(saved):
    path, _ = saved

    header = mrio.read_header(path)

    assert header.latitudes.ndim == 2
    assert header.latitudes[0, 0] >= header.latitudes[-1, 0]


def test_leadtime_resolution_is_pinned(saved):
    path, _ = saved

    leadtimes = mrio.read_header(path).leadtimes

    # Not just "a timedelta": the resolution too, because the raw integers differ
    # by 1000x between [us] and [ns] and any value-based indexing breaks.
    assert str(np.asarray(leadtimes).dtype) == 'timedelta64[ns]'
    assert pd.Timedelta(leadtimes[0]) == pd.Timedelta(days=1)


def test_monthly_leadtimes_survive_as_dateoffsets(monthly_saved):
    path, original = monthly_saved

    header = mrio.read_header(path)

    assert header.leadtime_format == 'pd.DateOffset:months'
    assert all(isinstance(lt, pd.DateOffset) for lt in header.leadtimes)
    assert [lt.months for lt in header.leadtimes] == [1, 2, 3]

    reloaded = MeteoRaster.load(path, verbose=False)
    assert [lt.months for lt in reloaded.leadtimes] == [lt.months for lt in original.leadtimes]


def test_leadtime_format_is_also_stored_as_an_attribute(monthly_saved):
    path, _ = monthly_saved

    # The scalar coordinate only survives while xarray keeps propagating dataset
    # scalar coords; the attribute survives any rewrite.
    with mrio.open_raster(path) as ds:
        assert ds['leadtime'].attrs.get('leadtime_format') == 'pd.DateOffset:months'


def test_leadtime_format_read_from_the_attribute_alone(monthly_saved):
    path, _ = monthly_saved

    with mrio.open_raster(path) as ds:
        stripped = ds.drop_vars('leadtime_format')
        assert mrio.read_leadtime_format(stripped) == 'pd.DateOffset:months'
        assert all(isinstance(lt, pd.DateOffset) for lt in mrio.read_leadtimes(stripped))


def test_variable_detection_tolerates_demoted_coordinates(saved):
    path, _ = saved

    with mrio.open_raster(path, decode_coords=False) as ds:
        # lat/lon/leadtime_format now appear as data variables, which the old
        # `len(ds.data_vars) != 1` check rejected outright.
        assert len(ds.data_vars) > 1
        assert mrio.detect_variable_name(ds) == 'precip'


def test_read_completeness(saved, tmp_path, sample_meteoraster):
    path, _ = saved
    assert mrio.read_completeness(path) is True

    # A single NaN pixel does not make a raster incomplete: with the default
    # space_completeness=False, one finite pixel per (production, leadtime) is
    # enough. Blank a whole slice to make it genuinely incomplete.
    incomplete = tmp_path / 'incomplete.nct'
    raster = sample_meteoraster.copy()
    raster.data[0, :, 0, :, :] = np.nan
    raster.save(incomplete)
    assert mrio.read_completeness(incomplete) is False


def test_missing_complete_flag_returns_none(saved):
    path, _ = saved

    with mrio.open_raster(path) as ds:
        assert mrio._read_complete(ds.drop_attrs()) is None


# --------------------------------------------------------------------------- #
# streaming
# --------------------------------------------------------------------------- #

def test_iter_productions_yields_each_production(saved):
    path, original = saved

    blocks = list(mrio.iter_productions(path))

    assert len(blocks) == original.data.shape[0]
    for position, (stamp, block) in enumerate(blocks):
        assert block.shape == original.data.shape[1:]
        assert stamp == original.production_datetime[position]
        np.testing.assert_allclose(block, original.data[position])


def test_iter_productions_reduces_members(saved):
    path, original = saved

    for name, function in (('mean', np.nanmean), ('median', np.nanmedian),
                           ('min', np.nanmin), ('max', np.nanmax)):
        _, block = next(iter(mrio.iter_productions(path, reduce_members=name)))
        assert block.shape == original.data.shape[2:]
        np.testing.assert_allclose(block, function(original.data[0], axis=0))


def test_iter_productions_selects_a_member_by_index(saved):
    path, original = saved

    _, block = next(iter(mrio.iter_productions(path, reduce_members=1)))

    np.testing.assert_allclose(block, original.data[0, 1])


def test_iter_productions_selects_by_value_and_slice(saved):
    path, original = saved

    blocks = list(mrio.iter_productions(path, productions=[original.production_datetime[1]],
                                        leadtimes=slice(0, 1)))

    assert len(blocks) == 1
    assert blocks[0][1].shape == (original.data.shape[1], 1) + original.data.shape[-2:]
    np.testing.assert_allclose(blocks[0][1], original.data[1][:, :1])


def test_iter_productions_selects_leadtimes_by_value(saved):
    path, original = saved

    _, block = next(iter(mrio.iter_productions(path, leadtimes=[original.leadtimes[1]])))

    np.testing.assert_allclose(block, original.data[0][:, 1:2])


def test_iter_productions_casts_dtype(saved):
    path, _ = saved

    _, block = next(iter(mrio.iter_productions(path, dtype='float32')))

    assert block.dtype == np.float32


def test_unknown_selector_is_reported(saved):
    path, _ = saved

    with pytest.raises(KeyError):
        list(mrio.iter_productions(path, productions=[np.datetime64('1999-01-01', 'ns')]))

    with pytest.raises(IndexError):
        list(mrio.iter_productions(path, productions=[99]))


def test_bad_reduce_members_is_reported(saved):
    path, _ = saved

    with pytest.raises(ValueError, match='reduce_members'):
        list(mrio.iter_productions(path, reduce_members='average'))
    with pytest.raises(TypeError):
        list(mrio.iter_productions(path, reduce_members=True))


# --------------------------------------------------------------------------- #
# read_cube
# --------------------------------------------------------------------------- #

def test_read_cube_matches_a_full_load(saved):
    path, original = saved

    data, productions, leadtimes, header = mrio.read_cube(path)

    np.testing.assert_allclose(data, original.data)
    np.testing.assert_array_equal(productions, original.production_datetime)
    assert len(leadtimes) == original.leadtimes.size
    assert header.variable == original.variable


def test_read_cube_subsets_and_casts(saved):
    path, original = saved

    data, productions, leadtimes, _ = mrio.read_cube(
        path, productions=slice(1, 2), leadtimes=[0], dtype='float32')

    assert data.shape == (1, original.data.shape[1], 1) + original.data.shape[-2:]
    assert data.dtype == np.float32
    assert len(productions) == 1 and len(leadtimes) == 1
    np.testing.assert_allclose(data[0], original.data[1][:, :1])


def test_streamed_blocks_agree_with_the_cube(saved):
    path, _ = saved

    data, _, _, _ = mrio.read_cube(path)
    for position, (_, block) in enumerate(mrio.iter_productions(path)):
        np.testing.assert_allclose(block, data[position])


def test_south_first_file_is_flipped_consistently(tmp_path):
    '''
    A file stored south-first must come back north-first, with the blocks flipped
    to match - getting this wrong is a silent vertical mirror.
    '''

    import xarray as xr

    latitudes, longitudes = np.meshgrid(np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0]),
                                        indexing='ij')
    values = np.arange(6, dtype=np.float32).reshape(1, 1, 1, 3, 2)
    array = xr.DataArray(
        data=values, dims=['production_datetime', 'ensemble_member', 'leadtime', 'y', 'x'],
        coords={'production_datetime': pd.to_datetime(['2024-01-01']).values,
                'ensemble_member': [0], 'leadtime': [pd.Timedelta(hours=0)],
                'lat': (['y', 'x'], latitudes), 'lon': (['y', 'x'], longitudes),
                'y': np.arange(3), 'x': np.arange(2), 'leadtime_format': 'pd.Timedelta'},
        name='v', attrs={'units': 'mm', 'variable': 'v'})
    path = tmp_path / 'south_first.nct'
    xr.Dataset({'v': array}).to_netcdf(path, engine=MeteoRaster.ENGINE)

    header = mrio.read_header(path)
    assert header.flipped is True
    assert header.latitudes[0, 0] > header.latitudes[-1, 0]

    _, block = next(iter(mrio.iter_productions(path)))
    # Row 0 of the block must be the row of header.latitudes[0], i.e. the original
    # last row.
    np.testing.assert_allclose(block[0, 0], values[0, 0, 0][::-1])
    np.testing.assert_allclose(MeteoRaster.load(path, verbose=False).data, block[None, ...])
