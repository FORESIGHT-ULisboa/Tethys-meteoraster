'''
Regression tests for defects fixed in v2.6.

Each test names the failure it pins, because most of these were silent: wrong
numbers or leaked state rather than an exception.
'''

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

from meteoraster import MeteoRaster


def _raster(**kwargs):
    defaults = dict(
        data=np.arange(2 * 2 * 2 * 3 * 4, dtype=np.float32).reshape(2, 2, 2, 3, 4),
        latitudes=np.array([10.0, 9.0, 8.0]),
        longitudes=np.array([0.0, 1.0, 2.0, 3.0]),
        production_datetime=pd.to_datetime(['2024-01-01', '2024-01-02']).values,
        leadtimes=np.array([pd.Timedelta(hours=0), pd.Timedelta(hours=1)]),
        verbose=False,
    )
    defaults.update(kwargs)

    return MeteoRaster(**defaults)


# --------------------------------------------------------------------------- #
# shared mutable default arguments
# --------------------------------------------------------------------------- #

def test_units_do_not_leak_between_instances():
    '''
    da_attrs/ds_attrs defaulted to {} and _update_attrs() wrote into them, so the
    dict created at class-definition time was shared by every instance. Since
    save() uses self.variable as the NetCDF variable name, the name written to disk
    could come from an unrelated object in the same process.
    '''

    first = _raster(units='mm', variable='precip')
    second = _raster()

    assert (first.units, first.variable) == ('mm', 'precip')
    assert (second.units, second.variable) == ('unknown', 'unknown')
    assert second.da_attrs.get('units') is None


def test_attrs_passed_in_are_not_mutated():
    supplied = {}
    _raster(units='mm', da_attrs=supplied)

    assert supplied == {}


def test_load_keeps_da_and_ds_attrs_separate(tmp_path):
    '''
    load() bound both names to ds.attrs, so the DataArray attributes were lost and
    _update_attrs' write leaked into the global attributes - every load/save cycle
    accreted `units` and `variable` there.
    '''

    path = tmp_path / 'r.nct'
    _raster(units='mm', variable='precip').save(path)

    once = MeteoRaster.load(path, verbose=False)
    assert once.da_attrs is not once.ds_attrs
    assert 'units' not in once.ds_attrs

    # A second cycle must not accrete anything either.
    second_path = tmp_path / 'r2.nct'
    once.save(second_path)
    twice = MeteoRaster.load(second_path, verbose=False)
    assert 'units' not in twice.ds_attrs
    assert twice.units == 'mm' and twice.variable == 'precip'


# --------------------------------------------------------------------------- #
# degenerate grids
# --------------------------------------------------------------------------- #

def test_single_latitude_row_raster_can_be_built_saved_and_loaded(tmp_path):
    '''
    __init__ tested latitudes[1, 0] unconditionally, so a one-row raster - which
    get_cropped can produce - raised IndexError. Such a file saved but could never
    be loaded again.
    '''

    raster = _raster(data=np.ones((1, 1, 1, 1, 4), dtype=np.float32),
                     latitudes=np.array([10.0]),
                     production_datetime=pd.to_datetime(['2024-01-01']).values,
                     leadtimes=np.array([pd.Timedelta(hours=0)]))

    path = tmp_path / 'thin.nct'
    raster.save(path)

    assert MeteoRaster.load(path, verbose=False).data.shape == (1, 1, 1, 1, 4)


def test_mixed_dimensionality_coordinates_are_refused():
    with pytest.raises(ValueError, match='must both be 1-D or both 2-D'):
        _raster(latitudes=np.zeros((3, 4)), longitudes=np.array([0.0, 1.0, 2.0, 3.0]))


# --------------------------------------------------------------------------- #
# dtype preservation
# --------------------------------------------------------------------------- #

def test_join_keeps_float32_when_padding_ensembles():
    '''
    np.empty(...) * np.nan is float64 whatever the source, so a float32 raster was
    silently doubled in memory and on disk.
    '''

    base = _raster()
    other = _raster(data=np.ones((2, 1, 2, 3, 4), dtype=np.float32),
                    production_datetime=pd.to_datetime(['2024-01-03', '2024-01-04']).values)

    with pytest.warns(UserWarning):
        base.join(other)

    assert base.data.dtype == np.float32


def test_resample_timestep_keeps_dtype_and_does_not_raise():
    '''
    The final diagnostic passed self.verbose positionally to strftime, so every
    successful call ended in TypeError.
    '''

    raster = _raster(units='mm', variable='precip')

    with pytest.warns(UserWarning, match='experimental'):
        raster.resample_timestep('1D')

    assert raster.data.dtype == np.float32
    assert raster.data.shape[0] == 2


def test_get_cropped_keeps_dtype():
    cropped = _raster().get_cropped(from_lat=8.5, to_lat=10.5)

    assert cropped.data.dtype == np.float32
    assert cropped.data.shape[-2] == 2


def test_get_cropped_does_not_alias_the_original():
    original = _raster()
    cropped = original.get_cropped(from_lon=0.5, to_lon=3.5)

    cropped.da_attrs['marker'] = True
    assert 'marker' not in original.da_attrs

    before = original.data[0, 0, 0, 0, 1]
    cropped.data[0, 0, 0, 0, 0] = -999.0
    assert original.data[0, 0, 0, 0, 1] == before


# --------------------------------------------------------------------------- #
# labelling
# --------------------------------------------------------------------------- #

def test_get_missing_labels_are_not_transposed():
    '''
    The reshape put leadtime fastest while the MultiIndex declared
    (leadtime, ensemble_member), so the two labels were swapped whenever there was
    more than one of each. Distinct per-slice values make it visible.
    '''

    raster = _raster()
    # leadtime 0 fully missing for member 0 only.
    raster.data[:, 0, 0, :, :] = np.nan

    missing = raster.get_missing()

    assert list(missing.index.names) == ['leadtime', 'ensemble_member']
    lead_zero = raster.leadtimes[0]
    lead_one = raster.leadtimes[1]
    assert missing.loc[(lead_zero, 0)].eq(1.0).all()
    assert missing.loc[(lead_zero, 1)].eq(0.0).all()
    assert missing.loc[(lead_one, 0)].eq(0.0).all()


def test_is_complete_semantics_are_unchanged():
    raster = _raster()
    assert raster.is_complete() is True

    # One NaN pixel: still complete, because space_completeness defaults to False.
    one_pixel = _raster()
    one_pixel.data[0, 0, 0, 0, 0] = np.nan
    assert one_pixel.is_complete() is True
    assert one_pixel.is_complete(space_completeness=True) is False

    # A whole (production, leadtime) slice missing: incomplete either way.
    whole_slice = _raster()
    whole_slice.data[0, :, 0, :, :] = np.nan
    assert whole_slice.is_complete() is False


def test_complete_index_matches_is_complete():
    raster = _raster()
    raster.data[1, :, 1, :, :] = np.nan

    index = raster.get_complete_index()

    assert index.shape == (2, 2)
    assert bool(index.iloc[0, 0]) and bool(index.iloc[0, 1]) and bool(index.iloc[1, 0])
    assert not bool(index.iloc[1, 1])


# --------------------------------------------------------------------------- #
# save mechanics
# --------------------------------------------------------------------------- #

def test_save_leaves_no_temporary_behind(tmp_path):
    path = tmp_path / 'r.nct'
    _raster().save(path)

    assert sorted(p.name for p in tmp_path.iterdir()) == ['r.nct']


def test_save_stages_in_the_destination_directory(tmp_path, monkeypatch):
    '''
    Nothing may be staged through the system temp dir: in a container that is often
    tmpfs (the whole file would be built in RAM) or a different filesystem from the
    output volume.
    '''

    monkeypatch.setenv('TMPDIR', str(tmp_path / 'nowhere'))
    monkeypatch.setenv('TEMP', str(tmp_path / 'nowhere'))
    monkeypatch.setenv('TMP', str(tmp_path / 'nowhere'))

    path = tmp_path / 'out' / 'r.nct'
    _raster().save(path)

    assert path.exists()


def test_save_replaces_an_existing_file_atomically(tmp_path):
    path = tmp_path / 'r.nct'
    _raster(variable='first').save(path)
    _raster(variable='second').save(path)

    assert MeteoRaster.load(path, verbose=False).variable == 'second'
    assert sorted(p.name for p in tmp_path.iterdir()) == ['r.nct']


@pytest.mark.skipif(sys.platform == 'win32', reason='POSIX permissions only')
def test_saved_file_is_group_readable(tmp_path):
    '''
    NamedTemporaryFile creates 0600 and the rename preserved it, so every saved
    file used to be readable only by the writing user.
    '''

    path = tmp_path / 'r.nct'
    _raster().save(path)

    assert os.stat(path).st_mode & 0o044


def test_save_accepts_an_explicit_engine(tmp_path):
    from meteoraster import io as mrio

    if 'netcdf4' not in mrio.available_engines():
        pytest.skip('netcdf4 engine not installed')

    path = tmp_path / 'r.nct'
    _raster(units='mm', variable='precip').save(path, engine='netcdf4')

    assert MeteoRaster.load(path, verbose=False).units == 'mm'


# --------------------------------------------------------------------------- #
# greenwich wrapping
# --------------------------------------------------------------------------- #

def test_greenwich_wrap_reorders_data_and_longitudes():
    values = np.arange(4, dtype=np.float32).reshape(1, 1, 1, 1, 4)
    raster = _raster(data=values,
                     latitudes=np.array([10.0]),
                     longitudes=np.array([0.0, 90.0, 190.0, 300.0]),
                     production_datetime=pd.to_datetime(['2024-01-01']).values,
                     leadtimes=np.array([pd.Timedelta(hours=0)]))

    np.testing.assert_allclose(raster.longitudes[0], [-170.0, -60.0, 0.0, 90.0])
    # The values must follow their longitudes.
    np.testing.assert_allclose(raster.data[0, 0, 0, 0], [2.0, 3.0, 0.0, 1.0])
    assert raster.data.dtype == np.float32
