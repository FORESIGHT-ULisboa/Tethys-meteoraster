'''
Portable NetCDF access for MeteoRaster files (.nc / .nct).

Two jobs:

1. `open_raster` is the only place a file is opened. It tries the read engines in
   turn and pins the decoding, so a file written by one install reads back the same
   on another. Measured drift it guards against: the same file decodes `leadtime` as
   timedelta64[us] on xarray 2026.7 and timedelta64[ns] on 2023.12 (raw integers
   differ 1000x); `engine='h5netcdf'` raises ValueError("unrecognized engine")
   wherever h5netcdf is absent even though netcdf4 could read the file; and v2.0/2.1
   wrote with xarray's default engine, so some archives are NETCDF3 classic and only
   scipy reads them.

2. `read_header` / `iter_productions` read metadata, and one production at a time, so
   a multi-GB archive can be aggregated without holding the whole cube. Nothing here
   imports MeteoRaster - the streaming path must not need a cube to exist.

Container note: h5py inherits HDF5's flock-based locking, which fails on NFS and some
bind mounts with "unable to lock file". Set HDF5_USE_FILE_LOCKING=FALSE in the
environment if that happens; this module deliberately does not set it, because
disabling locking for a whole process behind the caller's back is worse.
'''

import contextlib
import dataclasses
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

#: Read engines, tried in this order. h5netcdf is the write engine
#: (MeteoRaster.ENGINE); netcdf4 reads the same NETCDF4/HDF5 files, and scipy is
#: the only one that reads the NETCDF3 classic files some v2.0/v2.1 installs wrote.
READ_ENGINES = ('h5netcdf', 'netcdf4', 'scipy')

#: The canonical 5-D dimension names written by MeteoRaster.to_xarray().
DIMS = ('production_datetime', 'ensemble_member', 'leadtime', 'y', 'x')

#: Names that are coordinates, not data. Needed because any tool that rewrites a
#: file (nccopy, CDO, NCO) can demote lat/lon/leadtime_format to data variables.
COORD_NAMES = frozenset({'lat', 'lon', 'y', 'x', 'latitude', 'longitude',
                         'leadtime', 'leadtime_format',
                         'production_datetime', 'ensemble_member'})

DEFAULT_LEADTIME_FORMAT = 'pd.Timedelta'

#: Accepted string values for the `reduce_members` argument.
MEMBER_REDUCERS = ('mean', 'median', 'min', 'max', 'sum')

_REDUCERS = {'mean': np.nanmean, 'median': np.nanmedian,
             'min': np.nanmin, 'max': np.nanmax, 'sum': np.nansum}

# CF unit names -> numpy datetime64 unit codes, for the belt-and-braces path in
# read_leadtimes() where a future xarray hands back undecoded integers.
_CF_UNITS = {'weeks': 'W', 'days': 'D', 'hours': 'h', 'hour': 'h',
             'minutes': 'm', 'seconds': 's', 'milliseconds': 'ms',
             'microseconds': 'us', 'nanoseconds': 'ns'}


class RasterFormatError(ValueError):
    '''Raised when a NetCDF file is not shaped like a MeteoRaster file.'''


def available_engines():
    '''
    Returns the subset of READ_ENGINES that this install can actually use.
    '''

    try:
        backends = set(xr.backends.list_engines())
    except Exception:
        backends = set()

    return tuple(e for e in READ_ENGINES if e in backends)


@contextlib.contextmanager
def open_raster(file, engine=None, **open_kwargs):
    '''
    Opens a NetCDF file as an xarray Dataset, trying the read engines in turn.

    engine: force a single engine instead of falling back through READ_ENGINES.
    open_kwargs: passed to xr.open_dataset; they override the pinned defaults.
    '''

    file = Path(file)
    if not file.exists():
        raise FileNotFoundError('No such raster file: %s' % file)

    kwargs = {'decode_timedelta': True}
    kwargs.update(open_kwargs)

    engines = (engine,) if engine else (available_engines() or READ_ENGINES)

    dataset = None
    errors = []
    for name in engines:
        try:
            dataset = xr.open_dataset(file, engine=name, **kwargs)
            break
        except Exception as ex:
            errors.append('%s: %s: %s' % (name, type(ex).__name__, ex))

    if dataset is None:
        raise OSError('Could not open %s with any NetCDF engine.\n  %s'
                      % (file, '\n  '.join(errors)))

    try:
        yield dataset
    finally:
        dataset.close()


def detect_variable_name(ds):
    '''
    Returns the name of the data variable holding the raster.

    Tolerates files whose coordinates were demoted to data variables, which is
    what nccopy/CDO/NCO do and what the old `len(ds.data_vars) != 1` check
    rejected outright.
    '''

    names = [n for n in ds.data_vars if str(n) not in COORD_NAMES]

    canonical = [n for n in names if set(DIMS).issubset(ds[n].dims)]
    if len(canonical) == 1:
        return canonical[0]
    if len(canonical) > 1:
        raise RasterFormatError('Several variables carry the MeteoRaster dimensions: %s.'
                                % ', '.join(map(str, canonical)))
    if len(names) == 1:
        return names[0]
    if not names:
        raise RasterFormatError('No data variable found (variables: %s).'
                                % ', '.join(map(str, ds.data_vars)))

    raise RasterFormatError('Ambiguous file: %s could each be the raster. None carries '
                            'the expected dimensions %s.'
                            % (', '.join(map(str, names)), ', '.join(DIMS)))


def _as_text(value):
    '''
    Coerces an attribute/coordinate value to str, unwrapping 0-d arrays and bytes.

    The bytes case is real: some engines return b'pd.Timedelta', and str() on
    that yields "b'pd.Timedelta'", which silently fails the format comparison and
    turns monthly DateOffset leadtimes back into bare integers.
    '''

    if isinstance(value, np.ndarray):
        value = value.item() if value.ndim == 0 else value.ravel()[0]
    if isinstance(value, bytes):
        value = value.decode('utf-8', 'replace')

    return str(value)


def read_leadtime_format(ds):
    '''
    Returns the leadtime format string, looking in every place it may survive.
    '''

    for holder in (ds.coords, ds.data_vars):
        if 'leadtime_format' in holder:
            return _as_text(holder['leadtime_format'].values)

    if 'leadtime' in ds and 'leadtime_format' in ds['leadtime'].attrs:
        return _as_text(ds['leadtime'].attrs['leadtime_format'])

    if 'leadtime_format' in ds.attrs:
        return _as_text(ds.attrs['leadtime_format'])

    return DEFAULT_LEADTIME_FORMAT


def read_leadtimes(ds):
    '''
    Returns the leadtimes as timedelta64[ns], or a pd.Index of pd.DateOffset.
    '''

    raw = np.asarray(ds['leadtime'].values)
    fmt = read_leadtime_format(ds)
    kind, _, unit = fmt.partition(':')

    if kind == 'pd.DateOffset':
        if not unit:
            raise RasterFormatError('leadtime_format "%s" names no DateOffset unit.' % fmt)
        return pd.Index([pd.DateOffset(**{unit: int(v)}) for v in raw])

    if np.issubdtype(raw.dtype, np.timedelta64):
        # Normalise the resolution: the same file decodes to [us] on recent
        # xarray and [ns] on older releases.
        return raw.astype('timedelta64[ns]')

    if np.issubdtype(raw.dtype, np.integer):
        # A future xarray that stops decoding timedeltas by default lands here;
        # rebuild from the CF units rather than handing back bare counts.
        units = ds['leadtime'].attrs.get('units') or ds['leadtime'].encoding.get('units')
        code = _CF_UNITS.get(str(units).split(' since ')[0].strip().lower()) if units else None
        if code:
            return pd.to_timedelta(raw, unit=code).to_numpy().astype('timedelta64[ns]')
        warnings.warn('leadtime came back as %s and the file declares no usable units; '
                      'returning raw values.' % raw.dtype)

    return raw


def _read_complete(ds):
    '''
    Reads the `complete` flag. Returns None when the file does not carry one.
    '''

    value = ds.attrs.get('complete')
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return bool(value)

    return _as_text(value).strip().lower() in ('true', 'yes', '1')


def _grid_2d(latitudes, longitudes):
    '''
    Returns latitudes/longitudes as 2-D arrays, meshing 1-D inputs.
    '''

    latitudes = np.asarray(latitudes)
    longitudes = np.asarray(longitudes)
    if latitudes.ndim == 1 and longitudes.ndim == 1:
        longitudes, latitudes = np.meshgrid(longitudes, latitudes)

    return latitudes, longitudes


@dataclasses.dataclass(frozen=True)
class Header:
    '''
    Everything about a raster file except the values. Cheap to obtain.

    `latitudes` are north-first and `shape` describes the block layout as
    yielded by iter_productions, i.e. after `flipped` has been applied. A reader
    that bypasses this module and indexes the file directly must apply the same
    flip or it gets a vertically mirrored raster.
    '''

    variable: str
    units: str
    production_datetime: np.ndarray
    leadtimes: object
    leadtime_format: str
    latitudes: np.ndarray
    longitudes: np.ndarray
    shape: tuple
    dtype: np.dtype
    complete: object
    da_attrs: dict
    ds_attrs: dict
    flipped: bool

    @property
    def n_productions(self):
        return int(self.shape[0])

    @property
    def n_members(self):
        return int(self.shape[1])

    @property
    def n_leadtimes(self):
        return int(self.shape[2])

    @property
    def rows(self):
        return int(self.shape[3])

    @property
    def cols(self):
        return int(self.shape[4])


def header_from_dataset(ds):
    '''
    Builds a Header from an already-open Dataset.
    '''

    name = detect_variable_name(ds)
    da = ds[name]

    latitudes, longitudes = _grid_2d(ds['lat'].values, ds['lon'].values)

    flipped = latitudes.shape[0] > 1 and latitudes[0, 0] < latitudes[1, 0]
    if flipped:
        latitudes = latitudes[::-1].copy()
        longitudes = longitudes[::-1].copy()

    production_datetime = np.asarray(ds['production_datetime'].values)
    if np.issubdtype(production_datetime.dtype, np.datetime64):
        production_datetime = production_datetime.astype('datetime64[ns]')

    return Header(
        variable=str(da.attrs.get('variable', name)),
        units=str(da.attrs.get('units', 'unknown')),
        production_datetime=production_datetime,
        leadtimes=read_leadtimes(ds),
        leadtime_format=read_leadtime_format(ds),
        latitudes=latitudes,
        longitudes=longitudes,
        shape=tuple(int(s) for s in da.shape),
        dtype=np.dtype(da.dtype),
        complete=_read_complete(ds),
        da_attrs=dict(da.attrs),
        ds_attrs=dict(ds.attrs),
        flipped=bool(flipped),
    )


def read_header(file, engine=None):
    '''
    Reads a raster file's metadata without reading any values.
    '''

    with open_raster(file, engine=engine) as ds:
        return header_from_dataset(ds)


def read_completeness(file, engine=None):
    '''
    Reads only the `complete` flag. Returns None when the file carries none.
    '''

    with open_raster(file, engine=engine, decode_times=False,
                     decode_timedelta=False) as ds:
        return _read_complete(ds)


def _resolve_index(values, selector, what):
    '''
    Turns `selector` into positional indices into `values`.

    Accepts None (everything), a slice, integer or boolean indices, or the
    coordinate values themselves (timestamps / timedeltas / DateOffsets).
    '''

    count = len(values)
    if selector is None:
        return np.arange(count)
    if isinstance(selector, slice):
        return np.arange(count)[selector]

    wanted = np.asarray(selector)
    if wanted.ndim == 0:
        wanted = wanted.reshape(1)

    if wanted.dtype == bool:
        if wanted.size != count:
            raise ValueError('Boolean %s selector has %d entries for %d available.'
                             % (what, wanted.size, count))
        return np.flatnonzero(wanted)

    if np.issubdtype(wanted.dtype, np.integer):
        index = np.where(wanted < 0, wanted + count, wanted)
        if ((index < 0) | (index >= count)).any():
            raise IndexError('%s index out of range for %d available.' % (what, count))
        return index

    if np.issubdtype(wanted.dtype, np.datetime64):
        wanted = wanted.astype('datetime64[ns]')

    index = pd.Index(values).get_indexer(pd.Index(wanted.ravel()))
    if (index < 0).any():
        missing = wanted.ravel()[index < 0]
        raise KeyError('%s not present in the file: %s' % (what, list(missing)))

    return index


def member_reducer(reduce_members):
    '''
    Returns a callable collapsing the leading (member) axis, or None.

    Accepts None, a name from MEMBER_REDUCERS, an integer member index, or a
    callable taking (array, axis=0).
    '''

    if reduce_members is None:
        return None
    if callable(reduce_members):
        return lambda block: reduce_members(block, axis=0)
    if isinstance(reduce_members, (bool, np.bool_)):
        raise TypeError('reduce_members must not be a bool.')
    if isinstance(reduce_members, (int, np.integer)):
        position = int(reduce_members)
        return lambda block: block[position]

    name = str(reduce_members).lower()
    if name not in _REDUCERS:
        raise ValueError('reduce_members must be one of %s, an int or a callable (got %r).'
                         % (', '.join(MEMBER_REDUCERS), reduce_members))
    function = _REDUCERS[name]

    def reduce(block):
        with warnings.catch_warnings():
            # All-NaN members are normal in a pre-allocated forecast file.
            warnings.simplefilter('ignore', category=RuntimeWarning)
            return function(block, axis=0)

    return reduce


def iter_productions(file, productions=None, leadtimes=None, reduce_members=None,
                     dtype=None, header=None, engine=None):
    '''
    Yields (production_datetime, block) one production at a time.

    block is [ensemble_member, leadtime, y, x], or [leadtime, y, x] when
    reduce_members collapses the ensemble axis. Rows are north-first, matching
    header.latitudes and MeteoRaster.latitudes.

    Peak memory is one block, whatever the file size - this is the reason the
    module exists.
    '''

    with open_raster(file, engine=engine) as ds:
        head = header or header_from_dataset(ds)
        da = ds[detect_variable_name(ds)]

        production_index = _resolve_index(head.production_datetime, productions, 'production')
        leadtime_index = None
        if leadtimes is not None:
            leadtime_index = _resolve_index(head.leadtimes, leadtimes, 'leadtime')

        reduce = member_reducer(reduce_members)

        for position in production_index:
            block = da.isel(production_datetime=int(position))
            if leadtime_index is not None:
                block = block.isel(leadtime=leadtime_index)

            values = np.asarray(block.values)
            if head.flipped:
                values = np.ascontiguousarray(values[..., ::-1, :])
            if dtype is not None:
                values = values.astype(dtype, copy=False)
            if reduce is not None:
                values = reduce(values)

            yield head.production_datetime[position], values


def read_cube(file, productions=None, leadtimes=None, dtype=None, header=None,
              engine=None):
    '''
    Reads a whole (or partial) cube as a 5-D array, filling it production by
    production.

    Returns (data, production_datetime, leadtimes, header). Peak memory is the
    output array plus one block, so passing dtype='float32' for a float64 file
    genuinely halves it rather than casting after the fact.
    '''

    with open_raster(file, engine=engine) as ds:
        head = header or header_from_dataset(ds)
        da = ds[detect_variable_name(ds)]

        production_index = _resolve_index(head.production_datetime, productions, 'production')
        leadtime_index = _resolve_index(head.leadtimes, leadtimes, 'leadtime')

        shape = (len(production_index), head.n_members, len(leadtime_index),
                 head.rows, head.cols)
        data = np.empty(shape, dtype=np.dtype(dtype) if dtype is not None else head.dtype)

        for target, position in enumerate(production_index):
            block = da.isel(production_datetime=int(position))
            if len(leadtime_index) != head.n_leadtimes:
                block = block.isel(leadtime=leadtime_index)
            values = np.asarray(block.values)
            if head.flipped:
                values = values[..., ::-1, :]
            data[target] = values

        leadtime_values = head.leadtimes
        if len(leadtime_index) != head.n_leadtimes:
            leadtime_values = (leadtime_values[leadtime_index]
                               if isinstance(leadtime_values, pd.Index)
                               else np.asarray(leadtime_values)[leadtime_index])

        return data, head.production_datetime[production_index], leadtime_values, head
