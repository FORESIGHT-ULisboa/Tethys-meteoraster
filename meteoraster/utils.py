'''
Readers for ERA5-Land GRIB, returning a ready MeteoRaster.

Created on 10/02/2023

@author: Jose Pedro Matos

v3.0 removed readERA5_monthly, read_GFS, read_C3S and readCORDEX_monthly. Every
reader here used to populate data['productionDates'] while MeteoRaster.__init__
reads data['production_datetime'], so all six raised KeyError; only the two
ERA5-Land readers were worth keeping alive.

cfgrib needs the ecCodes binary. On Windows it lives in <env>/Library/bin and is
only found through PATH, so the conda environment must be activated - calling its
python.exe directly raises RuntimeError('Cannot find the ecCodes library').
'''

import warnings
import numpy as np
import xarray as xr
import pandas as pd
from meteoraster import MeteoRaster

#: variable -> units after the conversions below.
_MONTHLY_UNITS = {'tp': 'mm/month', 't2m': 'C', 'ro': 'mm/month', 'e': 'mm/month'}

#: variable -> GRIB shortName. A single monthly file holds several variables with
#: *different* stepTypes ('avgid' for 2t, 'avgas' for tp), so cfgrib cannot build one
#: hypercube from it and each read needs a filter. Filtering on stepType instead is
#: what broke: the value hard-coded here was 'avgad', and CDS now delivers 'avgas'.
_MONTHLY_SHORT_NAMES = {'tp': 'tp', 't2m': '2t', 'ro': 'ro', 'e': 'e'}

#: Variables stored as an accumulation since 00 UTC rather than an instantaneous value.
_CUMULATIVE = {'tp': True, 'ssr': True, 'u10': False, 'v10': False, 'sd': False,
               't2m': False}


def readERA5Land_monthly(file, variable, **kwargs):
    '''
    Reads ERA5-Land monthly means.

    https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land-monthly-means

    variable:
        t2m - 2 m temperature      -> C
        tp  - total precipitation  -> mm/month
        ro, e                      -> mm/month
    '''

    if variable not in _MONTHLY_UNITS:
        raise Exception(f'How to handle variable "{variable}" must be defined.')

    backend_kwargs = dict(kwargs.pop('backend_kwargs', {}))
    filter_by_keys = dict(backend_kwargs.pop('filter_by_keys', {}))
    filter_by_keys.setdefault('shortName', _MONTHLY_SHORT_NAMES[variable])
    backend_kwargs['filter_by_keys'] = filter_by_keys

    data = {}
    with xr.open_dataset(file, engine='cfgrib', indexpath='',
                         backend_kwargs=backend_kwargs, **kwargs) as ds:
        names = list(ds.data_vars)
        if len(names) != 1:
            raise Exception('Expected one variable for shortName "%s"; got %s. Narrow '
                            'the read with backend_kwargs={"filter_by_keys": ...}.'
                            % (filter_by_keys['shortName'], names))

        data['latitudes'] = ds.latitude.data
        data['longitudes'] = ds.longitude.data
        # A single month collapses the time dimension to a scalar coordinate, which
        # used to make ds[variable][:, :, :] raise and days_in_month return an int.
        production_datetime = pd.DatetimeIndex(np.atleast_1d(pd.to_datetime(ds.time.data)))
        data['production_datetime'] = production_datetime

        values = np.asarray(ds[names[0]].data, dtype=float)
        if values.ndim == 2:
            values = values[np.newaxis]
        data['data'] = np.expand_dims(values, [1, 2])

    data['leadtimes'] = np.array([pd.DateOffset(days=0)])

    if variable == 't2m':
        data['data'] -= 273.15
    else:
        data['data'] *= 1000                                  # m -> mm
        if variable == 'tp':
            # tp is a mean daily accumulation, so a month's total scales with its
            # length. Broadcast rather than tile: tiling this to a global 0.1 deg
            # grid allocates a second full-size array.
            days = production_datetime.days_in_month.to_numpy(dtype=float)
            data['data'] *= days.reshape(-1, 1, 1, 1, 1)

    return MeteoRaster(data, variable=variable, units=_MONTHLY_UNITS[variable])


def _read_ERA5Land_helper(file, variable):
    '''
    Reads the GRIB and normalises it to [date, step, y, x].

    cfgrib drops a length-1 dimension to a scalar coordinate, so which axis a 3-D
    array is missing depends on the request; the step count disambiguates it.
    '''

    data = {}
    with xr.open_dataset(file, engine='cfgrib', indexpath='') as ds:
        data['latitudes'] = ds.latitude.data
        data['longitudes'] = ds.longitude.data
        data['production_datetime'] = np.atleast_1d(ds.time.data)
        data['steps'] = np.atleast_1d(ds.step.data)
        values = ds[variable].data

    if values.ndim == 3:
        values = values[:, np.newaxis] if data['steps'].size == 1 else values[np.newaxis]
    data['data'] = values

    return data


def read_ERA5Land_hourly(file, variable, file_next_year=None):
    '''
    Reads ERA5-Land hourly data, returning one production stamp per hour.

    Accumulations are de-accumulated to per-hour increments. A CDS request for one
    period also returns the previous period's last hour (that date carrying its last
    step only) and omits the last hour of its own last day, which is delivered with
    the next period - pass that file as `file_next_year` to fill it in.

    Every value describes [stamp, stamp + 1h), the library's forward convention, so an
    accumulation at step s belongs at date + (s - s[0]). Instantaneous fields are valid
    at cfgrib's valid_time, date + s - an hour later.
    '''

    if variable not in _CUMULATIVE:
        raise Exception(f'How to handle variable "{variable}" must be defined.')

    data = _read_ERA5Land_helper(file, variable)
    steps = data['steps']

    if _CUMULATIVE[variable]:
        # Detected from the data rather than the calendar: these guards used to test
        # for December 31 and so never fired on a month boundary.
        if steps.size > 1:
            first = data['data'][0]
            if np.isnan(first[:-1]).all() and np.isfinite(first[-1]).any():
                data['data'] = data['data'][1:]
                data['production_datetime'] = data['production_datetime'][1:]
                warnings.warn('Last hour of the previous period dropped...')

            if np.isnan(data['data'][-1, -1]).all() and file_next_year is not None:
                following = _read_ERA5Land_helper(file_next_year, variable)
                data['data'][-1, -1] = following['data'][0, -1]
                warnings.warn('First hour of the following period added...')

        data['data'] = np.diff(data['data'], n=1, axis=1, prepend=0)

    if variable == 'tp':
        data['data'] *= 1000
        units = 'mm/hr'
    elif variable == 't2m':
        data['data'] -= 273.15
        units = 'C'
    elif variable == 'sd':
        data['data'] *= 1000
        units = 'mm'
    elif variable == 'ssr':
        data['data'] /= 3600
        units = 'W/m2'
    else:
        units = 'm/s'                                         # u10, v10

    offsets = steps - steps[0] if _CUMULATIVE[variable] else steps
    stamps = (np.asarray(data['production_datetime']).reshape(-1, 1)
              + np.asarray(offsets).reshape(1, -1))
    data['production_datetime'] = stamps.ravel()
    data['data'] = data['data'].reshape((-1,) + data['data'].shape[-2:])

    data['data'] = np.expand_dims(data['data'], [1, 2])
    data['leadtimes'] = np.array([pd.DateOffset(days=0)])

    raster = MeteoRaster(data, units=units, variable=variable)
    raster.trim()

    return raster
