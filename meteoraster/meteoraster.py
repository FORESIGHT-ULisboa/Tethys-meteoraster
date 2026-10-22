import math
import copy
import os
import warnings
import tempfile
import numpy as np
import xarray as xr
import datetime as dt
import pandas as pd
from pathlib import Path

from . import _lazy
from . import io as _io
from . import zonal as _zonal

# GDAL/OGR, matplotlib, seaborn and cartopy are imported on demand through
# _lazy so that `import meteoraster` works without them - see _lazy.py.

class MeteoRaster(object):
    '''
    Custom class to handle meteorological raster files, including ensembles and forecasts

    v3.1:
        get_values_from_KML / get_quantiles_from_KML weight by true WGS84 area
        (area_weighting='ellipsoidal', new default) instead of square degrees, which
        over-weighted the poleward side of a zone by ~tan(lat) * lat-span. Pass
        area_weighting='planar' to reproduce v3.0 exactly. A zone partly outside
        the grid now warns (on_partial=) and centroids carry its grid_fraction. A
        reused coverage_info built for another grid raises instead of aggregating
        the wrong cells.

    v3.0:
        BREAKING - four readers are gone. meteoraster.utils keeps only
        readERA5Land_monthly and read_ERA5Land_hourly; readERA5_monthly, read_GFS,
        read_C3S and readCORDEX_monthly are removed. All six were dead code - they
        populated data['productionDates'] while __init__ reads
        data['production_datetime'] (renamed in 9c79551, which missed the file), so
        every one raised KeyError. Converted Tethys output needs no reader: use
        MeteoRaster.load.

        BREAKING - get_values_from_KML returns different numbers. It now runs on
        meteoraster.zonal instead of the legacy coverage builder, which clipped its
        candidate cells to 0 < i < n-1 and so left the outermost row and column of the
        grid without coverage; because the weights were renormalised by the area it did
        find, the result looked plausible with the border silently missing (measured:
        1.0 on a 3x3 grid whose true area-weighted mean is 9.0). Its coverage threshold
        was also read from zone 0 and applied to every zone. Both are fixed.
        ensemble_member labels are now 0-based, matching get_values_from_latlon and
        get_quantiles_from_KML; `elementwise` is accepted, ignored and warns, since the
        missing-data renormalisation it skipped is now always applied. This also removes
        the last scipy LinearNDInterpolator from a library code path.

        BREAKING - single-leadtime up-sampling in Resample returns different numbers,
        as described under v2.6 below.

        The two surviving readers work, and are covered numerically against real
        ERA5-Land GRIB. Beyond the key rename: the monthly reader filtered on
        stepType='avgad' where CDS now delivers 'avgas', and could not read a file
        holding two variables with different stepTypes or a single month (a collapsed
        time dimension made ds[variable][:, :, :] raise and days_in_month return an
        int); the hourly reader's drop/borrow guards were hard-coded to December 31 and
        so never fired on a month boundary, it stamped instantaneous fields an hour
        early (cfgrib's valid_time is date + step, not date + step - step[0]), it
        hard-coded 24 steps per day, and `sd` skipped the reshape and so could not build
        a MeteoRaster at all.

        Resample handles unevenly spaced leadtimes (Resample.VERSION 2.1). Operational
        forecasts go hourly, then 3-hourly, then 6-hourly; the kernel now finds a
        sub-step's source column by searchsorted over the leadtimes and weights it by
        that column's own span, so 'sum' still conserves mass when a 1 h and a 6 h
        column land in the same output window. Uniform leadtimes are bit-for-bit
        unchanged. A forecast with one production row is accepted (only the time-series
        path needs two), and calendar leadtimes now raise a message that says so
        instead of pandas' "Invalid type for timedelta scalar".

        adjust_leadtimes works on monthly data. It measured its horizon with
        `.dt.days`, which raised AttributeError whenever the leadtimes were already
        pd.DateOffsets - that is, on exactly the seasonal forecasts it exists for.

        Testing: real ERA5-Land GRIB and converted Tethys forecasts are now part of the
        suite (ERA5-Land monthly and hourly, ECMWF SEAS5 seasonal, IPMA, NOAA GFS),
        covering de-accumulation, the monthly DateOffset adaptation, member reduction
        and the resampling units contract. Fixtures skip by name when absent, so a
        clone without them still passes. The GLEAM resource and its ad-hoc reader are
        gone.

    v2.6:
        Works in a slim container. GDAL, matplotlib, seaborn and cartopy are now
        imported on demand, so `import meteoraster` no longer fails without them -
        before this, `from osgeo import ogr` at module scope made the package
        unimportable in a pip-only image even for save/load, which need no GDAL.

        Reads are portable across installs. A new module, meteoraster.io, is the
        single door to a file: it tries the read engines in turn (h5netcdf, netcdf4,
        scipy) and pins the decoding choices that drift between xarray releases. The
        same file was decoding `leadtime` as timedelta64[us] on xarray 2026.7 and
        timedelta64[ns] on 2023.12; `engine='h5netcdf'` was raising
        ValueError("unrecognized engine") wherever h5netcdf was absent even though
        netcdf4 could read the file; and monthly DateOffset leadtimes could come
        back as bare integers. h5py is now declared explicitly, because h5netcdf
        >=1.6 no longer depends on it.

        Large files no longer have to be held whole. io.read_header reads metadata
        only, and io.iter_productions streams one production at a time - measured
        126 MB peak against 467 MB for load() on a 352 MB cube. load() also takes
        productions/leadtimes/dtype for a partial read.

        New module meteoraster.zonal: sparse coverage (per-zone cell indices and
        weights), exact fractional overlap, and area-weighted means and quantiles
        that renormalise for missing data per zone and per slice. It derives cell
        corners with numpy instead of two scipy interpolators, so the outermost row
        and column of the grid finally take part, and it reads namespaced KML
        without rewriting the caller's file. get_values_from_KML was still on the old
        builder at this point; v3.0 rewired it.

        get_quantiles_from_KML works. __groupByQuantile had an unconditional
        `raise Exception('This must be reviewed.')` with the rest of the method dead
        behind it. Quantiles are now exact weighted quantiles rather than replicating
        each cell round(weight/precision) times, which cost 10x the memory and
        dropped any cell covering less than one precision step; `precision` is
        accepted, ignored and warns. Which axes are pooled is explicit via `pool=`.

        Memory: get_cropped no longer deep-copies the whole uncropped cube nor
        indexes it four times over (973 MB -> 621 MB peak on that same file);
        join/adjust_leadtimes/resample_timestep keep float32 instead of silently
        promoting to float64; is_complete/get_complete_index/get_missing walk one
        production at a time instead of allocating a full-size boolean array.

        save() writes its temporary file in the destination directory and swaps it in
        with os.replace, so the swap is atomic and nothing is staged through /tmp -
        which in a container is often tmpfs or a different filesystem. Saved files are
        0644 rather than 0600, so a shared volume is readable.

        Fixed: da_attrs/ds_attrs defaulted to shared mutable dicts that
        _update_attrs() wrote into, so one raster's units and variable leaked into
        later ones - and since save() uses self.variable as the NetCDF variable name,
        the name written to disk could come from an unrelated object; load() bound
        both attribute dicts to ds.attrs, so each round trip accreted global
        attributes; a single-latitude-row raster raised IndexError in __init__, so it
        saved but could never be loaded; resample_timestep raised TypeError on its
        last diagnostic line; get_missing swapped its leadtime and ensemble_member
        labels; the size-1 level drop compared against 'Zone' instead of 'zone', so a
        single-zone layer lost its zone level entirely.

        Resample was rewritten (Resample.VERSION 2.0) around exact integer-nanosecond
        index arithmetic, and now documents its contract: 'sum' is extensive and
        rescales with the step, 'mean'/'linear'/'max' are intensive and
        step-invariant, every value describes [stamp, stamp + step). Down-sampling is
        unchanged; single-leadtime up-sampling returns different numbers, because
        'sum' now splits the total across the sub-steps instead of emitting a sliding
        total over the whole production step. Calendar steps ('MS', 'M') are rejected
        rather than silently treated as fixed durations. See AGENTS.md for the detail.

        numpy is no longer capped below 2, and CI now runs the suite - it never did
        before - on the full conda environment and on a deliberately minimal
        pip install.

    v2.5:
        Published on PyPI - installs with `pip install meteoraster`. Releases are
        built and uploaded automatically on tag push (GitHub Actions + PyPI
        Trusted Publishing), and attached to the GitHub release.
        Licensing made consistent: the LICENSE file carried GPL-3 text while the
        package metadata declared Apache-2.0. Apache-2.0 is now the license
        everywhere, declared as a PEP 639 SPDX expression, with a NOTICE file.

    v2.4:
        Packaging modernized (pyproject.toml, optional dependencies, h5netcdf and
        pytest declared as dependencies).
        Cleaned up deprecated numpy/pandas idioms (np.NaN -> np.nan,
        np.product -> np.prod, removed redundant axis= arguments) for
        forward-compatibility with numpy 2 and pandas 3.

    v2.3:
        Fixed a bug in the groupByMatrix function where the coverage was not updated to reflect missing data, which could lead to incorrect aggregations. The coverage is now set to 0 for pixels where data is not valid before performing the aggregation.

    v2.2:
        Specifies the saving engine as 'h5netcdf'

    v2.1:
        Updated to VSCode
        Updated to drop pickle and use cross-platfrom alternatives instead

        Added testing
        Removed support for reading multiple data formats
        A "latest" version is compiled and published

        Untested support for pd.DateOffset
    '''
    
    VERSION = '3.1'
    VERBOSE = True
    ENSEMBLEMEMBERpOSITION = 1

    ENGINE = 'h5netcdf'
    
    def __init__(self, data,
                 latitudes=None, longitudes=None, production_datetime=None, leadtimes=None,
                 units='unknown', variable='unknown', da_attrs=None, ds_attrs=None,
                 verbose=None,
                 ):
        '''
        data: 5-D numpy array [production_datetime, ensemble_member, leadtime, latitude, longitude]
        production dates: 1-D numpy array
        ensemble members: 1-D numpy array
        leadtimes: 1-D numpy array
        latitudes: 1-D numpy array (center)
        longitudes: 1-D numpy array (center)

        alternatively, data can be a dict with the previous fields
        '''

        if verbose is None:
            self.verbose = self.VERBOSE
        else:
            self.verbose = verbose

        self.units = units
        self.variable = variable
        # These default to None and are copied because _update_attrs() writes into
        # them. As shared mutable defaults ({}), one instance's units and variable
        # leaked into every later instance built without explicit attrs - and since
        # save() uses self.variable as the NetCDF variable name, the name written to
        # disk could come from an unrelated object in the same process.
        self.da_attrs = {} if da_attrs is None else dict(da_attrs)
        self.ds_attrs = {} if ds_attrs is None else dict(ds_attrs)
        self._update_attrs()

        if isinstance(data, dict):
            self.data = data['data']
            self.latitudes = data['latitudes']
            self.longitudes = data['longitudes']
            self.production_datetime = data['production_datetime']
            self.leadtimes = data['leadtimes']
        else:
            self.data = data

        if not latitudes is None:
            self.latitudes = latitudes
        if not longitudes is None:
            self.longitudes = longitudes
        if not production_datetime is None:
            self.production_datetime = production_datetime
        if not leadtimes is None:
            self.leadtimes = leadtimes

        # Fix Geomery
            #Greenwish
        if self.longitudes.ndim==1:
            if max(self.longitudes)>180:
                self._fixStartAtGreenwich()

            # Process geometry so that lats and lons are stored in 2D
        if self.longitudes.ndim==1 and self.latitudes.ndim==1:
            self.longitudes, self.latitudes = np.meshgrid(self.longitudes, self.latitudes)
        elif self.longitudes.ndim != self.latitudes.ndim:
            raise ValueError('latitudes (%d-D) and longitudes (%d-D) must both be 1-D or both 2-D.'
                             % (self.latitudes.ndim, self.longitudes.ndim))

            # Flip latitudes (if required). Guarded on the row count: a
            # single-latitude-row raster (get_cropped can produce one) used to raise
            # IndexError here, so such a file saved but could never be loaded.
        if self.latitudes.shape[0]>1 and self.latitudes[0, 0]<self.latitudes[1, 0]:
            self._flipLatitude()
    
    def _update_attrs(self):
        '''
        Docstring for _update_attrs
        
        :param self: Description
        '''

        if self.units == 'unknown':
            if 'units' in self.da_attrs.keys():
                self.units = self.da_attrs['units']
        else:
            self.da_attrs['units'] = self.units

        if self.variable == 'unknown':
            if 'variable' in self.da_attrs.keys():
                self.variable = self.da_attrs['variable']
        else:
            self.da_attrs['variable'] = self.variable
    
    def create_plot(self, central_longitude, projection_kwargs={}, figsize=None, projection_fun=None, crs=None, extent=None):
        '''
        extent [west, east, south, north]

        projection_fun and crs default to cartopy PlateCarree; they are None in
        the signature because cartopy is imported on demand.
        '''

        plt = _lazy.pyplot()
        ccrs = _lazy.ccrs()
        if projection_fun is None:
            projection_fun = ccrs.PlateCarree
        if crs is None:
            crs = ccrs.PlateCarree()

        plt.figure(figsize=figsize)
        projection = projection_fun(central_longitude=central_longitude, **projection_kwargs)
        ax = plt.axes(projection=projection)
        if not isinstance(extent, type(None)):
            ax.set_extent(extent, crs=crs)
        return ax

    def add_shapefile(self, ax, shapefile_path, crs=None, **kwargs):
        '''

        '''

        shpreader = _lazy.shapereader()
        if crs is None:
            crs = _lazy.ccrs().PlateCarree()

        ax.add_geometries(shpreader.Reader(shapefile_path).geometries(), crs=crs, **kwargs)

    def add_shape(self, ax, path, color='black', facecolor='none', linewidth=1, crs=None):
        '''

        '''
        plt = _lazy.pyplot()
        shpreader = _lazy.shapereader()
        if crs is None:
            crs = _lazy.ccrs().PlateCarree()

        reader = shpreader.Reader(path)

        for record in reader.records():
            geometry = record.geometry
            ax.add_geometries([geometry], crs=crs, edgecolor=color, facecolor=facecolor, linewidth=linewidth)
        plt.show(block=False)

    def plot_mean(self, ax=None, xarray=None, block=False, multiplier=1, coastline=False, borders=False, colorbar=True,
                  colorbar_label=None, cmap='viridis', central_longitude=None, central_latitude=None, *args, **kwargs):
        '''
        Plots the mean behavior for the full time series
        '''
          
        plt = _lazy.pyplot()
        ccrs = _lazy.ccrs()

        if isinstance(ax, type(None)):
            ax = self.create_plot(central_longitude)

        if isinstance(xarray, type(None)):
            xarray = self.to_xarray()
            xarray = xarray.mean(dim=['production_datetime', 'ensemble_member', 'leadtime'])*multiplier


        plot = xarray.plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(), x="lon", y="lat", add_colorbar=False, cmap=cmap, *args, **kwargs)
        if colorbar:
            if colorbar_label is None:
                colorbar_label = self.units
            cbar = plt.colorbar(plot, orientation='vertical', label=colorbar_label)
        else:
            cbar = None
            
        if coastline or borders:
            cf = _lazy.cfeature()
            if coastline:
                ax.add_feature(cf.COASTLINE)
            if borders:
                ax.add_feature(cf.BORDERS)

        plt.show(block=block)
    
        return ax, cbar
    
    def plot_mean_projected(self, ax=None, block=False, multiplier=1, *args, **kwargs):
        '''
        Plots the mean behavior for the full time series using the true coordinates
        '''

        plt = _lazy.pyplot()
        _lazy.seaborn().color_palette('viridis')

        tmp = np.nanmean(self.data, axis=(0,1,2))*multiplier

        if isinstance(ax, type(None)):
            plt.figure()

        plt.scatter(self.longitudes.ravel(), self.latitudes.ravel(), c=tmp.ravel(), *args, **kwargs)
        plt.colorbar()
        plt.show(block=block)

    def plot_coordinates(self, ax=None, block=False, *args, **kwargs):
        '''
        Plots the available coordinates in the lat and lon matrixes
        '''

        plt = _lazy.pyplot()

        if isinstance(ax, type(None)):
            plt.figure()

        plt.plot(self.longitudes.ravel(), self.latitudes.ravel(), 'k.', *args, **kwargs)
        plt.show(block=block)

    def plot_seasonal(self, lat, lon, window=31, leadtime=None, ax=None, block=False, multiplier=1, *args, **kwargs):
        
        data = self.get_values_from_latlon(lat, lon)
        if isinstance(leadtime, type(None)):
            leadtime = self.leadtimes[0]

        # Select specific leadtime columns (level 0 is leadtime)
        data = data.xs(leadtime, level='leadtime', axis=1)

        tmp = data.rolling(window=window, center=True).mean().dropna().stack().to_frame(name='values').reset_index()
        
        # Use proper column name from index
        idx_name = data.index.name if data.index.name else 'index'
        if idx_name in tmp.columns:
             tmp.rename(columns={idx_name: 'Production dates'}, inplace=True)

        tmp.loc[:, 'Day of year'] = tmp.loc[:, 'Production dates'].dt.day_of_year

        if isinstance(ax, type(None)):
            _lazy.pyplot().figure(figsize=(20,10))
        ax = _lazy.seaborn().lineplot(x='Day of year', y='values', data=tmp, errorbar=('pi', 95), ax=ax, *args, **kwargs)

        return ax

    def get_values_from_KML(self, kml, nameField=None, coverage_info=None, getCoverageInfo=False,
                            elementwise=False, min_coverage=None, on_error='raise',
                            area_weighting=None, on_partial='warn'):
        '''
        Area-weighted mean of each KML zone, per production, member and leadtime.

        Returns (agg, centroids). agg is indexed by production datetime and its columns
        are a MultiIndex over the surviving levels of (zone, leadtime, ensemble_member);
        size-1 levels are dropped except zone.

        v3.0 runs this on meteoraster.zonal instead of the legacy coverage builder, so
        the numbers changed for any zone touching the edge of the grid. The old builder
        clipped its candidate cells to 0 < i < n-1, which left the outermost row and
        column of the grid without coverage; because weights were renormalised by the
        area actually found, the result looked plausible with the border silently
        missing. Its coverage threshold was also read from zone 0 and applied to every
        zone. Both are fixed, and ensemble_member labels are now 0-based, matching
        get_values_from_latlon and get_quantiles_from_KML.

        min_coverage: fraction of a zone's weight that must carry data for a slice to
              be reported, tested per zone and per slice. Defaults to
              zonal.MIN_COVERAGE.
        elementwise: accepted for backwards compatibility and ignored. It skipped the
              missing-data renormalisation, which apply_coverage now always applies.
        area_weighting: 'ellipsoidal' (default, true WGS84 areas), 'cos_lat', or
              'planar' (square degrees) to reproduce v3.0 numbers. See zonal.
        on_partial: 'warn' (default), 'raise' or 'ignore' for zones partly outside
              the grid; centroids['grid_fraction'] holds the share inside.

        A reused coverage_info must have been built for this exact grid (checked by
        fingerprint for zonal.Coverage, by shape for the legacy tuple).
        '''

        if elementwise:
            warnings.warn('The `elementwise` argument of get_values_from_KML() is ignored: the '
                          'area weights are always renormalised over the cells that carry data. '
                          'Remove the argument.', DeprecationWarning, stacklevel=2)

        if min_coverage is None:
            min_coverage = _zonal.MIN_COVERAGE

        if isinstance(coverage_info, type(None)):
            coverage = _zonal.coverage_from_kml(kml, self.latitudes, self.longitudes,
                                                name_field=nameField, on_error=on_error,
                                                normalize='area',
                                                area_weighting=area_weighting or _zonal.DEFAULT_AREA_WEIGHTING,
                                                on_partial=on_partial)
            coverage_info = coverage
        else:
            coverage = self.__as_coverage(coverage_info)
            self.__check_coverage(coverage)

        agg = self.__groupByCoverage(coverage, min_coverage)

        centroids = coverage.centroids

        if getCoverageInfo:
            return agg, centroids, coverage_info
        else:
            return agg, centroids

    def get_quantiles_from_KML(self, kml, nameField=None, coverage_info=None, getCoverageInfo=False,
                               resampling=None, quantiles=(0.01, 0.1, 0.5, 0.9, 0.99),
                               pool=('space',), min_coverage=None, method='hazen',
                               precision=None, on_error='raise', area_weighting=None,
                               on_partial='warn'):
        '''
        Area-weighted spatial quantiles over the KML zones.

        Returns (agg, centroids). agg is indexed by production datetime and its
        columns are a MultiIndex over the surviving levels of
        (zone, leadtime, ensemble_member, quantile); size-1 levels are dropped
        except zone and quantile.

        A cell half inside a zone carries half the probability mass, which is what
        the coverage weights mean. Quantiles are computed exactly - see
        zonal.weighted_quantiles - rather than by replicating each cell
        round(weight / precision) times, which cost 10x the memory and dropped any
        cell covering less than one `precision` step.

        pool: which axes are pooled into the sample, from
              {'space', 'leadtime', 'production', 'members'}. 'space' is required.
              The default pools only the zone's cells, giving one distribution per
              (production, member, leadtime) - the direct spatial analogue of
              get_values_from_KML.

              'time' lives on a different axis depending on the layout, because the
              event time is production_datetime + leadtime, so the axis is named
              explicitly. For a forecast (many leadtimes) pool 'leadtime'; for an
              observation/reanalysis raster (one leadtime, many productions) pool
              'production'.

        resampling: a pandas offset giving the *block size* within the pooled time
              axis, not a relabelling. pool=('space', 'leadtime') with
              resampling='1D' on an hourly forecast pools all the values in a zone
              across each day into one distribution - the hourly-to-daily case.
              With resampling=None the whole pooled axis is a single block and its
              level is dropped from the columns.

        method: 'hazen' (default) or 'linear'. See zonal.weighted_quantiles.
        precision: accepted for backwards compatibility, ignored, and warns when
              passed - the result no longer depends on it.
        area_weighting, on_partial: as in get_values_from_KML.
        '''

        if precision is not None:
            warnings.warn('The `precision` argument of get_quantiles_from_KML() is ignored: '
                          'quantiles are now computed exactly instead of by replicating each '
                          'cell round(weight / precision) times. Remove the argument.',
                          DeprecationWarning, stacklevel=2)

        if min_coverage is None:
            min_coverage = _zonal.MIN_COVERAGE

        if isinstance(coverage_info, type(None)):
            coverage = _zonal.coverage_from_kml(kml, self.latitudes, self.longitudes,
                                                name_field=nameField, on_error=on_error,
                                                normalize='cell_fraction',
                                                area_weighting=area_weighting or _zonal.DEFAULT_AREA_WEIGHTING,
                                                on_partial=on_partial)
            coverage_info = coverage
        else:
            coverage = self.__as_coverage(coverage_info)
            self.__check_coverage(coverage)

        agg = self.__groupByQuantile(coverage, resampling, quantiles, pool, min_coverage, method)

        centroids = coverage.centroids

        if getCoverageInfo:
            return agg, centroids, coverage_info
        else:
            return agg, centroids

    @staticmethod
    def __as_coverage(coverage_info):
        '''
        Accepts a zonal.Coverage or the legacy (dense, columns, centroids) tuple.
        '''

        if isinstance(coverage_info, _zonal.Coverage):
            return coverage_info

        if isinstance(coverage_info, tuple) and len(coverage_info) == 3:
            dense, _, centroids = coverage_info
            zones = list(centroids.index) if centroids is not None else None
            return _zonal.Coverage.from_dense(dense, zones=zones, centroids=centroids)

        raise TypeError('coverage_info must be a zonal.Coverage or a legacy '
                        '(coverage, columns, centroids) tuple.')

    def __check_coverage(self, coverage):
        '''
        Raises if a reused coverage was built for a different grid.
        '''

        if not coverage.matches(self.latitudes, self.longitudes):
            raise ValueError('coverage_info was built for a different grid (shape %s or '
                             'coordinates differ from this raster %s). Rebuild it from the KML.'
                             % (coverage.shape, self.latitudes.shape))

    def get_cropped(self, from_prod_date=dt.datetime(1900, 1, 1), to_prod_date=dt.datetime(2199, 12, 31),
                   from_lat=-90, to_lat=90, from_lon=-180, to_lon=180,
                   from_leadtime=None, to_leadtime=None):
        '''
        returns a MeteoRaster cropped in time and space
        '''
        
        self._diag('Cropping meteorology...', self.verbose)
        
        date_idxs = np.arange(self.production_datetime.shape[0])[(self.production_datetime>=np.datetime64(from_prod_date)) & (self.production_datetime<=np.datetime64(to_prod_date))]
        
        inside = np.zeros_like(self.latitudes).astype(bool)
        inside[(np.round(self.longitudes,6)>=np.round(from_lon,6)) & (np.round(self.longitudes,6)<=np.round(to_lon,6)) & (np.round(self.latitudes,6)>=np.round(from_lat,6)) & (np.round(self.latitudes,6)<=np.round(to_lat,6))] = True
        tmp = np.where(inside)
        lat_idxs = np.arange(tmp[0].min(), tmp[0].max()+1)
        lon_idxs = np.arange(tmp[1].min(), tmp[1].max()+1)
        
        #=======================================================================
        # lat_idxs = np.arange(self.latitudes.shape[0])[(self.latitudes>=from_lat) & (self.latitudes<=to_lat)]
        # lon_idxs = np.arange(self.longitudes.shape[0])[(self.longitudes>=from_lon) & (self.longitudes<=to_lon)]
        #=======================================================================
        
        production_datetime = self.production_datetime[date_idxs]
        latitudes = np.round(self.latitudes[lat_idxs, :],6)
        latitudes = np.round(latitudes[:, lon_idxs],6)
        longitudes = np.round(self.longitudes[lat_idxs, :],6)
        longitudes = np.round(longitudes[:, lon_idxs],6)

        leadtimes = copy.deepcopy(self.leadtimes)
        leadtime_selection = slice(None)
        if from_leadtime or to_leadtime:
            keep = np.ones(len(leadtimes), dtype=bool)
            if from_leadtime:
                keep &= np.asarray(leadtimes >= from_leadtime)
            if to_leadtime:
                keep &= np.asarray(leadtimes <= to_leadtime)
            leadtimes = leadtimes[keep]
            leadtime_selection = self._as_slice(np.flatnonzero(keep))

        # The spatial ranges are contiguous by construction, so taking them first is
        # a free view; only then are the (possibly non-contiguous) date and leadtime
        # selections applied, at the already-reduced spatial size. The previous
        # version indexed four times in a row - the first with a fancy index over
        # every date, which copied the *whole* uncropped cube - on top of a
        # copy.deepcopy(self) that had copied it once more.
        # Note the selections are applied one subscript at a time on purpose: two
        # advanced indices in a single subscript would be broadcast against each
        # other rather than combined.
        view = self.data[:, :, :, self._as_slice(lat_idxs), self._as_slice(lon_idxs)]
        date_selection = self._as_slice(date_idxs)
        if not (isinstance(date_selection, slice) and date_selection == slice(0, self.data.shape[0])):
            view = view[date_selection]
        if not isinstance(leadtime_selection, slice) or leadtime_selection != slice(None):
            view = view[:, :, leadtime_selection]
        data = np.ascontiguousarray(view)

        cropped = copy.copy(self)
        cropped.da_attrs = dict(self.da_attrs)
        cropped.ds_attrs = dict(self.ds_attrs)
        cropped.leadtimes = leadtimes
        cropped.production_datetime = production_datetime
        cropped.latitudes = latitudes
        cropped.longitudes = longitudes
        cropped.data = data

        self._diag('    Done.', self.verbose)
        
        return cropped
    
    def trim(self):
        '''
        
        '''
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            tmp = np.where(np.isfinite(np.nanmean(self.data, axis=(1, 2, 3, 4))))[0]
        start = tmp[0]
        end = tmp[-1]+1
        
        self.data = self.data[start:end, ...]
        self.production_datetime = self.production_datetime[start:end]
    
    def join(self, meteoRaster, strickt=False, trim=False):
        '''
        joins a new meteoraster to self based on production dates
        keeps a minimum number of ensembles.
        keeps only matching leadtimes
        '''
        
        if trim:
            self.trim()
            meteoRaster.trim()
        
        newData = meteoRaster.data
        newproduction_datetime = meteoRaster.production_datetime
        try:
            newLeadtimes = meteoRaster.leadtimes.astype(self.leadtimes.dtype)
        except Exception as ex:
            newLeadtimes = meteoRaster.leadtimes
        newLatitudes = meteoRaster.latitudes
        newLongitudes = meteoRaster.longitudes
        
            
        # handle spatial coverage
        if (np.abs(self.latitudes-newLatitudes)>1E-10).any() or self.data.shape[-2]!=newData.shape[-2]:
            raise(Exception('Latitudes do not match.'))
        
        if (np.abs(self.longitudes-newLongitudes)>1E-10).any() or self.data.shape[-1]!=newData.shape[-1]:
            raise(Exception('Longitudes do not match.'))
        
        
        # handle ensembles
        e0 = self.data.shape[1]
        e1 = newData.shape[1]
        warningMessage = 'The ensemble numbers do not match (base: %d, joint: %d). Filled missing with NaN.' % (e0, e1)
        errorMessage = 'The ensemble numbers do not match (base: %d, joint: %d). Error issued in strickt mode.' % (e0, e1)
        if e0>e1:
            if strickt:
                raise(Exception(errorMessage))
            else:
                warnings.warn(warningMessage)
            shape = newData.shape
            data = self._nan_filled([shape[0], e0] + list(shape[2:]), newData.dtype)
            data[:, :e1, :, :, :] = newData
            newData = data
        elif e0<e1:
            if strickt:
                raise(Exception(errorMessage))
            else:
                warnings.warn(warningMessage)
            shape = self.data.shape
            data = self._nan_filled([shape[0], e1] + list(shape[2:]), self.data.dtype)
            data[:, :e0, :, :, :] = self.data
            self.data = data

        
        # handle leadtimes
        v0 = np.where(np.isin(self.leadtimes, newLeadtimes))[0]    
        v1 = np.where(np.isin(newLeadtimes, self.leadtimes))[0]
    
        if v0.size!=self.leadtimes.size or v1.size!=newLeadtimes.size:
            if strickt:
                raise(Exception('Leadtimes do not all match (base: %d, joint: %d). Error issued in strickt mode.' % (self.leadtimes.size, newLeadtimes.size)))
            else:
                warnings.warn('Leadtimes do not all match (base: %d, joint: %d). Keeping only matching values.' % (self.leadtimes.size, newLeadtimes.size)) 
                self.leadtimes = self.leadtimes[v0]
                self.data = self.data[:, :, v0, :, :]
                newData = newData[:, :, v1, :, :]
    
    
        # join
        if self.leadtimes.shape != newLeadtimes.shape or not np.array_equal(self.leadtimes, newLeadtimes):
            raise(Exception('Leadtimes do not match. Not implemented'))

        if self.production_datetime[-1]<newproduction_datetime[0]:
            # self if first
            self.production_datetime = np.concatenate((self.production_datetime, newproduction_datetime))
            self.data = np.concatenate((self.data, newData), axis=0)
            
        elif self.production_datetime[0]>newproduction_datetime[-1]: 
            # joint is first
            self.production_datetime = np.concatenate((newproduction_datetime, self.production_datetime))
            self.data = np.concatenate((newData, self.data), axis=0)
        else:
            # mixed/interleaved production dates: align on the union of dates, fill NaN, new data overwrites on overlap
            all_dates = np.sort(np.unique(np.concatenate((self.production_datetime, newproduction_datetime))))
            target_shape = (all_dates.size,) + tuple(self.data.shape[1:])
            target_dtype = np.result_type(self.data.dtype, np.float32)
            merged = np.full(target_shape, np.nan, dtype=target_dtype)

            # place existing data
            idx_self = np.searchsorted(all_dates, self.production_datetime)
            merged[idx_self, ...] = self.data.astype(target_dtype, copy=False)

            # place new data (overwrite on overlap)
            idx_new = np.searchsorted(all_dates, newproduction_datetime)
            newData_cast = newData.astype(target_dtype, copy=False)
            existing = merged[idx_new, ...]
            nan_mask = np.isnan(existing)
            if nan_mask.any():
                existing[nan_mask] = newData_cast[nan_mask]
                merged[idx_new, ...] = existing

            self.production_datetime = all_dates
            self.data = merged

    def adjust_leadtimes(self, period='months'):
        '''
        Adjust leadtimes to relative time steps (useful for monthly)
        '''
        self._diag('Adjusting lead times...', self.verbose)

        # The horizon is measured by applying the last leadtime to the first
        # production, which works for pd.Timedelta and pd.DateOffset alike. Reading
        # `.dt.days` off it instead raised AttributeError on a raster whose leadtimes
        # were already DateOffsets - i.e. on exactly the monthly data this is for.
        reference = pd.Timestamp(self.production_datetime[0])
        horizon_days = ((reference + self.leadtimes[-1]) - reference).days

        periodDefinition = {
            'months': pd.Series([pd.DateOffset(months=i) for i in range(1, math.ceil(horizon_days/30)+1)]),
            }

        if period not in periodDefinition.keys():
            raise(Exception('%s is not a valid period.' % period))
        
        warnings.warn('the function adjustLeadTimes() is experimental.')
    
        times = self.production_datetime[0] + self.leadtimes
        validTimes = pd.Timestamp(self.production_datetime[0]) + periodDefinition[period]
    
        idxs = self.closestIdx(times, validTimes)
        validPeriods =  periodDefinition[period][:idxs.max()+1]
        
        tmp = list(self.data.shape)
        data = self._nan_filled(tmp[:2] + [validPeriods.size] + tmp[-2:], self.data.dtype)
        
        for i0 in idxs:
            dataForLeadtime = self.data[:, :, idxs==i0, :, :]
            finite = np.isfinite(dataForLeadtime)
            count = np.sum(finite, axis=2)
            if count.max()>1:
                warnings.warn('Overlapping information detected when adjusting lead times (%s).' % str(validPeriods[i0]))
            
            tmp = np.nanmean(dataForLeadtime, axis=2)
            tmp[count==0] = np.nan
    
            if np.isnan(tmp).all():
                warnings.warn('Unused Lead time (%s)' % str(validPeriods[i0]))
            
            data[:, :, i0, :, :] = tmp
    
        self.data = data
        if period=='months':
            self.leadtimes = np.array([pd.DateOffset(months=i0.months-1) for i0 in validPeriods])
            
    
        self._diag('    Done.', self.verbose)
    
    def get_missing(self):
        '''
        Provides a diagnostic of missing data

        Rows are a (leadtime, ensemble_member) MultiIndex, columns are production
        datetimes.
        '''

        members = self.data.shape[self.ENSEMBLEMEMBERpOSITION]

        # One production at a time: np.isnan over the whole cube allocates a
        # full-size boolean array.
        missingFraction = np.empty((self.data.shape[0], self.data.shape[2], members))
        for production in range(self.data.shape[0]):
            fraction = np.isnan(self.data[production]).mean(axis=(-2, -1))  # [member, leadtime]
            missingFraction[production] = fraction.T

        # The axes are moved before flattening so that leadtime varies slowest and
        # ensemble_member fastest, which is the order the MultiIndex below declares.
        # Reshaping [production, member, leadtime] directly put leadtime fastest, so
        # the two labels were swapped whenever there was more than one of each.
        missingFraction = pd.DataFrame(missingFraction.reshape(missingFraction.shape[0], -1),
                                       index=self.production_datetime)
        missingFraction.index.name = 'production_datetime'
        missingFraction.columns = pd.MultiIndex.from_product((self.leadtimes, np.arange(members)),
                                                            names=['leadtime', 'ensemble_member'])

        return missingFraction.transpose()

    def get_values_from_latlon(self, lat, lon):
        '''
        Returns a dataframe with data from the pixel closest to the desired point
        '''

        idxLat, idxLon = self._getClosestPixels(lat, lon)

        if idxLat.size>1 or idxLon.size>1:
            warnings.warn('Several pixels are "closest". Averaging results.')
        tmp = []
        for iLat, iLon in zip(idxLat, idxLon):
            tmp.append(np.expand_dims(self.data[:, :, :, iLat, iLon], axis=0))
        tmp = np.concatenate(tmp, axis=0)
        tmp = np.nanmean(tmp, axis=0) # [production_datetime, ensemble_member, leadtimes]
        
        data = pd.DataFrame(tmp.reshape(tmp.shape[0]*tmp.shape[1], tmp.shape[2]))
        data.index = pd.MultiIndex.from_product((self.production_datetime.ravel(), np.arange(self.data.shape[1])), names=['production_datetime', 'ensemble_member'])
        data.columns = self.leadtimes
        data.columns.name = 'leadtime'

        data = data.unstack('ensemble_member')

        return data

    @staticmethod
    def get_values_from_latlon_by_event(production_date_dataframe):
        '''
        !!!!! Change the index name to Event dates
        '''
        
        ensembles = production_date_dataframe.columns.get_level_values('ensemble_member').unique()
        full_data_ = []
        for ensemble in ensembles:
            tmp = production_date_dataframe.loc[:, production_date_dataframe.columns.get_level_values('ensemble_member')==ensemble]
            data_ = []
            for i0 in range(tmp.shape[1]):
                tmp_ = tmp.iloc[:, [i0]]
                tmp_.index = tmp_.index + tmp_.columns.get_level_values('leadtime')[0]
                data_.append(tmp_)
            full_data_.append(pd.concat(data_, axis=1))
        
        return pd.concat(full_data_, axis=1)

    def plot_availability(self, missing=None, individualMembers=False):
        '''
        Plots the availability of the data
        '''
        
        if isinstance(missing, type(None)):
            missing = self.get_missing()

        available = missing < 0.95
        
        if individualMembers:
            tmp = available.stack().to_frame()
        else:
            virtualLevels = pd.to_datetime(self.production_datetime[0]) + self.leadtimes
            available.index = available.index.set_levels(virtualLevels, level=0)
            tmp = available.groupby(level=0).sum().transpose().stack().to_frame()
            tmp.index = tmp.index.set_levels(self.leadtimes, level=1)

        
        tmp.columns = ['available']
        tmp = tmp.loc[tmp['available']>0,:].reset_index()
        tmp.loc[:, 'event_dates'] = tmp.loc[:, 'production_datetime'] + tmp.loc[:, 'leadtime']
        
        plt = _lazy.pyplot()
        sns = _lazy.seaborn()

        sns.set_theme(style="whitegrid")
        if individualMembers:
            g = sns.relplot(
                data=tmp,
                x='event_dates', y='production_datetime', hue='ensemble_member', 
                palette='plasma',
                )
        else:
            g = sns.relplot(
                data=tmp,
                x='event_dates', y='production_datetime', hue='available', size='available',
                palette='viridis',
                sizes={s: s*10 for s in np.arange(1,max((11, tmp.loc[:, 'available'].max()+1)))},
                #===================================================================
                # size_order=np.arange(1, max((11, tmp.loc[:, 'available'].max()+1)))
                #===================================================================
                )
        g.ax.xaxis.grid(True, "minor", linewidth=.25)
        g.ax.yaxis.grid(True, "minor", linewidth=.25)
        
        plt.show(block=False)

    def resample_timestep(self, rule, fun=np.mean):
        '''
        Resamples data according to pandas conventions
        https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.resample.html
        
        rule='MS' resamples to the start of the month
        '''
        
        warnings.warn('the function resample_timestep() is experimental.')

        self._diag('    Resampling time step (rule: %s, %s)...' % (rule, str(fun)), self.verbose)        

        # prepare a template dataframe
        tmp = pd.DataFrame(np.arange(self.production_datetime.shape[0]), index=self.production_datetime)
        idxMax = tmp.resample(rule=rule).max().astype(int)+1
        
        resampled = self._nan_filled([idxMax.shape[0]] + list(self.data.shape[1:]), self.data.dtype)
        idx0 = 0
        for i0, i1 in enumerate(idxMax.values.ravel()):
            resampled[i0, :, :, :, :] = fun(self.data[idx0:i1, :, :, :, :], axis=0, keepdims=True)
            idx0 = i1
    
        self.data = resampled
        self.production_datetime = idxMax.index
    
        # The stray self.verbose used to be passed positionally to strftime, so this
        # line raised TypeError on every successful call.
        self._diag('        Done (%s to %s).' % (self.production_datetime[0].strftime('%Y-%m-%d %H:%M:%S'),
                                                 self.production_datetime[-1].strftime('%Y-%m-%d %H:%M:%S')),
                   self.verbose)
    
    def is_complete(self, full_ensemble:bool=True, space_completeness:bool=False) -> bool:
        '''
        Checks if the dataset is complete.

        - Always requires every production_datetime x leadtime slice to have finite data.
        - If full_ensemble is True, all ensemble members must be finite for each slice;
          otherwise, at least one ensemble member with finite data is enough.
        - If space_completeness is True, all spatial points (y, x) must be finite;
          otherwise, at least one spatial point with finite data is enough.

        :param full_ensemble: Require all ensemble members to be complete.
        :param space_completeness: Require full spatial coverage.
        :return: True if complete, False otherwise.
        '''
        if self.data is None:
            return False

        # Walked one production at a time, with an early exit. np.isfinite over the
        # whole cube allocates a full-size boolean array, and save() calls this by
        # default, so it used to add a grid-sized temporary to every write.
        for production in range(self.data.shape[0]):
            finite = np.isfinite(self.data[production])

            # Ensemble requirement
            if full_ensemble:
                finite = finite.all(axis=0)  # -> [leadtime, y, x]
            else:
                finite = finite.any(axis=0)  # -> [leadtime, y, x]

            # Spatial requirement
            if space_completeness:
                finite = finite.all(axis=(-2, -1))  # -> [leadtime]
            else:
                finite = finite.any(axis=(-2, -1))  # -> [leadtime]

            # Temporal requirement: every production_datetime and every leadtime
            # must be complete
            if not finite.all():
                return False

        return True
    
    def get_complete_index(self, full_ensemble:bool=True, space_completeness:bool=False) -> pd.DataFrame:
        '''
        
        '''

        self._diag('Retrieving complete index...', self.verbose)

        # One production at a time, for the same reason as is_complete().
        finite = np.empty((self.data.shape[0], self.data.shape[2]), dtype=bool)
        for production in range(self.data.shape[0]):
            slice_ = np.isfinite(self.data[production])

            # Ensemble requirement
            if full_ensemble:
                slice_ = slice_.all(axis=0)  # -> [leadtime, y, x]
            else:
                slice_ = slice_.any(axis=0)  # -> [leadtime, y, x]

            # Spatial requirement
            if space_completeness:
                finite[production] = slice_.all(axis=(-2, -1))  # -> [leadtime]
            else:
                finite[production] = slice_.any(axis=(-2, -1))  # -> [leadtime]

        completeness_index = pd.DataFrame(finite, index=self.production_datetime, columns=self.leadtimes)
        completeness_index.index.name = 'production_datetime'
        completeness_index.columns.name = 'leadtime'

        return completeness_index

    @classmethod
    def get_completeness(cls, file):
        '''
        Reads the `complete` flag of a saved file without reading any values.

        Returns None when the file carries no flag (it used to be a KeyError, and
        an AttributeError when a writer had stored a real boolean).
        '''

        return _io.read_completeness(file)

    @classmethod
    def read_header(cls, file):
        '''
        Reads a file's metadata (coordinates, shape, units) without its values.

        See meteoraster.io.Header.
        '''

        return _io.read_header(file)

    @classmethod
    def iter_productions(cls, file, **kwargs):
        '''
        Yields (production_datetime, block) one production at a time.

        Peak memory is one production block whatever the file size, which is what
        makes aggregating a multi-GB archive possible. See
        meteoraster.io.iter_productions for the arguments.
        '''

        return _io.iter_productions(file, **kwargs)


    def save(self, file, complevel=1, complete=None, tmp_dir=None, mode=0o644,
             engine=None) -> None:
        '''
        Saves a netcdf file (.nc / .nct)

        The file is written to a temporary name in the *destination directory* and
        then swapped in with os.replace, so the swap is atomic and nothing is
        staged through the system temp dir. That matters in a container, where
        /tmp is often tmpfs (the whole compressed file would be built in RAM) or a
        different filesystem from the output volume (shutil.move would degrade to
        copy + unlink, streaming the file twice).

        tmp_dir: stage elsewhere when the destination directory is not writable.
                 The swap is then a cross-device copy and no longer atomic.
        mode:    permissions for the finished file, 0o644 by default. A temporary
                 file is created 0o600 and the rename preserves it, so before this
                 every saved file was readable only by the writing user - a
                 regular problem for a container writing to a shared volume.
                 Pass None to keep 0o600.
        engine:  defaults to cls.ENGINE ('h5netcdf'). Use 'netcdf4' if h5netcdf
                 rejects the zlib/complevel encoding keys on your stack.
        '''

        if not isinstance(file, Path):
            file = Path(file)

        self._diag('Saving %s...' % file, self.verbose)

        data_array = self.to_xarray()
        ds = xr.Dataset({self.variable: data_array})
        ds['lat'].attrs['standard_name'] = 'latitude'
        ds['lon'].attrs['standard_name'] = 'longitude'

        ds.attrs = {k: v for k, v in self.ds_attrs.items()}
        ds.attrs['history'] = data_array.attrs['saved']
        ds.attrs['meteoraster_version'] = self.VERSION
        if complete is None:
            ds.attrs['complete'] = str(self.is_complete())
        else:
            ds.attrs['complete'] = str(complete)

        directory = Path(tmp_dir) if tmp_dir is not None else file.parent
        directory.mkdir(parents=True, exist_ok=True)

        handle, name = tempfile.mkstemp(prefix=file.name + '.', suffix='.tmp', dir=str(directory))
        os.close(handle)
        tmp_file = Path(name)
        try:
            if complevel>0:
                encoding = {
                    self.variable: {"zlib": True, "complevel": complevel},
                }
                ds.to_netcdf(tmp_file, encoding=encoding, engine=engine or self.ENGINE)
            else:
                ds.to_netcdf(tmp_file, engine=engine or self.ENGINE)

            if mode is not None:
                os.chmod(tmp_file, mode)

            os.replace(tmp_file, file)
        except Exception:
            if tmp_file.exists():
                try:
                    tmp_file.unlink()
                except Exception:
                    pass
            raise

        self._diag('    Saved.', self.verbose)
    
    def to_xarray(self) -> xr.DataArray:
        '''
        Docstring for to_xarray
        
        :param self: Description
        :return: Description
        :rtype: DataArray
        '''
        
        leadtimes = self.leadtimes
        leadtime_format = 'pd.Timedelta'
        if isinstance(leadtimes[-1], pd.DateOffset):
            if hasattr(leadtimes[-1], 'months'):
                leadtimes = [i.months for i in leadtimes]
                leadtime_format = 'pd.DateOffset:months'
            else:
                leadtimes = [i.years for i in leadtimes]
                leadtime_format = 'pd.DateOffset:years'

        data_array = xr.DataArray(
            data=self.data,
            dims=['production_datetime', 'ensemble_member', 'leadtime', 'y', 'x'],
            coords={
                'production_datetime': self.production_datetime,          # pd.Timestamp
                'ensemble_member': np.arange(self.data.shape[1], dtype=int),            # int
                'leadtime': leadtimes,                         # pd.Timedelta / DateOffset
                'lat': (['y', 'x'], self.latitudes),                   # float
                'lon': (['y', 'x'], self.longitudes),                  # float
                'y': np.arange(self.data.shape[-2], dtype=int),
                'x': np.arange(self.data.shape[-1], dtype=int),
                'leadtime_format': leadtime_format,
            },
            name=self.variable,
            attrs={'units': self.units, 'variable': self.variable, 'saved': f'Saved by {self.__class__.__name__} v{self.VERSION} @ {pd.Timestamp.now().isoformat()}'},
        )

        # Also record the leadtime format as an attribute on the leadtime
        # variable. The scalar coordinate above only survives while xarray
        # propagates dataset scalar coords onto extracted variables and the
        # `coordinates` attribute stays intact; any tool that rewrites the file
        # loses it, and DateOffset leadtimes then read back as bare integers.
        data_array['leadtime'].attrs['leadtime_format'] = leadtime_format

        return data_array
    
    @classmethod
    def load(cls, file, verbose=None, productions=None, leadtimes=None, dtype=None,
             engine=None) -> "MeteoRaster":
        '''
        Loads a MeteoRaster from a NetCDF (.nc / .nct) file written by save().

        Reading goes through meteoraster.io, which tries the available engines in
        turn and pins the decoding choices that drift between xarray versions - so
        a file written by another install reads back with the same dtypes. See
        io.open_raster.

        productions / leadtimes: read only part of the file. Accept indices, a
            slice, or the coordinate values themselves.
        dtype: cast while reading. The cube is filled one production at a time, so
            dtype='float32' on a float64 file genuinely halves peak memory instead
            of casting after the whole array is already resident.

        For a file too large to hold at all, use io.iter_productions (or
        MeteoRaster.iter_productions) and aggregate block by block.
        '''

        if not isinstance(file, Path):
            file = Path(file)

        kwargs = {}
        if verbose is None:
            verbose = cls.VERBOSE
        else:
            kwargs['verbose'] = verbose
        cls._diag('Loading %s...' % file, verbose)

        if file.suffix.lower() not in {'.nc', '.nct'}:
            raise ValueError('Only NetCDF files written by save() are supported.')

        data, production_datetime, leadtime_values, header = _io.read_cube(
            file, productions=productions, leadtimes=leadtimes, dtype=dtype, engine=engine)

        meteo = cls(
            data=data,
            latitudes=header.latitudes,
            longitudes=header.longitudes,
            production_datetime=production_datetime,
            leadtimes=leadtime_values,
            units=header.units,
            variable=header.variable,
            # These were both bound to ds.attrs, so the DataArray attributes were
            # lost and _update_attrs' write leaked into the global attributes -
            # every load/save cycle accreted `units` and `variable` there.
            da_attrs=dict(header.da_attrs),
            ds_attrs=dict(header.ds_attrs),
            **kwargs,
        )

        cls._diag('    Loaded.', verbose)

        return meteo


    def copy(self):
        '''
        Makes a copy of a MeteoRaster file
        '''
        
        mr = MeteoRaster(data=self.data.copy(),
                         latitudes=self.latitudes.copy(), longitudes=self.longitudes.copy(),
                         production_datetime=self.production_datetime.copy(), leadtimes=self.leadtimes.copy(),
                         da_attrs=self.da_attrs, ds_attrs=self.ds_attrs,
                         )
        return mr

    def _getClosestPixels(self, lat, lon):
        '''
        Returns the indexes of the pixel closer to a point 
        '''
    
        dist = ((self.longitudes-lon)**2+(self.latitudes-lat)**2)**0.5
        idx_lat, idx_lon = np.where(dist==dist.ravel().min())
                    
        return (idx_lat, idx_lon)
    
    @staticmethod
    def closestIdx(X, Y):
        '''
        Returns the list of Y indexes closest the values of X
        Dimension of X.
        Indexes to be used in Y.
        '''
    
        idxs = np.empty_like(X, dtype=int)
        for i0, x0 in enumerate(X):
            idxs[i0] = np.argmin(np.abs(x0-np.array(Y)))
                
        return idxs
  
    def _flipLatitude(self):
        '''
        flips the latitude of the raster
        '''
        
        self.data = np.flip(self.data, axis=3)
        self.latitudes = np.flip(self.latitudes, axis=0)
        self.longitudes = np.flip(self.longitudes, axis=0)
        
        self._diag('Flipped latitudes.', self.verbose)
    
    def _fixStartAtGreenwich(self):
        '''
        modifies rasters with longitudes starting at Greenwich to the more conveninent -180E.
        '''
        
        westIndexes = np.arange(len(self.longitudes))[self.longitudes>180]
        eastIndexes = np.arange(len(self.longitudes))[self.longitudes<=180]

        longitudes = np.concatenate((self.longitudes[westIndexes]-360, self.longitudes[eastIndexes]))

        # np.roll when the two halves are contiguous blocks, which is the usual
        # 0..360 layout: it produces one array instead of two fancy-indexed halves
        # plus the concatenation of them (3x peak for the largest array we hold).
        contiguous = (westIndexes.size and eastIndexes.size
                      and np.array_equal(westIndexes, np.arange(westIndexes[0], westIndexes[-1] + 1))
                      and np.array_equal(eastIndexes, np.arange(eastIndexes[0], eastIndexes[-1] + 1))
                      and eastIndexes[-1] + 1 == westIndexes[0])
        if contiguous:
            data = np.roll(self.data, westIndexes.size, axis=-1)
        else:
            data = np.concatenate((self.data[..., westIndexes], self.data[..., eastIndexes]), axis=-1)

        self.longitudes = longitudes
        self.data = data

        self._diag('Fixed start at Greenwich.', self.verbose)
    
    def __groupByCoverage(self, coverage, min_coverage):
        '''
        Area-weighted mean per zone, from a zonal.Coverage.

        Replaces __groupByMatrix, which densified the coverage to
        [1, 1, 1, y, x, zone] and then tiled it up to the full cube shape once per
        zone - the reason it needed explicit gc.collect() calls.
        '''

        self._diag('Executing spatial aggregation...', self.verbose)

        agg = _zonal.apply_coverage(coverage, self.data, min_coverage=min_coverage)

        # F-order flattens with ensemble_member fastest and zone slowest, which is the
        # order from_product declares below.
        flat = np.reshape(agg, (agg.shape[0], int(np.prod(agg.shape[1:]))), order='F')
        frame = pd.DataFrame(flat, index=np.asarray(self.production_datetime).ravel())
        frame.columns = pd.MultiIndex.from_product(
            (coverage.zones, self.leadtimes,
             np.arange(self.data.shape[self.ENSEMBLEMEMBERpOSITION])),
            names=('zone', 'leadtime', 'ensemble_member'))

        for name in ('leadtime', 'ensemble_member'):
            if frame.columns.get_level_values(name).nunique() == 1:
                self._diag('    Level %s is redundant and will be dropped.' % name, self.verbose)
                frame.columns = frame.columns.droplevel(name)

        frame.index.name = 'Production date'

        self._diag('    Done.', self.verbose)

        return frame


    POOLABLE_AXES = ('space', 'leadtime', 'production', 'members')

    #: ensemble_member label used when the members are pooled into the sample.
    POOLED_MEMBER = -1

    def __time_blocks(self, values, resampling, pooled):
        '''
        Returns (groups, labels) for one time axis.

        Each group is the positional indices pooled into a single distribution.
        When the axis is not pooled every position is its own group, so the axis
        survives in the columns unchanged.
        '''

        count = len(values)
        if not pooled:
            return [[i] for i in range(count)], list(values)

        if resampling is None:
            # The whole axis is one block; the level then drops out as redundant.
            return [list(range(count))], [values[0]]

        index = pd.Index(values)

        if isinstance(index, pd.DatetimeIndex):
            # resample() rather than floor() so calendar rules ('MS', 'W', 'QS')
            # work as well as fixed ones ('1D', '6h').
            grouped = pd.Series(np.arange(count), index=index).resample(resampling)
            blocks = [(label, np.asarray(positions))
                      for label, positions in grouped.indices.items()]
            blocks.sort(key=lambda item: item[0])

            return [list(positions) for _, positions in blocks], [label for label, _ in blocks]

        if not isinstance(index, pd.TimedeltaIndex):
            try:
                index = pd.TimedeltaIndex(index)
            except (TypeError, ValueError) as ex:
                raise TypeError('Cannot pool this axis with resampling=%r: its values are %s, '
                                'not timedeltas. Calendar-month leadtimes (pd.DateOffset) have '
                                'no fixed length, so pool the whole axis with resampling=None '
                                'instead.' % (resampling, index.dtype)) from ex

        keys = index.floor(resampling)
        groups, labels = [], []
        for key in pd.Index(keys).unique().sort_values():
            groups.append(list(np.flatnonzero(np.asarray(keys) == key)))
            labels.append(key)

        return groups, labels

    def __groupByQuantile(self, coverage, resampling, quantiles, pool, min_coverage, method):
        '''
        Area-weighted spatial quantiles per zone, optionally pooling a time axis.
        '''

        self._diag('Executing spatial aggregation...', self.verbose)

        pool = {str(p).lower() for p in ([pool] if isinstance(pool, str) else pool)}
        unknown = pool - set(self.POOLABLE_AXES)
        if unknown:
            raise ValueError('Unknown pool axes: %s. Choose from %s.'
                             % (', '.join(sorted(unknown)), ', '.join(self.POOLABLE_AXES)))
        if 'space' not in pool:
            raise ValueError("pool must include 'space': these are spatial quantiles over the "
                             "cells of each zone.")

        quantiles = np.atleast_1d(np.asarray(quantiles, dtype=float))
        members = np.arange(self.data.shape[self.ENSEMBLEMEMBERpOSITION])

        production_groups, production_labels = self.__time_blocks(
            self.production_datetime, resampling, 'production' in pool)
        leadtime_groups, leadtime_labels = self.__time_blocks(
            self.leadtimes, resampling, 'leadtime' in pool)

        pool_axes = []
        if 'production' in pool:
            pool_axes.append(0)
        if 'members' in pool:
            pool_axes.append(1)
        if 'leadtime' in pool:
            pool_axes.append(2)

        # Kept leading axes, in order. Production and leadtime are singletons here
        # whenever they are not pooled (each group holds one position), so they are
        # squeezed away and their labels come from the block instead.
        kept = [axis for axis in (0, 1, 2) if axis not in pool_axes]
        squeeze = tuple(position for position, axis in enumerate(kept) if axis in (0, 2))
        member_labels = [self.POOLED_MEMBER] if 'members' in pool else list(members)

        frames = []
        for leadtime_group, leadtime_label in zip(leadtime_groups, leadtime_labels):
            block = np.empty((len(production_groups), len(member_labels),
                              quantiles.size, len(coverage)))
            for row, production_group in enumerate(production_groups):
                values = self.data[np.asarray(production_group)][:, :, np.asarray(leadtime_group)]
                computed = _zonal.quantiles_from_coverage(
                    coverage, values, quantiles, pool_axes=pool_axes,
                    min_coverage=min_coverage, method=method)
                if squeeze:
                    computed = np.squeeze(computed, axis=squeeze)
                block[row] = computed.reshape(len(member_labels), quantiles.size, len(coverage))

            # C-order flatten of (member, quantile, zone) varies zone fastest, which
            # is the order from_product below declares. The leadtime level is a
            # size-1 axis inserted between them and does not affect the ordering.
            frame = pd.DataFrame(block.reshape(len(production_groups), -1), index=production_labels)
            frame.columns = pd.MultiIndex.from_product(
                (member_labels, [leadtime_label], quantiles, coverage.zones),
                names=('ensemble_member', 'leadtime', 'quantile', 'zone'))
            frames.append(frame)

        agg = pd.concat(frames, axis=1)
        agg = agg.reorder_levels(['zone', 'leadtime', 'ensemble_member', 'quantile'], axis=1)

        # Drop redundant levels, as __groupByMatrix does, but always keep zone and
        # quantile - they are what the caller asked for.
        for name in ('leadtime', 'ensemble_member'):
            if agg.columns.get_level_values(name).nunique() == 1:
                self._diag('    Level %s is redundant and will be dropped.' % name, self.verbose)
                agg.columns = agg.columns.droplevel(name)

        agg = agg.sort_index(axis=1)
        agg.index.name = 'Production date'

        self._diag('    Done.', self.verbose)

        return agg



    @staticmethod
    def _as_slice(indices):
        '''
        Returns an equivalent slice when `indices` is a contiguous ascending run,
        otherwise the array itself.

        Slicing gives a view where fancy indexing would copy, which is what keeps
        get_cropped down to a single allocation.
        '''

        indices = np.asarray(indices)
        if indices.size and np.array_equal(indices, np.arange(indices[0], indices[-1] + 1)):
            return slice(int(indices[0]), int(indices[-1]) + 1)

        return indices

    @staticmethod
    def _nan_filled(shape, dtype):
        '''
        Allocates a NaN-filled array, keeping dtype when it can hold NaN.

        `np.empty(shape) * np.nan` is float64 whatever the source dtype, so a
        float32 raster was silently doubled in memory - and on disk after the next
        save(), since save() writes whatever dtype it is handed.
        '''

        dtype = np.dtype(dtype)
        if not np.issubdtype(dtype, np.floating):
            dtype = np.result_type(dtype, np.float32)

        return np.full(tuple(shape), np.nan, dtype=dtype)

    @staticmethod
    def _diag(message, show=True):
        if show:
            print(message)

    def resampleTimeStep(self, *args, **kwargs):
        warnings.warn('resampleTimeStep() is deprecated and will be removed in a future release; call resample_timestep() directly instead.', DeprecationWarning, stacklevel=2)
        self.resample_timestep(*args, **kwargs)

    def getQuantilesFromKML(self, *args, **kwargs):
        warnings.warn('getQuantilesFromKML() is deprecated and will be removed in a future release; call get_quantiles_from_KML() directly instead.', DeprecationWarning, stacklevel=2)
        return self.get_quantiles_from_KML(*args, **kwargs)
    
    def getDataFromLatLon(self, *args, **kwargs):
        warnings.warn('getDataFromLatLon() is deprecated and will be removed in a future release; call get_values_from_latlon() directly instead.', DeprecationWarning, stacklevel=2)
        return self.get_values_from_latlon(*args, **kwargs)
    
    def getCropped(self, *args, **kwargs):
        warnings.warn('getCropped() is deprecated and will be removed in a future release; call get_cropped() directly instead.', DeprecationWarning, stacklevel=2)
        return self.get_cropped(*args, **kwargs)

    def getValuesFromKML(self, *args, **kwargs):
        warnings.warn('getValuesFromKML() is deprecated and will be removed in a future release; call get_values_from_KML() directly instead.', DeprecationWarning, stacklevel=2)
        return self.get_values_from_KML(*args, **kwargs)

if __name__=='__main__':
    
    pass
