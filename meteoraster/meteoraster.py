import math
import copy
import gc
import os
import warnings
import shutil
import tempfile
import numpy as np
import xarray as xr
import datetime as dt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import xml.etree.ElementTree as et
from pathlib import Path
from osgeo import ogr
import scipy.interpolate as interp
import cartopy.crs as ccrs
import cartopy.feature as cf
import cartopy.io.shapereader as shpreader

class MeteoRaster(object):
    '''
    Custom class to handle meteorological raster files, including ensembles and forecasts
    
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
    
    VERSION = '2.3'
    VERBOSE = True
    ENSEMBLEMEMBERpOSITION = 1

    ENGINE = 'h5netcdf'
    
    def __init__(self, data,
                 latitudes=None, longitudes=None, production_datetime=None, leadtimes=None,
                 units='unknown', variable='unknown', da_attrs={}, ds_attrs={},
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
        self.da_attrs = da_attrs
        self.ds_attrs = ds_attrs
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
    
            # Flip latitudes (if required)
        if self.latitudes[0, 0]<self.latitudes[1, 0]:
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
    
    def create_plot(self, central_longitude, projection_kwargs={}, figsize=None, projection_fun=ccrs.PlateCarree, crs=ccrs.PlateCarree(), extent=None):
        '''
        extent [west, east, south, north]
        '''
        
        plt.figure(figsize=figsize)
        projection = projection_fun(central_longitude=central_longitude, **projection_kwargs)
        ax = plt.axes(projection=projection)
        if not isinstance(extent, type(None)):
            ax.set_extent(extent, crs=crs)
        return ax

    def add_shapefile(self, ax, shapefile_path, crs=ccrs.PlateCarree(), **kwargs):
        '''
        
        '''

        ax.add_geometries(shpreader.Reader(shapefile_path).geometries(), crs=crs, **kwargs)

    def add_shape(self, ax, path, color='black', facecolor='none', linewidth=1, crs=ccrs.PlateCarree()):
        '''
        
        '''
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

        sns.color_palette('viridis')
    
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

        tmp = data.rolling(window=window, center=True, axis=0).mean().dropna().stack().to_frame(name='values').reset_index()
        
        # Use proper column name from index
        idx_name = data.index.name if data.index.name else 'index'
        if idx_name in tmp.columns:
             tmp.rename(columns={idx_name: 'Production dates'}, inplace=True)

        tmp.loc[:, 'Day of year'] = tmp.loc[:, 'Production dates'].dt.day_of_year

        if isinstance(ax, type(None)):
            plt.figure(figsize=(20,10))
        ax = sns.lineplot(x='Day of year', y='values', data=tmp, errorbar=('pi', 95), ax=ax, *args, **kwargs)
        
        return ax

    def get_values_from_KML(self, kml, nameField=None, coverage_info=None, getCoverageInfo=False, elementwise=False):
        '''
        Returns a pandas table with averaged values        
        '''
        
        if isinstance(coverage_info, type(None)):
            cvr, columns, centroids = self.__coverageMatrixFromKML(kml, nameField)
            coverage_info = (cvr, columns, centroids)
        else:
            cvr, columns, centroids = coverage_info
        
        agg = self.__groupByMatrix(cvr, columns, elementwise=elementwise)
        
        if getCoverageInfo:
            return agg, centroids, coverage_info
        else:
            return agg, centroids

    def get_quantiles_from_KML(self, kml, nameField=None, coverage_info=None, getCoverageInfo=False, resampling=None, precision=0.1, quantiles=[0.01, 0.1, 0.5, 0.9, 0.99]):
        '''
        Returns a pandas table with averaged values      
        '''
        
        if isinstance(coverage_info, type(None)):
            cvr, columns, centroids = self.__coverageMatrixFromKML(kml, nameField, normalize_by_area=False)
            coverage_info = (cvr, columns, centroids)
        else:
            cvr, columns, centroids = coverage_info
        
        agg = self.__groupByQuantile(cvr, columns, resampling, quantiles, precision)
        
        if getCoverageInfo:
            return agg, centroids, coverage_info
        else:
            return agg, centroids

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
        data = self.data[date_idxs, :, :, :, :]
        data = data[:, :, :, lat_idxs, :]
        data = data[:, :, :, :, lon_idxs]
        
        cropped = copy.deepcopy(self)
        cropped.production_datetime = production_datetime
        cropped.latitudes = latitudes
        cropped.longitudes = longitudes
        
        if from_leadtime:
            idxs = cropped.leadtimes>=from_leadtime
            cropped.leadtimes = cropped.leadtimes[idxs]
            data = data[:, :, idxs, :, :]
        
        if to_leadtime:
            idxs = cropped.leadtimes<=to_leadtime
            cropped.leadtimes = cropped.leadtimes[idxs]
            data = data[:, :, idxs, :, :]
                    
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
            data = np.empty([shape[0], e0] + list(shape[2:])) * np.NaN
            data[:, :e1, :, :, :] = newData
            newData = data
        elif e0<e1:
            if strickt:
                raise(Exception(errorMessage))
            else:
                warnings.warn(warningMessage)
            shape = self.data.shape
            data = np.empty([shape[0], e1] + list(shape[2:])) * np.NaN
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
        
        periodDefinition = {
            'months': pd.Series([pd.DateOffset(months=i) for i in range(1, math.ceil(pd.Series(self.leadtimes[-1]).dt.days.values[0]/30)+1)]),
            }
        
        if period not in periodDefinition.keys():
            raise(Exception('%s is not a valid period.' % period))
        
        warnings.warn('the function adjustLeadTimes() is experimental.')
    
        times = self.production_datetime[0] + self.leadtimes
        validTimes = pd.Timestamp(self.production_datetime[0]) + periodDefinition[period]
    
        idxs = self.closestIdx(times, validTimes)
        validPeriods =  periodDefinition[period][:idxs.max()+1]
        
        tmp = list(self.data.shape)
        data = np.empty(tmp[:2] + [validPeriods.size] + tmp[-2:]) * np.NaN
        
        for i0 in idxs:
            dataForLeadtime = self.data[:, :, idxs==i0, :, :]
            finite = np.isfinite(dataForLeadtime)
            count = np.sum(finite, axis=2)
            if count.max()>1:
                warnings.warn('Overlapping information detected when adjusting lead times (%s).' % str(validPeriods[i0]))
            
            tmp = np.nanmean(dataForLeadtime, axis=2)
            tmp[count==0] = np.NaN
    
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
        '''
        
        missingFraction = np.isnan(self.data).mean(axis=(-2, -1))
        missingFraction = pd.DataFrame(missingFraction.reshape((missingFraction.shape[0], np.prod(missingFraction.shape[1:]))), index=self.production_datetime)
        missingFraction.index.name = 'production_datetime'
        missingFraction.columns = pd.MultiIndex.from_product((self.leadtimes, np.arange(self.data.shape[1])), names=['leadtime', 'ensemble_member'])
        
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
            tmp = available.groupby(axis=0, level=0).sum().transpose().stack().to_frame()
            tmp.index = tmp.index.set_levels(self.leadtimes, level=1)

        
        tmp.columns = ['available']
        tmp = tmp.loc[tmp['available']>0,:].reset_index()
        tmp.loc[:, 'event_dates'] = tmp.loc[:, 'production_datetime'] + tmp.loc[:, 'leadtime']
        
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
        
        resampled = np.empty([idxMax.shape[0]] + list(self.data.shape[1:])) * np.NaN
        idx0 = 0
        for i0, i1 in enumerate(idxMax.values.ravel()):
            resampled[i0, :, :, :, :] = fun(self.data[idx0:i1, :, :, :, :], axis=0, keepdims=True)
            idx0 = i1
    
        self.data = resampled
        self.production_datetime = idxMax.index
    
        self._diag('        Done (%s to %s).' % (self.production_datetime[0].strftime('%Y-%m-%d %H:%M:%S', self.verbose),
                                                          self.production_datetime[-1].strftime('%Y-%m-%d %H:%M:%S')))
    
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

        finite = np.isfinite(self.data)

        # Ensemble requirement
        if full_ensemble:
            finite = finite.all(axis=1)  # -> [prod, leadtime, y, x]
        else:
            finite = finite.any(axis=1)  # -> [prod, leadtime, y, x]

        # Spatial requirement
        if space_completeness:
            finite = finite.all(axis=(-2, -1))  # -> [prod, leadtime]
        else:
            finite = finite.any(axis=(-2, -1))  # -> [prod, leadtime]

        # Temporal requirement: every production_datetime and every leadtime must be complete
        return finite.all()
    
    def get_complete_index(self, full_ensemble:bool=True, space_completeness:bool=False) -> pd.DataFrame:
        '''
        
        '''

        self._diag('Retrieving complete index...', self.verbose)

        finite = np.isfinite(self.data)

        # Ensemble requirement
        if full_ensemble:
            finite = finite.all(axis=1)  # -> [production_datetime, leadtime, y, x]
        else:
            finite = finite.any(axis=1)  # -> [production_datetime, leadtime, y, x]

        # Spatial requirement
        if space_completeness:
            finite = finite.all(axis=(-2, -1))  # -> [production_datetime, leadtime] or [production_datetime, ensemble_member, leadtime]
        else:
            finite = finite.any(axis=(-2, -1))  # -> [production_datetime, leadtime] or [production_datetime, ensemble_member, leadtime]

        completeness_index = pd.DataFrame(finite, index=self.production_datetime, columns=self.leadtimes)
        completeness_index.index.name = 'production_datetime'
        completeness_index.columns.name = 'leadtime'

        return completeness_index

    @classmethod
    def get_completeness(cls, file) -> bool:
        '''
        Docstring for check_completeness
        
        :param file: Description
        :return: Description
        :rtype: bool
        '''

        try:
            with xr.open_dataset(Path(file), decode_times=False, engine=cls.ENGINE) as ds:
                return ds.attrs['complete'].lower() in ['true', 'yes']
        except Exception as ex:
            raise(ex)
    
    def save(self, file, complevel=1, complete=None) -> None:
        '''
        Saves a netcdf file (.nc)
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
        if complete is None:
            ds.attrs['complete'] = str(self.is_complete())
        else:
            ds.attrs['complete'] = str(complete)

        tmp_file = None
        try:
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp_file = Path(tmp.name)

            if complevel>0:
                encoding = {
                    self.variable: {"zlib": True, "complevel": complevel},
                }
                ds.to_netcdf(tmp_file, encoding=encoding, engine=self.ENGINE)
            else:
                ds.to_netcdf(tmp_file, engine=self.ENGINE)

            shutil.move(tmp_file, file)
        except Exception:
            if tmp_file and tmp_file.exists():
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

        return data_array
    
    @classmethod
    def load(cls, file, verbose=None) -> "MeteoRaster":
        '''
        Loads a MeteoRaster from a NetCDF (.nc / .nct) file written by save().
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

        with xr.open_dataset(file, engine=cls.ENGINE) as ds:
            # pick the first data variable if name is unknown
            var_name = cls.__detect_variable_name(ds)
            da = ds[var_name]

            units = da.attrs.get('units', 'unknown')
            variable = da.attrs.get('variable', 'unknown')
            da_attrs = ds.attrs
            ds_attrs = ds.attrs

            leadtimes = ds['leadtime'].values
            if hasattr(ds['leadtime'], 'leadtime_format'):
                leadtime_format = str(ds['leadtime'].leadtime_format.values)
                if leadtime_format.split(':')[0]=='pd.DateOffset':
                    leadtimes = pd.Index([pd.DateOffset(**{leadtime_format.split(':')[1]: lt}) for lt in leadtimes])

            meteo = cls(
                data=da.values,
                latitudes=ds['lat'].values,
                longitudes=ds['lon'].values,
                production_datetime=ds['production_datetime'].values,
                leadtimes=leadtimes,
                units=units,
                variable=variable,
                da_attrs=da_attrs,
                ds_attrs=ds_attrs,
                **kwargs,
            )

        cls._diag('    Loaded.', verbose)

        return meteo

    @staticmethod
    def __detect_variable_name(ds: xr.Dataset) -> str:
        # Require single-variable datasets written by save()
        if len(ds.data_vars) != 1:
            raise ValueError('NetCDF file must contain exactly one data variable written by save().')
        return next(iter(ds.data_vars))
    
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
        
        westData = self.data[:, :, : ,: ,westIndexes]
        eastData = self.data[:, :, : ,: ,eastIndexes]
        
        data = np.concatenate((westData, eastData), axis=-1)
        
        self.longitudes = longitudes
        self.data = data
        
        self._diag('Fixed start at Greenwich.', self.verbose)
    
    def __groupByMatrix(self, coverage, columns, elementwise=False):
        '''
        Groups the data by a coverage matrix (see groupByKML for matrix definition)
        '''
        
        self._diag('Executing spatial aggregation...', self.verbose)

        #Verification of valid tiles (assumes first coverage zone to be representative (for speed and memory)
        val_ = np.isfinite(self.data).astype(bool)
        cvr = np.tile(coverage[:, :, :, :, :, 0].astype(np.single), list(val_.shape[:(val_.ndim-2)]) + [1, 1])
        val = np.nansum(cvr*val_, axis=(-2, -1))>0.8
        val = np.tile(np.expand_dims(val,-1), [1]*val.ndim + [coverage.shape[-1]])

        if elementwise:
            agg = np.empty(list(self.data.shape[:3]) + [len(columns)]).astype(np.single) * np.NaN
            
            data_single = self.data.astype(np.single)
            coverage_single = coverage.astype(np.single)
            
            for i0 in range(len(columns)):
                agg[:, :, :, i0] = np.nansum(data_single * np.tile(coverage_single[:, :, :, :, :, i0], list(data_single.shape[:(self.data.ndim-2)]) + [1, 1]), axis=(-2, -1))
                self._diag('    % 5.1f%%' % ((i0+1)/len(columns)*100), self.verbose)
        else:
            #Creation of matrices for calculation
            dat = np.tile(np.expand_dims(self.data.astype(np.single), axis=-1), [1]*self.data.ndim + [coverage.shape[-1]])
            cvr = np.tile(coverage.astype(np.single), list(dat.shape[:(dat.ndim-3)]) + [1, 1, 1])
            
            #Correction of the coveragefor missing data
            dat_finite = np.isfinite(dat).astype(bool)
            cvr_finite = cvr.copy()
            cvr_finite[~dat_finite] = 0
            weight_update = 1 / cvr_finite.sum(axis=(-3, -2))
            weight_update = np.tile(weight_update[:, :, :, np.newaxis, np.newaxis, :], [1, 1, 1] + list(cvr.shape[-3:-1]) + [1])

            dat *= cvr 
            dat *= weight_update 

            cvr = None
            gc.collect()
            
            #Calculation with garbage collection
            agg = np.nansum(dat, axis=(-3, -2))
            dat = None
            gc.collect()
            
        #Mark invalid tiles as NaN
        agg[~val] = np.NaN 
        
        tmp = np.reshape(agg, (agg.shape[0], np.prod(agg.shape[1:])), order='F')
        agg = pd.DataFrame(tmp, index=self.production_datetime.ravel())
        agg.columns = columns
        agg.index.name = 'Production date'

        self._diag('    Done.', self.verbose)

        return agg  
    
    def __groupByQuantile(self, coverage, columns, resampling, quantiles, precision=0.1):
        '''
        Groups the data by quantiles in time and space
        
        precision should be between 0 and 1.
        computation time will be proportional to 1/precision
        '''
        
        self._diag('Executing spatial aggregation...', self.verbose)

        if not resampling is None:
            resampled_idx = pd.DataFrame(self.production_datetime, self.production_datetime).resample(resampling).indices
            time_idx = list(resampled_idx.keys())
            time_groups = [i for _, i in resampled_idx.items()]
        else:
            time_idx = self.production_datetime
            time_groups = [[i] for i, _ in enumerate(time_idx)]
                    
        #Creation of matrices for calculation
        dat = np.tile(np.expand_dims(self.data.astype(np.single), axis=-1), [1]*self.data.ndim + [coverage.shape[-1]])
        coverage_ = coverage.astype(np.single)
        coverage_[coverage_==0] = np.nan
        gc.collect()
        
        #Calculation with garbage collection
        agg_ = []
        for g0 in time_groups:
            data = dat[g0,:, :, :, :].copy()
            
            # Verification of valid tiles (assumes first coverage zone to be representative (for speed and memory)
            cvr = np.tile(coverage_, list(data.shape[:(dat.ndim-3)]) + [1, 1, 1])
            val = cvr.copy()
            val[val<=0] = np.nan
            val[val>0] = 1

            data_ = data * val
            
            folds = np.round(cvr/precision)
            n_folds = int(np.nanmax(folds))
            data_unfolded = np.tile(np.expand_dims(data, 0), [n_folds] + [1] * data.ndim) * np.nan
            tmp = data_.copy()
            for i1 in range(n_folds):
                tmp[folds<i1+1] = np.nan
                data_unfolded[i1,:,:,:,:,:,:] = tmp 
           
            # Check what pixels are valid (more than 80% values available)
            val = np.nansum(np.isfinite(data_), axis=(0, -3, -2)) / np.nansum(np.isfinite(val), axis=(0, -3, -2)) > 0.8
            val = np.tile(np.expand_dims(val, (0, -1)), [1]*(val.ndim + 1) + [len(quantiles)])
            
            # Calculate the quantiles
            raise(Exception('This must be reviewed.'))
            tmp = np.rollaxis(np.nanquantile(np.expand_dims(data_unfolded, 0), quantiles, axis=(1, 2, -3, -2)), 0, len(quantiles)) #[time, lead, ensemble, zones, quantiles]
            tmp[~val] = np.nan 
            agg_.append(tmp)
    
        agg__ = np.r_[agg_] 
                
        tmp = np.reshape(agg__, (agg__.shape[0], np.prod(agg__.shape[1:])), order='F')
        agg = pd.DataFrame(tmp, index=time_idx)
        
        tmp = pd.concat([columns.to_frame()]*len(quantiles))
        tmp.loc[:, 'Quantile'] = np.array([[q]*np.product(agg__.shape[1:-1]) for q in quantiles]).ravel()
        agg.columns = pd.MultiIndex.from_frame(tmp)
        agg.index.name = 'Production date'
        agg.sort_index(axis=1, inplace=True)

        #=======================================================================
        # agg.to_clipboard(excel=True)
        #=======================================================================

        self._diag('    Done.', self.verbose)

        return agg  
    
    def __coverageMatrixFromKML(self, kml, nameField=None, buffer=2, normalize_by_area=True):
        '''
        Creates a coverage matrix from polygons contained in a kml file
        '''
        
        self._diag('Creating a coverage matrix for spatial aggregation...', self.verbose)
        
        # Correct kml file if required
        self.__correctKMLHeader(kml) 
        
        # Read geometry reference KML file
        ids = []
        coordinates = []
        document = et.parse(kml).getroot().find('Document')
        for i0, placemark in enumerate(document.findall('.//Placemark')):
            #https://docs.python.org/3/library/xml.etree.elementtree.html
            try:
                if not isinstance(nameField, type(None)):
                    id0 = placemark.find(".//SimpleData[@name='%s']" % nameField).text
                else:
                    id0 = str(i0)
                outer = [pd.Series(c.text.replace(' \n', '').replace('\n', '').replace('\t', '').split(' ')).str.split(',', n=2, expand=True) for c in placemark.findall('.//Polygon/outerBoundaryIs//coordinates')]
                outer = [c.astype(float) for c in outer]
                inner = [pd.Series(c.text.replace(' \n', '').replace('\n', '').replace('\t', '').split(' ')).str.split(',', n=2, expand=True) for c in placemark.findall('.//Polygon/innerBoundaryIs//coordinates')]
                inner = [c.astype(float) for c in inner]
                ids.append(id0)
                coordinates.append({'outer': outer, 'inner': inner})
            except Exception as ex:
                print(str(ex)) 
        
        cvr = np.zeros([1, 1, 1] + list(self.data.shape[3:]) + [len(ids)], dtype=np.single) # the last coordinate of the cvr represents the polygon
        
        longitudes = self.longitudes
        latitudes = self.latitudes
        idx_lat_ = np.tile(np.expand_dims(np.arange(longitudes.shape[0]),axis=1),(1, longitudes.shape[1])).ravel() 
        idx_lon_ = np.tile(np.arange(longitudes.shape[1]),(longitudes.shape[0], 1)).ravel()
        lonInterpolator = interp.LinearNDInterpolator(np.array([idx_lat_, idx_lon_]).T, longitudes.ravel()) # f(lat_id, lon_id)
        latInterpolator = interp.LinearNDInterpolator(np.array([idx_lat_, idx_lon_]).T, latitudes.ravel()) # f(lat_id, lon_id)
        
        # Create GDAL geometry
        centroids = {'x':[], 'y':[]}
        for i0, coords in enumerate(coordinates):
            try:
                # Add all "positive" rings
                poly = ogr.Geometry(ogr.wkbPolygon)
                for o1 in coords['outer']:
                    for r0 in o1:
                        polyTmp = ogr.Geometry(ogr.wkbPolygon)
                        ring = ogr.Geometry(ogr.wkbLinearRing)
                        for row in o1.iterrows():
                            ring.AddPoint(row[1][0], row[1][1])
                        _ = polyTmp.AddGeometry(ring)
                        poly = poly.Union(polyTmp)
                
                # Join all "negative rings
                polyHoles = ogr.Geometry(ogr.wkbPolygon)
                for o1 in coords['inner']:
                    for r0 in o1:
                        polyTmp = ogr.Geometry(ogr.wkbPolygon)
                        ring = ogr.Geometry(ogr.wkbLinearRing)
                        for row in o1.iterrows():
                            ring.AddPoint(row[1][0], row[1][1])
                        _ = polyTmp.AddGeometry(ring)
                        polyHoles = polyHoles.Union(polyTmp)
                    
                # Obtain the final polygon
                poly = poly.SymmetricDifference(polyHoles)
    
                # Select pixel subset
                #===============================================================
                # plt.figure(); plt.imshow(longitudes); plt.colorbar()
                # plt.figure(); plt.imshow(latitudes); plt.colorbar()
                # plt.figure(); plt.scatter(longitudes.ravel(), latitudes.ravel(),c=self.data.squeeze().mean(axis=0).ravel()); plt.colorbar()
                # plt.figure(); plt.scatter(longitudes.ravel(), latitudes.ravel(),c=longitudes.ravel()); plt.colorbar()
                # plt.figure(); plt.scatter(longitudes.ravel(), latitudes.ravel(),c=latitudes.ravel()); plt.colorbar()
                #===============================================================
                
                extent = poly.Buffer(buffer).GetEnvelope() #wesn
                inside = np.zeros_like(latitudes).astype(bool)
                inside[(longitudes>=extent[0]) & (longitudes<=extent[1]) & (latitudes>=extent[2]) & (latitudes<=extent[3])] = True
                tmp = np.where(inside)
                lat_idxs = np.arange(tmp[0].min()-1, tmp[0].max()+1)
                lat_idxs = lat_idxs[(lat_idxs>0) & (lat_idxs<latitudes.shape[0]-1)]
                lon_idxs = np.arange(tmp[1].min()-1, tmp[1].max()+1)
                lon_idxs = lon_idxs[(lon_idxs>0) & (lon_idxs<longitudes.shape[1]-1)]
                
                #===============================================================
                # plt.figure(); plt.imshow(inside); plt.colorbar()
                # plt.figure(); plt.scatter(longitudes.ravel(), latitudes.ravel(),c=inside.ravel()); plt.colorbar()
                #===============================================================
                
                #===============================================================
                # extent = [extent[0]-hdLon*2, extent[1]+hdLon*2, extent[2]-hdLat*2, extent[3]+hdLat*2]
                # lon_idxs = np.arange(self.longitudes.shape[0])[(self.longitudes>=extent[0]) & (self.longitudes<=extent[1])]
                # lat_idxs = np.arange(self.latitudes.shape[0])[(self.latitudes>=extent[2]) & (self.latitudes<=extent[3])]
                #===============================================================
    
                x, y, _ = poly.Centroid().GetPoint()
                centroids['x'].append(x)
                centroids['y'].append(y)
    
                for i2 in lon_idxs:
                    for i3 in lat_idxs:
                        #=======================================================
                        # lon = self.longitudes[i3, i2]
                        # lat = self.latitudes[i3, i2]
                        # plt.plot(lon, lat, 'rx')
                        # plt.plot(lonInterpolator(i3-0.5,i2+0.5), latInterpolator(i3-0.5,i2+0.5), '.m')
                        #=======================================================
                        
                        points = []
                        points.append((float(lonInterpolator(i3-0.5, i2+0.5)), float(latInterpolator(i3-0.5, i2+0.5))))
                        points.append((float(lonInterpolator(i3+0.5, i2+0.5)), float(latInterpolator(i3+0.5, i2+0.5))))
                        points.append((float(lonInterpolator(i3+0.5, i2-0.5)), float(latInterpolator(i3+0.5, i2-0.5))))
                        points.append((float(lonInterpolator(i3-0.5, i2-0.5)), float(latInterpolator(i3-0.5, i2-0.5))))
                        points.append((float(lonInterpolator(i3-0.5, i2+0.5)), float(latInterpolator(i3-0.5, i2+0.5))))
                        
                        ring = ogr.Geometry(ogr.wkbLinearRing)
                        for p in points:
                            ring.AddPoint(*p)
                            #===================================================
                            # plt.plot(*p,'.b')
                            #===================================================
                        
                        pixel = ogr.Geometry(ogr.wkbPolygon)
                        _ = pixel.AddGeometry(ring)
    
                        intersection = pixel.Intersection(poly)
                        area = intersection.Area()
    
                        if area>0:
                            cvr[0, 0, 0, i3, i2, i0] = area
            except Exception as ex:
                raise(ex)

        # Define columns (drops redundant levels (except zone))
        columns = pd.MultiIndex.from_product((ids, self.leadtimes, 1+np.arange(self.data.shape[self.ENSEMBLEMEMBERpOSITION])), names=('zone', 'leadtime', 'ensemble_member'))
        drop = []
        for i0, (n0, l0) in enumerate(zip(columns.names, list(columns.levels))):
            if l0.shape[0]==1 and n0!='Zone':
                self._diag('    Level %s is redundant and will be dropped.' % n0, self.verbose)
                drop.append(i0)
        for i0 in drop[::-1]:
            columns = columns.droplevel(i0)
                
        # Normalize
        if normalize_by_area:
            areas = cvr.sum(axis=(0, 1, 2, 3, 4))
            cvr /= np.tile(areas, list(cvr.shape[:-1]) + [1])
        else:
            max_pixel_area = cvr.max(axis=tuple(np.arange(cvr.ndim-1)))
            cvr /= np.tile(max_pixel_area, list(cvr.shape[:-1]) + [1])
        
        # Prepare centroids for export
        centroids = pd.DataFrame(centroids, index=ids)
        
        self._diag('    Done.', self.verbose)
        
        return cvr, columns, centroids
        
    def __correctKMLHeader(self, kml, encoding='utf-8'):
        '''
        Corrects the header of a KML file to that it can be read for model data preparation
        '''
        
        modificationRequired = False
        line = ''
        with open(kml, 'r', encoding=encoding) as file:
            for line in file:
                if line.strip().startswith('<kml'):
                    if line.strip() != '<kml>':
                        modificationRequired = True
                    break
                
        if modificationRequired:
            if not isinstance(kml, str):
                kml = str(kml)
                
            tmpKML = kml.replace('.kml', '__tmp.kml')
            with open(kml, 'r', encoding=encoding) as file_in:
                with open(tmpKML, 'w', encoding=encoding) as file_out:
                    for line in file_in:
                        if line.strip().startswith('<kml'):
                            if line.strip() != '<kml>':
                                line = '<kml>\n'
                        __ = file_out.write(line)
            os.remove(kml)
            os.rename(tmpKML, kml)

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
