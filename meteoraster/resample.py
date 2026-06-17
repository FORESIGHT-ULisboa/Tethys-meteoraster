
import math
import numpy as np
import pandas as pd

class Resample(object):
    '''
    Resamples meteorological data for use in hydrological and forecasting models
    handles pandas dataframes [production timestamps x leadtime timedeltas]
    Does not need initialization as the main method (resample) is a class method.
    
    
    v1.01:
        imported to meteoraster from tethys
    '''
    
    VERSION = '1.01'
    
    @classmethod
    def resample(cls, data, timestep_frequency, resamplingType, date_from=None, date_to=None, print_func=None):
        '''
        Main method of the class.
        Takes the following inputs:
            data: pandas dataframe [production timestamps x leadtime timedeltas]
            timestep_frequency: pandas pd.Timedelta or equivalent string (e.g., '1D', '3H', '30min', '30T', etc...).
            resamplingType: 'sum', 'mean', or 'linear. 'sum' and 'mean' are very similar. There is a difference only when the timestep_frequency is larger than the production frequency (1D case) or the leadtime frequency (2D case).
            date_from: first datetime that is displayed (rows)
            date_to: last datetime that is displayed (rows) - when upsampling additional values may be added.
            print_func: a function handle to print the evolution of the calculations
            
            
            ### THE VALIDITY CHECK SHOULD BE REVIEWED
        '''
        
        headers = list(data.columns.names)
        headers.remove('leadtime')
        
        # Default values for the date
        if isinstance(date_from, type(None)):
            date_from = data.index[0]
        if isinstance(date_to, type(None)):
            date_to = data.index[-1]
        
        if isinstance(timestep_frequency, dict):
            timestep_frequency = pd.Timedelta(**timestep_frequency)
        elif isinstance(timestep_frequency, str):
            timestep_frequency = pd.Timedelta(timestep_frequency)
            
        idx_map = None
        resampled = []
        for g0_, mis0_ in data.groupby(headers, axis=1):
            # Loop - separate handling of each header combination (indexes are calculated on the first pass)
            g0_ = [mis0_.columns.get_level_values(h)[0] for h in headers]
            mis0_ = mis0_.droplevel(headers, axis=1)
            
            # Progress message
            if not isinstance(print_func, type(None)):
                print_func(g0_[-1])
            
            # Infer frequencies
                # Production (rows)
            prod_frequency = pd.infer_freq(mis0_.index.values[0:3])
            if not prod_frequency:
                prod_frequency = pd.infer_freq(mis0_.index.values[3:6])
            if not prod_frequency[0].isdigit():
                prod_frequency = '1' + prod_frequency
            prod_frequency = pd.Timedelta(prod_frequency)
            
            date_from_ = date_from - timestep_frequency * math.ceil(prod_frequency / timestep_frequency)
            
                # Leadtime (columns)
            if mis0_.shape[1]>2:
                lead_frequency = pd.infer_freq(mis0_.columns.get_level_values('leadtime'))
                if not lead_frequency[0].isdigit():
                    lead_frequency = '1' + lead_frequency
                lead_frequency = pd.Timedelta(lead_frequency)
            elif mis0_.shape[1]==2:    
                tmp = mis0_.columns.get_level_values('leadtime')
                lead_frequency = tmp[1]-tmp[0]
            else:
                lead_frequency = prod_frequency
            
            # Check frequency ratio
            fr_ratio = prod_frequency/lead_frequency
            if mis0_.shape[1]!=1 and mis0_.shape[1]<fr_ratio:
                raise Exception('Problem with the leadtime [%u (multiple of %u) "%s"] . Does it agree with the production step ["%s"]?' % (mis0_.shape[1], fr_ratio, lead_frequency, prod_frequency))
                
            # Check timestep ratio
            ts_ratio = timestep_frequency/lead_frequency
            if mis0_.shape[1]!=1 and mis0_.shape[1]/ts_ratio!=int(mis0_.shape[1]/ts_ratio):
                additional_mis0_steps = int((int(mis0_.shape[1]/ts_ratio)+1) * ts_ratio)-mis0_.shape[1]
                additional_cols = pd.timedelta_range(start=mis0_.columns[-1], periods=additional_mis0_steps+1, freq=lead_frequency, closed='right')
                additional_mis0 = pd.DataFrame(np.nan, index=mis0_.index, columns=additional_cols)
                mis0_ = pd.concat((mis0_, additional_mis0), axis=1)
                #===============================================================
                # valid_mis0_steps = int(int(mis0_.shape[1]/ts_ratio) * ts_ratio)
                # mis0_ = mis0_.iloc[:, :valid_mis0_steps]
                #===============================================================
                
                print('            Issue with the leadtime [%u (multiple of %u) "%s"]. Does it agree with the time step ["%s"]? Augmented %u steps.' % (mis0_.shape[1], ts_ratio, lead_frequency, prod_frequency, additional_mis0_steps))
                #===============================================================
                # raise Exception('Problem with the leadtime [%u (multiple of %u) "%s"]. Does it agree with the time step ["%s"]?' % (mis0_.shape[1], ts_ratio, lead_frequency, prod_frequency))
                #===============================================================
            
            
            calculation_frequency = min((prod_frequency, pd.Timedelta('1H'), timestep_frequency, lead_frequency))
            if mis0_.shape[1]==1:
                resampled_ = cls._resample_aux_1D(mis0_, prod_frequency, calculation_frequency, timestep_frequency, resamplingType, date_from_, date_to)
            else:
                resampled_, idx_map = cls._resample_aux_2D(mis0_, prod_frequency, lead_frequency, calculation_frequency, timestep_frequency, resamplingType, date_from_, date_to, idx_map)
            
            #===================================================================
            # mis0_.to_clipboard(excel=True)
            # resampled_.reindex(production_range, axis=0).to_clipboard(excel=True)
            #===================================================================
            resampled_.columns = pd.MultiIndex.from_product([[i] for i in g0_] + [resampled_.columns.tolist()], names=headers + ['delay'])
            resampled.append(resampled_)
     
        # Aggregate to the desired time step
        resampled = pd.concat(resampled, axis=1)
        production_range = pd.date_range(start=date_from, end=date_to, freq=timestep_frequency)
    
        return resampled.reindex(production_range, axis=0)

    @classmethod
    def _resample_aux_1D(cls, data, prod_frequency, calculation_frequency, timestep_frequency, resamplingType, date_from, date_to):
        '''
        
        prod_frequency assumed to be compatible with timestep_frequency
        calculation frequency is at least hourly to account for timezone shifts
        '''
        
        # Calculate reference step windows    
        production_calculation_window = int(prod_frequency/calculation_frequency)
        timestep_calculation_window = int(timestep_frequency/calculation_frequency)    
        
        calculation_range = pd.date_range(start=date_from, end=date_to + prod_frequency, freq=calculation_frequency, inclusive='left')
        resampled_ = data.reindex(calculation_range, axis=0)
        
        if resamplingType=='sum' or resamplingType=='mean':
            if production_calculation_window>1:
                resampled_ = resampled_.ffill(limit=production_calculation_window-1) / production_calculation_window # value per calculation step
            resampled_ = resampled_.resample(rule=timestep_frequency, axis=0).sum(min_count=timestep_calculation_window) # sum per time step
            
            production_ratio = int(production_calculation_window / timestep_calculation_window)
            if  production_ratio > 1:
                if production_ratio != production_calculation_window / timestep_calculation_window:
                    raise Exception('Production step not multiple of calculation time step.')
                
                tmp = []
                for i0 in range(production_ratio):
                    tmp.append(resampled_.shift(-i0))
                tmp = pd.concat(tmp, axis=1)
                tmp.columns = pd.timedelta_range('0D', end=prod_frequency, freq=timestep_frequency, name='leadtimes', closed='left')
                resampled_ = tmp.sum(axis=1, min_count=production_ratio).to_frame()
                if resamplingType=='mean':
                    resampled_ /= production_ratio
                resampled_.columns = [pd.Timedelta('0D')]
                
            else:
                if resamplingType=='mean':
                    resampled_ *= production_calculation_window / timestep_calculation_window
        elif resamplingType=='linear':
            if production_calculation_window>1:
                resampled_ = resampled_.interpolate(axis=0, limit=production_calculation_window-1, method='linear', limit_area='inside')
        elif resamplingType=='max':
            if production_calculation_window>1:
                resampled_ = resampled_.ffill(limit=production_calculation_window-1) # value per calculation step
            resampled_ = resampled_.resample(rule=timestep_frequency, axis=0).max(min_count=timestep_calculation_window) # sum per time step
        else:
            raise Exception('Resampling type not implemented: %s.' % resamplingType) 
        
        return resampled_
        
    @classmethod
    def _resample_aux_2D(cls, data, prod_frequency, leadtime_frequency, calculation_frequency, timestep_frequency, resamplingType, date_from, date_to, idx_map):
        '''
        
        '''
        
        # Reindex to calculation delta when required
        if isinstance(idx_map, type(None)):
            idx_map = cls._resample_aux_2D_idx(data, prod_frequency, leadtime_frequency, calculation_frequency, timestep_frequency, resamplingType, date_from, date_to, idx_map)
        
        # Calculate reference step windows    
        leadtime_calculation_window = int(leadtime_frequency/calculation_frequency) 
        timestep_calculation_window = int(timestep_frequency/calculation_frequency)
         
        resampled_ = idx_map.copy()*np.nan
        tmp0 = idx_map.dropna().astype(int).values.ravel()
        tmp1 = np.isfinite(idx_map).astype(bool).values.ravel()
        resampled_.loc[tmp1,:] = data.values.ravel()[tmp0]
        resampled_ = resampled_.unstack('leadtime')
        resampled_ = resampled_.droplevel(0, axis=1)
         
        # Aggregate to desired time step (columns)      
        if timestep_calculation_window>1 or leadtime_calculation_window>1:  
            if resamplingType=='sum' or resamplingType=='mean':
                resampled_ = resampled_.resample(rule=timestep_frequency, axis=1).sum(min_count=timestep_calculation_window) / timestep_calculation_window
                if resamplingType=='sum':
                    resampled_ *= timestep_calculation_window / leadtime_calculation_window
            elif resamplingType=='linear':
                if leadtime_calculation_window>1:
                    resampled_ = resampled_.interpolate(axis=1, limit=leadtime_calculation_window-1, method='linear')
                leadtimes_ = pd.timedelta_range(start=max((pd.Timedelta(hours=0), resampled_.columns[0])), end=resampled_.columns[-1], freq=timestep_frequency)
                resampled_ = resampled_.reindex(leadtimes_, axis=1)
            elif resamplingType=='max':
                resampled_ = resampled_.resample(rule=timestep_frequency, axis=1).max(min_count=timestep_calculation_window)
            else:
                raise Exception('Resampling type not implemented: %s.' % resamplingType)
         
        return (resampled_, idx_map)
    
    @classmethod
    def _resample_aux_2D_idx(cls, data, prod_frequency, leadtime_frequency, calculation_frequency, timestep_frequency, resamplingType, date_from, date_to, idx_map):
        '''
        
        '''
        # Calculate reference step windows    
        production_calculation_window = int(prod_frequency/calculation_frequency)
        leadtime_calculation_window = int(leadtime_frequency/calculation_frequency)  
        
        # Reindex to calculation delta when required
        if isinstance(idx_map, type(None)):
            idx_map = data.copy()
            idx_map.loc[:,:] = np.arange(idx_map.size).reshape(idx_map.shape, order='C')
            idx_map = idx_map
            leadtimes = data.columns
             
            # Handle columns (leadtimes)
            if leadtime_calculation_window>1:
                if resamplingType=='linear':
                    leadtime_index = pd.timedelta_range(start=leadtimes[0] - leadtime_frequency + calculation_frequency, end=leadtimes[-1] + leadtime_frequency, freq=calculation_frequency, closed='left')
                else:
                    leadtime_index = pd.timedelta_range(start=leadtimes[0], end=leadtimes[-1]+leadtime_frequency, freq=calculation_frequency, closed='left')
                if resamplingType=='sum' or resamplingType=='mean' or resamplingType=='max':
                    idx_map = idx_map.reindex(leadtime_index, axis=1).ffill(limit=leadtime_calculation_window-1, axis=1)
                elif resamplingType=='linear':
                    idx_map = idx_map.reindex(leadtime_index, axis=1)
                else:
                    raise Exception('Resampling type not implemented: %s.' % resamplingType)
            idx_map.columns.name = 'leadtime'
             
            # Handle rows (production times)
            if production_calculation_window>1:
                calculation_index = pd.date_range(start=date_from, end=date_to+leadtime_frequency, freq=calculation_frequency, inclusive='left')
                 
                idx_map = idx_map.reindex(calculation_index, axis=0)
                valid = idx_map.iloc[0, :]*np.nan
                shift = 1
                for r0 in range(0, idx_map.shape[0]):
                    if np.isfinite(idx_map.iloc[r0, :]).any():
                        valid = idx_map.iloc[r0, :]
                        shift = 1
                    else:
                        idx_map.iloc[r0, :] = valid.shift(-shift)
                        shift += 1
             
            idx_map = idx_map.stack('leadtime', dropna=False).to_frame()
        
        return idx_map

    @classmethod
    def production_to_event(cls, data, prod_frequency):
        '''
        '''
        
        raise(Exception('Not implemented'))

