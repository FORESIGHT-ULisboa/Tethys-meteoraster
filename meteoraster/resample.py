
import math
import numpy as np
import pandas as pd

class Resample(object):
    '''
    Resamples meteorological data for use in hydrological and forecasting models.
    Handles pandas dataframes [production timestamps x leadtime timedeltas].
    Does not need initialization as the main method (resample) is a class method.


    DATA MODEL
    ----------
    Columns carry a 'leadtime' level plus any number of header levels (series,
    ensemble member, grid cell, ...). Every value V[production, leadtime] describes
    the window [production + leadtime, production + leadtime + leadtime_step) - the
    stamp OPENS its interval. Sources that stamp accumulations at the END of their
    interval (e.g. ERA5) must be shifted by the caller before resampling.

    Frames with a single leadtime column are plain time series: they are aggregated
    ALONG production time. Frames with several leadtimes are forecasts: production
    times are never mixed with each other (distinct forecasts stay distinct), only
    leadtimes are aggregated, and a production timestamp that falls between two
    issues reuses the latest available forecast with its leadtimes shifted.

    Leadtimes need NOT be evenly spaced - operational forecasts commonly go hourly,
    then 3-hourly, then 6-hourly. Each column spans the distance to the next one, and
    the last column reuses the preceding span, which is what fixes the horizon. A
    single forecast (one production row) is therefore enough; only the time-series
    path needs two, since it has nothing else to infer a production step from.


    RESAMPLING TYPES, AND WHAT THEY DO TO THE UNITS
    -----------------------------------------------
    type      quantity                        up-sample      down-sample     units
    --------  ------------------------------  -------------  --------------  -----------------
    'sum'     extensive: amount accumulated   split total    add             value rescales
              over the step (mm, J/m2)        over sub-steps                 with the step
    'mean'    intensive: interval average     replicate      time-weighted   preserved
              (degC, m/s, W/m2, mm/h)                        average
    'linear'  intensive: instantaneous value  interpolate    sample at the   preserved
              at the stamp                    linearly       stamp
    'max'     intensive: extreme              replicate      maximum         preserved

    Only 'sum' changes what a number means: the physical unit is unchanged (mm stays
    mm) but the accumulation period is not (mm/3h vs mm/day), so the value scales
    with the step. The other three are step-invariant - a rate in mm/h resampled with
    'mean' is still mm/h. The same physical series therefore needs 'sum' when stored
    as an accumulation per step and 'mean' when stored as a rate.

    'sum' conserves mass in both directions: the total over any period is preserved.

    Any missing input inside an output window makes that output value NaN.


    v2.1:
        Unevenly spaced leadtimes are supported: the forecast kernel finds a sub-step's
        source column by searchsorted over the leadtimes and weights it by that
        column's own span, instead of dividing by one scalar step. 'sum' therefore
        still conserves mass when a 1 h column and a 6 h column land in the same output
        window, and 'linear' interpolates over the local span. Uniform leadtimes are
        bit-for-bit unchanged.
        A forecast with a single production row is accepted; it used to raise.
    v2.0:
        Rewritten around exact integer-nanosecond index arithmetic.
        - Single-leadtime (time series) up-sampling now matches the forecast path:
          'sum' splits the total across the sub-steps (it used to emit a sliding
          total over the whole production step, which quadrupled a 1D->6H total)
          and 'mean' replicates the average (it used to divide it by the ratio).
          Down-sampling results are unchanged.
        - The internal grid is the exact gcd of the production, leadtime and target
          steps instead of min(..., 1H, ...): non-integer ratios are no longer
          truncated, and all-daily data no longer inflates to an hourly grid.
        - 'linear' no longer extrapolates flat past the forecast horizon.
        - Frequencies are inferred from the index spacing, so 2-row frames and
          unsorted leadtime columns work, and calendar steps ('MS', 'ME', ...) raise
          instead of being read as milliseconds.
        - Input column order is preserved; pandas >= 2.1 deprecations removed.
    v1.01:
        imported to meteoraster from tethys
    '''

    VERSION = '2.1'

    TYPES = ('sum', 'mean', 'linear', 'max')

    # Calendar aliases that pd.Timedelta silently misreads ('MS' parses as 1 millisecond).
    _CALENDAR_ALIASES = ('M', 'MS', 'ME', 'SM', 'SMS', 'BM', 'BMS', 'CBM', 'CBMS',
                         'Q', 'QS', 'QE', 'BQ', 'BQS', 'Y', 'YS', 'YE', 'BY', 'BYS',
                         'A', 'AS', 'BA', 'BAS', 'B', 'C')

    # Guards against a fine grid exploding when a frequency is mis-specified.
    MAX_SUBSTEPS = 100000
    MAX_OUTPUT_CELLS = 500000000

    _HOUR_NS = 3600 * 10 ** 9

    @classmethod
    def resample(cls, data, timestep_frequency, resamplingType, date_from=None, date_to=None, print_func=None):
        '''
        Main method of the class.
        Takes the following inputs:
            data: pandas dataframe [production timestamps x leadtime timedeltas].
                Columns must carry a 'leadtime' level; any other levels are headers
                and are resampled independently.
            timestep_frequency: pandas pd.Timedelta or equivalent string (e.g., '1D',
                '3H', '30min'). Calendar steps ('MS', 'ME', 'YS', ...) are rejected.
            resamplingType: 'sum', 'mean', 'linear' or 'max'. See the class docstring
                for what each one means and how it affects the units.
            date_from: first output timestamp (rows). Defaults to data.index[0].
            date_to: last output timestamp (rows). Defaults to data.index[-1].
            print_func: a function handle to print the evolution of the calculations

        Returns a dataframe whose rows run from date_from to date_to in exact
        timestep_frequency increments; the leadtime level is renamed 'delay' on output.
        '''

        if resamplingType not in cls.TYPES:
            raise Exception('Resampling type not implemented: %s (expected one of %s).'
                            % (resamplingType, ', '.join(cls.TYPES)))

        timestep = cls._as_timedelta(timestep_frequency, 'timestep_frequency')

        names = list(data.columns.names)
        if 'leadtime' not in names:
            raise Exception("The column index must carry a 'leadtime' level; got %s." % (names,))
        headers = [n for n in names if n != 'leadtime']

        if isinstance(date_from, type(None)):
            date_from = data.index[0]
        if isinstance(date_to, type(None)):
            date_to = data.index[-1]

        if not data.index.is_monotonic_increasing:
            data = data.sort_index(axis=0)

        # Time axes, as int64 nanoseconds (tz-aware indexes go through UTC, so a DST
        # jump is just an irregular spacing rather than a special case).
        prod_ns = cls._to_ns(data.index, 'production index')

        keys, col_pos, leadtime_ns = cls._column_layout(data, headers)
        n_lead = leadtime_ns.size

        # A single forecast is resamplable - only leadtimes are aggregated - so the
        # production step is only *needed* for the single-leadtime series path, which
        # is aggregated along production time and has nothing else to infer it from.
        prod_step, prod_diffs = cls._production_step(data.index, prod_ns,
                                                    allow_single=n_lead > 1)

        if n_lead > 1:
            lead_spans = cls._leadtime_spans(leadtime_ns)
            horizon = int(leadtime_ns[-1] + lead_spans[-1] - leadtime_ns[0])
            if prod_step is None:
                prod_step = horizon
        else:
            lead_spans = np.array([prod_step], dtype='int64')
            horizon = prod_step

        timestep_ns = int(timestep.value)
        calculation_ns = math.gcd(*(set(int(d) for d in prod_diffs)
                                    | set(int(s) for s in lead_spans)
                                    | {timestep_ns}))

        if timestep_ns // calculation_ns > cls.MAX_SUBSTEPS:
            raise Exception('The time step (%s) needs %u sub-steps of %s. Check the declared frequencies.'
                            % (timestep, timestep_ns // calculation_ns, pd.Timedelta(calculation_ns)))

        if timestep_ns < prod_step and horizon < prod_step:
            raise Exception('Problem with the leadtimes ["%s" to "%s", spanning "%s"]. They do not span the '
                            'production step ["%s"], so a "%s" output cannot be covered.'
                            % (pd.Timedelta(int(leadtime_ns[0])), pd.Timedelta(int(leadtime_ns[-1])),
                               pd.Timedelta(horizon), pd.Timedelta(prod_step), timestep))

        out_index, out_ns = cls._output_index(data.index, date_from, date_to, timestep_ns)

        if out_index.size * len(keys) * max(1, n_lead) > cls.MAX_OUTPUT_CELLS:
            raise Exception('The requested output (%u rows x %u headers) is too large. Narrow date_from/date_to.'
                            % (out_index.size, len(keys)))

        if not isinstance(print_func, type(None)):
            for key in keys:
                label = cls._as_tuple(key)
                print_func(label[-1] if label else 'leadtime')

        values = data.to_numpy(dtype=float)[:, col_pos.ravel()]
        # (production, leadtime, header): the gather below indexes the first two axes
        # and carries every header along for free.
        values = values.reshape(prod_ns.size, len(keys), n_lead).transpose(0, 2, 1)
        values = np.ascontiguousarray(values)

        if n_lead == 1:
            resampled = cls._resample_series(values[:, 0, :], prod_ns, prod_step,
                                             out_ns, timestep_ns, calculation_ns, resamplingType)
            resampled = resampled[:, None, :]
            delays = leadtime_ns[:1]
        else:
            resampled, delays = cls._resample_forecast(values, prod_ns, leadtime_ns, lead_spans,
                                                       out_ns, timestep_ns, calculation_ns, resamplingType)

        # (production, delay, header) -> flat columns ordered header-major
        resampled = resampled.transpose(0, 2, 1).reshape(out_index.size, -1)

        delay_index = pd.TimedeltaIndex(delays.astype('timedelta64[ns]'), name='delay')
        if headers:
            columns = pd.MultiIndex.from_tuples(
                [cls._as_tuple(k) + (d,) for k in keys for d in delay_index],
                names=headers + ['delay'])
        else:
            columns = delay_index

        return pd.DataFrame(resampled, index=out_index, columns=columns)

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _as_tuple(key):
        return key if isinstance(key, tuple) else (key,)

    @classmethod
    def _as_timedelta(cls, value, what):
        '''pd.Timedelta parses 'MS' as 1 millisecond, so calendar aliases are rejected.'''

        if isinstance(value, dict):
            value = pd.Timedelta(**value)
        elif isinstance(value, str):
            # case-sensitive on purpose: 'MS' is month start, 'ms' is milliseconds
            alias = value.lstrip('0123456789 ').strip()
            if alias.split('-')[0] in cls._CALENDAR_ALIASES:
                raise Exception('%s "%s" is a calendar step, which has no fixed duration and is not supported '
                                '(pandas would read it as a Timedelta of a different unit). Use a fixed step '
                                "such as '1D' or '3H'." % (what, value))
            value = pd.Timedelta(value)
        elif not isinstance(value, pd.Timedelta):
            value = pd.Timedelta(value)

        if value <= pd.Timedelta(0):
            raise Exception('%s must be positive; got %s.' % (what, value))
        return value

    @staticmethod
    def _to_ns(index, what):
        '''int64 nanoseconds; tz-aware datetimes are normalised through UTC.'''

        if isinstance(index, pd.DatetimeIndex) and not isinstance(index.tz, type(None)):
            index = index.tz_convert('UTC').tz_localize(None)
        array = np.asarray(index.to_numpy())
        if array.dtype.kind == 'M':
            array = array.astype('datetime64[ns]')
        elif array.dtype.kind == 'm':
            array = array.astype('timedelta64[ns]')
        else:
            raise Exception('The %s must hold datetimes or timedeltas; got dtype %s.' % (what, array.dtype))
        if np.isnat(array).any():
            raise Exception('The %s contains NaT.' % what)
        return array.view('int64')

    @classmethod
    def _output_index(cls, index, date_from, date_to, timestep_ns):
        '''
        Output stamps, spaced by exactly timestep_ns starting at date_from.

        Built by hand rather than with date_range because pd.Timedelta('1D') resolves
        to a calendar <Day> on pandas 2 but to a fixed <24 * Hours> on pandas 3, which
        shifts the grid of a tz-aware index across a DST switch. Here a step is always
        a fixed duration, on every pandas version; local clock times therefore drift by
        an hour after a switch. Convert to UTC upstream if that matters.
        '''

        tz = getattr(index, 'tz', None)
        bounds = []
        for value, what in ((date_from, 'date_from'), (date_to, 'date_to')):
            stamp = pd.Timestamp(value)
            if isinstance(stamp.tz, type(None)) != isinstance(tz, type(None)):
                raise Exception('%s (%s) and the production index (%s) must both be timezone aware or both naive.'
                                % (what, stamp, index[0]))
            bounds.append(stamp)

        start_ns, stop_ns = cls._to_ns(pd.DatetimeIndex(bounds), 'date_from/date_to')
        count = int((stop_ns - start_ns) // timestep_ns) + 1 if stop_ns >= start_ns else 0
        out_ns = start_ns + np.arange(count, dtype='int64') * timestep_ns

        out_index = pd.DatetimeIndex(out_ns.astype('datetime64[ns]'))
        if not isinstance(tz, type(None)):
            out_index = out_index.tz_localize('UTC').tz_convert(tz)
        return out_index, out_ns

    @staticmethod
    def _leadtime_spans(leadtime_ns):
        '''
        How long each leadtime column describes, in nanoseconds.

        Leadtimes need not be evenly spaced - a forecast commonly goes hourly, then
        3-hourly, then 6-hourly - so each column spans the distance to the next one.
        The last column has no successor and reuses the preceding span, which is what
        fixes the horizon.
        '''

        diffs = np.diff(leadtime_ns)

        return np.append(diffs, diffs[-1])

    @classmethod
    def _production_step(cls, index, prod_ns, allow_single=False):
        '''
        Nominal production step (modal spacing) plus every distinct spacing, for the gcd.

        Returns (None, empty) for a single production when allow_single, since a lone
        forecast has no production spacing to infer and does not need one.
        '''

        if prod_ns.size < 2:
            if allow_single and prod_ns.size == 1:
                return None, np.array([], dtype='int64')
            raise Exception('At least two production timestamps are needed to infer the production step; got %u.'
                            % prod_ns.size)
        diffs = np.diff(prod_ns)
        if (diffs <= 0).any():
            raise Exception('The production index must be strictly increasing (duplicate or unsorted timestamps).')

        unique, counts = np.unique(diffs, return_counts=True)
        step = int(unique[counts.argmax()])

        if unique.size > 1:
            # A tz-aware index legitimately drifts by one hour across a DST switch;
            # anything else (calendar months in particular) is rejected.
            tz_aware = not isinstance(getattr(index, 'tz', None), type(None))
            if not (tz_aware and (np.abs(unique - step) <= cls._HOUR_NS).all()):
                raise Exception('The production index is not evenly spaced (steps from %s to %s). Calendar steps '
                                'such as month starts have no fixed duration; resample to a fixed step first.'
                                % (pd.Timedelta(int(unique.min())), pd.Timedelta(int(unique.max()))))

        return step, unique

    @classmethod
    def _column_layout(cls, data, headers):
        '''
        Header keys in order of first appearance, the (header x leadtime) matrix of
        column positions sorted by leadtime, and the shared leadtime axis.
        '''

        try:
            leadtimes = pd.TimedeltaIndex(data.columns.get_level_values('leadtime'))
        except (TypeError, ValueError) as ex:
            # pd.DateOffset leadtimes (monthly seasonal forecasts) land here. Without
            # this the caller only saw pandas' "Invalid type for timedelta scalar".
            raise Exception('The leadtime level must hold fixed durations; got %s. Calendar leadtimes '
                            '(pd.DateOffset, as monthly forecasts use) have no fixed length and cannot be '
                            'resampled to a fixed step - aggregate them with MeteoRaster.adjust_leadtimes '
                            'or resample_timestep instead.'
                            % data.columns.get_level_values('leadtime').dtype) from ex
        lead_ns = cls._to_ns(leadtimes, 'leadtime level')

        if headers:
            codes, keys = pd.factorize(data.columns.droplevel('leadtime'))
            keys = list(keys)
        else:
            codes = np.zeros(data.shape[1], dtype='int64')
            keys = [()]

        counts = np.bincount(codes, minlength=len(keys))
        if (counts != counts[0]).any():
            raise Exception('Every header combination must carry the same leadtimes; found groups with %u and %u '
                            'columns.' % (counts.min(), counts.max()))

        col_pos = np.lexsort((lead_ns, codes)).reshape(len(keys), counts[0])
        grouped = lead_ns[col_pos]
        if not (grouped == grouped[0]).all():
            raise Exception('Every header combination must carry the same leadtimes.')

        leadtime_ns = grouped[0]
        if leadtime_ns.size > 1 and (np.diff(leadtime_ns) <= 0).any():
            raise Exception('Duplicate leadtimes found in the columns.')

        return keys, col_pos, leadtime_ns

    # ------------------------------------------------------------------ kernels

    @classmethod
    def _resample_series(cls, values, prod_ns, prod_step, out_ns, timestep_ns, calculation_ns, resamplingType):
        '''
        Single-leadtime frames: a time series, aggregated along production time.
        values is (production, header); the result is (output, header).
        Value i covers [P_i, P_i + prod_step), so each output window is the exact
        overlap-weighted combination of the values it touches.
        '''

        if resamplingType == 'linear':
            return cls._interpolate(values, prod_ns, out_ns)

        substeps = timestep_ns // calculation_ns
        resampled = None
        for k0 in range(substeps):
            stamps = out_ns + k0 * calculation_ns
            source = np.searchsorted(prod_ns, stamps, side='right') - 1
            safe = np.where(source >= 0, source, 0)
            valid = (source >= 0) & (stamps < prod_ns[safe] + prod_step)
            sample = np.where(valid[:, None], values[safe], np.nan)
            if isinstance(resampled, type(None)):
                resampled = sample
            elif resamplingType == 'max':
                resampled = np.maximum(resampled, sample)
            else:
                resampled = resampled + sample

        if resamplingType == 'sum':
            # each sub-step carries its share of the accumulation
            resampled = resampled * (calculation_ns / prod_step)
        elif resamplingType == 'mean':
            resampled = resampled / substeps
        return resampled

    @classmethod
    def _resample_forecast(cls, values, prod_ns, leadtime_ns, lead_spans, out_ns, timestep_ns, calculation_ns,
                           resamplingType):
        '''
        Multi-leadtime frames: each production row is a separate forecast.
        values is (production, leadtime, header); the result is (output, delay, header).
        An output row reuses the last forecast issued at or before its timestamp, with
        the leadtimes shifted by the elapsed time.

        lead_spans gives each column's duration, so leadtimes may be unevenly spaced;
        the source column for a fine sub-step is found by searchsorted rather than by
        dividing by a single step.
        '''

        lead0 = int(leadtime_ns[0])
        horizon = int(leadtime_ns[-1] + lead_spans[-1] - lead0)
        uniform = bool((lead_spans == lead_spans[0]).all())

        source = np.searchsorted(prod_ns, out_ns, side='right') - 1
        safe = np.where(source >= 0, source, 0)
        issued = source >= 0
        # elapsed time between the forecast issue and the output timestamp
        shift = out_ns - prod_ns[safe]

        if resamplingType == 'linear':
            n_delay = int((int(leadtime_ns[-1]) - lead0) // timestep_ns) + 1
        else:
            n_delay = int(horizon // timestep_ns)
        if n_delay < 1:
            raise Exception('The forecast horizon ("%s") is shorter than the requested time step ("%s").'
                            % (pd.Timedelta(horizon), pd.Timedelta(timestep_ns)))
        delays = lead0 + np.arange(n_delay, dtype='int64') * timestep_ns

        if resamplingType == 'linear':
            offsets = shift[:, None] + delays[None, :]
            resampled = cls._interpolate_leadtime(values, leadtime_ns, safe, offsets, issued)
            return resampled, delays

        substeps = timestep_ns // calculation_ns
        rows = safe[:, None]
        resampled = None
        for k0 in range(substeps):
            offsets = shift[:, None] + delays[None, :] + k0 * calculation_ns
            columns = np.searchsorted(leadtime_ns, offsets, side='right') - 1
            picked = np.where(columns >= 0, columns, 0)
            valid = issued[:, None] & (columns >= 0) & (offsets < leadtime_ns[picked] + lead_spans[picked])
            picked = np.where(valid, picked, 0)
            sample = values[rows, picked]
            if resamplingType == 'sum' and not uniform:
                # Each fine sub-step carries its share of its source column's total.
                # With uniform spans the share is constant and is applied once at the
                # end instead, which keeps those results bit-for-bit unchanged.
                sample = sample * (calculation_ns / lead_spans[picked])[:, :, None]
            sample = np.where(valid[:, :, None], sample, np.nan)
            if isinstance(resampled, type(None)):
                resampled = sample
            elif resamplingType == 'max':
                resampled = np.maximum(resampled, sample)
            else:
                resampled = resampled + sample

        if resamplingType == 'sum' and uniform:
            resampled = resampled * (calculation_ns / int(lead_spans[0]))
        elif resamplingType == 'mean':
            resampled = resampled / substeps
        return resampled, delays

    @staticmethod
    def _interpolate(values, source_ns, target_ns):
        '''Linear interpolation of (source, header) values onto target_ns. No extrapolation.'''

        n_source = source_ns.size
        if n_source < 2:
            raise Exception('At least two timestamps are needed to interpolate.')

        left = np.clip(np.searchsorted(source_ns, target_ns, side='right') - 1, 0, n_source - 2)
        span = (source_ns[left + 1] - source_ns[left]).astype(float)
        weight = (target_ns - source_ns[left]) / span
        inside = (target_ns >= source_ns[0]) & (target_ns <= source_ns[-1])

        low = values[left]
        high = values[left + 1]
        # weight 0 must not drag a NaN in from the next node (exact stamps stay exact)
        resampled = np.where(weight[:, None] == 0, low, low * (1 - weight[:, None]) + high * weight[:, None])
        return np.where(inside[:, None], resampled, np.nan)

    @staticmethod
    def _interpolate_leadtime(values, leadtime_ns, rows, offsets, issued):
        '''
        Linear interpolation along the leadtime axis of (production, leadtime, header)
        values, at `offsets` (production, delay) measured from each source forecast.

        The bracketing columns come from searchsorted, so unevenly spaced leadtimes
        interpolate over their own local span.
        '''

        n_lead = leadtime_ns.size

        left = np.clip(np.searchsorted(leadtime_ns, offsets, side='right') - 1, 0, n_lead - 2)
        span = (leadtime_ns[left + 1] - leadtime_ns[left]).astype(float)
        weight = (offsets - leadtime_ns[left]) / span
        inside = issued[:, None] & (offsets >= leadtime_ns[0]) & (offsets <= leadtime_ns[-1])

        row_index = rows[:, None]
        low = values[row_index, left]
        high = values[row_index, left + 1]
        weight = weight[:, :, None]
        resampled = np.where(weight == 0, low, low * (1 - weight) + high * weight)
        return np.where(inside[:, :, None], resampled, np.nan)

    @classmethod
    def production_to_event(cls, data, prod_frequency):
        '''
        '''

        raise(Exception('Not implemented'))
