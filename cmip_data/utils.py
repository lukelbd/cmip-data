#!/usr/bin/env python3
"""
Utilities for reading and handling results.
"""
import builtins

import cftime
import climopy as climo  # noqa: F401  # add accessor
import numpy as np
import xarray as xr
from icecream import ic  # noqa: F401

from .facets import STANDARD_LEVELS_CMIP5, STANDARD_LEVELS_CMIP6, _validate_ranges

__all__ = [
    'assign_dates',
    'average_periods',
    'average_regions',
    'load_file',
]


def assign_dates(data, year=None):
    """
    Assign a standard year and day for time coordinates in the file.

    Parameters
    ----------
    data : xarray.Dataset or xarray.DataArray
        The input data. Requires either time or month coordinate.
    year : int, optional
        The output year. Requires data with 12 time or month coordinates.

    Returns
    -------
    output : xarray.Dataset or xarray.DataArray
        The data with updated time coordinates.
    """
    # NOTE: This is used when *loading* datasets and combining along models. Should
    # not put this into per-model climate and feedback processing code.
    if 'time' in data.sizes and 'month' not in data.sizes:
        ntime = len(data.time)
        years = data.time.dt.year
        months = data.time.dt.month
    elif 'month' in data.sizes and 'time' not in data.sizes:
        ntime = len(data.month)
        months = data.month
        data = data.rename(month='time')
    else:
        raise ValueError('Unexpected input array. Expected months or times.')
    if ntime == 12 and year is None:
        raise ValueError('Monthly averages require an explicit input year.')
    if ntime != 12 and year is not None:
        raise ValueError('Cannot assign input year to data with more than 12 months.')
    if year is not None:
        time = [cftime.datetime(year, m, 15) for m in months]
    else:
        time = [cftime.datetime(y, m, 15) for y, m in zip(years, months)]
    time = xr.CFTimeIndex(time)  # prefer cftime
    time = xr.DataArray(time, dims='time', attrs=data.time.attrs)
    time.attrs.update({'axis': 'T', 'standard_name': 'time'})
    data = data.assign_coords(time=time)
    return data


def average_periods(input, annual=True, seasonal=True, monthly=True):
    """
    Convert time coordinates into two sets of coordinates: a ``'time'`` coordinate
    indicating the year, and a ``'period'`` coordinate inidicating the period (i.e.
    monthly, seasonal, or annual). If only one year is present ``'time'`` is removed.

    Parameters
    ----------
    input : xarray.Dataset
        The input dataset.
    annual, seasonal, monthly : bool, optional
        Whether to take various decompositions. Default is annual and seasonal.

    Returns
    -------
    output : xarray.Dataset
        The standardized data.
    """
    # NOTE: This is no longer used in feedacks.py. Instead regress monthly flux against
    # annual average temperature anomalies rebuild annual feedbacks when loading.
    # NOTE: The new climopy cell duration calculation will auto-detect monthly and
    # yearly data, but not yet done, so use explicit days-per-month weights for now.
    # NOTE: Here resample(time='AS') and groupby('time.year') yield identical results,
    # except latter creates an integer 'year' axis while former resamples the existing
    # time axis and preserves the datetime64 format. Test with the following code:
    # t = np.arange('2000-01-01', '2003-01-01', dtype='datetime64[M]')
    # d = xr.DataArray(np.arange(36), dims='time', coords={'time': t})
    # d.resample(time='AS').mean(dim='time'), d.groupby('time.year').mean(dim='time')
    times = {'ann': (None, None)} if annual else {}
    output = {}
    if seasonal:
        seasons = input.coords['time.season']  # strings indicators
        _, idxs = np.unique(seasons, return_index=True)  # unique strings
        seasons = seasons.isel(time=np.sort(idxs))  # coordinate selection
        names = seasons.str.lower()  # coordinate naming
        for name, season in zip(names.data.flat, seasons.data.flat):
            times[name] = ('time.season', season)
    if monthly:
        months = input.coords['time.month']  # numeric indicators
        names = input.time.dt.strftime('%b').str.lower()  # string indicators
        _, idxs = np.unique(months, return_index=True)  # unique strings
        months = months.isel(time=np.sort(idxs))  # coordinate selection
        names = names.isel(time=np.sort(idxs))  # coordinate naming
        for name, month in zip(names.data.flat, months.data.flat):
            times[name] = ('time.month', month)
    for name, (key, value) in times.items():
        data = input if key is None else input.sel(time=(input.coords[key] == value))
        days = data.time.dt.days_in_month.astype(np.float32)  # preserve float32
        wgts = days.groupby('time.year') / days.groupby('time.year').sum()
        with xr.set_options(keep_attrs=True):
            data = (data * wgts).groupby('time.year').sum(dim='time', skipna=False)
        output[name] = data
    period = xr.DataArray(
        list(output),
        dims='period',
        name='period',
        attrs={'long_name': 'time averaging period'},
    )
    output = xr.concat(
        output.values(),
        dim=period,
        coords='minimal',
        compat='equals',
        combine_attrs='drop_conflicts',
    )
    attrs = {'long_name': 'year', 'standard_name': 'time', 'units': 'yr', 'axis': 'T'}
    if isinstance(input, xr.DataArray):
        output.name = input.name
    if output.year.size == 1:  # avoid conflicts during merge
        output = output.isel(year=0, drop=True)
    else:
        output.year.attrs.update(attrs)
    return output


def average_regions(input, point=True, latitude=True, hemisphere=True, globe=True):
    """
    Convert space coordinates into three sets of coordinates: the original longitude
    and latitude coordinates, and a ``'region'`` coordinate indicating the average
    (i.e. points, latitudes, hemispheres, or global). See `response_feedbacks`.

    Parameters
    ----------
    input : xarray.Dataset
        The input dataset.
    point, latitude, hemisphere, globe : optional
        Whether to add each type of average.

    Returns
    -------
    output : xarray.Dataset
        The standardized data.
    """
    # NOTE: Previously stored separate 'numerator' and 'denominator' averaging
    # methods with numerator='globe' and denominator='globe' corresponding to global
    # feedbacks. Now just take global average of pointwise global feedback.
    # NOTE: Previously used a 2D array for storing (rows) the global average, sh
    # average, nh average, and fully-resolved latitude data, plus (cols) the global
    # average and fully-resolved longitude data. This was built as a nested 4x2 list
    # with longitude and latitude coordinates of NaN, ... and NaN, -Inf, Inf, ..., and
    # default NaN-filled arrays in each slot, subsequently combined with combine_nested.
    # However this was overly complicated and no longer saves space now that we do not
    # save average feedbacks. Note .sel() does not work with NaN and Inf coordinates.
    eps = 1e-10  # avoid assigning twice
    input = input.climo.add_cell_measures()
    outputs = []
    for data in input.climo._iter_data_vars():
        output = {'point': data} if point else {}
        if latitude:
            avgs = xr.ones_like(data) * data.climo.average('lon')
            avgs.name = data.name
            avgs.attrs.update(data.attrs)
            output['latitude'] = avgs
        if hemisphere:
            shemi = data.climo.sel_hemisphere('sh').climo.average('area')
            nhemi = data.climo.sel_hemisphere('nh').climo.average('area')
            shemi = shemi.drop_vars(('lon', 'lat')).expand_dims(('lon', 'lat'))
            nhemi = nhemi.drop_vars(('lon', 'lat')).expand_dims(('lon', 'lat'))
            shemi = shemi.transpose(*data.sizes)
            nhemi = nhemi.transpose(*data.sizes)
            dim = 'lat'  # assignment coordinate
            avgs = data.astype(np.float64)
            avgs.loc[{dim: slice(None, 0 - eps)}] = shemi
            avgs.loc[{dim: slice(0 + eps, None)}] = nhemi
            avgs.loc[{dim: slice(0 - eps, 0 + eps)}] = 0.5 * (shemi + nhemi)
            output['hemisphere'] = avgs
        if globe:
            avgs = xr.ones_like(data) * data.climo.average('area')
            avgs.name = data.name
            avgs.attrs.update(data.attrs)
            output['globe'] = avgs
        outputs.append(output)
    if isinstance(input, xr.DataArray):
        output, = outputs
    else:
        output = {k: xr.Dataset({o[k].name: o[k] for o in outputs}) for k in outputs[0]}
    region = xr.DataArray(
        list(output),
        dims='region',
        name='region',
        attrs={'long_name': 'spatial averaging region'},
    )
    output = xr.concat(
        output.values(),
        dim=region,
        coords='minimal',
        compat='equals',
        combine_attrs='drop_conflicts',
    )
    return output


def load_file(
    path, variable=None, project=None, validate=False,
    lazy=False, demote=None, printer=None, **kwargs,
):
    """
    Load an output dataset and possibly standardize time and pressure coordinates.

    Parameters
    ----------
    path : path-like
        The path.
    variable : str or sequence, optional
        The variable(s). If string a data array is returned.
    project : str, optional
        The project. If passed pressure levels are standardized.
    validate : bool, default: False
        Whether to validate against quality control ranges.
    lazy : bool, default: False
        Whether to load the dataset into memory immediately.
    demote : bool, default: ``not lazy``
        Whether to demote the precision to float32 for speed.
    printer : callable, optional
        The printer.
    **kwargs
        Passed to `xarray.load_dataset` or `xarray.open_dataset`

    Returns
    -------
    output : xarray.Dataset or xarray.DataArray
        The resulting data.
    """
    # Load the dataset
    # TODO: Currently ensembles are made with xr.concat() which requires loading into
    # memory. In future should make open_files() function that uses open_mfdataset()
    # and refactor open_dataset() utility in coupled/results.py (remove averaging
    # utility, standardize after concatenating). For now since loading required anyway
    # below uses load_dataset() and demotes float64 to float32 by default to help
    # reduce memory usage. See: https://github.com/pydata/xarray/issues/4628
    # NOTE: Use dataset[name].variable._data to check whether data is loaded (tried
    # ic() on dataset[name] and reading the array output but this seems to sometimes
    # / inconsistently trigger loading itself). In future along with open_mfdataset()
    # may use dask for its lazy loading / lazy operations even if chunking not needed.
    # Also note cache=False prevents storing loaded values on original dataset when
    # operation triggers load, but if result of operation is same size as the dataset
    # (i.e. did not slice first) and remains as a session variable (e.g. not a function
    # variable) then may still cause memory issues. Should be lazy-loading aware in
    # all dataset manipulations. See: https://stackoverflow.com/a/45644654/4970632.
    variable = variable if isinstance(variable, str) else list(variable or ())
    demote = not lazy if demote is None else demote
    print = printer or builtins.print
    kwargs.setdefault('use_cftime', True)
    if lazy:  # open read-only view
        data = xr.open_dataset(path, **kwargs)
    else:  # open and load into memory
        data = xr.load_dataset(path, **kwargs)
    drop = data.coords.keys() - data.sizes.keys()
    data = data.drop_vars(drop)
    if variable:
        data = data[variable]
    for coord in data.coords.values():  # remove missing bounds variables
        coord.attrs.pop('bounds', None)
    if demote:  # also triggers lazy load so should remove if switching workflows
        data = data.astype(np.float32)

    # Validate pressure coordinates
    # WARNING: Merging files with ostensibly the same pressure levels can result in
    # new staggered levels due to inexact float pressure coordinates. Fix this when
    # building multi-model datasets by using the standard level array for coordinates.
    # NOTE: CMIP6 has two more levels than CMIP5. Have to drop them to avoid issues
    # with concatenation issues or kernel integration e.g. due to improper weights or
    # NaN results from NaN data levels (see also _fluxes_from_anomalies() kernel data).
    # NOTE: Some datasets have a pressure 'bounds' variable but appears calculated
    # naively as halfway points between levels. Since inconsistent between files
    # just strip all bounds attribute and rely on climopy calculations.
    if project and 'plev' in data.coords and data.plev.size > 1:
        plev = data.plev.values
        project = project and project.lower()
        if project == 'cmip5':
            levels = STANDARD_LEVELS_CMIP5
        elif project == 'cmip6':
            levels = STANDARD_LEVELS_CMIP6
        else:
            raise ValueError(f'Invalid {project=} for determining levels.')
        levs = [levels[idx.item()] for p in plev if len(idx := np.where(np.isclose(p, levels))[0])]  # noqa: E501
        levs = xr.DataArray(levs, name='plev', dims='plev', attrs=data.plev.attrs)
        drop = [p for p in plev if not np.any(np.isclose(p, levels))]
        missing = [p for p in levels if not np.any(np.isclose(p, plev))]
        data = data.drop_sel(plev=drop)  # no message since this is common
        data = data.assign_coords(plev=levs)  # retain original attributes
        if message := ', '.join(format(p / 100, '.0f') for p in missing):
            print(
                f'Warning: File {path.name!r} has {len(missing)} missing '
                f'pressure levels ({plev.size} out of {levels.size}): {message}.'
            )

    # Validate time coordinates
    # NOTE: Some files will have duplicate times (i.e. due to using cdo mergetime on
    # files with overlapping time coordinates) and drop_duplicates(time, keep='first')
    # does not work since it is only available on DataArrays, so use manual method.
    # See: https://github.com/pydata/xarray/issues/1072
    # See: https://github.com/pydata/xarray/discussions/6297
    # NOTE: Testing reveals that calling 'cdo ymonmean' on files that extend from
    # december-november instead of january-december will result in january-december
    # climate files with non-monotonic time steps. This will mess up the .groupby()
    # operations in average_periods, so we auto-convert time array to be monotonic
    # and standardize the days (so that cell_duration calculations are correct). Note
    # this is distinct from assign_dates() required for concatenating ensemble models.
    if 'time' in data.coords and data.time.size > 1:
        time = data.time.values
        mask = data.get_index('time').duplicated(keep='first')
        if message := ', '.join(str(t) for t in time[mask]):
            data = data.isel(time=~mask)  # WARNING: drop_isel fails here
            print(
                f'Warning: File {path.name!r} has {mask.sum().item()} duplicate '
                f'time values: {message}. Kept only the first values.'
            )
        years, months = data.time.dt.year, data.time.dt.month
        if sorted(months.values) == sorted(range(1, 13)):  # standardize times
            cls = type(data.time[0].item())  # e.g. CFTimeNoLeap
            std = [cls(max(years), t.month, 15) for t in data.time.values]
            std = xr.DataArray(std, dims='time', name='time', attrs=data.time.attrs)
            data = data.assign_coords(time=std)  # assign with same attributes
            data = data.sortby('time')  # sort in case months are non-monotonic

    # Validate variable data
    # NOTE: Here the model id recorded under 'source_id' or 'model_id' often differs
    # from the model id in the file name. Annoying but have to use the file version.
    # NOTE: Found negative radiative flux values for rldscs in IITM-ESM model. For
    # now just automatically invert these values but should contact developer.
    # NOTE: Found negative specific humidity values in IITM-ESM model. Add constant
    # offset to avoid insane humidity kernels (inspection of abrupt time series with
    # cdo -infon -timmin -selname,hus revealed pretty large anomalies up to -5e-4
    # at the surface so use dynamic adjustment with floor offset of 1e-6).
    model = path.name.split('_')[2] if path.name.count('_') >= 4 else None
    datas = {} if not validate else {data.name: data} if variable else data.data_vars
    ranges = {name: _validate_ranges(name, 'Amon') for name in datas}
    if 'pr' in datas and model == 'CIESM':  # kludge
        print(
            'Warning: Adjusting CIESM precipitation flux with obviously '
            'incorrect units by 1e3 (guess to recover correct units).'
        )
        datas['pr'].data *= 1e3
    if 'hus' in datas and np.any(datas['hus'].values <= 0.0):
        print(
            'Warning: Adjusting negative specific humidity values by enforcing '
            'absolute minimum of 1e-6 (consistent with other model stratospheres).'
        )
        datas['hus'].data[datas['hus'].data < 1e-6] = 1e-6
    for name, array in datas.items():
        sample = array  # default
        if 'bnds' in array.sizes or 'bounds' in array.sizes:
            continue
        if 'time' in array.sizes:  # TODO: ensure robust to individual timesteps
            sample = array.isel(time=0)
        min_, max_ = sample.min().item(), sample.max().item()
        pmin, pmax, *_ = ranges[name]
        skip_range = name == 'rsut' and model == 'MCM-UA-1-0'
        skip_identical = name == 'rsdt' and model == 'MCM-UA-1-0'
        if not skip_identical and array.size > 1 and np.isclose(min_, max_):
            array.data[:] = np.nan
            print(
                f'Warning: Variable {name!r} has the identical value {min_} '
                'across entire domain. Set all data to NaN.'
            )
        elif not skip_range and pmin and pmax and (
            np.sign(pmin) == np.sign(pmax) and min_ >= -pmax and max_ <= -pmin
        ):
            array.data[:] = -1 * array.data[:]
            print(
                f'Warning: Variable {name!r} range ({min_}, {max_}) is inside '
                f'negative of valid cmip range ({pmin}, {pmax}). Multiplied data by -1.'
            )
        elif not skip_range and (
            pmin is not None and min_ < pmin or pmax is not None and max_ > pmax
        ):
            array.data[:] = np.nan
            print(
                f'Warning: Variable {name!r} range ({min_}, {max_}) is outside '
                f'valid cmip range ({pmin}, {pmax}). Set all data to NaN.'
            )
    return data
