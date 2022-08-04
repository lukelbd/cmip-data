#!/usr/bin/env python3
"""
Utilities for reading and handling results.
"""
import builtins
import re

import climopy as climo  # noqa: F401  # add accessor
import numpy as np
import xarray as xr
from icecream import ic  # noqa: F401

from .internals import STANDARD_LEVELS_CMIP5, STANDARD_LEVELS_CMIP6, _validate_ranges

__all__ = [
    'open_file',
    'average_periods',
    'average_regions',
]

# Automatically updated mapping from model names to institution ids
# NOTE: This is used inside plotting functions to group cmip5 and/or cmip6 models from
# the same center. Previously tried to detect alike models from their names but this
# was unreliable (for example ACCESS and CSIRO models are both from CSIRO in Australia,
# MPI and ICON are both from MPI in Germany, and CESM and CCSM are both from NCAR). Can
# survey diferent institution ids with following code: { models=(); for f in ts_Amon_*;
# do model=$(echo $f | cut -d_ -f3); [[ " ${models[*]} " =~ " $model " ]] && continue;
# models+=("$model"); echo $f; ncinfo $f | grep -E 'institut(e|ion)_id'; done; } | less
MODELS_INSTITUTIONS = {}


def open_file(path, variable=None, validate=True, project=None, printer=None):
    """
    Open an output dataset and repair possible coordinate issues.

    Parameters
    ----------
    path : path-like
        The path.
    variable : str, optional
        The variable. If passed a data array is returned.
    validate : bool, optional
        Whether to validate against quality control ranges.
    project : str, optional
        The project. If passed vertical levels are restricted.
    printer : callable, optional
        The printer.
    """
    # Initial stuff
    # NOTE: Since cmip6 includes 2 extra levels have to drop them to get it to work
    # with cmip5 data (this is used to get standard kernels to work with cmip5 data).
    # NOTE: Here drop_duplicates is only available for arrays. Monitor this
    # thread for updates: https://github.com/pydata/xarray/pull/5239
    # WARNING: Here the model id recorded under 'source_id' or 'model_id' often differs
    # from the model id in the file name. Annoying but have to use the file version.
    print = printer or builtins.print
    project = project and project.lower()
    dataset = xr.open_dataset(path, use_cftime=True)
    if 'numerator' in dataset and 'denominator' in dataset and 'region' in dataset.sizes:  # noqa: E501
        dataset = dataset.set_index(region=('numerator', 'denominator'))
    dataset = dataset.drop_vars(dataset.coords.keys() - dataset.sizes.keys())
    for coord in dataset.coords.values():  # remove missing bounds variables
        coord.attrs.pop('bounds', None)
    model = path.name.split('_')[2] if path.name.count('_') >= 4 else None
    institution = dataset.attrs.get('institute_id', dataset.attrs.get('institution_id'))
    if institution:
        if 'FGOALS' in model and 'CAS' in institution:
            institution = 'LASG'  # changed in cmip6 to accomodate CAS (Chinese center)
        else:
            institution = institution.split()[0].split('-')[0]

    # Validate pressure level data
    # NOTE: Some files will have duplicate times (i.e. due to using cdo mergetime on
    # files with overlapping time coordinates) and drop_duplicates(time, keep='first')
    # does not work since it is only available on DataArrays, so use manual method.
    # See: https://github.com/pydata/xarray/issues/1072
    # See: https://github.com/pydata/xarray/discussions/6297
    # NOTE: Testing reveals that calling 'cdo ymonmean' on files that extend from
    # december-november instead of january-december will result in january-december
    # climate files with non-monotonic time steps. This will mess up the .groupby()
    # operations in average_periods, so we auto-convert time array to be monotonic
    # and standardize the days (so that cell_duration calculations are correct).
    if project and 'plev' in dataset.coords and dataset.plev.size > 1:
        plev = dataset.plev.values
        if project == 'cmip5':
            levels = STANDARD_LEVELS_CMIP5
        elif project == 'cmip6':
            levels = STANDARD_LEVELS_CMIP6
        else:
            raise ValueError(f'Invalid {project=} for determining levels.')
        extra = [p for p in plev if not np.any(np.isclose(p, levels))]
        missing = [p for p in levels if not np.any(np.isclose(p, plev))]
        dataset = dataset.drop_sel(plev=extra)  # no message since this is common
        if message := ', '.join(format(p / 100, '.0f') for p in missing):
            print(
                f'Warning: File {path.name!r} has {len(missing)} missing '
                f'pressure levels ({plev.size} out of {levels.size}): {message}.'
            )
    if 'time' in dataset.coords and dataset.time.size > 1:
        time = dataset.time.values
        mask = dataset.get_index('time').duplicated(keep='first')
        if message := ', '.join(format(t, '.0f') for t in time[mask]):
            dataset = dataset.isel(time=~mask)  # WARNING: drop_isel fails here
            print(
                f'Warning: File {path.name!r} has {mask.sum().item()} duplicate '
                f'time values: {message}. Kept only the first values.'
            )
        years, months = dataset.time.dt.year, dataset.time.dt.month
        if years.size == 12 and sorted(months.values) == sorted(range(1, 13)):
            cls = type(dataset.time[0].item())  # e.g. CFTimeNoLeap
            std = [cls(min(years), t.month, 1) for t in dataset.time.values]
            std = xr.DataArray(std, dims='time', name='time', attrs=dataset.time.attrs)
            dataset = dataset.assign_coords(time=std)  # assign with same attributes
            dataset = dataset.sortby('time')  # sort in case months are non-monotonic

    # Validate variable data
    # NOTE: Found negative radiative flux values for rldscs in IITM-ESM model. For
    # now just automatically invert these values but should contact developer.
    names = dataset.data_vars if validate else ()
    ranges = {name: _validate_ranges(name, 'Amon') for name in names}
    if 'pr' in dataset.data_vars and model == 'CIESM':  # kludge
        print(
            'Warning: Adjusting CIESM precipitation flux with obviously '
            'incorrect units by 1e3 (guess to recover correct units).'
        )
        with xr.set_options(keep_attrs=True):
            dataset['pr'].data *= 1e3
    for name in names:
        array = test = dataset[name]
        if 'bnds' in array.sizes or 'bounds' in array.sizes:
            continue
        if 'time' in array.sizes:
            test = array.mean('time')
        min_, max_ = test.min().item(), test.max().item()
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

    # Possibly retrieve variable and update mapping
    # NOTE: Some models from the same lineage have different suffixes in their
    # institution id (for example ACCESS in CMIP6 is CSIRO-ARCCSS while ACCESS
    # in CMIP5 is CSIRO-BOM). Therefore only preserve first part of id.
    if model and institution:
        MODELS_INSTITUTIONS[model] = institution
    if variable is None:  # demote precision for speed
        return dataset
    else:
        return dataset[variable].astype(np.float32)


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
    """
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
        seasons = input.coords['time.season']
        _, idxs = np.unique(seasons, return_index=True)
        seasons = seasons.isel(time=np.sort(idxs))  # coordinate selection
        names = seasons.str.lower()  # coordinate naming
        for name, season in zip(names.data.flat, seasons.data.flat):
            times[name] = ('time.season', season)
    if monthly:
        months = input.coords['time.month']
        names = input.time.dt.strftime('%b').str.lower()
        _, idxs = np.unique(months, return_index=True)
        months = months.isel(time=np.sort(idxs))  # coordinate selection
        names = names.isel(time=np.sort(idxs))  # coordinate naming
        for name, month in zip(names.data.flat, months.data.flat):
            times[name] = ('time.month', month)
    for name, (key, value) in times.items():
        data = input if key is None else input.sel(time=(input.coords[key] == value))
        days = data.time.dt.days_in_month
        wgts = days.groupby('time.year') / days.groupby('time.year').sum()
        wgts = wgts.astype(np.float32)  # preserve float32 variables
        ones = xr.where(data.isnull(), 0, 1)
        ones = ones.astype(np.float32)  # preserve float32 variables
        with xr.set_options(keep_attrs=True):
            numerator = (data * wgts).groupby('time.year').sum(dim='time')
            denominator = (ones * wgts).groupby('time.year').sum(dim='time')
            output[name] = numerator / denominator
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
    attrs = {'long_name': 'time', 'standard_name': 'time', 'units': 'yr', 'axis': 'T'}
    output = output.rename(year='time')
    if isinstance(input, xr.DataArray):
        output.name = input.name
    if output.time.size == 1:  # avoid conflicts during merge
        output = output.isel(time=0, drop=True)
    else:
        output.time.attrs.update(attrs)
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
    """
    # NOTE: Alternative approach to feedbacks 'numerator' and 'denominator' is to
    # get pointwise fluxes relative to different averages, then when we want non-local
    # feedbacks, simply take averages of the local values. Will consider after testing.
    # NOTE: Previously used a 2D array ((1x1, 1xM), (1x1, 1xM), (1x1, 1xM), (Nx1, NxM))
    # for storing (rows) the global average, sh average, nh average, and fully-resolved
    # latitude data, plus (cols) the global average and fully-resolved longitude data.
    # This was built as a nested 4x2 list (with longitude and latitude coordinates of
    # NaN, ... and NaN, -Inf, Inf, ..., and default NaN-filled arrays in each slot),
    # subsequently combined with combine_nested. However this was unnecessary... way
    # easier to just broadcast averages in existing space coordinates. Also normal
    # xarray-style slice selection fails with NaN or +/-Inf coordinates so selections
    # would always need to use .isel() rather than .sel()... even more awkward.
    eps = 1e-10  # avoid assigning twice
    input = input.climo.add_cell_measures()
    outputs = []
    for data in input.climo._iter_data_vars():
        output = {'point': data} if point else {}
        if latitude:
            da = xr.ones_like(data) * data.climo.average('lon')
            da.name = data.name
            da.attrs.update(data.attrs)
            output['latitude'] = da
        if hemisphere:
            sh = data.climo.sel_hemisphere('sh').climo.average('area')
            nh = data.climo.sel_hemisphere('nh').climo.average('area')
            sh = sh.drop_vars(('lon', 'lat')).expand_dims(('lon', 'lat'))
            nh = nh.drop_vars(('lon', 'lat')).expand_dims(('lon', 'lat'))
            sh = sh.transpose(*data.sizes)
            nh = nh.transpose(*data.sizes)
            da = data.astype(np.float64)
            da.loc[{'lat': slice(None, 0 - eps)}] = sh
            da.loc[{'lat': slice(0 + eps, None)}] = nh
            da.loc[{'lat': slice(0 - eps, 0 + eps)}] = 0.5 * (sh + nh)
            output['hemisphere'] = da
        if globe:
            da = xr.ones_like(data) * data.climo.average('area')
            da.name = data.name
            da.attrs.update(data.attrs)
            output['globe'] = da
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
