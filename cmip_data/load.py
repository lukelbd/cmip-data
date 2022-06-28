#!/usr/bin/env python
"""
Load datasets downloaded from external sources and downloaded here.
"""
import builtins
import json
import warnings
from pathlib import Path

import climopy as climo  # noqa: F401  # add accessor
import numpy as np
import pandas as pd
import xarray as xr
from climopy import ureg
from icecream import ic  # noqa: F401

from .facets import _get_ranges, _glob_files, LEVELS_CMIP5, LEVELS_CMIP6


__all__ = [
    'load_file',
    'load_tables',
    'load_datasets',
    'concat_tables',
    'concat_datasets',
]


def load_file(path, variable=None, project=None, printer=None):
    """
    Open an output dataset and repair possible coordinate issues.

    Parameters
    ----------
    path : path-like
        The path.
    project : str, optional
        The project. Used in level checking.
    variable : str, optional
        The variable. If passed a data array is returned.
    printer : callable, optional
        The printer.
    """
    # Initial stuff
    project = (project or 'cmip6').lower()
    levels = LEVELS_CMIP5 if project == 'cmip5' else LEVELS_CMIP6
    print = printer or builtins.print
    data = xr.open_dataset(path, use_cftime=True)
    if variable is not None:
        data = data[variable]
    data.load()
    for coord in data.coords.values():  # remove missing bounds variables
        coord.attrs.pop('bounds', None)

    # Validate variable data
    # NOTE: Here monthly temperature can be out of range in the stratosphere so we
    # are conservative and only test annual means. Also don't bother with global
    # tests for now (compare with 'summarize_ranges' function in process.py).
    if variable is not None:
        points, _ = _get_ranges()
        test = data
        if 'time' in data.sizes:  # use more liberal test
            test = data.resample(time='AS').mean('time')
        min_, max_ = test.min().item(), test.max().item()
        if data.size > 1 and np.isclose(min_, max_):
            data.data[:] = np.nan
            print(
                f'Warning: Variable {variable!r} has the identical value {min_} '
                'across entire domain. Set all values to NaN.'
            )
        pmin, pmax = points.get(('Amon', variable), (None, None))
        if pmin is not None and min_ < pmin or pmax is not None and max_ > pmax:
            data.data[:] = np.nan
            print(
                f'Warning: Pointwise {variable!r} range ({min_}, {max_}) is outside '
                f'the valid cmip range ({pmin}, {pmax}). Set all values to NaN.'
            )
    # Validate coordinate data
    # NOTE: Here drop_duplicates is only available for arrays. Monitor this
    # thread for updates: https://github.com/pydata/xarray/pull/5239
    # NOTE: Since cmip6 includes 2 extra levels have to drop them to get it to work
    # with cmip5 data (this is used to get standard kernels to work with cmip5 data).
    if 'plev' in data.coords:
        plev = data.coords['plev']
        vals = [v for v in plev.data.flat if not np.any(np.isclose(v, levels))]
        if vals:
            message = ', '.join(format(v, '.0f') for v in vals)
            data = data.drop_sel(plev=vals)
            print(
                f'Warning: File {path.name!r} has {len(vals)} extra (stratospheric?) '
                f'pressure levels: {message}. Kept only the standard 19 levels.'
            )
    if variable is not None and 'time' in data.coords:  # only arrays (xarray #5239)
        time = data.coords['time']
        data = data.drop_duplicates('time', keep='first')
        vals = [v for v in time.data.flat if v not in data.coords['time'].data]
        if vals:  # error with standardize_time? or outdated file?
            message = ', '.join(format(v, '.0f') for v in vals)
            print(
                f'Warning: File {path.name!r} has {len(vals)} duplicate '
                f'time values: {message}. Kept only the first values.'
            )
    return data


def load_tables(path='~/data'):
    """
    Load forcing-feedback data from each source. Return a dictionary of dataframes.

    Parameters
    ----------
    path : path-like, optional
        The base path. Searches for a ``cmip-tables`` subfolder.
    """
    path = Path(path).expanduser()
    path = path / 'cmip-tables'
    if not path.is_dir():
        raise RuntimeError(f'Path {path!r} not found.')
    files = [file for ext in ('.txt', '.json') for file in path.glob('cmip*' + ext)]
    tables = {}
    for file in files:
        key = file.stem.split('_')[-1]
        if file.suffix == '.json':
            with open(file, 'r') as f:
                src = json.load(f)
            index = pd.MultiIndex.from_tuples([], names=('project', 'model', 'variant'))
            for project in ('CMIP5', 'CMIP6'):
                df = pd.DataFrame(index=index)
                for model, variants in src[project].items():
                    for variant, data in variants.items():
                        for name, value in data.items():
                            df.loc[(project, model, variant), name] = value
                tables[project + '-' + key] = df
        elif file.suffix == '.txt':
            if 'zelinka' in file.stem:
                pass
            else:
                df = pd.read_table(
                    file,
                    header=1,
                    skiprows=[2],
                    index_col=0,
                    delimiter=r'\s{2,}',
                    engine='python'
                )
                df.index = pd.MultiIndex.from_product(
                    (('CMIP5',), df.index, ('r1i1p1',)),
                    names=('project', 'model', 'variant')
                )
                tables['CMIP5' + '-' + key] = df
        else:
            raise RuntimeError(f'Unexpected path {file}.')
    for key, table in tables.items():
        print('Table: ' + key)
        print('Sensitivity and feedbacks: ' + ', '.join(table.columns))
        print('Number of models: ' + str(len(table.index.levels[1])))
        print('Model names: ', ', '.join(table.index.levels[1]))
    return tables


def load_datasets(path='~/data', **constraints):
    """
    Load CMIP variables for each model. Return a dictionary of datasets.

    Parameters
    ----------
    path : path-like, optional
        The data location. Passed to `_glob_files`.
    **constraints
        The constraints. Passed to `_glob_files`. Default is to take an average.
    """
    # NOTE: Currently standardize_coords renames bounds and converts coordinate
    # units but does not convert bounds units... however that is irrelavant since
    # we just need bounds for applying the weighting levels.
    monthly, seasonal, annual = {}, {}, {}
    for project in ('CMIP5', 'CMIP6'):
        # Iterate over unique models
        print(f'Project: {project}')
        files = _glob_files(path, **constraints)
        if not files:
            raise RuntimeError(
                f'Files with constraints {constraints} '
                f'not found in path {path}.'
            )
        models = tuple(file.name.split('_')[2] for file in files)
        print('Model:', end=' ')
        for i, model in enumerate(sorted(set(models))):
            # Model dataset
            ads = xr.Dataset()
            sds = xr.Dataset()
            mds = xr.Dataset()
            filtered = tuple(file for m, file in zip(models, files) if m == model)
            messages = []
            print(model, end=' (')
            for j, file in enumerate(filtered):
                # Load data
                var = file.name.split('_')[0]
                ds = xr.open_dataset(file, use_cftime=True)
                print(var, end='')
                if any(str(t) == 'NaT' for t in ds.time.values):  # time reading error
                    messages.append('Invalid time data.')
                    continue

                # Standardize coordinates and attributes
                ds = ds.climo.standardize_coords(verbose=False)
                ds = ds.climo.add_scalar_coords(verbose=False)  # e.g. add back lon
                for dim in ('lon', 'lat', 'lev'):
                    if ds.coords[dim].ndim > 1:
                        messages.append(f'Invalid {dim} data.')
                        continue
                if 'lev' in ds.dims and ds.lev[1] < ds.lev[0]:
                    ds = ds.isel(lev=slice(None, None, -1))  # always ascending
                    if 'bnds' in ds.dims:
                        ds = ds.isel(bnds=slice(None, None, -1))
                ds = ds.climo.add_cell_measures(verbose=False)
                da = ds[var].squeeze(drop=True)  # e.g. already averaged longitude
                if var[:3] in ('clw', 'cli'):
                    if da.attrs['units'] == 'kg m-2':
                        raise RuntimeError('Unexpected units for cloud water.')
                print('', da.attrs['units'], end='')
                ds.close()

                # Calculate annual and seasonal zonal averages
                ads[var] = da.mean('time', keep_attrs=True)
                sds[var] = da.groupby('time.season').mean('time', keep_attrs=True)
                mds[var] = da.climo.replace_coords(time=da['time.month']).rename(time='month')  # noqa: E501
                if 'lev_bnds' in sds:  # bugfix caused by groupby
                    sds['lev_bnds'] = sds['lev_bnds'].isel(season=0)
                print('', end='' if j == len(filtered) - 1 else ', ')

            # Save results
            if messages := set(messages):  # skipped critical file above
                warnings.warn(f'Skipping model {model!r}. ' + ' '.join(messages))
                continue
            key = project + '-' + model
            annual[key] = ads
            seasonal[key] = sds
            monthly[key] = mds
            print(')', end='\n' if i == len(set(models)) - 1 else ', ')
        print(f'Number of models: {len(set(models))}.')

    return monthly, seasonal, annual


def concat_datasets(datasets, verbose=False):
    """
    Concatenate datasets along a model dimension
    accounting for missing variables.

    Parameters
    ----------
    datasets : dict
        The datasets passed as dictionaries.
    """
    if not isinstance(datasets, dict) or not all(
        isinstance(_, xr.Dataset) for _ in datasets.values()
    ):
        raise ValueError('Invalid input. Must be dictionary of datasets.')
    names = sorted(set(name for ds in datasets.values() for name in ds.data_vars))
    print('Concatenating datasets...')
    if verbose:
        print('Variables:', *names)
    for ds in datasets.values():  # interpolated datasets
        for name in names:
            if name in ds:
                continue
            da = tuple(ds.data_vars.values())[0]
            da = xr.full_like(da, np.nan)
            da.attrs.clear()
            ds[name] = da
    models = xr.DataArray(
        list(datasets),
        dims='model',
        name='model',
        attrs={'long_name': 'CMIP version and model ID'}
    )
    concat = xr.concat(
        datasets.values(),
        dim=models,
        coords='minimal',
        compat='equals',  # problems with e.g. both surface and integrated values
        combine_attrs='drop_conflicts',  # drop e.g. history but keep e.g. long_name
    )
    for suffix in ('', 'vi'):  # in-place and vertially integrated
        ice = 'cli' + suffix  # solid
        water = 'clw' + suffix  # liquid + solid (read description)
        if ice in concat and water in concat:
            da = 100 * concat[ice] / concat[water]
            da = da.clip(0, 100)
            da.attrs['long_name'] = 'ice water ratio'
            da.attrs['units'] = '%'
            concat['clr' + suffix] = da
    for name in concat.data_vars:
        da = concat[name]
        if da.climo.units == ureg.Pa:
            da.attrs['standard_units'] = 'hPa'  # default standard units
        if 'title' not in da.attrs:
            pass
        elif 'long_name' not in da.attrs:
            da.attrs['long_name'] = da.attrs['title'].lower()
        else:
            del da.attrs['title']
        if da.climo.standard_units:
            concat[name] = da.climo.to_standard_units()
    return concat.climo.add_cell_measures(verbose=False)


def concat_tables(tables, datasets):
    """
    Concatenate pandas tables of feedback values along with the input
    dataset.

    Parameters
    ----------
    tables : dict
        A dictionary of dataframes with feedback and sensitivity info. The
        keys should indicate the source.
    datasets : dict or xarray.Dataset
        A dictionary of datasets with cliamte info. The keys should indicate
        the model.

    Returns
    -------
    dataset : xarray.Dataset
        The combined feedback and climate dataset. Contains ``'model'``
        dimension whose values are built with ``project-model'``.
    """
    # Combine datasets
    # WARNING: Critical to have same variables in each dataset
    if not isinstance(tables, dict) or not all(
        isinstance(_, pd.DataFrame) for _ in tables.values()
    ):
        raise ValueError('Invalid input. Must be dictionary of dataframes.')
    print('Adding feedbacks and sensitivity...')
    message = lambda label, array: None if len(array) == 0 else print(
        f'{len(array)} {label}:', ', '.join(map(repr, sorted(array)))
    )
    if isinstance(datasets, xr.Dataset):
        dataset = datasets
    else: # possibly raise type error inside this
        dataset = concat_datasets(datasets)
    for project in ('CMIP5', 'CMIP6'):
        for src, table in tables.items():
            # Filter the table values
            try:
                table = table.xs(project, level='project')
            except KeyError:
                continue  # project not in this table
            added = set()
            initial = set(m for m in table.index.levels[0] if ' ' not in m)
            for model, variant in table.index:
                if ' ' in model:  # human-readable summary statistic
                    continue
                if variant[:6] == 'r1i1p1' and variant[-1] == '1' and model not in added:  # noqa: E501
                    added.add(model)
            table = table.loc[added, :]  # automatically assumes first index
            final = set(m for m in table.index.levels[0] if project + '-' + m in dataset.model.values)  # noqa: E501
            table = table.loc[final, :]
            # Add the values to the dataset
            for var, series in table.items():
                key = src.split('-', 1)[1] + '_' + var.lower()
                if key not in dataset:
                    full = np.full(dataset.sizes['model'], np.nan)
                    dataset[key] = ('model', full)
                    dataset[key].attrs.update(
                        long_name=var.upper(),
                        units=(
                            'K' if var[:3] == 'ECS'
                            else 'W / m^2' if var[0] == 'F'
                            else 'W / m^2 / K'
                        ),
                    )
                index = series.index.get_level_values('model')
                index = [project + '-' + m for m in index]
                dataset[key].loc[index] = series.values
            # Messages
            print(f'Table {src!r}...')
            message('initial models', () if final == initial else initial)
            message('from other variant', initial - added)
            message('missing models', initial - final)
            message('final models', final)

    return dataset
