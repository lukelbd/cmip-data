#!/usr/bin/env python
"""
Load various CMIP datasets.
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from icecream import ic  # noqa: F401

import climopy as climo  # noqa: F401  # add accessor
from climopy import ureg

# Constants
DATA = Path.home() / 'data'


def load_cmip_tables():
    """
    Load forcing-feedback data from each source. Return a dictionary of dataframes.
    """
    path = DATA / 'cmip-tables'
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
        print('Number of models: ' + str(len(table.index)))  # number of columns
    return tables


def load_cmip_xsections(name='ta', experiment='piControl', table='Amon'):
    """
    Load CMIP variables for each model. Return a dictionary of datasets.
    """
    # TODO: Support arbitrary multiple variables, experiments,
    # etc. rather than just multiple projects and models.
    monthly, seasonal, annual = {}, {}, {}
    for project in ('CMIP5', 'CMIP6'):
        path = DATA / f'{project.lower()}-{experiment}-{table}-avg'
        models = set()
        if not path.is_dir():
            raise RuntimeError(f'Path {path!s} not found.')
        for file in path.glob(f'{name}_{table}_*_{experiment}_{project.lower()}.nc'):
            # Load data
            ds = xr.open_dataset(file, use_cftime=True).squeeze(drop=True)
            _, _, model, *_ = file.name.split('_')
            if any(str(t) == 'NaT' for t in ds.time.values):  # no time reading error
                warnings.warn(f'Ignoring model {model!r}. Invalid time data.')
                continue
            # Standardize coordinates and measures
            ds = ds.rename(plev='lev')
            if ds.lev[1] < ds.lev[0]:
                ds = ds.isel(lev=slice(None, None, -1))  # always ascending
                if 'bnds' in ds.dims:
                    ds = ds.isel(bnds=slice(None, None, -1))
            if ds.lev.climo.units == ureg.Pa:
                ds = ds.climo.replace_coords(lev=ds.lev / 100)
                ds['lev'].attrs['units'] = 'hPa'
            if 'plev_bnds' in ds:
                ds = ds.rename(plev_bnds='lev_bnds')
                ds['lev'].attrs['bounds'] = 'lev_bnds'
            ds = ds.climo.add_scalar_coords(verbose=False)
            ds = ds.climo.add_cell_measures(verbose=False)
            # Time averages
            ads = ds.mean('time', keep_attrs=True)
            sds = ds.groupby('time.season').mean('time', keep_attrs=True)
            if 'lev_bnds' in sds:
                sds['lev_bnds'] = sds['lev_bnds'].isel(season=0)
            mds = ds.climo.replace_coords(time=ds['time.month']).rename(time='month')
            # Save results
            key = project + '-' + model
            models.add(model)
            monthly[key] = mds
            seasonal[key] = sds
            annual[key] = ads
        print(f'Project {project!r} experiment {experiment!r} variable {name!r}: ' + ', '.join(models))  # noqa: E501
        print(f'Number of models: {len(models)}.')
    return monthly, seasonal, annual


def concat_datasets(tables, datasets, lat=10, lev=50):
    """
    Interpolate and concatenate dictionaries of datasets. Input is dictionary of
    tables from different sources and dictionary of datasets from each model.
    """
    if not isinstance(tables, dict) or any(not isinstance(_, pd.DataFrame) for _ in tables.values()):  # noqa: E501
        raise ValueError('Invalid input. Must be dataframe.')
    if not isinstance(datasets, dict) or any(not isinstance(_, xr.Dataset) for _ in datasets.values()):  # noqa: E501
        raise ValueError('Invalid input. Must be dictionary of datasets.')
    if np.iterable(lat):
        pass
    elif 90 % lat == 0:
        lat = np.arange(-90, 90 + lat / 2, lat)
    else:
        raise ValueError(f'Invalid {lat=}.')
    if np.iterable(lev):
        pass
    elif 1000 % lev == 0:
        lev = np.arange(lev, 1000 + lev / 2, lev)  # not including zero
    else:
        raise ValueError(f'Invalid {lev=}.')

    # Standardize datasets
    print('Interpolating datasets...')
    concat = {}
    for model, dataset in datasets.items():
        for key in ('lev_bnds', 'cell_width', 'cell_depth', 'cell_height'):
            if key in dataset:
                dataset = dataset.drop_vars(key)
        interp = dataset.interp(lat=lat, lev=lev)
        concat[model] = interp

    # Combine datasets
    print('Concatenating datasets...')
    models = xr.DataArray(
        list(concat),
        dims='model',
        name='model',
        attrs={'long_name': 'CMIP version and model ID'}
    )
    concat = xr.concat(
        concat.values(),
        dim=models,
        coords='minimal',
        compat='equals',
    )
    concat = concat.climo.add_cell_measures(verbose=False)

    # Add tables
    print('Adding feedbacks and sensitivity...')
    for project, variant in zip(('CMIP5', 'CMIP6'), ('r1i1p1', 'r1i1p1f1')):
        print(f'Project {project!r}...')
        for src, table in tables.items():
            try:
                table = table.xs((project, variant), level=('project', 'variant'))
            except KeyError:
                continue
            table = table.drop(m for m in table.index if ' ' in m)  # summary statistics
            models = [m for m in table.index if project + '-' + m in concat.coords['model']]  # noqa: E501
            missing = set(table.index) - set(models)
            if True:
                print(f'{src} models ({len(models)}): ' + ', '.join(map(repr, sorted(models))))  # noqa: E501
            elif missing:
                print(f'{src} missing ({len(missing)}): ' + ', '.join(map(repr, sorted(missing))))  # noqa: E501
            table = table.loc[models, :]
            for var, series in table.items():
                key = src.split('-', 1)[1] + '_' + var.lower()
                if key not in concat:
                    full = np.full(concat.sizes['model'], np.nan)
                    concat[key] = ('model', full)
                index = [project + '-' + m for m in series.index]
                concat[key].loc[index] = series.values

    return concat
