#!/usr/bin/env python
"""
Load various CMIP datasets.
"""
import json
import pathlib
import warnings

import numpy as np
import climopy as climo  # noqa: F401  # add accessor
import pandas as pd
import xarray as xr

from climopy import ureg


# Global constants
ZELINKA_FEEDBACKS = [
    'ECS',
    'ERF2x',
    'PL',
    'PL*',  # constant relative humidity
    'LR',
    'LR*',  # constant relative humidity
    'WV',
    'RH',
    'ALB',
    'CLD',
    'SWCLD',
    'LWCLD',
    'NET',  # kernel total
    'ERR',  # kernel residual
]


def load_feedbacks_sensitivity(
    version=5,
    path='~/data/cmip56_forcing_feedback_ecs/cmip56_forcing_feedback_ecs.json',
    # name='ECS',
    # experiment='r1i1p1',
):
    """
    Load a dataframe of model sensitivities.
    """
    path = pathlib.Path(path)
    path = path.expanduser()
    with open(path, 'r') as f:
        src = json.load(f)
        src = src[f'CMIP{version}']
    data = pd.DataFrame(index=tuple(src), columns=ZELINKA_FEEDBACKS)
    for model, experiments in src.items():
        # print(f'{model} options: ' + ', '.join(experiments))
        experiment = tuple(experiments)[0]  # always use the first one?
        for key, value in experiments[experiment].items():
            data.loc[model, key] = value
    print('Sensitivity and feedbacks: ' + ', '.join(data.index))
    print(f'Number of datasets: {len(data)}')  # number of columns
    return data


def load_cross_sections(
    name='ta', forcing='piControl', path='~/data/cmip5_Amon'
):
    """
    Load a single CMIP variable.
    """
    path = pathlib.Path(path)
    path = path.expanduser()
    files = [
        file for file in path.glob('*.nc')
        if file.name.startswith('_'.join((name, forcing)))
    ]
    monthly, seasonal, annual = {}, {}, {}
    for file in files:
        # Load data
        ds = xr.open_dataset(file, use_cftime=True).squeeze(drop=True)
        _, model = file.name.split('-', maxsplit=1)
        model, _ = model.split('-mon.nc', maxsplit=1)  # ignore suffix
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
        ds = ds.climo.add_scalar_coords(verbose=True)
        ds = ds.climo.add_cell_measures(verbose=True)
        # Time averages
        ads = ds.mean('time', keep_attrs=True)
        sds = ds.groupby('time.season').mean('time', keep_attrs=True)
        if 'lev_bnds' in sds:
            sds['lev_bnds'] = sds['lev_bnds'].isel(season=0)
        mds = ds.climo.replace_coords(time=ds['time.month']).rename(time='month')
        # Save results
        monthly[model] = mds
        seasonal[model] = sds
        annual[model] = ads
    print(f'Variable {name!r} forcing {forcing!r}: ' + ', '.join(monthly))
    print(f'Number of datasets: {len(monthly)}.')
    return monthly, seasonal, annual


def concat_datasets(table, datasets, lat=10, lev=50):
    """
    Interpolate and concatenate dictionaries of datasets.
    """
    if not isinstance(table, pd.DataFrame):
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
    print('Interpolating datasets...')
    concat = {}
    for model, dataset in datasets.items():
        for key in ('lev_bnds', 'cell_width', 'cell_depth', 'cell_height'):
            if key in dataset:
                dataset = dataset.drop_vars(key)
        interp = dataset.interp(lat=lat, lev=lev)
        concat[model] = interp
    print('Concatenating datasets...')
    models = xr.DataArray(
        list(concat),
        dims='model',
        name='model',
        attrs={'long_name': 'CMIP model id'}
    )
    concat = xr.concat(
        concat.values(),
        dim=models,
        coords='minimal',
        compat='equals',
    )
    concat = concat.climo.add_cell_measures(verbose=True)
    print('Adding feedbacks and sensitivity...')
    models = [model for model in table.index if model in concat.coords['model']]
    missing = set(table.index) - set(models)
    print('Found models: ' + ', '.join(map(repr, sorted(models))))
    if missing:
        print('Missing models: ' + ', '.join(map(repr, sorted(missing))))
    table = table.loc[models, :]
    for name, series in table.items():
        data = np.full(concat.sizes['model'], np.nan)
        concat[name] = ('model', data)
        concat[name].loc[series.index] = series.values
    return concat
