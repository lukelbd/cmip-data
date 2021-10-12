#!/usr/bin/env python
"""
Load various CMIP datasets.
"""
import json
import pathlib
import warnings

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


def load_table(
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


def load_netcdf(
    name='ta', forcing='piControl', path='~/data/cmip5_Amon'
):
    """
    Load a single CMIP variable.
    """
    path = pathlib.Path(path)
    path = path.expanduser()
    print(path.glob('*.nc'))
    files = [
        file for file in path.glob('*.nc')
        if file.name.startswith('_'.join((name, forcing)))
    ]
    datasets = {}
    for file in files:
        # Load data
        ds = xr.open_dataset(file, use_cftime=True).squeeze(drop=True)
        _, model = file.name.split('-', maxsplit=1)
        model, _ = model.split('-mon.nc', maxsplit=1)  # ignore suffix
        if not any(str(t) == 'NaT' for t in ds.time.values):  # no time reading error
            ds = ds.groupby('time.season').mean('time', keep_attrs=True)
        else:
            warnings.warn(f'Ignoring model {model!r}. Invalid time data.')
            continue
        ds.coords['model'] = model
        # Standardize vertical coordinates
        ds = ds.rename(plev='lev')
        if ds.lev.climo.units == ureg.Pa:
            ds = ds.climo.replace_coords(lev=ds.lev / 100)
            ds['lev'].attrs['units'] = 'hPa'
        if 'plev_bnds' in ds:
            ds = ds.rename(plev_bnds='lev_bnds')
            ds['lev'].attrs['bounds'] = 'lev_bnds'
        datasets[model] = ds
    print(f'Variable {name!r} forcing {forcing!r}: ' + ', '.join(datasets))
    print(f'Number of datasets: {len(datasets)}.')
    return datasets
