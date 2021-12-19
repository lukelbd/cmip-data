#!/usr/bin/env python
"""
Load various CMIP datasets.
"""
import json
import warnings
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from icecream import ic  # noqa: F401

import climopy as climo  # noqa: F401  # add accessor
from climopy import ureg

# Constants and utilities
DATA = Path.home() / 'data'
if sys.platform == 'darwin':
    ROOT = Path.home() / 'data'
else:  # TODO: add conditionals?
    ROOT = Path('/mdata5') / 'ldavis'

# Results of get_facet_options() called on SearchContext(project='CMIP5')
# and SearchContext(project='CMIP6') using https://esgf-node.llnl.gov/esg-search
# for the SearchConnection URL. Conventions changed between projects so e.g.
# 'experiment', 'ensemble', 'cmor_table', and 'time_frequency' in CMIP5 must be
# changed to 'experiment_id', 'variant_label', 'table_id', and 'frequency' in CMIP6.
# Note 'member_id' is equivalent to 'variant_label' if 'sub_experiment_id' is unset
# and for some reason 'variable' and 'variable_id' are kepts synonyms in CMIP5.
# URL https://esgf-node.llnl.gov/esg-search:     11900116 hits for CMIP6 (use this one!)
# URL https://esgf-data.dkrz.de/esg-search:      01009809 hits for CMIP6
# URL https://esgf-node.ipsl.upmc.fr/esg-search: 01452125 hits for CMIP6
CMIP5_FACETS = [
    'access', 'cera_acronym', 'cf_standard_name', 'cmor_table', 'data_node',
    'ensemble', 'experiment', 'experiment_family', 'forcing', 'format',
    'index_node', 'institute', 'model', 'product', 'realm', 'time_frequency',
    'variable', 'variable_long_name', 'version'
]
CMIP6_FACETS = [
    'access', 'activity_drs', 'activity_id', 'branch_method', 'creation_date',
    'cf_standard_name', 'data_node', 'data_specs_version', 'datetime_end',
    'experiment_id', 'experiment_title', 'frequency', 'grid', 'grid_label',
    'index_node', 'institution_id', 'member_id', 'nominal_resolution', 'realm',
    'short_description', 'source_id', 'source_type', 'sub_experiment_id', 'table_id',
    'variable', 'variable_id', 'variable_long_name', 'variant_label', 'version'
]


def download_cmip_wgets(**kwargs):  # noqa: U100
    """
    Download wget files using pyesgf. Will eventually repace download_wgets.sh. This
    will leverage utilities eventually built into climopy.
    """
    # Log on with OpenID
    # LLNL: https://esgf-node.llnl.gov/
    # CEDA: https://esgf-index1.ceda.ac.uk/
    # DKRZ: https://esgf-data.dkrz.de/
    # GFDL: https://esgdata.gfdl.noaa.gov/
    # IPSL: https://esgf-node.ipsl.upmc.fr/
    # JPL:  https://esgf-node.jpl.nasa.gov/
    # LIU:  https://esg-dn1.nsc.liu.se/
    # NCI:  https://esgf.nci.org.au/
    # NCCS: https://esgf.nccs.nasa.gov/
    # Nodes listed here: https://esgf.llnl.gov/nodes.html
    from pyesgf.logon import LogonManager
    lm = LogonManager()
    host = 'esgf-node.llnl.gov'
    if not lm.is_logged_on():
        lm.logon(username='lukelbd', password=None, hostname=host)

    # Iterate over tables
    from pyesgf.search import SearchConnection
    url = 'https://esgf-node.llnl.gov/esg-search'
    conn = SearchConnection(url, distrib=True)
    cmip6 = conn.new_context(project='CMIP6')
    cmip5 = conn.new_context(project='CMIP5')
    cmip6_control = cmip6.constrain(
        experiment_id=['piControl'],
        variable_id=['ta'],
        variant_label=['r1i1p1f1'],
        table_id=['Amon'],
    )
    cmip5_control = cmip5.constrain(
        experiment=['piControl'],
        variable=['ta'],
        ensemble=['r1i1p1'],
        cmor_table=['Amon'],
    )
    cmip6_response = cmip6.constrain(
        experiment_id=['abrupt-4xCO2'],
        variable_id=['rlut', 'rsut', 'rlutcs', 'rsutcs', 'tas'],
        variant_label=['r1i1p1f1'],
        table_id=['Amon'],
    )
    cmip5_response = cmip5.constrain(
        experiment=['abrupt4xCO2'],  # no dash
        variable=['rlut', 'rsut', 'rlutcs', 'rsutcs', 'tas'],
        ensemble=['r1i1p1'],
        cmor_table=['Amon'],
    )
    ctxs = (cmip6_control, cmip5_control, cmip6_response, cmip5_response)
    print(cmip5.facet_constraints)
    for i in range(4):
        ctx = ctxs[i]
        print(f'Context {i}:', ctx, ctx.facet_constraints)
        print(f'Hit count {i}:', ctx.hit_count)
        keys = ('project', ('experiment', 'experiment_id'), ('cmor_table', 'table_id'))
        parts = []
        for key in keys:  # constraint components to use in file name
            key = (key,) if isinstance(key, str) else key
            opts = sum((ctx.facet_constraints.getall(k) for k in key), start=[])
            if not opts:
                raise RuntimeError
            elif len(opts) > 1:
                raise NotImplementedError
            part = opts[0].replace('-', '')
            if 'project' in key:
                part = part.lower()
            parts.append(part)
        for j, ds in enumerate(ctx.search()):  # iterate over models and dates
            print(f'Dataset {j}:', ds)
            fc = ds.file_context()
            wget = fc.get_download_script()
            name = 'wget_' + '-'.join((*parts, str(j))) + '.sh'
            path = Path(ROOT, 'wgets', name)
            with open(path, 'w') as f:
                f.write(wget)
            print('Created:', name)


def load_cmip_tables(project='cmip5'):
    """
    Load forcing-feedback data from each source. Return a dictionary of dataframes.
    """
    path = DATA / 'cmip-tables'
    paths = [path for ext in ('.txt', '.json') for path in dir.glob('cmip*' + ext)]
    tables = {}
    for path in paths:
        if path.suffix == '.json':
            with open(path, 'r') as f:
                src = json.load(f)
            index = pd.MultiIndex.from_tuples([], names=('model', 'variant'))
            df = pd.DataFrame(index=index)
            for model, variants in src[project.upper()].items():
                for variant, data in variants.items():
                    for name, value in data.items():
                        df.loc[(model, variant), name] = value
            tables[path.stem.split('_')[-1]] = df
        elif path.suffix == '.txt':
            if project[-1] != '5' or 'zelinka' in path.stem:
                pass
            else:
                df = pd.read_table(
                    path,
                    header=1,
                    skiprows=[2],
                    index_col=0,
                    delimiter=r'\s{2,}',
                    engine='python'
                )
                df.index = pd.MultiIndex.from_product(
                    (df.index, ('r1i1p1',)), names=('model', 'variant')
                )
                tables[path.stem.split('_')[-1]] = df
        else:
            raise RuntimeError(f'Unexpected path {path}.')
    for path, table in tables.items():
        print('Table: ' + path)
        print('Sensitivity and feedbacks: ' + ', '.join(table.columns))
        print('Number of models: ' + str(len(table.index)))  # number of columns
    return tables


def load_cmip_xsections(
    name='ta', project='cmip5', experiment='piControl', table='Amon',
):
    """
    Load CMIP variables for each model. Return a dictionary of datasets.
    """
    path = DATA / f'{project}-{experiment}-{table}-avg'
    if not path.is_dir():
        raise RuntimeError(f'Path {path!s} not found.')
    monthly, seasonal, annual = {}, {}, {}
    for file in path.glob(f'{name}_{table}_*_{experiment}_{project}.nc'):
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
    print(f'Variable {name!r} experiment {experiment!r}: ' + ', '.join(monthly))
    print(f'Number of models: {len(monthly)}.')
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
    for src, table in tables.items():
        if isinstance(table.index, pd.MultiIndex):
            table = table.xs('r1i1p1', level='variant')
        table = table.drop(m for m in table.index if ' ' in m)
        models = [m for m in table.index if m in concat.coords['model']]
        missing = set(table.index) - set(models)
        if True:
            print(f'{src.title()} models ({len(models)}): ' + ', '.join(map(repr, sorted(models))))  # noqa: E501
        if missing:
            print(f'{src.title()} missing ({len(missing)}): ' + ', '.join(map(repr, sorted(missing))))  # noqa: E501
        table = table.loc[models, :]
        for name, series in table.items():
            data = np.full(concat.sizes['model'], np.nan)
            name = '_'.join((src, name.lower()))
            concat[name] = ('model', data)
            concat[name].loc[series.index] = series.values
    return concat
