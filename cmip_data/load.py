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
import metpy.interpolate as interp
from icecream import ic  # noqa: F401

import climopy as climo  # noqa: F401  # add accessor
from climopy import ureg

# Constants
DATA = Path.home() / 'data'


def combine_cam_kernels(
    path='~/data/kernels-cam5/cam5_plev.nc',
    # ref='~/data/timescales_constants/hs1_t85l60e.nc',
    month=None,
    season=None,
    shortwave=True,
):
    """
    Create averages of

    Parameters
    ----------
    path : path-like, optional
        CAM data location.
    month : month-spec, optional
        Month(s) to be averaged.
    season : season-spec, optional
        Season(s) to be averaged.
    shortwave : bool, optional
        Whether to include shortwave components.
    """
    # Load kernel data, fixing sign convention so both LW and SW are +ve toward
    # atmosphere (default is SW positive down, LW positive up, understandably).
    # Also get cloudy-sky and net atmospheric components.
    # NOTE: See load.py:load_cam for details
    # NOTE: Xarray caches already-loaded datasets so this is fast after one run
    print('Calculating averages...')
    ds_cam5 = xr.open_dataset(path)
    keep = ('ps', 'bnd', 'fln') + (('fsn',) if shortwave else ())
    drop = tuple(var for var in ds_cam5 if not any(s in var for s in keep))
    ds_cam5 = ds_cam5.drop_vars(drop)  # drop surface kernels, and maybe shortwave
    ds_cam5 = ds_cam5.climo.sel_time(season=season, month=month)
    ds_cam5 = ds_cam5.climo.mean('time')
    ds_cam5['ps'].attrs['standard_name'] = 'surface_air_pressure'
    with xr.set_options(keep_attrs=True):
        for var in ds_cam5:
            if 'flnt' in var or 'fsns' in var:
                ds_cam5[var] *= -1
        for var in ('t_fln', 'q_fln', 'q_fsn', 'ts_fln', 'alb_fsn'):
            for s in ('c', ''):
                if var + 't' + s in ds_cam5 and var + 's' + s in ds_cam5:
                    ds_cam5[var + 'a' + s] = ds_cam5[var + 't' + s] + ds_cam5[var + 's' + s]  # noqa: E501
            for s in ('t', 's', 'a'):
                if var + s + 'c' in ds_cam5 and var + s in ds_cam5:
                    ds_cam5[var + s + 'l'] = ds_cam5[var + s] - ds_cam5[var + s + 'c']
    ds_cam5 = ds_cam5.climo.standardize_coords()
    ds_cam5 = ds_cam5.climo.add_cell_measures(verbose=True)
    ds_cam5 = ds_cam5.climo.quantify()
    return ds_cam5


def load_cam_kernels(
    path='~/data/kernels-cam5/',
    folders=('forcing', 'kernels'),
    ref='~/data/timescales_constants/hs1_t85l60e.nc',
):
    """
    Load the CAM forcing and feedback kernel data.

    Parameters
    ----------
    base : path-like
        The base directory in which `folders` is indexed.
    ref : path-like
        The path used for destination parameters.
    """
    # Get list of files
    ps = set()
    ref = Path(ref).expanduser()
    path = Path(path).expanduser()
    files = [file for folder in folders for file in (path / folder).glob('*.nc')]
    files = [file for file in files if file.name != 'PS.nc' or ps.add(file)]
    if not ps:
        raise RuntimeError('Surface pressure data not found.')

    # Load data using PS.nc for time coords (times are messed up on other datasets)
    print('Loading data...')
    ignore = ('gw', 'nlon', 'ntrk', 'ntrm', 'ntrn', 'w_stag', 'wnummax')
    ds_mlev = xr.Dataset()  # will add variables one-by-one
    for file in (*ps, *files):  # put PS.nc first
        data = xr.open_dataset(file, use_cftime=True)
        time_dim = data.time.dims[0]
        if 'ncl' in time_dim:  # fix issue where 'time' coord dim is an NCL dummy name
            data = data.swap_dims({time_dim: 'time'})
        if file.name != 'PS.nc':  # ignore bad time indices
            data = data.reset_index('time', drop=True)
        var, *_ = file.name.split('.')  # first thing with dot
        for name, da in data.items():  # iterates through variables, not coordinates
            if name in ignore:
                continue
            name = name.lower()
            if name[:2] in ('fl', 'fs'):  # longwave or shortwave radiative flux
                name = var + '_' + name
            ds_mlev[name] = da

    # Normalize by pressure thickness and standardize units
    print('Normalizing data...')
    pi = ds_mlev.hyai * ds_mlev.p0 + ds_mlev.hybi * ds_mlev.ps
    pi = pi.transpose('time', 'ilev', 'lat', 'lon')
    pm = 0.5 * (pi.data[:, 1:, ...] + pi.data[:, :-1, ...])
    dp = pi.data[:, 1:, ...] - pi.data[:, :-1, ...]
    for name, da in ds_mlev.items():
        if 'lev' in da.dims and da.ndim > 1:
            da.data /= dp
            da.attrs['units'] = 'W m^-2 K^-1 Pa^-1'
        elif name[:2] == 'ts':
            da.attrs['units'] = 'W m^-2 K^-1'
        elif name[:3] == 'alb':
            da.attrs['units'] = 'W m^-2 %^-1'
        elif da.attrs.get('units', None) == 'W/m2':
            da.attrs['units'] = 'W m^-2'
    dims = ('time', 'lev', 'lat', 'lon')
    attrs = {'units': 'Pa', 'long_name': 'approximate pressure thickness'}
    ds_mlev['lev_delta'] = (dims, dp, attrs)
    ds_mlev.lon.attrs['axis'] = 'X'
    ds_mlev.lat.attrs['axis'] = 'Y'
    ds_mlev.lev.attrs['axis'] = 'Z'
    ds_mlev.time.attrs['axis'] = 'T'

    # Interpolate onto standard pressure levels using metpy
    # NOTE: Also normalize by pressure level thickness, giving units W m^-2 100hPa^-1
    # TODO: Are copies of data arrays necessary?
    ds_hs94 = xr.open_dataset(ref, decode_times=False)
    if 'plev' in ds_hs94.coords:
        ds_hs94 = ds_hs94.rename(plev='lev', plev_bnds='lev_bnds')
    ds_plev = xr.Dataset(
        coords={
            'lon': ds_mlev.lon,
            'lat': ds_mlev.lat,
            'lev': ds_hs94.lev,
            'time': ds_mlev.time,
        }
    )
    ds_plev['lev'].attrs['bounds'] = 'lev_bnds'
    ds_plev['lev_bnds'] = ds_hs94.lev_bnds
    for name, da in ds_mlev.items():
        if 'lev' in da.dims and da.ndim > 1 and name != 'lev_delta':
            print(f'Interpolating {name!r}...')
            dims = list(da.dims)
            shape = list(da.shape)
            dims[1] = 'lev'
            shape[1] = ds_plev.lev.size
            data = np.empty(shape)
            for i in range(shape[0]):  # iterate through times
                for j in range(shape[1]):  # iterate through levels
                    data[i, j, ...] = interp.interpolate_to_isosurface(
                        pm[i, ...], da.data[i, ...], 100 * ds_plev.lev.data[j]
                    )
            ds_plev[name] = (dims, data, da.attrs)
        elif not any(dim in da.dims for dim in ('lev', 'ilev')):
            print(f'Copying {name!r}.')
            ds_plev[name] = da

    # Save and return
    try:
        ds_mlev.to_netcdf(f'{path}/cam5_mlev.nc')
        ds_plev.to_netcdf(f'{path}/cam5_plev.nc')
    except Exception:
        warnings.warn('Failed to save datasets. Possible permissions error.')
    return ds_mlev, ds_plev


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
        print('Number of models: ' + str(len(table.index.levels[1])))
        print('Model names: ', ', '.join(table.index.levels[1]))
    return tables


def load_cmip_xsections(
    name='ta', experiment='piControl', table='Amon', mode='latlev'
):
    """
    Load CMIP variables for each model. Return a dictionary of datasets.
    """
    # TODO: Support arbitrary multiple variables, experiments,
    # etc. rather than just multiple projects and models.
    names = name
    if isinstance(names, str):
        names = (names,)
    monthly, seasonal, annual = {}, {}, {}
    for project in ('CMIP5', 'CMIP6'):
        print(f'Project: {project}')
        path = DATA / f'{project.lower()}-{experiment}-{table}'
        if not path.is_dir():
            raise RuntimeError(f'Path {path!s} not found.')
        files = sorted(file for name in names for file in path.glob(f'{name}_{table}_*_{experiment}_{project.lower()}.nc'))  # noqa: E501
        models = tuple(file.name.split('_')[2] for file in files)
        print('Model:', end=' ')
        for i, model in enumerate(sorted(set(models))):
            # Model dataset
            ads = xr.Dataset()
            sds = xr.Dataset()
            mds = xr.Dataset()
            mfiles = tuple(file for m, file in zip(models, files) if m == model)
            mnames = tuple(file.name.split('_')[0] for file in mfiles)
            messages = []
            print(model, end=' (')
            for j, (name, file) in enumerate(zip(mnames, mfiles)):
                # Load data
                name = file.name.split('_')[0]
                ds = xr.open_dataset(file, use_cftime=True)
                print(name, end='')
                if any(str(t) == 'NaT' for t in ds.time.values):  # time reading error
                    messages.append('Invalid time data.')
                    continue

                # Standardize coordinates and attributes
                # TODO: Translate latitude_bnds and longitude_bnds
                # NOTE: Critical to work with dataset to retain bounds variables
                # NOTE: This renames bounds and converts coordinate units but does not
                # convert bounds units... however that is irrelavant since we just
                # need bounds for applying the weighting levels.
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
                da = ds[name].squeeze(drop=True)  # e.g. already averaged longitude
                if name[:3] in ('clw', 'cli'):
                    if da.attrs['units'] == 'kg m-2':
                        raise RuntimeError('Unexpected units for cloud water.')
                print('', da.attrs['units'], end='')
                ds.close()

                # Calculate annual and seasonal zonal averages
                # TODO: Re-compute with 100 years instead of 50 years.
                # TODO: Re-download data that only exists in longitude
                # means originally computed and transfered from laptop.
                # if any(dim not in da.coords for dim in ('lon', 'lat', 'lev')):
                if any(dim not in da.coords for dim in ('lon', 'lat',)):
                    messages.append('Missing spatial coordinates.')
                    continue
                if mode == 'latlev':  # cross-section
                    if 'lon' in da.dims:
                        da = da.mean('lon', keep_attrs=True)
                    elif da.coords['lon'].size == 1:  # TODO: re-download and remove
                        pass
                    else:
                        messages.append('Missing lon dimension.')
                        continue
                elif mode == 'lonlat':
                    if 'lon' not in da.dims:
                        messages.append('Missing lon dimension.')
                        continue
                    elif da.name in ('hur', 'hus'):  # weighted average with cell_height
                        da = da.climo.average('lev', keep_attrs=True)
                    elif 'lev' in da.dims:  # surface level selection
                        sfc = da.coords['lev'].max(keep_attrs=True)
                        da = da.sel(lev=sfc.item())
                        if any(name in ('hur', 'hus') for name in names):  # kludge
                            lab = fr'{sfc.item():.0f}$\,${sfc.climo.units_label}'
                            da.attrs['long_suffix'] = lab
                            da = da.climo.replace_coords(lev=np.nan)
                else:
                    raise RuntimeError(f'Invalid reduction mode {mode!r}.')
                ads[name] = da.mean('time', keep_attrs=True)
                sds[name] = da.groupby('time.season').mean('time', keep_attrs=True)
                mds[name] = da.climo.replace_coords(time=da['time.month']).rename(time='month')  # noqa: E501
                if 'lev_bnds' in sds:  # bugfix caused by groupby
                    sds['lev_bnds'] = sds['lev_bnds'].isel(season=0)
                print('', end='' if j == len(mnames) - 1 else ', ')

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


def concat_datasets(tables, datasets, lat=10, lon=20, lev=50):
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
    if np.iterable(lon):
        pass
    elif 360 % lon == 0:
        lon = np.arange(0, 360 + lon / 2, lon)
    else:
        raise ValueError(f'Invalid {lon=}.')
    if np.iterable(lev):
        pass
    elif 1000 % lev == 0:
        lev = np.arange(lev, 1000 + lev / 2, lev)  # not including zero
    else:
        raise ValueError(f'Invalid {lev=}.')

    # Interpolate datasets
    print('Interpolating datasets...')
    print('Model:', end=' ')
    concat = {}
    for i, (model, dataset) in enumerate(datasets.items()):
        print(model, end='\n' if i == len(datasets) - 1 else ', ')
        for key in ('lev_bnds', 'cell_width', 'cell_depth', 'cell_height'):
            if key in dataset:
                dataset = dataset.drop_vars(key)
        interp = {}
        for key, val in zip(('lon', 'lat', 'lev'), (lon, lat, lev)):
            if key in dataset.dims:  # not coords
                interp[key] = val
        if 'lev' not in interp and 'lev' in dataset.coords:  # round pressuer levs
            dataset = dataset.climo.replace_coords(lev=np.round(dataset.coords['lev']))
        concat[model] = dataset.interp(**interp)

    # Combine datasets
    # WARNING: Critical to have same variables in each dataset
    names = {name for ds in concat.values() for name in ds.data_vars}
    print('Concatenating datasets...')
    print(f'Variables names: {names}')
    for ds in concat.values():  # interpolated datasets
        for name in names:
            if name not in ds:
                da = tuple(ds.data_vars.values())[0]
                da = xr.full_like(da, np.nan)
                da.attrs.clear()
                ds[name] = da
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
        compat='equals',  # problems with e.g. both surface and integrated values
        combine_attrs='drop_conflicts',  # drop e.g. history but keep e.g. long_name
    )
    for suffix in ('', 'vi'):  # in-place and vertially integrated
        ice = 'cli' + suffix  # solid
        water = 'clw' + suffix  # liquid + solid (read description)
        if ice in names and water in names:
            da = 100 * concat[ice] / concat[water]
            da = da.clip(0, 100)
            da.attrs['long_name'] = 'ice water ratio'
            da.attrs['units'] = '%'
            concat['clr' + suffix] = da
    for name in names:
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
    concat = concat.climo.add_cell_measures(verbose=False)

    # Add tables
    # NOTE: Keep models that are only available with different forcing variants. This
    # still indicates simulations belonging to same branch but slightly different
    # forcing procedure. The control data downloaded is all r1i1p1 though so should
    # still match those values.
    print('Adding feedbacks and sensitivity...')
    message = lambda label, array: None if len(array) == 0 else print(
        f'{len(array)} {label}:', ', '.join(map(repr, sorted(array)))
    )
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
            final = set(m for m in table.index.levels[0] if project + '-' + m in models.values)  # noqa: E501
            table = table.loc[final, :]
            # Add the values to the dataset
            for var, series in table.items():
                key = src.split('-', 1)[1] + '_' + var.lower()
                if key not in concat:
                    full = np.full(concat.sizes['model'], np.nan)
                    concat[key] = ('model', full)
                    concat[key].attrs.update(
                        long_name=var.upper(),
                        units=(
                            'K' if var[:3] == 'ECS'
                            else 'W / m^2' if var[0] == 'F'
                            else 'W / m^2 / K'
                        ),
                    )
                index = series.index.get_level_values('model')
                index = [project + '-' + m for m in index]
                concat[key].loc[index] = series.values
            # Messages
            print(f'Table {src!r}...')
            message('initial models', () if final == initial else initial)
            message('from other variant', initial - added)
            message('missing models', initial - final)
            message('final models', final)

    return concat
