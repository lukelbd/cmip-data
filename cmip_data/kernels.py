#!/usr/bin/env python
"""
Standardize radiative kernel estimates into formats compatible with ESGF data.
"""
import warnings
from pathlib import Path

import climopy as climo  # noqa: F401  # add accessor
import numpy as np
import xarray as xr
from icecream import ic  # noqa: F401

from . import process

__all__ = [
    'combine_cam_kernels',
    'combine_era_kernels',
]


def _standardize_kernels(
    ds, path='~/data/kernels-standard', suffix=None, standardize=True, **kwargs
):
    """
    Standardize the radiative kernel data. Add cloudy sky and atmospheric kernels, take
    month selections or season averages, and optionally drop shortwave water vapor.

    Parameters
    ----------
    ds : xarray.Dataset
        The radiative kernel dataset.
    path : path-like, optional
        The location for the ouput kernels.
    suffix : str, optional
        The suffix for the output file. Will be ``f'kernels_{suffix}.nc'``.
    standardize : bool, optional
        Whether to standardize coordinates names and horizontal and vertical levels.
    **kwargs
        Passed to `standardize_horizontal` and `standardize_vertical`.
    """
    # Standardize coordinates and variables
    print('Standardizing kernel data...')
    suffix = suffix and '_' + suffix or ''
    path = Path(path).expanduser()
    path = path / f'kernels{suffix}.nc'
    vars_keep = ('ps', 'bnd', 'rln', 'rsn')
    vars_drop = tuple(var for var in ds if not any(s in var for s in vars_keep))
    ds = ds.drop_vars(vars_drop)  # drop surface kernels, and maybe shortwave kernels
    if standardize:
        ds = ds.climo.standardize_coords(
            pressure_units='Pa', descending_levels=True,  # match cmip conventions
        )
    with xr.set_options(keep_attrs=True):
        for var in ds:
            if 'rlnt' in var or 'rsns' in var:
                ds[var] *= -1
        for var in ('t_rln', 'ts_rln', 'hus_rln', 'hus_rsn', 'alb_rsn'):
            for sky in ('cs', ''):
                if f'{var}t{sky}' in ds and f'{var}s{sky}' in ds:
                    da = ds[f'{var}t{sky}'] + ds[f'{var}s{sky}']
                    nm = da.attrs['long_name']
                    da.attrs['long_name'] = (
                        nm.replace('at top of model', 'across atmospheric column')
                    )
                    ds[f'{var}a{sky}'] = da
            for lev in ('t', 's', 'a'):
                if var + lev + 'cs' in ds and var + lev in ds:
                    da = ds[var + lev] - ds[var + lev + 'cs']
                    nm = da.attrs['long_name']
                    da.attrs['long_name'] = (
                        nm.replace('Net', 'Cloud forcing')
                    )
                    ds[var + lev + 'cf'] = da

    # Standardize horizontal and vertical grid
    # TODO: Consider supporting xarray dataset input to the standardize functions
    # and then passing them to `cdo`, but difficult due to chained operators.
    path.unlink(missing_ok=True)
    ds = ds.climo.dequantify()
    ds.to_netcdf(path)
    print(tuple(ds))
    kwvert = {'levels': kwargs.pop('levels', None)}
    kwhori = {'gridspec': kwargs.pop('gridspec', None), 'method': kwargs.pop('method', None)}  # noqa: E501
    if kwargs:
        raise ValueError(f'Invalid keyword args {kwargs}.')
    if standardize:
        print('Standardizing kernel coordinates...')
        temp = path.parent / 'kernels_tmp.nc'
        process.standardize_vertical(
            path, output=temp, overwrite=True, **kwvert,
        )
        process.standardize_horizontal(
            temp, output=path, overwrite=True, **kwhori,
        )
        temp.unlink(missing_ok=True)
    ds = xr.open_dataset(path)
    ds = ds.climo.add_cell_measures(verbose=True)
    ds = ds.climo.quantify()
    return ds


def combine_era_kernels(
    path='~/data/kernels-eraint/', **kwargs
):
    """
    Combine and standardize the ERA-Interim kernel data downloaded from Yi Huang to
    match naming convention with CAM5.

    Parameters
    ----------
    path : path-like, optional
        The kernel data location.
    **kwargs
        Passed to `_standardize_kernels`.
    """
    # Initialize dataset
    # TODO: Currently right side of bnds must be larger than left. Perhaps
    # order should correspond to order of vertical coordinate.
    path = Path(path).expanduser()
    time = np.arange('2000-01', '2001-01', dtype='M')
    src = xr.open_dataset(path / 'dp1.nc')  # seems to be identical duplicate of dp2.nc
    levs = src['plevel']  # level edges as opposed to 'player' level centers
    bnds = np.vstack((levs.data[1:], levs.data[:-1]))
    time = xr.DataArray(time, dims='time', attrs={'axis': 'T', 'standard_name': 'time'})
    ds = xr.Dataset(coords={'time': time})
    ds['lev_bnds'] = (('lev', 'bnds'), bnds.T[::-1, :])

    # Load and combine files
    # NOTE: Unlike CAM5 data ERA-Interim data includes two 1000hPa feedbacks, where
    # the first one is intended to be multiplied by surface temperature kernel since
    # radiation across lower layer changes substantially (see Yi Haung ncl code). For
    # simplicity we add the two 1000hPa feedback together but in reality this will
    # differ from the "correct" feedback by the temp difference Ts - Ta(1000hPa).
    print('Loading kernel data...')
    vars = {'t': 'ta', 'ts': 'ts', 'wv': 'hus', 'alb': 'alb'}
    files = sorted(path.glob('RRTM*.nc'))
    for file in files:
        # Standardize variable name
        # NOTE: Only water vapor has longwave/shortwave indicator
        parts = file.name.split('_')
        if len(parts) == 6:  # longwave or shortwave water vapor
            _, var, wav, lev, sky, _ = parts
        else:  # always longwave unless albedo
            _, var, lev, sky, _ = parts
            wav = 'sw' if var == 'alb' else 'lw'
        var = vars[var]
        wav = wav.replace('w', 'n')  # i.e. 'sw' to 'sn' for 'shortwave net'
        lev = lev[0]  # i.e. 'surface' to 's' and 'toa' to 't'
        sky = '' if sky == 'cld' else 'cs'  # 'cld' == including clouds i.e. all-sky
        name = f'{var}_r{wav}{lev}{sky}'

        # Standardize data
        # TODO: When to multiply by -1
        da = xr.open_dataarray(file)
        with xr.set_options(keep_attrs=True):
            da = da * -1
        da = da.isel(lat=slice(None, None, -1))  # match CAM5 data
        da = da.rename(month='time')
        da = da.assign_coords(time=time)
        units = da.attrs.get('units', None)
        if 'longname' in da.attrs:
            da.attrs['long_name'] = da.attrs.pop('longname')
        if units is None:
            da.attrs['units'] = 'W/m2/K/100mb'  # match other unit
        elif var == 'alb':
            da.attrs['units'] = units.replace('0.01', '%')  # repair albedo units
        if 'player' in da.coords:
            da = da.rename(player='lev')
            da = da.isel(lev=slice(None, None, -1))  # match CAM5 data
            lev = da.coords['lev']
            lev.attrs['bounds'] = 'lev_bnds'
            if lev[-1] == lev[-2]:  # duplicate 1000 hPa level bug
                print(f'Warning: Combining 1000hPa pressure feedbacks for {name!r}.')
                da[{'lev': -2}] += da[{'lev': -1}]
                da = da.isel(lev=slice(None, -1))
        ds[name] = da  # note this adds coords and silently replaces da.name

    # Standardize and save data
    ds = _standardize_kernels(ds, suffix='eraint', **kwargs)
    return ds


def combine_cam_kernels(
    path='~/data/kernels-cam5/', folders=('forcing', 'kernels'), **kwargs,
):
    """
    Combine and interpolate the model-level CAM5 feedback kernel data downloaded
    from Angie Pendergrass.

    Parameters
    ----------
    path : path-like
        The base directory in which `folders` is indexed.
    folders : sequence of str, optional
        The subfolders storing forcing and kernel data.
    **kwargs
        Passed to `_standardize_kernels`.
    """
    # Get list of files
    ps = set()
    path = Path(path).expanduser()
    files = [file for folder in folders for file in (path / folder).glob('*.nc')]
    files = [file for file in files if file.name != 'PS.nc' or ps.add(file)]
    if not ps:  # attempted to record separately
        raise RuntimeError('Surface pressure data not found.')

    # Load data using PS.nc for time coords (times are messed up on other datasets)
    print('Loading kernel data...')
    ds = xr.Dataset()  # will add variables one-by-one
    vars = {'t': 'ta', 'ts': 'ts', 'q': 'hus', 'alb': 'alb'}
    ignore = ('gw', 'nlon', 'ntrk', 'ntrm', 'ntrn', 'w_stag', 'wnummax')
    for file in (*ps, *files):  # put PS.nc first
        data = xr.open_dataset(file, use_cftime=True)
        time_dim = data.time.dims[0]
        if 'ncl' in time_dim:  # fix issue where 'time' dim is a dummy ncl name
            data = data.swap_dims({time_dim: 'time'})
        if file.name == 'PS.nc':  # ignore bad time indices
            data = data.rename(PS='ps')
        else:
            data = data.reset_index('time', drop=True)  # drop time coordinates
        if 'ps' in data:
            data['ps'].attrs['standard_name'] = 'surface_air_pressure'
        var, *_ = file.name.split('.')  # format is varname.(kernel|forcing).nc
        var = vars[var]
        for name, da in data.items():  # iterates through variables not coordinates
            if name in ignore:
                continue
            name = name.lower()
            if name[:2] in ('fl', 'fs'):  # longwave or shortwave radiative flux
                name = var + '_r' + name[1:]  # translate f to r consistent with cmip
            ds[name] = da

    # Normalize by pressure thickness and standardize units
    print('Standardizing kernel data...')
    pi = ds.hyai * ds.p0 + ds.hybi * ds.ps
    pi = pi.transpose('time', 'ilev', 'lat', 'lon')
    dp = pi.data[:, 1:, ...] - pi.data[:, :-1, ...]
    for name, da in ds.items():
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
    ds['lev_delta'] = (dims, dp, attrs)
    ds.lon.attrs['axis'] = 'X'
    ds.lat.attrs['axis'] = 'Y'
    ds.lev.attrs['axis'] = 'Z'
    ds.time.attrs['axis'] = 'T'

    # Interpolate onto standard pressure levels using metpy
    # NOTE: Also normalize by pressure level thickness, giving units W m^-2 100hPa^-1
    ds_plev = _standardize_kernels(ds, suffix='cam5_plev', **kwargs)
    ds_mlev = _standardize_kernels(ds, suffix='cam5_mlev', standardize=False, **kwargs)
    return ds_mlev, ds_plev
