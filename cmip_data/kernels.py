#!/usr/bin/env python
"""
Standardize radiative kernel estimates into formats compatible with ESGF data.
"""
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
    ds, path='~/data/cmip-kernels', suffix=None, standardize=True, **kwargs
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
    # NOTE: The CAM5 kernels are stored so that positive for longwave is always up and
    # positive for shortwave is always down (see readme). However we want positive for
    # both shortwave and longwave, surface and toa to mean into the atmosphere. This
    # requires multiplication by -1 for toa longwave and surface shortwave kernels.
    print('Standardizing kernel data...')
    suffix = suffix and '_' + suffix or ''
    path = Path(path).expanduser()
    path = path / f'kernels{suffix}.nc'
    if 'time_bnds' in ds:
        ds = ds.drop_vars('time_bnds')
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
    ds = ds.climo.dequantify()
    kw_std = {'prefix_levels': True, 'pressure_units': 'Pa', 'descending_levels': True}
    kw_vert = {'levels': kwargs.pop('levels', None)}
    kw_hori = {'gridspec': kwargs.pop('gridspec', None), 'method': kwargs.pop('method', None)}  # noqa: E501
    if kwargs:
        raise ValueError(f'Invalid keyword args {kwargs}.')
    if standardize:
        print('Standardizing kernel coordinates...')
        path.unlink(missing_ok=True)
        temp = path.parent / 'kernels_tmp.nc'
        if pressure := ('a' not in ds and 'b' not in ds):
            ds = ds.climo.standardize_coords(**kw_std)  # for pressure interpolation
        ds.to_netcdf(path)
        process.standardize_vertical(path, output=temp, overwrite=True, **kw_vert)
        process.standardize_horizontal(temp, output=path, overwrite=True, **kw_hori)
        ds = xr.open_dataset(path)
        if not pressure:
            ds = ds.climo.standardize_coords(**kw_std)  # after hybrid interpolation
        temp.unlink(missing_ok=True)

    # Save the final data and add cell measures
    # WARNING: Never save cell measures to variables because can cause later issues
    # working with anomaly data. See feedbacks.py for details.
    path.unlink(missing_ok=True)
    ds.to_netcdf(path)
    if standardize:
        ds = ds.isel(plev=slice(None, None, -1))
        ds = ds.climo.add_cell_measures(verbose=True)
        ds = ds.isel(plev=slice(None, None, -1))
    return ds.climo.quantify()


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
    # TODO: Currently right side of bnds must be larger than left.
    # Perhaps order should correspond to order of vertical coordinate.
    path = Path(path).expanduser()
    time = np.arange('2000-01', '2001-01', dtype='M')
    src = xr.open_dataset(path / 'dp1.nc')  # seems to be identical duplicate of dp2.nc
    levs = src['plevel']  # level edges as opposed to 'player' level centers
    bnds = np.vstack((levs.data[1:], levs.data[:-1]))
    time = xr.DataArray(time, dims='time', attrs={'axis': 'T', 'standard_name': 'time'})
    ds = xr.Dataset(coords={'time': time})
    ds['lev_bnds'] = (('lev', 'bnds'), bnds.T[::-1, :])

    # Load and combine files
    print('Loading kernel data...')
    vars = {'t': 'ta', 'ts': 'ts', 'wv': 'hus', 'alb': 'alb'}
    files = sorted(path.glob('RRTM*.nc'))
    for file in files:
        # Standardize variable name
        # NOTE: Only water vapor has longwave/shortwave indicator. Otherwise
        # assumed longwave for temperature shortwave for albedo.
        parts = file.name.split('_')
        if len(parts) == 6:  # longwave or shortwave water vapor
            _, var, wav, lev, sky, _ = parts
        else:  # always longwave unless albedo
            _, var, lev, sky, _ = parts
            wav = 'sw' if var == 'alb' else 'lw'
        var = vars.get(var, var)
        wav = wav.replace('w', 'n')  # i.e. 'sw' to 'sn' for 'shortwave net'
        lev = lev[0]  # i.e. 'surface' to 's' and 'toa' to 't'
        sky = '' if sky == 'cld' else 'cs'  # 'cld' == including clouds i.e. all-sky
        name = f'{var}_r{wav}{lev}{sky}'

        # Standardize data
        # NOTE: Unlike CAM5 there are no notes about which fluxes are positive upward
        # or downward. Instead tested manually by comparing with cam5 and seems all
        # kernels are positive down. So convert longwave to positive up. Try this code:
        # for var in alb ts ta hus; do for flux in rlnt rlns rsnt rsns; do for file in
        # kernels_eraint.nc kernels_cam5.nc; do name=${var}_${flux}; echo
        # "$file: $name"; ncvartable $name $file | tail -n +2; done; done; done
        da = xr.open_dataarray(file)
        with xr.set_options(keep_attrs=True):
            if wav[0] == 'l':  # convert positive down to positive up
                da = da * -1
        da = da.isel(lat=slice(None, None, -1))  # match cam5 direction
        da = da.rename(month='time')
        da = da.assign_coords(time=time)
        units = da.attrs.get('units', None)
        if 'longname' in da.attrs:
            da.attrs['long_name'] = da.attrs.pop('longname')
        if units is None:
            da.attrs['units'] = 'W/m2/K/100mb'  # match other unit
        elif var == 'alb':
            da.attrs['units'] = units.replace('0.01', '%')  # repair albedo units

        # Standardize vertical units
        # NOTE: Unlike CAM5 data ERA-Interim data includes two 1000hPa feedbacks -- the
        # first one is intended to be multiplied by surface temperature kernel since
        # radiation across lower layer changes substantially (see Yi Haung ncl code).
        # For simplicity we add the two feedbacks together but in reality this will
        # differ from the "correct" feedback by the temp difference Ts - Ta(1000hPa).
        if 'player' in da.coords:
            da = da.rename(player='lev')
            da = da.isel(lev=slice(None, None, -1))  # match CAM5 data
            lev = da.coords['lev']
            lev.attrs['bounds'] = 'lev_bnds'
            if lev[-1] == lev[-2]:  # duplicate 1000hPa level bug
                print(f'Warning: Combining 1000hPa pressure feedbacks for {name!r}.')
                da[{'lev': -2}] += da[{'lev': -1}]
                da = da.isel(lev=slice(None, -1))
        if var == 'alb':
            da = da.climo.to_units('W m^-2 %^-1')
        elif var == 'ts':
            da = da.climo.to_units('W m^-2 K^-1')
        else:
            da = da.climo.to_units('W m^-2 K^-1 Pa^-1')
        da.name = name
        ds[name] = da  # this also adds coords

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
        var, *_ = file.name.split('.')  # format is varname.(kernel|forcing).nc
        var = vars.get(var, var)
        data = xr.open_dataset(file, use_cftime=True)
        time_dim = data.time.dims[0]
        if 'ncl' in time_dim:  # fix issue where 'time' dim is a dummy ncl name
            data = data.swap_dims({time_dim: 'time'})
        if file.name != 'PS.nc':  # ignore bad time indices
            data = data.reset_index('time', drop=True)  # drop time coordinates
        for name, da in data.items():  # iterates through variables not coordinates
            if name in ignore:
                continue
            name = name.lower()
            if name[:2] in ('fl', 'fs'):  # longwave or shortwave radiative flux
                name = var + '_r' + name[1:]  # translate f to r consistent with cmip
            if name[-1:] == 'c':  # clear-sky indicator from 'c' to 'cs'
                name = name + 's'
            if name == 'ps':
                da.attrs['standard_name'] = 'surface_air_pressure'
            if formula := da.attrs.get('formula_terms', None):
                da.attrs['formula_terms'] = formula.lower()
            ds[name] = da

    # Normalize by pressure thickness and standardize units
    # NOTE: Without this step variables interpolated to pressure are nonsense. Also
    # tried to set 'coordinates' attribute to None on levels with variables or else
    # xarray will set it to 'a b' and cdo will emit annoying 'inconsistent variable
    # definition' and 'coordinates variable X cannot be assigned' warnings... but for
    # some reason despite the merge shown below it fails. Not sure why.
    # Issue: https://github.com/pydata/xarray/issues/5510
    # Fixed: https://github.com/pydata/xarray/pull/5514 (July 2021)
    pi = ds.hyai * ds.p0 + ds.hybi * ds.ps  # both already in Pa
    pi = pi.transpose('time', 'ilev', 'lat', 'lon')
    dp = pi.data[:, 1:, ...] - pi.data[:, :-1, ...]
    pi_min = ', '.join(format(p, '.0f') for p in pi.data.min((0, 2, 3)))
    pi_max = ', '.join(format(p, '.0f') for p in pi.data.max((0, 2, 3)))
    dp_avg = ', '.join(format(p, '.0f') for p in dp.mean((0, 2, 3)))
    print('Minimum level interfaces:', pi_min)
    print('Maximum level interfaces:', pi_max)
    print('Average level thicknesses:', dp_avg)
    print('Normalizing by level thickness...')
    for name, da in ds.items():
        if 'lev' in da.dims and da.ndim > 1:
            da.data /= dp
            da.attrs['units'] = 'W m^-2 K^-1 Pa^-1'
            # da.encoding['coordinates'] = None  # suppress adding 'a' and 'b'
        elif name[:3] == 'alb':
            da.attrs['units'] = 'W m^-2 %^-1'
        elif name[:2] == 'ts':
            da.attrs['units'] = 'W m^-2 K^-1'
        elif da.attrs.get('units', None) == 'W/m2':
            da.attrs['units'] = 'W m^-2'

    # Normalize attributes and interpolate onto standard pressure using cdo ml2pl.
    # WARNING: Here _is_bounds will fail for a_bnds, b_bnds if a, b are not also
    # stored as coordiantes, so ensure they are.
    # WARNING: Here 'lev' must come last due to reset_index line required
    # for concatenating data array selections with conflicting level indices.
    print('Preparing attributes and levels...')
    ds = ds.rename(hyam='a', hybm='b', hyai='ia', hybi='ib')
    attrs = {'axis': 'Z', 'units': 'Pa', 'positive': 'down'}
    attrs['formula'] = 'p = a*p0 + b*ps'
    attrs['standard_name'] = 'atmosphere_hybrid_sigma_pressure_coordinate'
    for name in ('a', 'b', 'lev'):
        bnds = ds[f'i{name}'].reset_index('ilev', drop=True).rename(ilev='lev')
        bnds = xr.concat((bnds[:-1], bnds[1:]), dim='bnds')
        ds[f'{name}_bnds'] = bnds.transpose('lev', 'bnds')
        ds[name].attrs['bounds'] = f'{name}_bnds'
        ds = ds.drop_vars(f'i{name}')
        ds = ds.set_coords(name)
    with xr.set_options(keep_attrs=True):
        ds = ds.assign_coords(lev=100.0 * ds.lev)  # removes attributes
        ds['lev_bnds'] *= 100.0  # match surface pressure units
        ds.lev.attrs.update(attrs)
        ds.lev_bnds.attrs.update(attrs)
        ds.lev.attrs['formula_terms'] = 'p0: p0 a: a b: b ps: ps'
        ds.lev_bnds.attrs['formula_terms'] = 'p0: p0 a: a_bnds b: b_bnds ps: ps'
    print('Saving model level data...')
    ds_mlev = _standardize_kernels(ds, suffix='cam5_orig', standardize=False, **kwargs)
    print('Saving pressure level data...')
    ds_plev = _standardize_kernels(ds, suffix='cam5', standardize=True, **kwargs)

    return ds_plev, ds_mlev
