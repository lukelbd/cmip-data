#!/usr/bin/env python3
"""
Calculate feedback, forcing, and sensitivity estimates for ESGF data.
"""
import builtins
import gc
import itertools
import traceback
from pathlib import Path

import numpy as np  # noqa: F401
import xarray as xr
from climopy import const, ureg
from icecream import ic  # noqa: F401
from idealized import definitions  # noqa: F401
from metpy import calc  # noqa: F401
from metpy.units import units as mreg  # noqa: F401
from numpy import ma  # noqa: F401

from .internals import (
    Database,
    Logger,
    glob_files,
    _item_dates,
    _item_join,
    _item_parts,
)
from .utils import average_periods, average_regions, load_file

__all__ = [
    'get_feedbacks',
    'process_feedbacks',
]


# Global constants
# NOTE: Here 'n' stands for net (follows Angie convention). These variables are built
# from the 'u' and 'd' components of cmip fluxes and to prefix the kernel-implied net
# fluxes 'alb', 'pl', 'lr', 'hu', 'cl', and 'resid' in _fluxes_from_anomalies.
# NOTE: Here all-sky fluxes are used for albedo (follows Angie convention). This is
# because several models are missing clear-sky surface shortwave fluxes and ratio
# of upwelling to downwelling should be identical between components (although for some
# reason Yi uses "all-sky" vs. "clear-sky" albedo with all-sky vs. clear-sky kernels).
FEEDBACK_DESCRIPTIONS = {
    '': '',  # net all-sky feedback
    'cs': 'clear-sky',  # net clear-sky feedback
    'alb': 'albedo',
    'pl': 'Planck',
    'lr': 'lapse rate',
    'hus': 'specific humidity',
    'pl*': 'adjusted Planck',
    'lr*': 'adjusted lapse rate',
    'hur': 'relative humidity',
    'cl': 'cloud',
    'resid': 'residual',
}
FEEDBACK_DEPENDENCIES = {  # (longwave, shortwave) tuples of dependencies
    '': {'longwave': (), 'shortwave': ()},
    'cs': {'longwave': (), 'shortwave': ()},
    'alb': {'longwave': (), 'shortwave': ('alb',)},
    'pl': {'longwave': ('ts', 'ta'), 'shortwave': ()},
    'lr': {'longwave': ('ts', 'ta'), 'shortwave': ()},
    'hus': {'longwave': ('hus',), 'shortwave': ('hus',)},
    'pl*': {'longwave': ('ts', 'ta', 'hus'), 'shortwave': ('hus',)},
    'lr*': {'longwave': ('ts', 'ta', 'hus'), 'shortwave': ('hus',)},
    'hur': {'longwave': ('hus', 'ta'), 'shortwave': ('hus',)},
    'cl': {'longwave': ('ts', 'ta', 'hus'), 'shortwave': ('alb', 'hus')},
    'resid': {'longwave': ('ts', 'ta', 'hus'), 'shortwave': ('alb', 'hus')},
}
VARIABLE_DEPENDENCIES = {
    'pspi': ('ps',),  # control data for integration bounds
    'tapi': ('ta',),  # control data for humidity kernels and integration bounds
    # 'ps4x': ('ps',),  # response data for integration bounds
    # 'ta4x': ('ta',),  # response data for integration bounds
    'ts': ('ts',),  # anomaly data
    'ta': ('ta',),  # anomaly data
    'hus': ('hus',),  # anomaly data
    'alb': ('rsus', 'rsds'),  # out of and into the surface
    'rlnt': ('rlut',),  # out of the atmosphere
    'rsnt': ('rsut',),  # out of the atmosphere (ignore constant rsdt)
    'rlns': ('rlds', 'rlus'),  # out of and into the atmosphere
    'rsns': ('rsds', 'rsus'),  # out of and into the atmosphere
    'rlntcs': ('rlutcs',),  # out of the atmosphere
    'rsntcs': ('rsutcs',),  # out of the atmosphere (ignore constant rsdt)
    'rlnscs': ('rldscs', 'rlus'),  # out of and into the atmosphere
    'rsnscs': ('rsdscs', 'rsuscs'),  # out of and into the atmosphere
}


def _get_file_pairs(data, *args, printer=None):
    """
    Return a dictionary mapping of variables to ``(control, response)`` file
    pairs from a database of files with ``(experiment, ..., variable)`` keys.

    Parameters
    ----------
    data : dict
        The database of file lists.
    *args : str
        The experiment and file suffixes.
    printer : callable, default: `print`
        The print function.
    """
    print = printer or builtins.print
    pairs = []
    if len(args) != 4:
        raise TypeError('Exactly four input arguments required.')
    for experiment, suffix in zip(args[:-1:2], args[1:None:2]):
        paths = {
            var: [file for file in files if _item_dates(file) == suffix]
            for (exp, *_, var), files in data.items() if exp == experiment
        }
        for variable, files in tuple(paths.items()):
            names = ', '.join(file.name for file in files)
            print(f'  {experiment} {variable} files ({len(files)}):', names)
            if message := not files and 'Missing' or len(files) > 1 and 'Ambiguous':
                print(f'Warning: {message} files (expected 1, found {len(files)}).')
                del paths[variable]
            else:
                (file,) = files
                paths[variable] = file
        pairs.append(paths)
    if message := ', '.join(pairs[0].keys() - pairs[1].keys()):
        print(f'Warning: Missing response files for control data {message}.')
    if message := ', '.join(pairs[1].keys() - pairs[0].keys()):
        print(f'Warning: Missing control files for response data {message}.')
    pairs = {
        variable: (pairs[0][variable], pairs[1][variable])
        for variable in sorted(pairs[0].keys() & pairs[1].keys())
    }
    if not pairs:
        raise RuntimeError(f'No pairs found for groups {tuple(data)}.')
    return pairs


def _get_clausius_scaling(ta, pa=None, liquid=True):
    r"""
    Return the inverse of the approximate change in the logarithm saturation specific
    humidity associated with a 1 Kelvin change in atmospheric temperature, defined as
    :math:`1 / (\mathrm{d} \log q / \mathrm{d} T) = \mathrm{d} T (q / \mathrm{d} q)`.

    Parameters
    ----------
    ta : xarray.DataArray
        The air temperature.
    pa : xarray.DataArray, optional
        The air pressure. If provided an exact specific humidity conversion is used.
    liquid : bool, optional
        Whether to only use the saturation estimate with-respect-to liquid water.

    Returns
    -------
    scale : xarray.DataArray
        The scaling in units Kelvin.

    Note
    ----
    This uses the more accurate Murphy and Koop (2005) method rather than Bolton et al.
    (1980). Soden et al. (2008) and Shell et al. (2008) recommend using the logarithmic
    scale factor :math:`(\log q_1 - \log q_0) / (\mathrm{d} \log q / \mathrm{d} T)`
    since the radiative response is roughly proportional to the logarithm of water vapor
    and thus linearity of the response to perturbations is a better assumption in log
    space. See written notes for details on approximation and replacing `q` with `e`.
    """
    # Add helper functions
    # NOTE: In example NCL script Angie uses ta and hus from 'basefields.nc' for dq/dt
    # (metadata indicates these were fields from kernel calculations) while Yi uses
    # temperature from the picontrol climatology. Here we use latter methodology.
    ta = ta.climo.to_units('K')
    ta = ta.climo.dequantify()
    if pa is not None:
        pa = pa.climo.to_units('Pa')
        pa = pa.climo.dequantify()
        pa, ta = xr.broadcast(pa, ta)
    es_ice = lambda t: np.exp(  # input kelvin, output pascals  # noqa: F841
        9.550426
        - 5723.265 / t
        + 3.53068 * np.log(t)
        - 0.00728332 * t
    )
    es_liq = lambda t: np.exp(  # input kelvin, output pascals
        54.842763
        - 6763.22 / t
        - 4.21 * np.log(t)
        + 0.000367 * t
        + np.tanh(0.0415 * (t - 218.8))
        * (53.878 - 1331.22 / t - 9.44523 * np.log(t) + 0.014025 * t)
    )

    # Get scaling
    # WARNING: If using metpy, critical to replace climopy units with metpy units, or
    # else get hard-to-debug errors, e.g. temperature units mysteriously stripped. In
    # the end decided to go manual since metpy uses inaccurate Bolton (1980) method.
    es_hus = lambda e, p: np.where(
        p < 5000, np.nan,  # ignore stratospheric values
        const.eps.magnitude * e / (p.data - (1 - const.eps.magnitude) * e)
    )
    if liquid:
        es0 = es_liq(ta.data)
        es1 = es_liq(ta.data + 1)
    else:
        es0, ei0 = es_liq(ta.data), es_ice(ta.data)
        es1, ei1 = es_liq(ta.data + 1), es_ice(ta.data + 1)
        es0[mask0] = ei0[mask0 := ta.data < 273.15]
        es1[mask1] = ei1[mask1 := ta.data + 1 < 273.15]
    if pa is None:  # vapor pressure approximation (see written notes)
        scale = 1 / np.log(es1 / es0)
    else:  # exact specific humidity calculation
        scale = 1 / np.log(es_hus(es1, pa.data) / es_hus(es0, pa.data))
    scale = xr.DataArray(
        scale,
        dims=tuple(ta.sizes),
        coords=dict(ta.coords),
        attrs={'units': 'K', 'long_name': 'inverse Clausius-Clapeyron scaling'}
    )
    scale.data[~np.isfinite(scale.data)] = np.nan
    scale = scale.climo.quantify()
    return scale


def _anomalies_from_files(
    project=None, select=None, dryrun=False, printer=None, **inputs,
):
    """
    Return dataset containing response minus control anomalies for the variables given
    pairs of paths to its dependencies (e.g. ``'rlds'`` and ``'rlus'`` for ``'rlns'``).

    Parameters
    ----------
    **inputs : 2-tuple of path-like
        Tuples of response and control paths for the dependencies by name.
    project : str, optional
        The project. Used in level checking.
    select : int, optional
        The start and stop years for pattern effects.
    dryrun : bool, optional
        Whether to run with only three years of data.
    printer : callable, default: `print`
        The print function.
    """
    # Iterate over dependencies
    # NOTE: Since incoming solar is constant, and since all our regressions are
    # with anomalies with respect to base climate, the incoming solar terms cancel
    # out and we can just consider upwelling solar at the top-of-atmosphere.
    print = printer or builtins.print
    project = (project or 'cmip6').lower()
    start, stop = select or (None, None)
    if start is not None:
        start = int(start) * 12
    if stop is not None:
        stop = int(stop) * 12
    output = xr.Dataset()
    print('Calculating response minus control anomalies.')
    for variable, dependencies in VARIABLE_DEPENDENCIES.items():
        # Load datasets and prepare the variables
        # NOTE: Critical to simply ignore 'start' and 'stop' for e.g. ratio
        # feedbacks as here times were just used to directly pick climate files.
        # NOTE: The signs of the kernels are matched so that negative always means
        # tending to cool the atmosphere and positive means tending to warm the
        # atmosphere. We scale the positive-definite upwelling and dowwelling
        # components by sorting dependency fluxes in the format (out,) or (out, in).
        if missing := tuple(name for name in dependencies if name not in inputs):
            print(
                f'Warning: Anomaly variable {variable!r} is missing its '
                f'dependenc(ies) {missing}. Skipping anomaly calculation.'
            )
            continue
        controls, responses = [], []
        for name in dependencies:  # upwelling and downwelling component
            control, response = inputs[name]  # control and response paths
            for path, datas in zip((control, response), (controls, responses)):
                path = Path(path).expanduser()
                data = load_file(path, name, project=project, printer=print)
                if data.sizes.get('time', 12) <= 12:
                    pass
                elif dryrun:
                    data = data.isel(time=slice(None, 36))
                elif start is not None or stop is not None:
                    data = data.isel(time=slice(start, stop))
                data.name = variable
                datas.append(data)

        # Calculate variable anomalies using dependencies
        # NOTE: Here subtraction from 'groupby' objects can only be used if the array
        # has a coordinate with the corresponding indices, i.e. in this case 'month'
        # instead of 'time'. Use .groupby('time.month').mean() to turn time indices
        # into month indices (works for both time series and pre-averaged data).
        with xr.set_options(keep_attrs=True):
            if variable in ('tapi', 'pspi'):  # control data itself
                (control,), (response,) = controls, responses
                control = control
                response = None
            elif variable in ('ta4x', 'ps4x'):  # response data itself
                (control,), (response,) = controls, responses
                control = None
                response = response
            elif variable in ('ts', 'ta'):  # surface and atmosphere temperature
                (control,), (response,) = controls, responses
                control = control
                response = response
            elif variable == 'hus':  # logarithm for Clausius-Clapeyron scaling
                (control,), (response,) = controls, responses
                control = np.log(control)
                response = np.log(response)
            elif variable == 'alb':  # ratio of outward and inward flux
                (control_out, control_in), (response_out, response_in) = controls, responses  # noqa: E501
                control = 100 * (control_out / control_in)
                response = 100 * (response_out / response_in)
            elif len(controls) == 1:  # outward flux to inward flux
                (control_out,), (response_out,) = controls, responses
                control = -1 * control_out
                response = -1 * response_out
            else:  # inward flux minus outward flux
                (control_out, control_in), (response_out, response_in) = controls, responses  # noqa: E501
                control = control_in - control_out
                response = response_in - response_out

        # Add attributes and update dataset
        # NOTE: Albedo is taken above from ratio of upwelling to downwelling all-sky
        # surface shortwave radiation. Feedback is simply (rsus - rsds) / (rsus / rsds)
        with xr.set_options(keep_attrs=True):
            if control is None:
                data = response
            elif response is None:
                data = control.groupby('time.month').mean()  # match to response times
            else:
                data = response.groupby('time.month') - control.groupby('time.month').mean()  # noqa: E501
        if variable == 'alb':  # enforce new name and new units
            long_name = 'surface albedo'
            data.attrs['units'] = '%'
        elif variable == 'hus':  # enforce new name and new units
            long_name = 'specific humidity logarithm'
            data.attrs['units'] = '1'
        else:  # enforce lower-case name for consistency
            parts = data.attrs['long_name'].split()
            long_name = ' '.join(s if s == 'TOA' else s.lower() for s in parts)
        if variable in ('tapi', 'pspi', 'ta4x', 'ps4x'):
            prefix = 'response' if '4x' in variable else 'control'
            long_name = f'{prefix} {long_name}'  # NOTE: keep standard name for ptop
        else:
            long_name = f'{long_name} anomaly'
            data.attrs.pop('standard_name', None)
        size = data.sizes['time' if 'time' in data.sizes else 'month']
        data.attrs['long_name'] = long_name
        output[variable] = data
        print(f'  {variable} ({long_name}): {size}')

    return output


def _fluxes_from_anomalies(anoms, kernels, printer=None):
    """
    Return a dataset containing the actual radiative flux responses and the radiative
    flux responses implied by the radiative kernels and input anomaly data.

    Parameters
    ----------
    anoms : xarray.Dataset
        The climate anomaly data.
    kernels : xarray.Dataset
        The radiative kernel data.
    printer : callable, default: `print`
        The print function.
    """
    # Prepare kernel dataset before combining with anomaly dataset
    # TODO: Increase efficiency of tropopause calculation now that it is merely
    # control data repeated onto response data series. Possibly consider assigning
    # cell measures to kernels and use groupby, or build cell height from a sample
    # selection of 12 timesteps then assign repetition to anomaly data.
    # WARNING: Had unbelievably frustrating issue where pressure level coordinates
    # on kernels and anomalies would be identical (same type, value, etc.) but then
    # anoms.plev == kernels.plev returned empty array and assigning time-varying cell
    # heights to data before integration in Planck feedback block created array of
    # all NaNs. Still not sure why this happened but assign_coords seems to fix it.
    print = printer or builtins.print
    print('Calculating kernel-derived radiative fluxes.')
    plev = anoms.plev
    with xr.set_options(keep_attrs=True):
        kernels = kernels.groupby('time.month').mean()  # see _anomalies_from_files
    levs = kernels.plev.values  # NESM3 abrupt-4xCO2 data contains missing levels
    if anoms.sizes.get('plev', np.inf) < kernels.sizes['plev']:
        levs = [lev for lev in levs if np.any(np.isclose(lev, plev.values))]
    kernels = kernels.sel(plev=levs)  # note load_file() already restricts levels
    kernels = kernels.climo.quantify()
    kernels = kernels.assign_coords(plev=plev)  # critical (see above)

    # Get cell height measures and Clausius-Clapeyron adjustments
    # NOTE: Initially used response time series for tropopause consistent with Zelinka
    # stated methodology of 'time-varying tropopause' but got bizarre issue with Planck
    # feedback biased positive for certain models (seemingly due to higher surface
    # temperature associated with higher tropopause and more negative fluxes... so
    # should be negative bias... but whatever). This also made little sense because it
    # corrected for changing troposphere depth without correcting for reduced strength
    # of kernels under increased depth. Now try with control data average, and below
    # code works with both averages and response time series (see top of file).
    # WARNING: Currently cell_height() in physics.py automatically adds surface bounds,
    # tropopause bounds, and cell heights as *coordinates* (for consistency with xarray
    # objects). Must promote to variable before working with other arrays to avoid
    # conflicts during reassignment.
    # WARNING: Using assign_coords(cell_height=height) with height 'month' coordinate
    # will cause 'month' to remain as a coordinate on the dataset, but not on other
    # relevant data arrays (tapi, pbot, ptop). Can only fix this by manually assigning
    # the 'month' coord *within the same assign_coords call*. Version: xarray 0.21.1.
    print('Calculating vertical cell measures and Clausius-Clapeyron scaling.')
    anoms = anoms.climo.add_cell_measures(surface=True, tropopause=True)
    anoms = anoms.reset_coords(('plev_bot', 'plev_top'))  # critical (see above)
    height = anoms.cell_height.climo.quantify()
    scalar = ureg.Quantity(0, 'Pa')
    array = anoms['ta' if 'ta' in anoms else 'hus' if 'hus' in anoms else 'ts']
    array = ureg.Pa * xr.zeros_like(array)  # for matching time coordinates
    base = array.groupby('time.month') if 'month' in height.dims else scalar
    with xr.set_options(keep_attrs=True):  # WARNING: critical for 'base' to be second
        height = (const.g * height).climo.to_units('Pa') + base
    if 'plev' in array.dims:
        array = array.isel(plev=0, drop=True)
    output = {}
    coords = {'month': anoms.month, 'cell_height': height.climo.dequantify()}
    anoms = anoms.assign_coords(coords)
    anoms = anoms.climo.quantify()
    min_, max_, mean = height.min().item(), height.max().item(), height.mean().item()
    del height
    print(f'Cell height range: min {min_:.0f} max {max_:.0f} mean {mean:.0f}')
    pbot = anoms.plev_bot.climo.quantify()
    base = array.groupby('time.month') if 'month' in pbot.dims else scalar
    pbot = pbot.climo.to_units('Pa') + base  # possibly expand along response times
    pbot.attrs['long_name'] = 'surface pressure'
    output['pbot'] = pbot.climo.dequantify()
    min_, max_, mean = pbot.min().item(), pbot.max().item(), pbot.mean().item()
    del pbot
    print(f'Surface pressure range: min {min_:.0f} max {max_:.0f} mean {mean:.0f}')
    ptop = anoms.plev_top.climo.quantify().climo.to_units('Pa')
    base = array.groupby('time.month') if 'month' in ptop.dims else scalar
    ptop = ptop.climo.to_units('Pa') + base  # possibly expand along response times
    ptop.attrs['long_name'] = 'tropopause pressure'
    output['ptop'] = ptop.climo.dequantify()
    min_, max_, mean = ptop.min().item(), ptop.max().item(), ptop.mean().item()
    del ptop
    print(f'Tropopause pressure range: min {min_:.0f} max {max_:.0f} mean {mean:.0f}')
    # pa = anoms.climo.coords.plev.climo.dequantify()
    ta = anoms.tapi.climo.dequantify()
    scale = _get_clausius_scaling(ta)  # _get_clausius_scaling(ta, pa)
    min_, max_, mean = scale.min().item(), scale.max().item(), scale.mean().item()
    del ta, base, array  # del pa, ta, base
    print(f'Clausius-Clapeyron range: min {min_:.2f} max {max_:.2f} mean {mean:.2f}')
    output['ts'] = anoms.ts.climo.dequantify()

    # Iterate over flux components
    # NOTE: Here cloud feedback comes about from change in cloud radiative forcing (i.e.
    # R_cloud_abrupt - R_cloud_control where R_cloud = R_clear - R_all) then corrected
    # for masking of clear-sky effects (e.g. by adding dt * K_t_cloud where similarly
    # K_t_cloud = K_t_clear - K_t_all). Additional greenhouse forcing masking
    # correction is applied when we estimate feedbacks.
    for boundary, wavelength, (component, descrip) in itertools.product(
        ('TOA', 'surface'), ('longwave', 'shortwave'), FEEDBACK_DESCRIPTIONS.items()
    ):
        # Skip feedbacks without shortwave or longwave components.
        # NOTE: Cloud and residual feedbacks also require the full model radiative
        # flux to be estimated, so include those terms as dependencies.
        # NOTE: This function uses kernel names standardized by _standardize_kernels
        # in kernels.py (cmip variable name, underscore, 4-character flux name).
        rads = (rad,) = (f'r{wavelength[0]}n{boundary[0].lower()}',)
        variables = FEEDBACK_DEPENDENCIES[component][wavelength]  # empty for fluxes
        dependencies = list(variables)  # anomaly and kernel variables
        if component == '':
            print(f'Calculating {wavelength} {boundary} fluxes using kernel method.')
        if component == '':
            running = 0.0 * ureg('W m^-2')
        if component == '':
            dependencies.append(rad)
        elif component == 'cs':
            dependencies.extend(rads := (rad := f'{rad}cs',))
        elif component == 'cl' or component == 'resid':
            dependencies.extend(rads := (rad, f'{rad}cs'))  # masking adjustments
        elif not dependencies:
            continue
        names = list(f'{var}_{rad}' for var in variables for rad in rads)
        if message := ', '.join(repr(name) for name in dependencies if name not in anoms):  # noqa: E501
            print(
                f'Warning: {wavelength.title()} flux {component=} is missing '
                f'variable dependencies {message}. Skipping {wavelength} flux estimate.'
            )
            continue
        if message := ', '.join(repr(name) for name in names if name not in kernels):
            print(
                f'Warning: {wavelength.title()} flux {component=} is missing '
                f'kernel dependencies {message}. Skipping {wavelength} flux estimate.'
            )
            continue

        # Component fluxes
        # NOTE: Need to assign cell height coordinates for Planck feedback since
        # they are not defined on the kernel data array prior to multiplication.
        anom = mask = kernel = None  # ensure exists
        if component == '' or component == 'cs':  # net flux
            data = 1.0 * anoms[rad]
        elif component == 'resid':  # residual flux
            data = anoms[rad] - running
        elif component == 'alb':  # albedo flux (longwave skipped above)
            kernel = kernels[f'alb_{rad}']
            anom = anoms['alb']
            data = kernel * anom.groupby('time.month')
        elif component[:2] == 'pl':  # planck flux (shortwave skipped for 'pl')
            kernel = 0.0 * ureg('W m^-2 Pa^-1 K^-1')
            if wavelength[0] == 'l':
                kernel = kernel + kernels[f'ta_{rad}']
            if component == 'pl*':
                kernel = kernel + kernels[f'hus_{rad}']
            anom = anoms['ts']  # has no cell height
            data = kernel * anom.groupby('time.month')
            data = data.assign_coords(cell_height=anoms.cell_height)
            data = data.climo.integral('plev')
            if wavelength[0] == 'l':
                kernel = kernels[f'ts_{rad}']
                data = data + kernel * anom.groupby('time.month')
        elif component[:2] == 'lr':  # lapse rate flux (shortwave skipped for 'lr')
            kernel = 0.0 * ureg('W m^-2 Pa^-1 K^-1')
            if wavelength[0] == 'l':
                kernel = kernel + kernels[f'ta_{rad}']
            if component == 'lr*':
                kernel = kernel + kernels[f'hus_{rad}']
            anom = anoms['ta'] - anoms['ts']
            data = kernel * anom.groupby('time.month')
            data = data.climo.integral('plev')
        elif component[:2] == 'hu':  # specific and relative humidity fluxes
            kernel = kernels[f'hus_{rad}']
            anom = scale * anoms['hus'].groupby('time.month')
            if component == 'hur':
                anom = anom - anoms['ta']
            data = kernel * anom.groupby('time.month')
            data = data.climo.integral('plev')
        elif component == 'cl':  # shortwave and longwave cloud fluxes
            data = anoms[rad] - anoms[f'{rad}cs']
            for var in variables:  # relevant longwave or shortwave variables
                kernel = kernels[f'{var}_{rad}'] - kernels[f'{var}_{rad}cs']
                mask = kernel * anoms[var].groupby('time.month')  # mask adjustment
                if var == 'hus':
                    mask = scale * mask.groupby('time.month')
                if var in ('ta', 'hus'):
                    mask = mask.climo.integral('plev')
                data = data - mask.climo.to_units('W m^-2')  # apply adjustment
        else:
            raise RuntimeError(f'Invalid flux component {component!r}.')
        del anom, mask, kernel
        gc.collect()

        # Update the input dataset with resulting component fluxes.
        # NOTE: Calculate residual feedback by summing traditional component
        # feedbacks from Soden and Held.
        name = rad if component in ('', 'cs') else f'{component}_{rad}'
        descrip = descrip and f'{descrip} flux' or 'flux'  # for lon gname
        component = component or 'net'  # for print message
        data.name = name
        data.attrs['long_name'] = f'net {boundary} {wavelength} {descrip}'
        data = data.climo.to_units('W m^-2')
        if 'plev' in data.coords:  # scalar NaN value
            data = data.drop_vars('plev')
        if component in ('pl', 'lr', 'hus', 'alb', 'cl'):
            running = data + running
        output[data.name] = data = data.climo.dequantify()
        min_, max_, mean = data.min().item(), data.max().item(), data.mean().item()
        print(format(f'  {component} flux:', ' <15s'), end=' ')
        print(f'min {min_: <+7.2f} max {max_: <+7.2f} mean {mean: <+7.2f} ({descrip})')
        del data
        gc.collect()

    # Combine shortwave and longwave components
    # NOTE: Could also compute feedbacks and forcings for just shortwave and longwave
    # components, then average them to get the net (linearity of offset and slope
    # estimators), but this is slightly easier and has only small computational cost.
    for boundary, (component, descrip) in itertools.product(
        ('TOA', 'surface'), FEEDBACK_DESCRIPTIONS.items()
    ):
        # Determine fluxes that should be summed to provide net balance. For
        # some feedbacks this is just longwave or just shortwave.
        # NOTE: Warning prevents e.g. only shortwave or longwave water vapor
        # feedback or cloud feedback from being counted as the 'net' feedback.
        if component == '':
            print(f'Calculating shortwave plus longwave {boundary} fluxes.')
        long = f'rln{boundary[0].lower()}'
        short = f'rsn{boundary[0].lower()}'
        if component in ('', 'cs'):
            names = (f'{long}{component}', f'{short}{component}')
        elif component in ('alb',):
            names = (f'{component}_{short}',)
        elif component in ('lr', 'pl'):  # traditional feedbacks
            names = (f'{component}_{long}',)
        else:
            names = (f'{component}_{long}', f'{component}_{short}')
        if message := ', '.join(name for name in names if name not in output):
            print(
                f'Warning: Net radiative flux {component=} is missing longwave or '
                f'shortwave dependencies {message}. Skipping net flux estimate.'
            )
            continue

        # Sum component fluxes into net longwave plus shortwave fluxes
        # if both dependencies exist. Otherwise emit warning.
        net = f'rfn{boundary[0].lower()}'  # use 'f' i.e. 'full' instead of wavelength
        name = f'{net}{component}' if component in ('', 'cs') else f'{component}_{net}'
        descrip = descrip and f'{descrip} flux' or 'flux'  # for long name
        component = component or 'net'  # for print message
        data = sum(output[name].climo.quantify() for name in names)  # no leakage here
        data.name = name
        data.attrs['long_name'] = f'net {boundary} {descrip}'
        data = data.climo.to_units('W m^-2')
        if len(names) == 1:  # e.g. drop the shortwave albedo 'component'
            output.pop(names[0])
        output[name] = data = data.climo.dequantify()
        min_, max_, mean = data.min().item(), data.max().item(), data.mean().item()
        print(format(f'  {component} flux:', ' <15s'), end=' ')
        print(f'min {min_: <+7.2f} max {max_: <+7.2f} mean {mean: <+7.2f} ({descrip})')
        del data
        gc.collect()

    # Construct dataset and return
    output = xr.Dataset(output)
    return output


def _feedbacks_from_fluxes(fluxes, forcing=None, pattern=True, printer=None, **kwargs):
    """
    Return a dataset containing feedbacks calculated along all `average_periods` periods
    (annual averages, seasonal averages, and month averages) and along all combinations
    of `average_regions` (points, latitudes, hemispheres, and global, with different
    averaging conventions used in the denominator indicated by the ``region`` coord).

    Parameters
    ----------
    fluxes : xarray.Dataset
        The source for the flux anomaly data. Should have been generated
        in `_fluxes_from_anomalies` using standardized radiative kernel data.
    forcing : path-like, optional
        Source for the forcing data. If passed then the time-average anomalies are
        used rather than regressions. This can be used with response-climate data.
    pattern : bool, optional
        Whether to include the local temperature pattern term in the output dataset.
    printer : callable, default: `print`
        The print function.
    **kwargs
        Passed to `average_periods` and `average_regions`.
    """
    # Load data and perform averages
    # NOTE: Need to extract 'plev' top and bottom because for some reason they get
    # dropped from 'input' since they are defined on the coordinate dictionary.
    print = printer or builtins.print
    statistic = 'slope' if forcing is None else 'ratio'
    print(f'Calculating radiative feedbacks using {statistic}s.')
    fluxes = fluxes.climo.add_cell_measures()
    fluxes = fluxes.climo.quantify()
    print('Getting average time periods.')
    keys_periods = ('annual', 'seasonal', 'monthly')
    kw_periods = {key: kwargs.pop(key) for key in keys_periods if key in kwargs}
    fluxes = average_periods(fluxes, **kw_periods)
    print('Getting average spatial regions.')
    keys_regions = ('point', 'latitude', 'hemisphere', 'globe')
    kw_regions = {key: kwargs.pop(key) for key in keys_regions if key in kwargs}
    denoms = average_regions(fluxes['ts'], **kw_regions)
    if kwargs:
        raise ValueError(f'Got unexpected keyword argument(s): {kwargs}')

    # Iterate over fluxes
    # NOTE: Since forcing is constant, the cloud masking adjustment has no effect on
    # feedback estimates, but does affect the 'cl_erf' cloud adjustment forcing. Also
    # shortwave clear-sky effects are just albedo and small water vapor effect, so very
    # small, but longwave component is always very significant.
    outputs = {}
    for region, boundary, wavelength, component in itertools.product(
        ('point', 'latitude', 'hemisphere', 'globe'),
        ('TOA', 'surface'),
        ('full', 'longwave', 'shortwave'),
        FEEDBACK_DESCRIPTIONS,
    ):
        # Get the flux names
        # NOTE: Here if a component has no 'dependencies' it does not
        # exist so skip (e.g. temperature shortwave).
        if region not in denoms.region.values:
            continue
        if wavelength == 'full' and component == '':
            print(f'Calculating {region} {boundary} forcing and feedback.')
        if wavelength == 'full' and component == '' and boundary == 'TOA':
            outputs[region] = output = {}
        rad = f'r{wavelength[0]}n{boundary[0].lower()}'
        descrip = FEEDBACK_DESCRIPTIONS[component]
        if component in ('', 'cs'):
            name = f'{rad}{component}'
        elif wavelength == 'full':
            name = f'{component}_{rad}'
        elif all(keys for keys in FEEDBACK_DEPENDENCIES[component].values()):
            name = f'{component}_{rad}'
        else:  # skips e.g. non-full planck and albedo
            continue

        # Warning message for missing components
        # NOTE: This depends on calculating the net flux feedbacks before
        # the cloud and residual components.
        skies = ('', 'cs') if component in ('cl', 'resid') else ()
        masks = list(f'{rad}{sky}_erf' for sky in skies)
        if (missing := name) not in fluxes or (missing := 'ts') not in fluxes:
            print(
                'Warning: Input dataset is missing the feedback '
                f'dependency {missing!r}. Skipping calculation.'
            )
            continue
        if message := ', '.join(repr(mask) for mask in masks if mask not in output):
            print(
                'Warning: Output dataset is missing cloud-masking forcing '
                f'adjustment variable(s) {message}. Cannot make adjustment.'
            )

        # Get the components and possibly adjust for forcing masking
        # NOTE: Soden et al. (2008) used standard horizontally uniform value of 15%
        # the full forcing but no reason not to use regressed forcing estimates.
        numer = fluxes[name]  # pointwise radiative flux
        denom = denoms.sel(region=region)  # possibly averaged temperature
        component = 'net' if component == '' else component
        if masks and all(mask in output for mask in masks):
            mask = output[masks[0]] - output[masks[1]]
            if wavelength == 'full' and component == 'cl':
                min_, max_, mean = mask.min().item(), mask.max().item(), mask.mean().item()  # noqa: E501
                print(format('  mask flux:', ' <12s'), end=' ')
                print(f'min {min_: <+7.2f} max {max_: <+7.2f} mean {mean: <+7.2f}')
            with xr.set_options(keep_attrs=True):
                if component == 'cl':  # remove masking effect
                    numer = numer - mask * ureg('W m^-2')
                else:  # add back masking effect
                    numer = numer + mask * ureg('W m^-2')
            del mask
            gc.collect()

        # Perform the regression or division
        # See: https://en.wikipedia.org/wiki/Simple_linear_regression
        # NOTE: When forcing is provided time-varying input will produce time-varying
        # feedbacks similar to Armour 2015 or just simple scalar values if data is
        # already time-averaged. When not provided time coordinate is required.
        # NOTE: Previously did separate regressions with and without intercept...
        # but piControl regressions are *always* centered on origin because they
        # are deviations from the average by *construction*. So unnecessary.
        prefix = boundary if wavelength == 'full' else f'{boundary} {wavelength}'
        descrip = 'net' if wavelength == 'full' and component == '' else descrip
        if forcing is not None:
            erf = forcing[f'{name}_erf'].sel(region=region, drop=True)
            erf = erf.climo.quantify()
            lam = (numer - 2.0 * erf) / denom  # time already averaged
        elif 'time' in numer.sizes:
            nm, dm = numer.mean('time', skipna=False), denom.mean('time', skipna=False)
            lam = ((denom - dm) * (numer - nm)).sum('time') / ((denom - dm) ** 2).sum('time')  # noqa: E501
            erf = 0.5 * (nm - lam * dm)  # possibly zero minus zero
        else:
            raise ValueError('Time coordinate required for slope-style feedbacks.')
        del numer, denom
        gc.collect()

        # Standardize the results
        # NOTE: Always keep non-net forcing estimates as these represent rapid
        # adjustments. Also previously also recored equilibrium climate sensitivity
        # but this is always nonsense on local scales, so now compute a posterior only.
        lam = lam.climo.to_units('W m^-2 K^-1')
        lam = lam.climo.dequantify()
        lam.name = f'{name}_lam'
        lam.attrs['long_name'] = f'{prefix} {descrip} feedback parameter'
        output[lam.name] = lam
        if wavelength == 'full':
            min_, max_, mean = lam.min().item(), lam.max().item(), lam.mean().item()
            print(format(f'  {component} lam:', ' <12s'), end=' ')
            print(f'min {min_: <+7.2f} max {max_: <+7.2f} mean {mean: <+7.2f} ({descrip})')  # noqa: E501
        erf = erf.climo.to_units('W m^-2')
        erf = erf.climo.dequantify()
        erf.name = f'{name}_erf'  # halved from quadrupled co2
        erf.attrs['long_name'] = f'{prefix} {descrip} effective forcing'
        output[erf.name] = erf
        if wavelength == 'full':
            min_, max_, mean = erf.min().item(), erf.max().item(), erf.mean().item()
            print(format(f'  {component} erf:', ' <12s'), end=' ')
            print(f'min {min_: <+7.2f} max {max_: <+7.2f} mean {mean: <+7.2f} ({descrip})')  # noqa: E501
        del erf, lam
        gc.collect()

    # Concatenate result along 'region' axis
    # NOTE: This should have skipped impossible combinations
    print('Concatenating feedback regions.')
    coord = xr.DataArray(
        list(outputs),
        dims='region',
        name='region',
        attrs={'long_name': 'temperature averaging region'}
    )
    output = xr.concat(
        tuple(xr.Dataset(output) for output in outputs.values()),
        dim=coord,
        coords='minimal',
        compat='equals',
    )
    output.attrs['title'] = (
        f'{statistic}-derived forcing-feedback decompositions'
    )
    for key in ('pbot', 'ptop'):
        if key not in fluxes:
            continue
        data = fluxes[key]
        if 'time' in data.dims:  # i.e. seasonal averages as function of year
            data = data.mean(dim='time', keep_attrs=True)
        output[key] = data.climo.dequantify()

    # Add final pattern effect term
    # NOTE: Region coordinate is necessary for pattern effect so it can be shown
    # for different feedback versions. Non-globe values get auto-filled with nan.
    print('Calculating pattern effect term.')
    if pattern:
        point = denoms.sel(region='point')
        globe = denoms.sel(region='globe')
        if forcing is not None:
            data = point / globe  # simply the ratio of differences
        elif 'time' in point.sizes:
            pm, gm = point.mean('time', skipna=False), globe.mean('time', skipna=False)
            data = ((globe - gm) * (point - pm)).sum('time') / ((globe - gm) ** 2).sum('time')  # noqa: E501
        else:
            raise ValueError('Time coordinte required for slope-style pattern effect.')
        data = data.climo.dequantify()
        data.attrs['units'] = 'K / K'
        data.attrs['long_name'] = 'relative surface warming'
        output['tpat'] = data.assign_coords(region='globe').expand_dims('region')
    return output


def get_feedbacks(
    feedbacks='~/data',
    fluxes='~/data',
    kernels='~/data/cmip-kernels',
    select=None,
    response=None,
    source=None,
    ratio=None,
    project=None,
    experiment=None,
    ensemble=None,
    table=None,
    model=None,
    nodrift=False,
    overwrite=False,
    printer=None,
    dryrun=False,
    **inputs
):
    """
    Calculate the net surface and top-of-atmosphere feedback components with a variety
    of averaging conventions and save the result in an automatically-named file. Also
    save the flux component calculations in an intermediate location.

    Parameters
    ----------
    feedbacks : path-like, optional
        The feedback directory. Subfolder ``{project}-{experiment}-feedbacks`` is used.
    fluxes : path-like, optional
        The flux directory. Subfolder ``{project}-{experiment}-fluxes`` is used.
    kernels : path-like, optional
        The kernel data directory. Default folder is ``~/cmip-kernels``.
    select : 2-tuple of int, optional
        The start and stop years. Used in the output file name. If relevant the anomaly
        data is filtered to these years using ``.isel(slice(12 * start, 12 * stop))``.
    response : 2-tuple of int, optional
        The full response start and stop years. Used to load flux data containing a
        superset of `select` times and to load forcing data for ratio-style feedbacks.
    source : str, default: 'eraint'
        The source for the kernel data (e.g. ``'eraint'``). This searches for files
        in the `kernels` directory formatted as ``kernels_{source}.nc``.
    ratio : bool, optional
        Whether to compute feedbacks with a ratio of a regression. The latter is
        only possible with abrupt response data where there is timescale separation.
    **inputs : tuple of path-like lists
        Tuples of ``(control_inputs, response_inputs)`` for the variables
        required to compute the feedbacks, passed as keyword arguments for
        each variable. The feedbacks computed will depend on the variables passed.

    Other parameters
    ----------------
    project : str, optional
        The project. Used in the output folder and for level checking.
    experiment : str, optional
        The experiment. Used in the output folder and file name.
    table, ensemble, model : str, optional
        The table id, ensemble name, model name. Used in the output file name.
    nodrift : bool, optional
        Whether to append a ``-nodrift`` indicator to the end of the filename.
    overwrite : bool, default: False
        Whether to overwrite existing output flux and feedback files.
    printer : callable, default: `print`
        The print function.
    dryrun : bool, optional
        Whether to run with only three years of data.

    Returns
    -------
    path : Path
        The path of the output file, formatted according to cmip conventions as
        ``feedbacks_{table}_{model}_{experiment}_{ensemble}_{options}.nc`` where
        `table`, `model`, `experiment`, and `ensemble` are applied only if passed
        by the user and `options` is set to ``{source}-{statistic}-{nodrift}``,
        where `source` is the kernel source; `statistic` is one of ``regression``
        or ``ratio``; and ``-nodrift`` is added only if ``nodrift`` is ``True``.

    Important
    ---------
    The output file will contain the results of three different feedback breakdowns.
    For the latter two breakdowns, a residual is computed by subtracting the canonical
    net feedback, calculated as net all-sky longwave plus shortwave versus surface
    temperature, from the sum of the individual component feedbacks.

    1. Simple all-sky and clear-sky regressions of the net longwave and net shortwave
       response against surface temperature. The clear-sky regressions are needed
       for cloud feedback estimates (see paragraph below).
    2. Traditional feedback kernel breakdowns, where the temperature feedback is split
       into a constant-temperature planck feedback and difference-with-height lapse
       rate feedback. The water vapor feedback is kept separate.
    3. Relative humidity feedback breakdowns, where the temperature and water vapor
       feedbacks are split into relative humidity-preserving constant-temperature
       response, relative humidity-preserving deviations from the constant-temperature
       response, and the relative humidity change at each level.

    The all-sky regressions are useful where more elaborate breakdowns are
    unavailable (note dominant longwave and shortwave inter-model spread is from
    longwave and shortwave cloud feedbacks). They are also needed for the ratio
    feedback calculations to plug in as estimates for the effective forcing (when
    ``ratio=True`` this function looks for ``'slope'`` files to retrieve the
    forcing estimates). The all-sky and clear-sky regressions are both needed for an
    estimate of cloud masking of the radiative forcing to plug into the Soden et al.
    adjusted cloud radiative effect estimates to get the effective cloud forcing.
    """
    # Parse input arguments
    # NOTE: Here 'ratio' should only ever be used with abrupt forcing experiments
    # where there is timescale separation of anomaly magnitudes.
    # NOTE: The 'fluxes' files also need a regression/ratio indicator because former
    # is loaded as a time series while latter is loaded as climate averages.
    print = printer or builtins.print
    project = (project or 'cmip6').lower()
    select = select or (0, 150)
    response = response or (0, 150)
    source = source or 'eraint'
    statistic = 'ratio' if ratio else 'slope'
    nodrift = nodrift and 'nodrift' or ''
    outputs = []
    subfolder = _item_join((project, experiment, table))
    tuples = (  # try to load from parent flux file if possible
        (fluxes, 'fluxes', select),
        (fluxes, 'fluxes', response),
        (feedbacks, 'feedbacks', select),
        (feedbacks, 'feedbacks', response),
    )
    for folder, prefix, times in tuples:
        file = _item_join(
            prefix, table, model, experiment, ensemble,
            (*(format(int(t), '04d') for t in times), source, statistic, nodrift),
            modify=False
        ) + '.nc'
        path = Path(folder).expanduser() / subfolder / file
        path.parent.mkdir(exist_ok=True)
        outputs.append(path)
    if message := ', '.join(
        f'{variable}={paths}' for variable, paths in inputs.items()
        if not isinstance(paths, (tuple, list))
        or len(paths) != 2 or not all(isinstance(path, (str, Path)) for path in paths)
    ):
        raise TypeError(f'Unexpected kwargs {message}. Must be 2-tuple of paths.')

    # Load kernels and cimpute flux components and feedback estimates
    # NOTE: Try to load from same fluxes files for feedbacks estimated from subset
    # of full time series to avoid duplicating expensive calculations.
    # NOTE: Always overwrite if at least flux data is missing so that we never have
    # feedback data inconsistent with the flux data on storage.
    # NOTE: Even for kernel-derived flux responses, rapid adjustments and associated
    # pattern effects may make the effective radiative forcing estimate non-zero (see
    # Andrews et al.) so we always save the regression intercept data.
    *fluxes, feedbacks, forcing = outputs
    fluxes_exist = tuple(flux.is_file() and flux.stat().st_size > 0 for flux in fluxes)
    feedbacks_exist = feedbacks.is_file() and feedbacks.stat().st_size > 0
    overwrite = overwrite or feedbacks_exist and not any(fluxes_exist)
    if not overwrite and not dryrun and any(fluxes_exist):
        output = fluxes[0] if fluxes_exist[0] else fluxes[1]
        print(f'Loading flux data from file: {output.name}')
        fluxes = load_file(output, validate=False)
        if not fluxes_exist[0]:  # must subselect from loaded data
            init, start, stop = *response[:1], *select
            if init is not None:
                init = int(init) * 12
            if start is not None:
                start = int(start) * 12
            if stop is not None:
                stop = int(stop) * 12
            if init is not None:
                start, stop = start - init, stop - init  # convert to relative years
            if fluxes.sizes.get('time', 12) > 12:
                fluxes = fluxes.isel(time=slice(start, stop))
    else:  # here _anomalies_from_files will filter time selection
        output = fluxes[0]  # fluxes[1] only used as a source; saved file is fluxes[0]
        print(f'Output flux file: {output.name}')
        kernels = Path(kernels).expanduser() / f'kernels_{source}.nc'
        print(f'Loading kernel data from file: {kernels.name}')
        kernels = load_file(kernels, project=project, validate=False)
        kwargs = dict(select=select, project=project, dryrun=dryrun)
        anoms = _anomalies_from_files(printer=print, **kwargs, **inputs)
        fluxes = _fluxes_from_anomalies(anoms, kernels=kernels, printer=print)
        for dataset in (anoms, kernels):
            dataset.close()
        if not dryrun:  # save after compressing repeated values
            encoding = {key: {'zlib': True, 'complevel': 5} for key in fluxes.data_vars}
            output.unlink(missing_ok=True)
            fluxes.to_netcdf(output, engine='netcdf4', encoding=encoding)
            print(f'Created output file: {output.name}')
    if not overwrite and not dryrun and feedbacks_exist:
        output = feedbacks
        print(f'Loading feedback data from file: {output.name}')
        feedbacks = load_file(output, validate=False)
    else:
        output = feedbacks  # use the name 'feedbacks' for dataset
        print(f'Output feedback file: {output.name}')
        if ratio:
            forcing = forcing.parent / forcing.name.replace('ratio', 'slope')
            print(f'Loading forcing data from file: {forcing.name}')
            forcing = load_file(forcing, project=project, validate=False)
        else:
            forcing = None
        feedbacks = _feedbacks_from_fluxes(fluxes, forcing=forcing, printer=print)
        if forcing:
            forcing.close()
        if not dryrun:  # save after compressing repeated values
            encoding = {key: {'zlib': True, 'complevel': 5} for key in feedbacks.data_vars}  # noqa: E501
            output.unlink(missing_ok=True)
            feedbacks.to_netcdf(output, engine='netcdf4', encoding=encoding)
            print(f'Created feedbacks file: {output.name}')
    return feedbacks, fluxes


def process_feedbacks(
    *paths,
    ratio=None,
    source=None,
    control=None,
    response=None,
    experiment=None,
    project=None,
    nodrift=False,
    logging=False,
    dryrun=False,
    nowarn=False,
    **kwargs
):
    """
    Generate feedback deompositions using the output of `process_files`.

    Parameters
    ----------
    *paths : path-like, optional
        Location(s) for the climate and series data.
    ratio : bool, optional
        Whether to use ratio feedbacks.
    source : str, default: 'eraint'
        The source for the kernels.
    control : 2-tuple of int, default: (0, 150)
        The year range for the ``piControl`` climate data.
    response : 2-tuple of int, default: (0, 150) or (120, 150)
        The year range for the "response" climate data.
    experiment : str, optional
        The experiment to use for the "response" data.
    project : str, optional
        The project to use.
    nodrift : bool, default: False
        Whether to use drift-corrected data.
    logging : bool, optional
        Whether to build a custom logger.
    dryrun : bool, optional
        Whether to run with only three years of data.
    nowarn : bool, optional
        Whether to always raise errors instead of warnings.
    **kwargs
        Passed to `get_feedbacks`.
    """
    # Find files and restrict to unique constraints
    # NOTE: This requires flagship translation or else models with different control
    # and abrupt runs are not grouped together. Not sure how to handle e.g. non-flagship
    # abrupt runs from flagship control runs but cross that bridge when we come to it.
    # NOTE: Paradigm is to use climate monthly mean surface pressure when interpolating
    # to model levels and keep surface pressure time series when getting feedback
    # kernel integrals. Helps improve accuracy since so much stuff depends on kernels.
    logging = logging and not dryrun
    statistic = 'ratio' if ratio else 'slope'
    experiment = experiment or 'abrupt4xCO2'
    source = source or 'eraint'
    nodrift = 'nodrift' if nodrift else ''
    suffix = nodrift and '-' + nodrift
    control = control or (0, 150)
    if ratio:  # ratio-type
        response = response or (120, 150)
        control_suffix = f'{control[0]:04d}-{control[1]:04d}-climate{suffix}'
        response_suffix = f'{response[0]:04d}-{response[1]:04d}-climate{suffix}'
    else:  # NOTE: get anomaly data from time series file matching control years
        response = response or (0, 150)
        control_suffix = f'{control[0]:04d}-{control[1]:04d}-climate{suffix}'
        response_suffix = f'{control[0]:04d}-{control[1]:04d}-series{suffix}'
    response, select = control, response
    suffixes = (*(format(int(t), '04d') for t in select), source, statistic, nodrift)
    if logging:
        print = Logger('feedbacks', *suffixes, project=project, experiment=experiment, table='Amon')  # noqa: E501
    else:
        print = builtins.print
    print('Generating database.')
    constraints = {
        'variable': sorted(set(k for d in VARIABLE_DEPENDENCIES.values() for k in d)),
        'experiment': ['piControl', experiment],
        'table': 'Amon',
    }
    constraints.update({
        key: kwargs.pop(key) for key in tuple(kwargs)
        if any(s in key for s in ('model', 'flagship', 'ensemble'))
    })
    files, *_ = glob_files(*paths, project=project)
    facets = ('project', 'model', 'ensemble', 'grid')
    database = Database(files, facets, project=project, **constraints)
    if experiment == 'piControl':  # otherwise translate for cmip5 or cmip6
        control_experiment = response_experiment = experiment
    else:
        control_experiment, response_experiment = database.constraints['experiment']

    # Calculate clear and all-sky feedbacks surface and TOA files
    # NOTE: Unlike the method that uses a fixed SST experiment to deduce forcing and
    # takes the ratio of local radiative flux change to global temperature change as
    # the feedback, the average of a regression of radiative flux change against global
    # temperature change will not necessarily equal the regression of average radiative
    # flux change. Therefore compute both and compare to get idea of the residual. Also
    # consider local temperature change for point of comparison (Hedemman et al. 2022).
    print(f'Input files ({len(database)}):')
    print(*(f'{key}: ' + ' '.join(opts) for key, opts in database.constraints.items()), sep='\n')  # noqa: E501
    print(f'Control data: experiment {control_experiment} suffix {control_suffix}')
    print(f'Response data: experiment {response_experiment} suffix {response_suffix}')
    _print_error = lambda error: print(
        ' '.join(traceback.format_exception(None, error, error.__traceback__))
    )
    for group, data in database.items():
        group = dict(zip(database.group, group))
        print()
        print(f'Computing {statistic} feedbacks:')
        print(', '.join(f'{key}: {value}' for key, value in group.items()))
        files = _get_file_pairs(
            data,
            control_experiment,
            control_suffix,
            response_experiment,
            response_suffix,
            printer=print,
        )
        try:
            datasets = get_feedbacks(
                project=group['project'],
                experiment=_item_parts['experiment'](tuple(files.values())[0][1]),
                ensemble=_item_parts['ensemble'](tuple(files.values())[0][1]),
                table=_item_parts['table'](tuple(files.values())[0][1]),
                model=group['model'],
                select=select,
                response=response,
                ratio=ratio,
                source=source,
                nodrift=nodrift,
                printer=print,
                dryrun=dryrun,
                **kwargs,
                **files,
            )
            for dataset in datasets:
                dataset.close()
        except Exception as error:
            if dryrun or nowarn:
                raise error
            else:
                _print_error(error)
            print('Warning: Failed to compute feedbacks.')
            continue
