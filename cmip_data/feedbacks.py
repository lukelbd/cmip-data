#!/usr/bin/env python3
"""
Calculate feedback, forcing, and sensitivity estimates for ESGF data.
"""
import builtins
import itertools
import traceback
from pathlib import Path

import numpy as np  # noqa: F401
import xarray as xr
from climopy import const, ureg
from icecream import ic  # noqa: F401
from idealized import physics  # noqa: F401
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
from .utils import open_file, average_periods, average_regions

__all__ = [
    'compute_feedbacks',
    'process_feedbacks',
    'clausius_clapeyron',
    'response_anomalies',
    'response_fluxes',
    'response_feedbacks',
]


# Global constants
# NOTE: Here 'n' stands for net (follows Angie convention). These variables are built
# from the 'u' and 'd' components of cmip fluxes and to prefix the kernel-implied net
# fluxes 'alb', 'pl', 'lr', 'hu', 'cl', and 'resid' in response_fluxes.
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
    'ts': ('ts',),  # anomaly data
    'ta': ('ta',),  # anomaly data
    'hus': ('hus',),  # anomaly data
    'alb': ('rsus', 'rsds'),  # out of and into the surface
    'tapi': ('ta',),  # control data for humidity kernels and integration bounds
    'pspi': ('ps',),  # control data for integration bounds
    # 'ta4x': ('ta',),  # response data for integration bounds
    # 'ps4x': ('ps',),  # response data for integration bounds
    'rlnt': ('rlut',),  # out of the atmosphere
    'rsnt': ('rsut',),  # out of the atmosphere (ignore constant rsdt)
    'rlntcs': ('rlutcs',),  # out of the atmosphere
    'rsntcs': ('rsutcs',),  # out of the atmosphere (ignore constant rsdt)
    'rlns': ('rlds', 'rlus'),  # out of and into the atmosphere
    'rsns': ('rsds', 'rsus'),  # out of and into the atmosphere
    'rlnscs': ('rldscs', 'rlus'),  # out of and into the atmosphere
    'rsnscs': ('rsdscs', 'rsuscs'),  # out of and into the atmosphere
}


def _file_pairs(data, *args, printer=None):
    """
    Return a dictionary mapping of variables to ``(control, response)`` file
    pairs from a database of files with ``(experiment, ..., variable)`` keys.

    Parameters
    ----------
    data : dict
        The database of file lists.
    *experiments : str
        The two control and response experiments.
    *suffixes
        The two experiment suffixes.
    printer : callable, default: `print`
        The print function.
    """
    print = printer or builtins.print
    pairs = []
    for experiment, suffix in zip(args[:2], args[2:]):
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
    return pairs


def clausius_clapeyron(ta, pa=None):
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
    es_liq = lambda t: np.exp(  # input kelvin, output pascals
        54.842763
        - 6763.22 / t
        - 4.21 * np.log(t)
        + 0.000367 * t
        + np.tanh(0.0415 * (t - 218.8))
        * (53.878 - 1331.22 / t - 9.44523 * np.log(t) + 0.014025 * t)
    )
    es_ice = lambda t: np.exp(  # input kelvin, output pascals
        9.550426
        - 5723.265 / t
        + 3.53068 * np.log(t)
        - 0.00728332 * t
    )

    # Get scaling
    # WARNING: If using metpy, critical to replace climopy units with metpy units, or
    # else get hard-to-debug errors, e.g. temperature units mysteriously stripped. In
    # the end decided to go manual since metpy uses inaccurate Bolton (1980) method.
    el0, el1 = es_liq(ta.data - 0.5), es_liq(ta.data + 0.5)
    ei0, ei1 = es_ice(ta.data - 0.5), es_ice(ta.data + 0.5)
    es0, es1 = el0.copy(), el1.copy()
    es0[mask] = ei0[mask := ta.data - 0.5 < 273.15]
    es1[mask] = ei1[mask := ta.data + 0.5 < 273.15]
    es_hus = lambda e, p: (
        const.eps.magnitude * e / (p.data - (1 - const.eps.magnitude) * e)
    )
    if pa is None:  # i.e. d(K) / d(log(es)) = es / d(es)
        scale = 1 / np.log(es1 / es0)
    else:  # i.e. 1 / d(log(qs)) = 1 / (log(qs1) - log(qs))
        scale = 1 / np.log(es_hus(es1, pa.data) / es_hus(es0, pa.data))
    scale = xr.DataArray(
        scale,
        dims=tuple(ta.sizes),
        coords=dict(ta.coords),
        attrs={'units': 'K', 'long_name': 'inverse Clausius-Clapeyron scaling'}
    )
    return scale.climo.quantify()


def response_anomalies(printer=None, project=None, dryrun=False, **inputs):
    """
    Return dataset containing response minus control anomalies for the variables given
    pairs of paths to its dependencies (e.g. ``'rlds'`` and ``'rlus'`` for ``'rlns'``).

    Parameters
    ----------
    **inputs : 2-tuple of path-like
        Tuples of response and control paths for the dependencies by name.
    project : str, optional
        The project. Used in level checking.
    printer : callable, default: `print`
        The print function.
    dryrun : bool, optional
        Whether to run with only three years of data.
    """
    # Iterate over dependencies
    # NOTE: Since incoming solar is constant, and since all our regressions are
    # with anomalies with respect to base climate, the incoming solar terms cancel
    # out and we can just consider upwelling solar at the top-of-atmosphere.
    print = printer or builtins.print
    project = (project or 'cmip6').lower()
    output = xr.Dataset()
    print('Calculating response minus control anomalies.')
    for variable, dependencies in VARIABLE_DEPENDENCIES.items():
        # Load datasets and prepare the variables
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
        for dependency in dependencies:  # upwelling and downwelling component
            control, response = inputs[dependency]  # control and response paths
            for path, datas in zip((control, response), (controls, responses)):
                path = Path(path).expanduser()
                data = open_file(path, dependency, project=project, printer=print)
                if dryrun:
                    data = data.isel(time=slice(None, 36))
                data.name = variable
                datas.append(data)

        # Calculate variable anomalies using dependencies
        # NOTE: Here subtraction from 'groupby' objects can only be used if the array
        # has a coordinate with the corresponding indices, i.e. in this case 'month'
        # instead of 'time'. Use .groupby('time.month').mean() to turn time indices
        # into month indices (works for both control time series and pre-averaged data).
        with xr.set_options(keep_attrs=True):
            if variable in ('tapi', 'pspi'):  # control data matched to response times
                (control,), (response,) = controls, responses
                control = (0 * response).groupby('time.month') + control.groupby('time.month').mean()  # noqa: E501
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
            if response is None:
                data = control
            elif control is None:
                data = response
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
        if variable in ('tapi', 'pspi'):
            long_name = f'control {long_name}'
            pass  # WARNING: keep standard name to use as tropopause
        elif variable in ('ta4x', 'ps4x'):
            long_name = f'response {long_name}'
            data.attrs.pop('standard_name', None)  # WARNING: remove to use tropopause
        else:
            long_name = f'{long_name} anomaly'
            data.attrs.pop('standard_name', None)
        data.attrs['long_name'] = long_name
        output[variable] = data
        print(f'  {variable} ({long_name}): {len(data.time)}')

    return output


def response_fluxes(anoms, kernels, printer=None):
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
        kernels = kernels.groupby('time.month').mean()  # see response_anomalies notes
    levs = kernels.plev.values  # NESM3 abrupt-4xCO2 data contains missing levels
    if anoms.sizes.get('plev', np.inf) < kernels.sizes['plev']:
        levs = [lev for lev in levs if np.any(np.isclose(lev, plev.values))]
    kernels = kernels.sel(plev=levs)  # note open_file() already restricts levels
    kernels = kernels.climo.quantify()
    kernels = kernels.assign_coords(plev=plev)  # critical (see above)

    # Get cell height measures and Clausius-Clapeyron adjustments
    # NOTE: Initially used response time series for tropopause consistent with
    # Zelinka stated methodology of 'time-varying tropopause' but got bizarre issue
    # with Planck feedback biased positive for certain models (seemingly due to higher
    # surface temperature associated with higher tropopause and more negative
    # fluxes... so should be negative bias... but whatever). This also made little
    # sense because it corrected for changing troposphere depth without correcting
    # for reduced strength of kernels under increased depth. Now try with both control
    # data and response data average (note that groupby operations may be no-ops if
    # we are already working with control data matched to the response times).
    # WARNING: Currently cell_height() in physics.py automatically adds surface bounds,
    # tropopause bounds, and cell heights as *coordinates* (for consistency with xarray
    # objects). Promote to variable before working with other arrays to avoid conflicts.
    print('Calculating cell measures for vertical integration.')
    anoms = anoms.climo.add_cell_measures(surface=True, tropopause=True)
    anoms = anoms.reset_coords(('plev_bot', 'plev_top'))  # critical (see above)
    with xr.set_options(keep_attrs=True):  # critical (preserve cell_measures)
        height = const.g * anoms.climo.coords.cell_height
        height = height.climo.to_units('Pa').climo.dequantify()
    anoms = anoms.assign_coords(cell_height=height)
    anoms = anoms.climo.quantify()
    min_, max_, mean = height.min().item(), height.max().item(), height.mean().item()
    print(f'Cell height range: min {min_:.0f} max {max_:.0f} mean {mean:.0f}')
    pbot = anoms.plev_bot.climo.dequantify()
    pbot.attrs['long_name'] = 'surface pressure'
    min_, max_, mean = pbot.min().item(), pbot.max().item(), pbot.mean().item()
    print(f'Surface pressure range: min {min_:.0f} max {max_:.0f} mean {mean:.0f}')
    ptop = anoms.plev_top.climo.dequantify()
    ptop.attrs['long_name'] = 'tropopause pressure'
    min_, max_, mean = ptop.min().item(), ptop.max().item(), ptop.mean().item()
    print(f'Tropopause pressure range: min {min_:.0f} max {max_:.0f} mean {mean:.0f}')
    ta, pa = anoms.tapi.climo.dequantify(), anoms.climo.coords.plev.climo.dequantify()
    scale = clausius_clapeyron(ta, pa)
    min_, max_, mean = scale.min().item(), scale.max().item(), scale.mean().item()
    print(f'Clausius-Clapeyron range: min {min_:.2f} max {max_:.2f} mean {mean:.2f}')

    # Iterate over flux components
    # NOTE: Here cloud feedback comes about from change in cloud radiative forcing (i.e.
    # R_cloud_abrupt - R_cloud_control where R_cloud = R_clear - R_all) then corrected
    # for masking of clear-sky effects (e.g. by adding dt * K_t_cloud where similarly
    # K_t_cloud = K_t_clear - K_t_all). Additional greenhouse forcing masking
    # correction is applied when we estimate feedbacks.
    dict_ = {'pbot': pbot, 'ptop': ptop, 'ts': anoms.ts.climo.dequantify()}
    output = xr.Dataset(dict_)
    for boundary, wavelength, (component, descrip) in itertools.product(
        ('TOA', 'surface'), ('longwave', 'shortwave'), FEEDBACK_DESCRIPTIONS.items()
    ):
        # Skip feedbacks without shortwave or longwave components.
        # NOTE: If either longwave or shortwave component is missing it will be excluded
        # from the "longwave plus shortwave" calculations below. Also only exclude cloud
        # and residual components when the actual model radiative flux is missing.
        rad = f'r{wavelength[0]}n{boundary[0].lower()}'
        rads = (rad,)  # kernel fluxes
        variables = FEEDBACK_DEPENDENCIES[component][wavelength]  # empty for fluxes
        dependencies = list(variables)  # anomaly and kernel variables
        if component == '':
            dependencies.append(rad)
            running = 0.0 * ureg('W m^-2')
            print(f'Calculating {wavelength} {boundary} fluxes using kernel method.')
        else:
            if component == 'cs':
                dependencies.append(rad := f'{rad}cs')
            elif component == 'cl' or component == 'resid':
                dependencies.extend(rads := (rad, f'{rad}cs'))  # used for cloud masking
            elif not dependencies:
                continue
        if message := ', '.join(
            repr(name) for name in dependencies if name not in anoms
        ):
            print(
                f'Warning: {wavelength.title()} flux {component=} is missing '
                f'variable dependencies {message}. Skipping {wavelength} flux estimate.'  # noqa: E501
            )
            continue
        if message := ', '.join(
            repr(name) for var in variables for rad in rads
            if (name := f'{var}_{rad}') not in kernels
        ):
            print(
                f'Warning: {wavelength.title()} flux {component=} is missing '
                f'kernel dependencies {message}. Skipping {wavelength} flux estimate.'
            )
            continue

        # Component fluxes
        # NOTE: Need to assign cell height coordinates for Planck feedback since
        # they are not defined on the kernel data array before multiplication.
        if component == '' or component == 'cs':  # net flux
            data = 1.0 * anoms[rad]
        elif component == 'resid':  # residual flux
            data = anoms[rad] - running
        elif component == 'alb':  # albedo flux (longwave skipped above)
            anom = anoms['alb']
            data = anom.groupby('time.month') * kernels[f'alb_{rad}']
        elif component[:2] == 'pl':  # planck flux (shortwave skipped for 'pl')
            kernel = 0.0 * ureg('W m^-2 Pa^-1 K^-1')
            if wavelength[0] == 'l':
                kernel = kernel + kernels[f'ta_{rad}']
            if component == 'pl*':
                kernel = kernel + kernels[f'hus_{rad}']
            anom = anoms['ts']  # has no cell height
            data = kernel * anom.groupby('time.month')
            data = data.assign_coords(cell_height=height).climo.integral('plev')
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
            if component == 'hur':
                anom = scale * anoms['hus'] - anoms['ta']
            else:
                anom = scale * anoms['hus']
            data = kernel * anom.groupby('time.month')
            data = data.climo.integral('plev')
        elif component == 'cl':  # shortwave and longwave cloud fluxes
            data = anoms[rad] - anoms[f'{rad}cs']
            for var in variables:  # relevant longwave or shortwave variables
                kernel = kernels[f'{var}_{rad}'] - kernels[f'{var}_{rad}cs']
                part = kernel * anoms[var].groupby('time.month')  # mask adjustment
                if var == 'hus':
                    part = scale * part
                if var in ('ta', 'hus'):
                    part = part.climo.integral('plev')
                data = data - part.climo.to_units('W m^-2')  # apply adjustment
        else:
            raise RuntimeError

        # Update the input dataset with resulting component fluxes.
        # NOTE: Calculate residual feedback by summing traditional component
        # feedbacks from Soden and Held.
        name = rad if component in ('', 'cs') else f'{component}_{rad}'
        descrip = descrip and f'{descrip} flux' or 'flux'  # for lon gname
        component = component or 'net'  # for print message
        data.name = name
        data.attrs['long_name'] = f'net {boundary} {wavelength} {descrip}'
        data = data.climo.to_units('W m^-2')
        if component in ('pl', 'lr', 'hus', 'alb', 'cl'):
            running = data + running
        output[data.name] = data = data.climo.dequantify()
        min_, max_, mean = data.min().item(), data.max().item(), data.mean().item()
        print(format(f'  {component} flux:', ' <15s'), end=' ')
        print(f'min {min_: <+7.2f} max {max_: <+7.2f} mean {mean: <+7.2f} ({descrip})')

    # Combine shortwave and longwave components
    # NOTE: Could also compute feedbacks and forcings for just shortwave and longwave
    # components, then average them to get the net (linearity of offset and slope
    # estimators), but this is slightly easier and has only small computational cost.
    for boundary, (component, descrip) in itertools.product(
        ('TOA', 'surface'), FEEDBACK_DESCRIPTIONS.items()
    ):
        # Determine fluxes that should be summed to provide net balance. For
        # some feedbacks this is just longwave or just shortwave.
        if component == '':
            print(f'Calculating shortwave plus longwave {boundary} fluxes.')
        net = f'rfn{boundary[0].lower()}'  # use 'f' i.e. 'full' instead of wavelength
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

        # Sum component fluxes into net longwave plus shortwave fluxes
        # if both dependencies exist. Otherwise emit warning.
        if message := ', '.join(name for name in names if name not in output):
            print(
                f'Warning: Net radiative flux {component=} is missing longwave or '
                f'shortwave dependencies {message}. Skipping net flux estimate.'
            )
            continue
        name = f'{net}{component}' if component in ('', 'cs') else f'{component}_{net}'
        descrip = descrip and f'{descrip} flux' or 'flux'  # for long name
        component = component or 'net'  # for print message
        data = sum(output[name].climo.quantify() for name in names)  # no leakage here
        data.name = name
        data.attrs['long_name'] = f'net {boundary} {descrip}'
        data = data.climo.to_units('W m^-2')
        if len(names) == 1:  # e.g. drop the shortwave albedo 'component'
            output = output.drop_vars(names[0])
        output[name] = data = data.climo.dequantify()
        min_, max_, mean = data.min().item(), data.max().item(), data.mean().item()
        print(format(f'  {component} flux:', ' <15s'), end=' ')
        print(f'min {min_: <+7.2f} max {max_: <+7.2f} mean {mean: <+7.2f} ({descrip})')

    return output


def response_feedbacks(fluxes, forcing=None, printer=None):
    """
    Return a dataset containing feedbacks calculated along all `average_periods` periods
    (annual averages, seasonal averages, and month averages) and along all combinations
    of `average_regions` (points, latitudes, hemispheres, and global, with different
    averaging conventions used in the denominator indicated by the ``region`` coord).

    Parameters
    ----------
    fluxes : xarray.Dataset
        The source for the flux anomaly data. Should have been generated
        in `response_fluxes` using standardized radiative kernel data.
    forcing : path-like, optional
        Source for the forcing data. If passed then the time-average anomalies are
        used rather than regressions. This can be used with response-climate data.
    printer : callable, default: `print`
        The print function.
    """
    # Load data and perform averages
    # NOTE: Need to extract 'plev' top and bottom because for some reason they get
    # dropped from 'input' since they are defined on the coordinate dictionary.
    print = printer or builtins.print
    statistic = 'regression' if forcing is None else 'ratio'
    print(f'Calculating radiative feedbacks using {statistic}s.')
    fluxes = fluxes.climo.add_cell_measures()
    fluxes = fluxes.climo.quantify()
    print('Getting average time periods and spatial regions.')
    fluxes = average_periods(fluxes)
    denoms = average_regions(fluxes['ts'])  # concatenate possible averages

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
        if boundary == 'TOA' and wavelength == 'full' and component == '':
            outputs[region] = output = xr.Dataset()
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
        if wavelength == 'full' and component == '':
            print(
                f'Calculating {region}-average surface temperature '
                f'{boundary} forcing and feedback.'
            )
        if (missing := name) not in fluxes or (missing := 'ts') not in fluxes:
            print(
                'Warning: Input dataset is missing the feedback '
                f'dependency {missing!r}. Skipping calculation.'
            )
            continue

        # Get the components and possibly adjust for forcing masking
        # NOTE: Soden et al. (2008) used standard horizontally uniform value of 15%
        # the full forcing but no reason not to use regressed forcing estimates.
        numer = fluxes[name]  # pointwise radiative flux
        denom = denoms.sel(region=region)  # possibly averaged temperature
        prefix = boundary if wavelength == 'full' else f'{boundary} {wavelength}'
        descrip = 'net' if wavelength == 'full' and component == '' else descrip
        component = 'net' if component == '' else component
        if component not in ('cl', 'resid'):
            pass
        elif message := ', '.join(
            repr(erf) for sky in ('', 'cs')
            if (erf := f'{rad}{sky}_erf') not in output
        ):
            print(
                'Warning: Output dataset is missing cloud-masking forcing '
                f'adjustment variable(s) {message}. Cannot make adjustment.'
            )
        else:
            all_, clear = output[f'{rad}_erf'], output[f'{rad}cs_erf']
            prop = (mask := all_ - clear) / all_
            if wavelength == 'full' and component == 'cl':
                min_, max_, mean = mask.min().item(), mask.max().item(), mask.mean().item()  # noqa: E501
                print(format('  mask flux:', ' <12s'), end=' ')
                print(f'min {min_: <+7.2f} max {max_: <+7.2f} mean {mean: <+7.2f}')
                min_, max_, mean = prop.min().item(), prop.max().item(), prop.mean().item()  # noqa: E501
                print(format('  mask frac:', ' <12s'), end=' ')
                print(f'min {min_: <+7.2f} max {max_: <+7.2f} mean {mean: <+7.2f}')
            with xr.set_options(keep_attrs=True):
                if component == 'cl':  # remove masking effect
                    numer = numer - mask * ureg('W m^-2')
                else:  # add back masking effect
                    numer = numer + mask * ureg('W m^-2')

        # Perform the regression or division. Calculate climate sensitivity for
        # net fluxes only and always calculate forcing to include adjustments.
        # See: https://en.wikipedia.org/wiki/Simple_linear_regression
        # See: https://doi.org/10.1175/JCLI-D-12-00544.1 (Equation 2)
        # NOTE: Previously did separate regressions with and without intercept...
        # but piControl regressions are *always* centered on origin because they
        # are deviations from the average by *construction*. So unnecessary.
        if forcing is not None:
            erf = forcing[f'{name}_erf'].sel(region=region, drop=True)
            erf = erf.climo.quantify()
            lam = (numer - 2.0 * erf) / denom  # time already averaged
        else:
            nm, dm = numer.mean('time', skipna=False), denom.mean('time', skipna=False)
            lam = ((denom - dm) * (numer - nm)).sum('time') / ((denom - dm) ** 2).sum('time')  # noqa: E501
            erf = 0.5 * (nm - lam * dm)  # possibly zero minus zero
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

    # Concatenate result along 'region' axis
    # NOTE: This should have skipped impossible combinations
    coord = xr.DataArray(
        list(outputs),
        dims='region',
        name='region',
        attrs={'long_name': 'temperature averaging region'}
    )
    output = xr.concat(
        outputs.values(),
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
        if 'time' in data.coords:  # i.e. regression data
            data = data.mean(dim='time', keep_attrs=True)
        data = data.climo.dequantify()
        output[key] = data
    return output


def compute_feedbacks(
    feedbacks='~/data',
    fluxes='~/data',
    kernels='~/data',
    ratio=None,
    source=None,
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
        The feedback directory (subfolder ``{project}-{experiment}-feedbacks`` is used).
    fluxes : path-like, optional
        The flux directory (subfolder ``{project}-{experiment}-feedbacks`` is used).
    kernels : path-like, optional
        The kernel data directory (subfolder ``cmip-kernels`` is used).
    ratio : bool, optional
        Whether to compute feedbacks with a ratio of a regression. The latter is
        only possible with abrupt response data where there is timescale separation.
    source : str, default: 'eraint'
        The source for the kernel data (e.g. ``'eraint'``). This searches for files
        in the `kernels` directory formatted as ``kernels_{source}.nc``.
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
    ``ratio=True`` this function looks for ``'regression'`` files to retrieve the
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
    source = source or 'eraint'
    statistic = 'ratio' if ratio else 'regression'
    folder = _item_join((project, experiment, 'feedbacks'))
    file = _item_join(
        'feedbacks', table, model, experiment, ensemble,
        (source, statistic, nodrift and 'nodrift' or ''),
        modify=False
    ) + '.nc'
    feedbacks = Path(feedbacks).expanduser() / folder / file
    feedbacks.parent.mkdir(exist_ok=True)
    folder = _item_join((project, experiment, 'fluxes'))
    file = _item_join(
        'fluxes', table, model, experiment, ensemble,
        (source, statistic, nodrift and 'nodrift'),
        modify=False
    ) + '.nc'
    fluxes = Path(fluxes).expanduser() / folder / file
    fluxes.parent.mkdir(exist_ok=True)
    if message := ', '.join(
        f'{variable}={paths}' for variable, paths in inputs.items()
        if not isinstance(paths, (tuple, list))
        or len(paths) != 2 or not all(isinstance(path, (str, Path)) for path in paths)
    ):
        raise TypeError(f'Unexpected kwargs {message}. Must be 2-tuple of paths.')

    # Load kernels and cimpute flux components and feedback estimates
    # NOTE: Try to permit updating feedback conventions from expensive flux component
    # calculations, but never record new flux data potentially inconsistent with saved
    # feedback data, so always overwrite if at least flux data is missing.
    # NOTE: Even for kernel-derived flux responses, rapid adjustments and associated
    # pattern effects may make the effective radiative forcing estimate non-zero (see
    # Andrews et al.) so we always save the regression intercept data.
    fluxes_exist = fluxes.is_file() and fluxes.stat().st_size > 0
    feedbacks_exist = feedbacks.is_file() and feedbacks.stat().st_size > 0
    overwrite = overwrite or feedbacks_exist and not fluxes_exist
    output = fluxes  # use the name 'fluxes' for dataset
    print(f'Output flux file: {output.name}')
    if not overwrite and not dryrun and fluxes_exist:
        print(f'Loading flux data from file: {output.name}')
        fluxes = open_file(output, validate=False)
    else:
        kernels = Path(kernels).expanduser() / 'cmip-kernels' / f'kernels_{source}.nc'
        print(f'Loading kernel data from file: {kernels.name}')
        kernels = open_file(kernels, project=project, validate=False)
        fluxes = response_anomalies(project=project, printer=print, dryrun=dryrun, **inputs)  # noqa: E501
        fluxes = response_fluxes(fluxes, kernels=kernels, printer=print)
        if not dryrun:  # save after compressing repeated values
            encoding = {k: {'zlib': True, 'complevel': 5} for k in fluxes.data_vars}
            output.unlink(missing_ok=True)
            fluxes.to_netcdf(output, engine='netcdf4', encoding=encoding)
            print(f'Created output file: {output.name}')
    output = feedbacks  # use the name 'feedbacks' for dataset
    print(f'Output feedback file: {output.name}')
    if not overwrite and not dryrun and feedbacks_exist:
        print(f'Loading feedback data from file: {output.name}')
        feedbacks = open_file(output, validate=False)
    else:
        if ratio:
            forcing = output.parent / output.name.replace('ratio', 'regression')
            print(f'Loading forcing data from file: {forcing.name}')
            forcing = open_file(forcing, project=project, validate=False)
        else:
            forcing = None
        feedbacks = response_feedbacks(fluxes, forcing=forcing, printer=print)
        if not dryrun:  # save after compressing repeated values
            encoding = {k: {'zlib': True, 'complevel': 5} for k in feedbacks.data_vars}
            output.unlink(missing_ok=True)
            feedbacks.to_netcdf(output, engine='netcdf4', encoding=encoding)
            print(f'Created feedbacks file: {output.name}')
    return feedbacks, fluxes


def process_feedbacks(
    *paths,
    ratio=None,
    source=None,
    series=None,
    control=None,
    response=None,
    experiment=None,
    project=None,
    nodrift=False,
    logging=False,
    dryrun=False,
    **kwargs
):
    """
    Generate feedback deompositions using the output of `process_files`.

    Parameters
    ----------
    *paths : path-like, optional
        Location(s) of the climate and series data.
    ratio : bool, optional
        Whether to use ratio feedbacks.
    source : str, default: 'eraint'
        The source for the kernels.
    series : 2-tuple of int, default: (0, 150)
        The year range for the "response" time series data.
    control : 2-tuple of int, default: (0, 150)
        The year range for the ``piControl`` climate data.
    response : 2-tuple of int, default: (120, 150)
        The year range for the "response" climate data.
    experiment : str, optional
        The experiment to use for the "response" data.
    project : str, optional
        The project to search for.
    nodrift : bool, default: False
        Whether to use drift-corrected data.
    logging : bool, optional
        Whether to build a custom logger.
    dryrun : bool, optional
        Whether to run with only three years of data.
    **kwargs
        Passed to `compute_feedbacks`.
    """
    # Find files and restrict to unique constraints
    # NOTE: This requires flagship translation or else models with different control
    # and abrupt runs are not grouped together. Not sure how to handle e.g. non-flagship
    # abrupt runs from flagship control runs but cross that bridge when we come to it.
    # NOTE: Paradigm is to use climate monthly mean surface pressure when interpolating
    # to model levels and keep surface pressure time series when getting feedback
    # kernel integrals. Helps improve accuracy since so much stuff depends on kernels.
    logging = logging and not dryrun
    statistic = 'ratio' if ratio else 'regression'
    experiment = experiment or 'abrupt4xCO2'
    source = source or 'eraint'
    series = series or (0, 150)
    control = control or (0, 150)
    response = response or (120, 150)
    nodrift = 'nodrift' if nodrift else ''
    suffix = nodrift and '-' + nodrift
    parts = ('feedbacks', experiment, source, statistic, nodrift)
    print = Logger('summary', *parts, project=project) if logging else builtins.print
    control_suffix = f'{control[0]:04d}-{control[1]:04d}-climate{suffix}'
    if ratio:  # ratio-type
        response_suffix = f'{response[0]:04d}-{response[1]:04d}-climate{suffix}'
    else:
        response_suffix = f'{series[0]:04d}-{series[1]:04d}-series{suffix}'
    print('Generating database.')
    constraints = {
        'project': project,
        'variable': sorted(set(k for d in VARIABLE_DEPENDENCIES.values() for k in d)),
        'experiment': ['piControl', experiment],
    }
    constraints.update({
        key: kwargs.pop(key) for key in tuple(kwargs)
        if any(s in key for s in ('model', 'flagship', 'ensemble'))
    })
    files, *_ = glob_files(*paths, project=project)
    facets = ('project', 'model', 'ensemble', 'grid')
    database = Database(files, facets, **constraints)
    if experiment == 'piControl':  # otherwise translate for cmip5 or cmip6
        control_experiment = response_experiment = 'piControl'
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
        files = _file_pairs(
            data,
            control_experiment,
            response_experiment,
            control_suffix,
            response_suffix,
            printer=print,
        )
        try:
            datasets = compute_feedbacks(
                project=group['project'],
                experiment=_item_parts['experiment'](tuple(files.values())[0][1]),
                ensemble=_item_parts['ensemble'](tuple(files.values())[0][1]),
                table=_item_parts['table'](tuple(files.values())[0][1]),
                model=group['model'],
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
            if dryrun:
                raise error
            else:
                _print_error(error)
            print('Warning: Failed to compute feedbacks.')
            continue
