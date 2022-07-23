#!/usr/bin/env python3
"""
Calculate feedback, forcing, and sensitivity estimates for ESGF data.
"""
import builtins
import itertools
import traceback
from pathlib import Path

import numpy as np  # noqa: F401
import pandas as pd
import xarray as xr
from climopy import const, ureg
from icecream import ic  # noqa: F401
from shared import physics  # noqa: F401
from metpy import calc  # noqa: F401
from metpy.units import units as mreg  # noqa: F401
from numpy import ma  # noqa: F401

from .internals import (
    Database,
    Printer,
    _glob_files,
    _item_dates,
    _item_join,
    _item_parts,
)
from .utils import open_file, space_averages, time_averages

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
    'cta': ('ta',),  # control data for humidity kernels
    'rta': ('ta',),  # response data for integration bounds
    'rps': ('ps',),  # response data for integration bounds
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


def clausius_clapeyron(ta):
    r"""
    Return the inverse of the approximate change in the logarithm saturation specific
    humidity associated with a 1 Kelvin change in atmospheric temperature, defined as
    :math:`1 / (\mathrm{d} \log q / \mathrm{d} T) = \mathrm{d} T (q / \mathrm{d} q)`.

    Parameters
    ----------
    ta : xarray.DataArray
        The temperature.

    Note
    ----
    This uses the more accurate Murphy and Koop (2005) method rather than Bolton et al.
    (1980). Soden et al. (2008) and Shell et al. (2008) recommend using the logarithmic
    scale factor :math:`(\log q_1 - \log q_0) / (\mathrm{d} \log q / \mathrm{d} T)`
    since the radiative response is roughly proportional to the logarithm of water vapor
    and thus linearity of the response to perturbations is a better assumption in log
    space. This is also more convenient, since under constant relative humidity the
    scale factor is independent of base state relative humidity or specific humidity:

    .. math::

       \mathrm{d} \log e
       = \mathrm{d} \log(r e_\mathrm{s})
       = \mathrm{d} \log r + \mathrm{d} \log e_\mathrm{s}
       = \mathrm{d} \log e_\mathrm{s}

    i.e. we only need temperature to calculate the scale factor (which helps address
    issues with negative specific humidities or invalid relative humidities induced by
    drift corrections). Note that this also neglects second-order terms by assuming
    :math:`\mathrm{d}\log e_\mathrm{s} \approx \mathrm{d}\log q_\mathrm{s}` as follows:

    .. math::

       q_\mathrm{s} = (\epsilon * e_\mathrm{s}) / (p - (1 - \epsilon) * e_\mathrm{s}) \\
       \mathrm{d} q_\mathrm{s} / q_\mathrm{s}
       = \mathrm{d} e_\mathrm{s} * \left(%
         1 / e_\mathrm{s} + (1 - \epsilon) / (p - (1 - \epsilon) * e_\mathrm{s})
       \right) \approx \mathrm{d} e_\mathrm{s} / e_\mathrm{s}

    which is valid since picontrol has a maximum :math:`T \approx 305 \, \mathrm{K}`,
    which implies :math:`e_\mathrm{s} = 50 \, \mathrm{hPa}` and an approximation error
    of only :math:`(0.204 - 0.200) / 0.204 \approx 0.02` i.e. just 2 percent.
    """
    # NOTE: In example NCL script Angie uses ta and hus from 'basefields.nc' for dq/dt
    # (metadata indicates these were fields from kernel calculations) while Yi uses
    # temperature from the picontrol climatology. Here we use latter methodology.
    # WARNING: Critical to replace climopy units with metpy units if using metpy methods
    # or else get hard-to-debug errors, e.g. temperature units mysteriously stripped.
    # However use manual calc since metpy uses less accurate Bolton (1980) method.
    # WARNING: Critical to build separate array and only preserve dimension coordinates
    # or else can get 'cell_height' conflict if this is called before gravity-scaled
    # 'cell_height' are assigned which later causes issues with vertical integration.
    ta = ta.climo.to_units('K').climo.dequantify()
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
    el, ei = es_liq(ta.data), es_ice(ta.data)
    el1, ei1 = es_liq(ta.data + 1), es_ice(ta.data + 1)
    es, es1 = el.copy(), el1.copy()
    es[mask] = ei[mask := ta.data < 273.15]
    es1[mask] = ei1[mask := ta.data + 1 < 273.15]
    scale = es / (es1 - es)  # i.e. 1 / d(log(es)) = es / d(es)
    scale = xr.DataArray(
        scale,
        dims=tuple(ta.sizes),
        coords={dim: ta.coords[dim] for dim in ta.sizes},
        attrs={'units': 'K', 'long_name': 'inverse Clausius-Clapeyron scaling'}
    )
    return scale.climo.quantify()


def response_anomalies(printer=None, project=None, testing=False, **inputs):
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
    testing : bool, optional
        Whether to run this in 'testing mode' with only three years of data.
    """
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
        # NOTE: Some datasets have a pressure 'bounds' variable but appears calculated
        # naively as halfway points between levels. Since inconsistent between files
        # just strip all bounds attribute and rely on climopy auto calculations. see:
        # ta_Amon_ACCESS-CM2_piControl_r1i1p1f1_gn_0000-0150-climate-nodrift.nc
        # NOTE: Some datasets seem to have overlapping time series years. Check
        # against this by removing duplicate indices.
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
                if testing:
                    data = data.isel(time=slice(None, 24))
                data.name = variable
                datas.append(data)

        # Calculate variable anomalies using dependencies
        # NOTE: Since incoming solar is constant, and since all our regressions are
        # with anomalies with respect to base climate, the incoming solar terms cancel
        # out and we can just consider upwelling solar at the top-of-atmosphere.
        # NOTE: Here subtraction from 'groupby' objects can only be used if the array
        # has a coordinate with the corresponding indices, i.e. in this case 'month'
        # instead of 'time'. Use the dummy operation .groupby('time.month').mean()
        # to turn time indices into month indices. Note this also makes this function
        # robust to using control data time series rather than climatologies.
        with xr.set_options(keep_attrs=True):
            if variable in ('cta',):  # the control data matched to response data times
                (control,), (response,) = controls, responses
                control = (0 * response).groupby('time.month') + control.groupby('time.month').mean()  # noqa: E501
                response = None
            elif variable in ('rta', 'rps'):  # the response data itself
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
        if response is None:
            data = control
        elif control is None:
            data = response
        else:
            with xr.set_options(keep_attrs=True):
                data = response.groupby('time.month') - control.groupby('time.month').mean()  # noqa: E501
        if variable == 'alb':  # enforce new name and new units
            long_name = 'albedo'
            data.attrs['units'] = '%'
        elif variable == 'hus':
            long_name = 'specific humidity logarithm'
            data.attrs['units'] = '1'
        else:  # enforce lower-case name for consistency
            parts = data.attrs['long_name'].split()
            long_name = ' '.join(s if s == 'TOA' else s.lower() for s in parts)
        if variable in ('cta',):
            long_name = f'control {long_name}'
            output.coords[variable] = data
        elif variable in ('rta', 'rps'):
            long_name = f'response {long_name}'
            output.coords[variable] = data
        else:
            long_name = f'{long_name} anomaly'
            output[variable] = data
        data = output[variable]
        data.attrs['long_name'] = long_name
        if variable not in ('rta', 'rps'):  # keep standard name for cell heights
            data.attrs.pop('standard_name', None)
        print(f'  {variable} ({long_name}): {len(data.time)}')

    return output


def response_fluxes(input, kernels, printer=None):
    """
    Return a dataset containing the actual radiative flux responses and the radiative
    flux responses implied by the radiative kernels and input anomaly data.

    Parameters
    ----------
    input : xarray.Dataset
        The climate anomaly data.
    kernels : xarray.Dataset
        The radiative kernel data.
    printer : callable, default: `print`
        The print function.
    """
    # Initial stuff
    # WARNING: Cell height returns zero everywhere for descending levels consistent
    # with cmip conventions rather than ascending levels. Should fix this. Also should
    # support selections like e.g. sel(50, 100) on reversed coordinates in .sel wrapper
    # and permit bypassing the automatic scaling to 'kg m^-2' units.
    # WARNING: Had unbelievably frustrating issue where pressure level coordinates on
    # kernels and anomalies would be identical (same type, value, etc.) but then
    # input.plev == kernels.plev returned empty array and assigning time-varying cell
    # heights to data before integration in Planck feedback block created array of
    # all NaNs. Still not sure why this happened but assign_coords seems to fix it.
    print = printer or builtins.print
    print('Calculating cell measures for vertical integration.')
    input = input.isel(plev=slice(None, None, -1))
    input = input.climo.add_cell_measures(surface=True, tropopause=True)
    input = input.isel(plev=slice(None, None, -1))
    with xr.set_options(keep_attrs=True):
        height = const.g * input.climo.coords['cell_height']
        kernels = kernels.groupby('time.month').mean()  # see _get_anomalies notes
    height = height.climo.dequantify()
    min_, max_, mean = height.min().item(), height.max().item(), height.mean().item()
    print(f'Cell height range: min {min_:.0f} max {max_:.0f} mean {mean:.0f}')
    bot = input.coords['plev_bot'].climo.dequantify()
    min_, max_, mean = bot.min().item(), bot.max().item(), bot.mean().item()
    print(f'Vertical bottom range: min {min_:.0f} max {max_:.0f} mean {mean:.0f}')
    top = input.coords['plev_top'].climo.dequantify()
    min_, max_, mean = top.min().item(), top.max().item(), top.mean().item()
    print(f'Vertical top range: min {min_:.0f} max {max_:.0f} mean {mean:.0f}')
    scale = clausius_clapeyron(input.coords['cta'])
    min_, max_, mean = scale.min().item(), scale.max().item(), scale.mean().item()
    print(f'Clausius-Clapeyron range: min {min_:.2f} max {max_:.2f} mean {mean:.2f}')
    input = input.climo.quantify()
    input = input.assign_coords(cell_height=height)  # critical (see above)
    kernels = kernels.climo.quantify()
    kernels = kernels.assign_coords(plev=input.plev)  # critical (see above)
    if input.sizes.get('plev', np.inf) < kernels.sizes['plev']:
        kernels = kernels.sel(plev=input.plev.values)  # NESM3 abrupt-4xCO2 data
    output = xr.Dataset({'ts': input.ts})

    # Iterate over flux components
    # NOTE: Here cloud feedback comes about from change in cloud radiative forcing (i.e.
    # R_cloud_abrupt - R_cloud_control where R_cloud = R_clear - R_all) then corrected
    # for masking of clear-sky effects (e.g. by adding dt * K_t_cloud where similarly
    # K_t_cloud = K_t_clear - K_t_all). Additional greenhouse forcing masking correction
    # is applied when we estimate feedbacks. Note that since shortwave clear-sky effects
    # are just albedo and small water vapor effect, the mask correction is far less
    # important. Only longwave component will have huge masked clear-sky effects.
    for boundary, wavelength, (component, descrip) in itertools.product(  # noqa: E501
        ('TOA', 'surface'), ('longwave', 'shortwave'), FEEDBACK_DESCRIPTIONS.items()
    ):
        # Skip feedbacks without shortwave or longwave components.
        # NOTE: If either longwave or shortwave component is missing it will be excluded
        # from the "longwave plus shortwave" calculations below. Also only exclude cloud
        # and residual components when the actual model radiative flux is missing.
        flux = f'r{wavelength[0]}n{boundary[0].lower()}'
        fluxes = (flux,)  # kernel fluxes
        variables = FEEDBACK_DEPENDENCIES[component][wavelength]  # empty for fluxes
        dependencies = list(variables)  # anomaly and kernel variables
        if component == '':
            dependencies.append(flux)
        elif component == 'cs':
            dependencies.append(flux := f'{flux}cs')
        elif component == 'cl' or component == 'resid':
            dependencies.extend(fluxes := (flux, f'{flux}cs'))  # used for cloud masking
        elif not dependencies:
            continue
        if component == '':
            running = 0.0 * ureg('W m^-2')
        if component == '':
            print(f'Calculating {wavelength} {boundary} fluxes using kernel method.')
        if message := ', '.join(
            repr(name) for name in dependencies if name not in input
        ):
            print(
                f'Warning: {wavelength.title()} flux {component=} is missing '
                f'variable dependencies {message}. Skipping {wavelength} flux estimate.'  # noqa: E501
            )
            continue
        if message := ', '.join(
            repr(name) for var in variables for flux in fluxes
            if (name := f'{var}_{flux}') not in kernels
        ):
            print(
                f'Warning: {wavelength.title()} flux {component=} is missing '
                f'kernel dependencies {message}. Skipping {wavelength} flux estimate.'
            )
            continue

        # Comopnent fluxes
        # NOTE: This data is then fed into response_feedbacks and converted
        # into feedback parameter and forcing estimates using these names.
        if component == 'alb':  # albedo flux (longwave skipped above)
            grp = input['alb'].groupby('time.month')
            data = grp * kernels[f'alb_{flux}']
        elif component[:2] == 'pl':  # planck flux (shortwave skipped for 'pl')
            grp = input['ts'].groupby('time.month')
            data = 0.0 * ureg('W m^-2 Pa^-1')
            if wavelength[0] == 'l':
                data = data + grp * kernels[f'ta_{flux}']
            if component == 'pl*':
                data = data + grp * kernels[f'hus_{flux}']
            data = data.assign_coords(cell_height=input.cell_height)  # necessary
            data = data.climo.integral('plev')
            if wavelength[0] == 'l':
                data = data + grp * kernels[f'ts_{flux}']
        elif component[:2] == 'lr':  # lapse rate flux (shortwave skipped for 'lr')
            grp = (input['ta'] - input['ts']).groupby('time.month')
            data = 0.0 * ureg('W m^-2 Pa^-1')
            if wavelength[0] == 'l':
                data = data + grp * kernels[f'ta_{flux}']
            if component == 'lr*':
                data = data + grp * kernels[f'hus_{flux}']
            data = data.climo.integral('plev')
        elif component[:2] == 'hu':  # specific and relative humidity fluxes
            grp = (scale * input['hus']).groupby('time.month')
            data = grp * kernels[f'hus_{flux}']
            if component == 'hur':
                grp = input['ta'].groupby('time.month')
                data = data - grp * kernels[f'hus_{flux}']
            data = data.climo.integral('plev')
        elif component == 'cl':  # shortwave and longwave cloud fluxes
            data = input[flux] - input[f'{flux}cs']
            for var in variables:  # relevant longwave or shortwave variables
                grp = input[var].groupby('time.month')
                adj = grp * (kernels[f'{var}_{flux}'] - kernels[f'{var}_{flux}cs'])
                if var == 'hus':
                    adj = scale * adj
                if var in ('ta', 'hus'):
                    adj = adj.climo.integral('plev')
                data = data - adj.climo.to_units('W m^-2')  # cloud masking adjustment
        elif component == 'resid':  # residual flux
            data = input[flux] - running
        else:  # net flux
            data = 1.0 * input[flux]

        # Update the input dataset with component fluxes
        name = flux if component in ('', 'cs') else f'{component}_{flux}'
        descrip = descrip and f'{descrip} flux' or 'flux'  # for lon gname
        component = component or 'net'  # for print message
        data.name = name
        data.attrs['long_name'] = f'net {boundary} {wavelength} {descrip}'
        data = data.climo.to_units('W m^-2')
        if component in ('pl', 'lr', 'hus', 'alb', 'cl'):
            running = data + running
        data = data.climo.dequantify()
        min_, max_, mean = data.min().item(), data.max().item(), data.mean().item()
        print(format(f'  {component} flux:', ' <15s'), end=' ')
        print(f'min {min_: <+7.2f} max {max_: <+7.2f} mean {mean: <+7.2f} ({descrip})')
        output[data.name] = data

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
            fluxes = (f'{long}{component}', f'{short}{component}')
        elif component in ('alb',):
            fluxes = (f'{component}_{short}',)
        elif component in ('lr', 'pl'):  # traditional feedbacks
            fluxes = (f'{component}_{long}',)
        else:
            fluxes = (f'{component}_{long}', f'{component}_{short}')

        # Sum component fluxes into net longwave plus shortwave fluxes
        # if both dependencies exist. Otherwise emit warning.
        if message := ', '.join(flux for flux in fluxes if flux not in output):
            print(
                f'Warning: Net radiative flux {component=} is missing longwave or '
                f'shortwave dependencies {message}. Skipping net flux estimate.'
            )
            continue
        name = f'{net}{component}' if component in ('', 'cs') else f'{component}_{net}'  # noqa: E501
        descrip = descrip and f'{descrip} flux' or 'flux'  # for long name
        component = component or 'net'  # for print message
        with xr.set_options(keep_attrs=True):
            data = sum(output[flux] for flux in fluxes)
        data.name = name
        data.attrs['long_name'] = f'net {boundary} {descrip}'
        output[name] = data
        if len(fluxes) == 1:  # e.g. drop the shortwave albedo 'component'
            output = output.drop_vars(fluxes[0])
        min_, max_, mean = data.min().item(), data.max().item(), data.mean().item()
        print(format(f'  {component} flux:', ' <15s'), end=' ')
        print(f'min {min_: <+7.2f} max {max_: <+7.2f} mean {mean: <+7.2f} ({descrip})')

    return output


def response_feedbacks(input, forcing=None, printer=None):
    """
    Return a dataset containing feedbacks calculated along all `time_averages` periods
    (annual averages, seasonal averages, and month averages) and along all combinations
    of `space_averages` (points, latitudes, hemispheres, and global, with different
    averaging conventions used in the denominator indicated by the ``region`` coord).

    Parameters
    ----------
    input : xarray.Dataset
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
    plev_top = time_averages(input['plev_top'])
    plev_bot = time_averages(input['plev_bot'])
    title = f'{statistic}-derived forcing-feedback decompositions'
    input = input.climo.add_cell_measures()
    input = input.climo.quantify()
    input = time_averages(input)
    input = space_averages(input)

    # Iterate over fluxes
    outputs = {}
    regions = ('point', 'latitude', 'hemisphere', 'globe')
    for numerator, denominator in itertools.product(regions, regions):
        if regions.index(numerator) > regions.index(denominator):
            continue
        key = (numerator, denominator)  # used as multi-index
        outputs[key] = output = xr.Dataset()
        for boundary, wavelength, component in itertools.product(
            ('TOA', 'surface'),
            ('full', 'longwave', 'shortwave'),
            FEEDBACK_DESCRIPTIONS,
        ):
            # Get the flux names
            # NOTE: Here if a component has no 'dependencies' it does not
            # exist so skip (e.g. temperature shortwave).
            flux = f'r{wavelength[0]}n{boundary[0].lower()}'
            descrip = FEEDBACK_DESCRIPTIONS[component]
            if component in ('', 'cs'):
                name = f'{flux}{component}'
            elif wavelength == 'full':
                name = f'{component}_{flux}'
            elif all(keys for keys in FEEDBACK_DEPENDENCIES[component].values()):
                name = f'{component}_{flux}'
            else:  # skips e.g. non-full planck and albedo
                continue
            if wavelength == 'full' and component == '':
                print(
                    f'Calculating {numerator} vs. {denominator} '
                    f'{wavelength} {boundary} forcing and feedback.'
                )
            if name not in input:
                print(
                    'Warning: Input dataset is missing radiative flux '
                    f'component {name!r}. Skipping feedback estimate.'
                )
                continue
            numer = input[name].sel(region=numerator)
            denom = input['ts'].sel(region=denominator)

            # Possibly add forcing masking adjustments
            # NOTE: Since forcing is constant this has no effect on feedback estimate
            # but does effect the '..._cl_erf2x' cloud adjustment forcing estimates.
            # NOTE: Soden et al. (2008) used standard horizontally uniform value of 15%
            # the full forcing but no reason not to use regressed forcing estimates.
            if component in ('cl', 'resid'):
                if message := ', '.join(
                    repr(erf) for sky in ('', 'cs')
                    if (erf := f'{flux}{sky}_erf2x') not in output
                ):
                    print(
                        'Warning: Output dataset is missing cloud-masking forcing '
                        f'adjustment variable(s) {message}. Cannot make adjustment.'
                    )
                else:
                    all_, clear = output[f'{flux}_erf2x'], output[f'{flux}cs_erf2x']
                    prop = (mask := all_ - clear) / all_
                    if wavelength == 'full' and component == 'cl':
                        min_, max_, mean = mask.min().item(), mask.max().item(), mask.mean().item()  # noqa: E501
                        print(format('  masking flux:', ' <15s'), end=' ')
                        print(f'min {min_: <+7.2f} max {max_: <+7.2f} mean {mean: <+7.2f}')  # noqa: E501
                        min_, max_, mean = prop.min().item(), prop.max().item(), prop.mean().item()  # noqa: E501
                        print(format('  masking frac:', ' <15s'), end=' ')
                        print(f'min {min_: <+7.2f} max {max_: <+7.2f} mean {mean: <+7.2f}')  # noqa: E501
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
                forcing = Path(forcing).expanduser()
                erf = xr.open_dataset(forcing, use_cftime=True)[f'{name}_erf2x']
                erf = erf.climo.quantify()
                lam = (numer.mean('time') - 2.0 * erf) / denom.mean('time')
            else:
                nm = numer.mean('time', skipna=False)  # require data for entire period
                dm = denom.mean('time', skipna=False)
                lam = ((denom - dm) * (numer - nm)).sum('time') / ((denom - dm) ** 2).sum('time')  # noqa: E501
                erf = 0.5 * (nm - lam * dm)  # possibly zero minus zero
            descrip = descrip or 'net'  # for long names
            component = component or 'net'  # for print messages
            lam = lam.climo.to_units('W m^-2 K^-1')
            lam = lam.climo.dequantify()
            lam.name = f'{name}_lambda'
            lam.attrs['long_name'] = f'{descrip} feedback'
            output[lam.name] = lam
            if wavelength == 'full':
                min_, max_, mean = lam.min().item(), lam.max().item(), lam.mean().item()
                print(format(f'  {component} lambda:', ' <15s'), end=' ')
                print(f'min {min_: <+7.2f} max {max_: <+7.2f} mean {mean: <+7.2f} ({descrip})')  # noqa: E501
            erf = erf.climo.to_units('W m^-2')
            erf = erf.climo.dequantify()
            erf.name = f'{name}_erf2x'  # halved from quadrupled co2
            erf.attrs['long_name'] = f'{descrip} effective forcing'
            output[erf.name] = erf
            if wavelength == 'full':
                min_, max_, mean = erf.min().item(), erf.max().item(), erf.mean().item()
                print(format(f'  {component} erf2x:', ' <15s'), end=' ')
                print(f'min {min_: <+7.2f} max {max_: <+7.2f} mean {mean: <+7.2f} ({descrip})')  # noqa: E501
            if wavelength == 'full' and component == '':  # for convenience only
                ecs = -1 * (erf / lam)  # already standardized and dequantified
                ecs.name = f'{name}_ecs2x'  # already halved relative to quadrupled co2
                ecs.attrs['units'] = 'K'
                ecs.attrs['long_name'] = f'{descrip} effective sensitivity'
                output[ecs.name] = ecs
                min_, max_, mean = ecs.min().item(), ecs.max().item(), ecs.mean().item()  # noqa: E501
                print(format(f'  {component} ecf2x:', ' <15s'), end=' ')
                print(f'min {min_: <+7.2f} max {max_: <+7.2f} mean {mean: <+7.2f} ({descrip})')  # noqa: E501

    # Concatenate result along 'region' axis
    # NOTE: This should have skipped impossible combinations like
    coord = xr.DataArray(
        pd.MultiIndex.from_tuples(outputs, names=('numerator', 'denominator')),
        dims='region',
        name='region',
        attrs={'long_name': 'spatial averaging regions'}
    )
    output = xr.concat(
        outputs.values(),
        dim=coord,
        coords='minimal',
        compat='equals',
    )
    output.attrs['title'] = title
    output.update({'plev_bot': plev_bot, 'plev_top': plev_top})
    return output


def compute_feedbacks(
    input='~/data',
    output='~/data',
    kernels=None,
    ratio=None,
    project=None,
    experiment=None,
    ensemble=None,
    table=None,
    model=None,
    nodrift=False,
    overwrite=False,
    printer=None,
    testing=False,
    **inputs
):
    """
    Calculate the feedbacks and generate three files containing local vs. local,
    local vs. global, and global vs. global comparisons of net surface or
    top-of-atmosphere radiative flux against surface temperature.

    Parameters
    ----------
    input : path-like, optional
        The kernel data directory. The subfolder ``cmip-kernels`` is used.
    output : path-like, optional
        The output data directory. The subfolder ``cmip[56]-feedbacks`` is used.
    kernels : str, default: 'eraint'
        The source for the kernel data (e.g. ``'eraint'``). This searches for files
        in the `kernels` directory formatted as ``kernels_{kernels}.nc``.
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
    ensemble, model : str, optional
        The ensemble and model name. Used in the output file name.
    nodrift : bool, optional
        Whether to append a ``-nodrift`` indicator to the end of the filename.
    overwrite : bool, default: False
        Whether to overwrite existing output files.
    printer : callable, default: `print`
        The print function.
    testing : bool, optional
        Whether to run this in 'testing mode' with only three years of data.

    Returns
    -------
    *paths : Path
        The paths of the three output files. They file names are formatted as
        ``feedbacks_{model}_{experiment}_{regions}_{options}.nc`` where
        `model` and `experiment` are applied only if passed by the user, `regions`
        is set to ``local-local``, ``local-global``, or ``global-global`` for each
        feedback type, `options` is set to ``{kernels}-{statistic}-{nodrift}`` where
        `kernels` is the kernel source; `statistic` is one of ``regression`` or
        ``ratio``; and ``-nodrift`` is added only if ``nodrift`` is ``True``.

    Important
    ---------
    Each file will contain the results of three different feedback breakdowns. For
    each breakdown, a residual is computed by subtracting the canonical net feedback,
    calculated as net all-sky longwave plus shortwave versus surface temperature,
    from the sum of the notional component feedbacks.

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
    adjusted cloud radiative forcing estimates of the cloud feedback parameters.
    """
    # Parse input arguments
    # NOTE: Here 'ratio' should only ever be used with abrupt forcing experiments
    # where there is timescale separation of anomaly magnitudes.
    print = printer or builtins.print
    project = (project or 'cmip6').lower()
    kernels = kernels or 'eraint'
    statistic = 'ratio' if ratio else 'regression'
    output = Path(output).expanduser()
    output = output / _item_join((project, experiment, 'feedbacks'))
    output.mkdir(exist_ok=True)
    prefix = '_'.join(filter(None, ('feedbacks', table, model, experiment, ensemble)))
    suffix = '-'.join(filter(None, (kernels, statistic, nodrift and 'nodrift' or '')))
    output = output / f'{prefix}_{suffix}.nc'
    print(f'Creating output file: {output.name}')
    if not overwrite and not testing and output.is_file() and output.stat().st_size > 0:
        print('Output file already exists.')
        return output
    if message := ', '.join(
        f'{variable}={paths}' for variable, paths in inputs.items()
        if not isinstance(paths, (tuple, list))
        or len(paths) != 2 or not all(isinstance(path, (str, Path)) for path in paths)
    ):
        raise TypeError(f'Unexpected kwargs {message}. Must be 2-tuple of paths.')

    # Load kernels and flux components and compute feedback estimates
    # NOTE: Even for kernel-derived flux responses, rapid adjustments and associated
    # pattern effects may make the effective radiative forcing estimate non-zero
    # (see Andrews et al.) so we always permit a non-zero regression intercept.
    input = Path(input).expanduser()
    input = input / 'cmip-kernels' / f'kernels_{kernels}.nc'
    kernels = open_file(input, project=project, validate=False)
    print(f'Loaded kernel data from file: {input}')
    anoms = response_anomalies(project=project, printer=print, testing=testing, **inputs)  # noqa: E501
    forcing = None
    if ratio:
        input = output.parent
        input = input / output.name.replace('ratio', 'regression')
        forcing = open_file(input, project=project, validate=False)
        print(f'Loaded forcing data from file: {input}')
    fluxes = response_fluxes(anoms, kernels, printer=print)
    feedbacks = response_feedbacks(fluxes, forcing=forcing, printer=print)
    if not testing:  # save after compressing repeated NaN values
        output.unlink(missing_ok=True)
        comp = dict(zlib=True, complevel=5)
        encoding = {var: comp for var in feedbacks.data_vars}
        feedbacks = feedbacks.reset_index('region')  # cannot save regions
        feedbacks.to_netcdf(output, engine='netcdf4', encoding=encoding)
        feedbacks = feedbacks.set_index(region=('numerator', 'denominator'))
        print(f'Created output file: {output.name}')
    return feedbacks


def process_feedbacks(
    *paths,
    kernels=None,
    ratio=None,
    series=None,
    control=None,
    response=None,
    experiment=None,
    project=None,
    nodrift=False,
    printer=None,
    testing=False,
    **kwargs
):
    """
    Generate feedback deompositions using the output of `process_files`.

    Parameters
    ----------
    *paths : path-like, optional
        Location(s) of the climate and series data.
    kernels : str, default: 'eraint'
        The source for the kernels.
    ratio : bool, optional
        Whether to use ratio feedbacks.
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
    printer : callable, default: `print`
        The print function.
    testing : bool, optional
        Whether testing is enabled.
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
    printer = testing and not printer and builtins.print or printer
    statistic = 'ratio' if ratio else 'regression'
    experiment = experiment or 'abrupt4xCO2'
    kernels = kernels or 'eraint'
    series = series or (0, 150)
    control = control or (0, 150)
    response = response or (120, 150)
    nodrift = 'nodrift' if nodrift else ''
    suffix = nodrift and '-' + nodrift
    parts = ('feedbacks', experiment, kernels, statistic, nodrift)
    print = printer or Printer('summary', *parts, project=project)
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
        **{key: kwargs.pop(key) for key in tuple(kwargs) if 'flagship' in key}
    }
    files, *_ = _glob_files(*paths, project=project)
    facets = ('project', 'model', 'ensemble', 'grid')
    database = Database(files, facets, **constraints)
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
            compute_feedbacks(
                project=group['project'],
                experiment=_item_parts['experiment'](tuple(files.values())[0][1]),
                ensemble=_item_parts['ensemble'](tuple(files.values())[0][1]),
                table=_item_parts['table'](tuple(files.values())[0][1]),
                model=group['model'],
                ratio=ratio,
                kernels=kernels,
                nodrift=nodrift,
                printer=print,
                testing=testing,
                **kwargs,
                **files,
            )
        except Exception as error:
            _print_error(error)
            print('Warning: Failed to compute feedbacks.')
            continue
