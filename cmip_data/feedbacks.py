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

from .facets import FacetDatabase, FacetPrinter, _glob_files, _item_dates, _item_parts
from .load import load_file

__all__ = [
    'compute_feedbacks',
    'process_feedbacks',
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
    '': 'radiative',  # net all-sky feedback
    'cs': 'clear-sky radiative',  # net clear-sky feedback
    'alb': 'albedo',
    'pl': 'traditional Planck',
    'lr': 'traditional lapse rate',
    'hus': 'specific humidity',
    'pl*': 'relative humidity-preserving Planck',
    'lr*': 'relative humidity-preserving lapse rate',
    'hur': 'relative humidity',
    'cl': 'adjusted cloud radiative forcing',
    'resid': 'kernel residual',
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
    'tactl': ('ta',),  # control data for humidity kernels
    'tarsp': ('ta',),  # response data for integration bounds
    'psrsp': ('ps',),  # response data for integration bounds
    'rsnt': ('rsut',),  # out of the atmosphere
    'rlntcs': ('rlutcs',),  # out of the atmosphere
    'rsntcs': ('rsutcs',),  # out of the atmosphere
    'rlns': ('rlds', 'rlus'),  # out of and into the atmosphere
    'rsns': ('rsds', 'rsus'),  # out of and into the atmosphere
    'rlnscs': ('rldscs', 'rlus'),  # out of and into the atmosphere
    'rsnscs': ('rsdscs', 'rsuscs'),  # out of and into the atmosphere
}


def _file_pairs(
    data,
    control_experiment='piControl',
    response_experiment='abrupt-4xCO2',
    control_suffix='0000-0150-climate',
    response_suffix='0000-0150-series',
    printer=None,
):
    """
    Return a dictionary mapping of variables to ``(control, response)`` file
    pairs from a database of files with ``(experiment, ..., variable)`` keys.

    Parameters
    ----------
    data : dict
        The database of file lists.
    control_experiment, response_experiment : str
        The control and response experiment names.
    control_suffix, response_suffix : str
        The control and response experiment suffixes.
    printer : callable, default: `print`
        The print function.
    """
    print = printer or builtins.print
    pairs = []
    for experiment, suffix in zip(
        (control_experiment, response_experiment),
        (control_suffix, response_suffix),
    ):
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
    pairs = {
        variable: (pairs[0][variable], pairs[1][variable])
        for variable in sorted(pairs[0].keys() & pairs[1].keys())
    }
    return pairs


def log_clausius_clapeyron(ta):
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
       \Longrightarrow = \mathrm{d} q_\mathrm{s} / q_\mathrm{s}
       = \mathrm{d} e_\mathrm{s} * \left(%
         1 / e_\mathrm{s} + (1 - \epsilon) / (p - (1 - \epsilon) * e_\mathrm{s})
       \right) \approx \mathrm{d} e_\mathrm{s} / e_\mathrm{s}

    which is valid since picontrol has a maximum :math:`T \approx 305 \, \mathrm{K}`,
    which implies :math:`e_\mathrm{s} = 5 \, \mathrm{kPa}` and an approximation error
    of only :math:`(0.204 - 0.200) / 0.204 \approx 0.02` i.e. just 2 percent.
    """
    # NOTE: In example NCL script Angie uses ta and hus from 'basefields.nc' for dq/dt
    # (metadata indicates these were fields from kernel calculations) while Yi uses
    # temperature from the picontrol climatology. Here we use latter methodology.
    # NOTE: If using metpy method must replace units or else get hard-to-debug errors,
    # e.g. temperature units mysteriously stripped. However use manual calculation
    # since they use less accurate Bolton (1980) method for supercooled water only.
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
    el, ei = es_liq(ta), es_ice(ta)
    el1, ei1 = es_liq(ta + 1), es_ice(ta + 1)
    es, es1 = el.copy(), el1.copy()
    es.data[mask] = ei.data[mask := ta < 273.15]
    es1.data[mask] = ei1.data[mask := ta + 1 < 273.15]
    scale = es / (es1 - es)  # i.e. 1 / d(log(es)) = es / d(es)
    scale.attrs['units'] = 'K'
    scale.attrs['long_name'] = 'inverse Clausius-Clapeyron scaling'
    scale = scale.climo.quantify()
    return scale


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
                data = load_file(path, dependency, project=project, printer=print)
                if testing:
                    data = data.isel(time=slice(None, 36))
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
            control = response = None
            if variable[:2] in ('ts', 'ta') or variable[:3] in ('hus', 'alb'):
                if variable == 'alb':  # ratio of outward and inward flux
                    (control_out, control_in), (response_out, response_in) = controls, responses  # noqa: E501
                    control = 100 * (control_out / control_in)
                    response = 100 * (response_out / response_in)
                elif 'ctl' in variable:
                    (control,) = controls
                elif 'rsp' in variable:
                    (response,) = responses
                else:
                    (control,), (response,) = controls, responses
            else:
                if len(controls) == 2:  # inward flux minus outward flux
                    (control_out, control_in), (response_out, response_in) = controls, responses  # noqa: E501
                    control = control_in - control_out
                    response = response_in - response_out
                else:
                    (control_out,), (response_out,) = controls, responses
                    control = -1 * control_out
                    response = -1 * response_out
            if response is None:
                data = control
            elif control is None:
                data = response
            else:
                data = response.groupby('time.month') - control.groupby('time.month').mean()  # noqa: E501

        # Add attributes and update dataset
        # NOTE: Albedo is taken above from ratio of upwelling to downwelling all-sky
        # surface shortwave radiation. Feedback is simply (rsus - rsds) / (rsus / rsds)
        if variable == 'alb':  # enforce new name and new units
            data.attrs['units'] = '%'
            data.attrs['long_name'] = 'albedo'
        else:  # enforce lower-case name for consistency
            strs = data.attrs['long_name'].split()
            data.attrs['long_name'] = ' '.join(s if s == 'TOA' else s.lower() for s in strs)  # noqa: E501
        if variable in ('rps', 'rta', 'rhus'):  # keep standard name for cell heights
            data.attrs.update({'long_name': 'raw ' + data.attrs['long_name']})
            output.coords[variable] = data
        else:  # pop standard name so anomalies are not used in cell heights
            data.attrs.pop('standard_name', None)
            output[variable] = data
        print(f'  {variable} ({data.attrs["long_name"]}): {len(data.time)}')

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
    coords = input.climo.coords  # coordinate source
    with xr.set_options(keep_attrs=True):
        height = const.g * coords['cell_height']
        kernels = kernels.groupby('time.month').mean()  # see _get_anomalies notes
    height = height.climo.to_units('hPa').climo.dequantify()
    min_, max_, mean = height.min().item(), height.max().item(), height.mean().item()
    print(f'Cell height range: min {min_:.2f} max {max_:.2f} mean {mean:.2f}')
    lower = coords['plev_bot'].climo.to_units('hPa').climo.dequantify()
    min_, max_, mean = lower.min().item(), lower.max().item(), lower.mean().item()
    print(f'Lower bounds range: min {min_:.2f} max {max_:.2f} mean {mean:.2f}')
    upper = coords['upper'].climo.to_units('hPa').climo.dequantify()
    min_, max_, mean = upper.min().item(), upper.max().item(), upper.mean().item()
    print(f'Upper bounds range: min {min_:.2f} max {max_:.2f} mean {mean:.2f}')
    scale = log_clausius_clapeyron(coords['plev'], coords['rta'], coords['rhus'], printer=print)  # noqa: E501
    min_, max_, mean = scale.min().item(), scale.max().item(), scale.mean().item()
    print(f'Clausius-Clapeyron scaling range: min {min_:.5f} max {max_:.5f} mean {mean:.5f}')  # noqa: E501
    input = input.climo.quantify()
    input = input.assign_coords(cell_height=height)  # critical (see above)
    kernels = kernels.climo.quantify()
    kernels = kernels.assign_coords(plev=input.plev)  # critical (see above)
    print(input.ts.coords)
    output = xr.Dataset(
        {'ts': input.ts, 'plev_bot': input.plev_bot, 'plev_top': input.plev_top}
    )
    raise Exception

    # Iterate over flux components
    # NOTE: Here cloud feedback comes about from change in cloud radiative forcing (i.e.
    # R_cloud_abrupt - R_cloud_control where R_cloud = R_clear - R_all) then corrected
    # for masking of clear-sky effects (e.g. by adding dt * K_t_cloud where similarly
    # K_t_cloud = K_t_clear - K_t_all). Additional greenhouse forcing masking correction
    # is applied when we estimate feedbacks. Note that since shortwave clear-sky effects
    # are just albedo and small water vapor effect, the mask correction is far less
    # important. Only longwave component will have huge masked clear-sky effects.
    for boundary, wavelength, (component, description) in itertools.product(  # noqa: E501
        ('TOA', 'surface'), ('longwave', 'shortwave'), FEEDBACK_DESCRIPTIONS.items()
    ):
        # Skip feedbacks without shortwave or longwave components.
        # NOTE: If either longwave or shortwave component is missing it will be excluded
        # from the "longwave plus shortwave" calculations below. Also only exclude cloud
        # and residual components when the actual model radiative flux is missing.
        flux = f'r{wavelength[0]}n{boundary[0].lower()}'
        variables = FEEDBACK_DEPENDENCIES[component][wavelength]  # empty for fluxes
        kerns = ()  # kernel fluxes
        dependencies = list(variables)  # anomaly and kernel variables
        if component == '':
            dependencies.append(flux)
        elif component == 'cs':
            dependencies.append(flux := f'{flux}cs')
        elif component == 'cl' or component == 'resid':
            dependencies.extend(kerns := (flux, f'{flux}cs'))  # used for cloud masking
        elif dependencies:
            kerns = (flux,)
        else:
            continue
        if component == '':
            running = 0.0 * ureg('W m^-2')
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
            repr(name) for var in variables for flux in kerns
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
            data = data.assign_coords(cell_height=input.cell_height)
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
        data.name = flux if component in ('', 'cs') else f'{component}_{flux}'
        data.attrs['long_name'] = f'net {boundary} {wavelength} {description}'
        data = data.climo.to_units('W m^-2')
        if component in ('pl', 'lr', 'hus', 'alb', 'cl'):
            running = data + running
        data = data.climo.dequantify()
        min_, max_, mean = data.min().item(), data.max().item(), data.mean().item()
        print(f'  component: {component: >5s}', end=' ')
        print(f'min {min_: >7.2f} max {max_: >7.2f} mean {mean: >7.2f} ({description})')
        output[data.name] = data

    # Combine shortwave and longwave components
    # NOTE: Could also compute feedbacks and forcings for just shortwave and longwave
    # components, then average them to get the net (linearity of offset and slope
    # estimators), but this is slightly easier and has only small computational cost.
    for boundary, (component, description) in itertools.product(
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
        # if they both exist. Otherwise emit warning.
        if message := ', '.join(flux for flux in fluxes if flux not in output):
            print(
                f'Warning: Net radiative flux {component=} is missing longwave or '
                f'shortwave dependencies {message}. Skipping net flux estimate.'
            )
            continue
        data = output[fluxes[0]]
        if len(fluxes) == 2:
            with xr.set_options(keep_attrs=True):
                data = data + output[fluxes[1]]  # avoid overwriting original with '+='
        long_name = data.attrs['long_name']
        long_name = long_name.replace('longwave ', '').replace('shortwave ', '')
        data.name = f'{net}{component}' if component in ('', 'cs') else f'{component}_{net}'  # noqa: E501
        data.attrs['long_name'] = long_name
        min_, max_, mean = data.min().item(), data.max().item(), data.mean().item()
        print(f'  component: {component: >5s}', end=' ')
        print(f'min {min_: >7.2f} max {max_: >7.2f} mean {mean: >7.2f} ({description})')
        output[data.name] = data

    return output


def response_feedbacks(
    input, data_global=False, temp_global=False, forcing=None, printer=None,
):
    """
    Return a dataset containing estimates for the feedback parameter, effective
    radiative forcing, and effective climate sensitivity for each radiative flux.

    Parameters
    ----------
    input : xarray.Dataset
        The source for the flux anomaly data.
    data_global : bool, optional
        Whether to average the radiative flux fields.
    temp_global : bool, optional
        Whether to average the surface temperature field.
    forcing : path-like, optional
        Source for the forcing data. If passed the raw flux anomalies are used
        instead of regressions and the time dimension must have length 1.
    printer : callable, default: `print`
        The print function.
    """
    # Helper function for getting days-per-month weighted annual averages
    # See: https://ncar.github.io/esds/posts/2021/yearly-averages-xarray
    def _global_average(input, b=True):  # noqa: E301
        result = input
        if b:
            with xr.set_options(keep_attrs=True):
                result = input.climo.average(('lon', 'lat'))
        return result
    def _annual_average(input):  # noqa: E301
        days = input.coords['cell_duration']
        wgts = days.groupby('time.year') / days.groupby('time.year').sum()
        ones = xr.where(input.isnull(), 0.0, 1.0)
        with xr.set_options(keep_attrs=True):
            numerator = (input * wgts).resample(time='AS').sum(dim='time')
            denominator = (ones * wgts).resample(time='AS').sum(dim='time')
            result = numerator / denominator
        return result

    # Load surface temperature
    print = printer or builtins.print
    print('Calculating radiative feedbacks using regression method.')
    print('Use global radiative fluxes?', ('no', 'yes')[data_global])
    print('Use global surface temperature?', ('no', 'yes')[temp_global])
    output = xr.Dataset()
    if 'ts' not in input:
        print('Warning: Surface temperature is unavailable. Skipping feedback estimate.')  # noqa: E501
        return output
    ts = input['ts']
    ts = ts.climo.quantify()
    ts = ts.climo.add_cell_measures()  # add cell duration
    ts = _annual_average(ts)
    ts = _global_average(ts, temp_global)
    if not temp_global or not data_global:  # for debugging
        output['lower'] = input['lower'].climo.average('time')
        output['upper'] = input['upper'].climo.average('time')

    # Iterate over fluxes
    for boundary, wavelength, (component, description) in itertools.product(
        ('TOA', 'surface'),
        ('full', 'longwave', 'shortwave'),
        FEEDBACK_DESCRIPTIONS.items()
    ):
        # Standardize variables and possibly take weighted averages.
        variables = FEEDBACK_DEPENDENCIES[component].get(wavelength, object())
        flux = f'r{wavelength[0]}n{boundary[0].lower()}'
        if component in ('', 'cs'):
            name = f'{flux}{component}'
        elif variables:  # always true for 'full' shortwave plus longwave
            name = f'{component}_{flux}'
        else:  # skips e.g. shortwave traditional planck and longwave albedo
            continue
        if component == '':
            print(f'Calculating {wavelength} {boundary} forcing and feedback.')
        if name not in input:
            print(
                'Warning: Input dataset is missing radiative flux'
                f'component {name!r}. Skipping feedback estimate.'
            )
            continue

        # Get the flux and possibly add forcing masking adjustments
        # NOTE: The forcing masking is only component of the flux that cannot be
        # applied above. Have to correct both 'cl' and 'resid' with masking here.
        # NOTE: Soden et al. (2008) used standard horizontally uniform value of around
        # 15% the full forcing but no reason not to use regressed forcing estimates.
        data = input[name]
        data = data.climo.quantify()
        data = data.climo.add_cell_measures()
        data = _annual_average(data)
        data = _global_average(data, data_global)
        if component in ('cl', 'resid'):
            if message := ', '.join(
                repr(erf) for sky in ('', 'cs')
                if (erf := f'{flux}{sky}_erf2x') not in output
            ):
                print(
                    'Warning: Output dataset is missing cloud-masking forcing '
                    f'adjustment variable(s) {message}. Skipping feedback estimate.'
                )
                continue
            full, clear = output[f'{flux}_erf2x'], output[f'{flux}cs_erf2x']
            prop = (mask := full - clear) / full
            min_, max_, mean = mask.min().item(), mask.max().item(), mask.mean().item()
            print(f'  masking amount:    min {min_: >7.2f} max {max_: >7.2f} mean {mean: >7.2f}')  # noqa: E501
            min_, max_, mean = prop.min().item(), prop.max().item(), prop.mean().item()
            print(f'  masking fraction:  min {min_: >7.2f} max {max_: >7.2f} mean {mean: >7.2f}')  # noqa: E501
            with xr.set_options(keep_attrs=True):
                if component == 'cl':
                    data = data - mask * ureg('W m^-2')
                else:
                    data = data + mask * ureg('W m^-2')

        # Perform the regression or division after cloud mask adjustments
        # See: https://en.wikipedia.org/wiki/Simple_linear_regression
        if forcing is not None:
            if ts.time.size != 1:
                raise ValueError('Expected scalar time for anomaly feedbacks.')
            forcing = Path(forcing).expanduser()
            erf = xr.open_dataset(forcing, use_cftime=True)[f'{name}_erf2x']
            erf = erf.climo.quantify()
            lam = (data - 2.0 * erf) / ts  # back to quadrupled forcing
        else:
            if ts.time.size <= 1:
                raise ValueError('Expected non-scalar time for regression feedbacks.')
            ts_bar = ts.mean('time')  # simple annual average
            data_bar = data.mean('time')
            lam = ((ts - ts_bar) * (data - data_bar)).sum('time') / ((ts - ts_bar) ** 2).sum('time')  # noqa: E501
            erf = 0.5 * (data_bar - lam * ts_bar)

        # Calculate climate sensitivity for net fluxes only
        # See: https://doi.org/10.1175/JCLI-D-12-00544.1 (Equation 2)
        if component in ('', 'cs'):
            ecs = -1 * (erf / lam).climo.to_units('K')
            ecs = ecs.climo.dequantify()
            ecs.name = f'{name}_ecs2x'  # halved from quadrupled co2
            ecs.attrs['long_name'] = data.attrs['long_name'] + ' effective climate sensitivity'  # noqa: E501
            output[ecs.name] = ecs
            min_, max_, mean = ecs.min().item(), ecs.max().item(), ecs.mean().item()
            print(f'  ecs2x:  {component: >5s}', end=' ')
            print(f'min {min_: >7.2f} max {max_: >7.2f} mean {mean: >7.2f} ({description})')  # noqa: E501

        # Calculate feedbacks for all fluxes
        erf = erf.climo.to_units('W m^-2')
        erf = erf.climo.dequantify()
        erf.name = f'{name}_erf2x'  # halved from quadrupled co2
        erf.attrs['long_name'] = data.attrs['long_name'] + ' effective radiative forcing'  # noqa: E501
        output[erf.name] = erf
        min_, max_, mean = erf.min().item(), erf.max().item(), erf.mean().item()
        print(f'  erf2x:  {component: >5s}', end=' ')
        print(f'min {min_: >7.2f} max {max_: >7.2f} mean {mean: >7.2f} ({description})')

        # Calculate forcing for all fluxes
        # NOTE: This is needed even for component fluxes when getting anomaly feedbacks
        lam = lam.climo.to_units('W m^-2 K^-1')
        lam = lam.climo.dequantify()
        lam.name = f'{name}_lambda'
        lam.attrs['long_name'] = data.attrs['long_name'] + ' feedback parameter'
        output[lam.name] = lam
        min_, max_, mean = lam.min().item(), lam.max().item(), lam.mean().item()
        print(f'  lambda: {component: >5s}', end=' ')
        print(f'min {min_: >7.2f} max {max_: >7.2f} mean {mean: >7.2f} ({description})')

    return output


def compute_feedbacks(
    output='~/data',
    kernels='~/data',
    source=None,
    project=None,
    model=None,
    experiment=None,
    ensemble=None,
    nodrift=False,
    anomalies=False,
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
    output : path-like, optional
        The output data directory. The subfolder ``cmip[56]-feedbacks`` is used.
    kernels : path-like, optional
        The kernel data directory. The subfolder ``cmip-kernels`` is used.
    source : str, default: 'eraint'
        The source for the kernels (e.g. ``'eraint'``). This searches for files
        in the `kernels` directory formatted as ``kernels_{source}.nc``.
    project : str, optional
        The project. Used in the output folder and for level checking.
    model, experiment, ensemble : str, optional
        The model, experiment, and ensemble to use in the output file name.
    nodrift : bool, optional
        Whether to append a ``-nodrift`` indicator to the end of the filename.
    anomalies : bool, default: False
        Whether input files should be used for anomaly differences instead of
        regressions. In this case forcing estimates are loaded from disk.
    overwrite : bool, default: False
        Whether to overwrite existing output files.
    printer : callable, default: `print`
        The print function.
    testing : bool, optional
        Whether to run this in 'testing mode' with only three years of data.
    **inputs : tuple of path-like lists
        Tuples of ``(control_inputs, response_inputs)`` for the variables
        required to compute the feedbacks, passed as keyword arguments for
        each variable. The feedbacks computed will depend on the variables passed.

    Returns
    -------
    *paths : Path
        The paths of the three output files. They file names are formatted as
        ``feedbacks_{model}_{experiment}_{regions}_{options}.nc`` where
        `model` and `experiment` are applied only if passed by the user, `regions`
        is set to ``local-local``, ``local-global``, or ``global-global`` for each
        feedback type, `options` is set to ``{source}-{type}-{nodrift}`` where `source`
        is the kernel source, `type` is ``anomalies`` if `anomalies` is ``True`` and
        ``regression`` otherwise, and ``-nodrift`` is added if ``nodrift`` is ``True``.

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
    longwave and shortwave cloud feedbacks). They are also needed for the anomaly
    feedbacks to plug in as estimates for the effective forcing (when
    ``anomalies=True`` this function looks for regression files to load the
    forcing estimates). The all-sky and clear-sky regressions are both needed for an
    estimate of cloud masking of the radiative forcing to plug into the Soden et al.
    adjusted cloud radiative forcing estimates of the cloud feedback parameters.
    """
    # Parse input arguments and load kernels
    print = printer or builtins.print
    project = (project or 'cmip6').lower()
    source = source or 'eraint'
    output = Path(output).expanduser()
    output = output / f'{project}-feedbacks'
    output.mkdir(exist_ok=True)
    prefix = ('feedbacks', model, experiment, ensemble)
    suffix = source + '-'
    suffix += ('regression', 'anomalies')[bool(anomalies)]
    suffix += ('', '-nodrift')[bool(nodrift)]
    outputs = {
        regions: output / (
            '_'.join(filter(None, (*prefix, f'{regions[0]}-{regions[1]}', suffix)))
            + '.nc'
        )
        for regions in (('local', 'local'), ('local', 'global'), ('global', 'global'))
    }
    if message := ', '.join(
        output.name for output in outputs.values() if not overwrite
        and output.is_file() and output.stat().st_size > 0
    ) and not testing:
        print(f'Output files already exist: {message}')
        return (*outputs.values(),)
    database = {}
    for variable, paths in inputs.items():
        if (
            not isinstance(paths, (tuple, list)) or len(paths) != 2
            or not all(isinstance(path, (str, Path)) for path in paths)
        ):
            raise TypeError(
                f'Unexpected argument {variable}={paths}. Must be 2-tuple of paths.'
            )
        database[variable] = list(map(Path, paths))

    # Load radiative kernels
    # NOTE: Unlike CMIP5, CMIP6
    kernels = Path(kernels).expanduser()
    kernels = kernels / 'cmip-kernels'
    kernels = kernels / f'kernels_{source}.nc'
    print(f'Loading kernel data from file: {kernels}')
    kernels = load_file(kernels, project=project)

    # Get flux component anomalies and iterate over local and global combinations
    # NOTE: For all fluxes (both observed and kernel-implied) still record the
    # forcing estimates. Kernel-implied fluxes will not necessary intercept at
    # zero due to rapid adjustments so still use for the forcing data.
    anoms = response_anomalies(**database, project=project, printer=print, testing=testing)  # noqa: E501
    fluxes = response_fluxes(anoms, kernels=kernels, printer=print)
    for regions, output in outputs.items():
        print(f'Creating output file: {output.name}')
        forcing = None
        if anomalies:
            forcing = output.parent / output.name.replace('anomalies', 'regression')
            print(f'Loading forcing data from file: {forcing}')
            forcing = xr.open_dataset(forcing, use_cftime=True)
        feedbacks = response_feedbacks(
            fluxes,
            forcing=forcing,
            printer=print,
            data_global=(regions[0] == 'global'),
            temp_global=(regions[1] == 'global'),
        )
        if not testing:
            output.unlink(missing_ok=True)
            feedbacks.to_netcdf(output)
        print(f'Finished output file: {output.name}')

    return (*outputs.values(),)


def process_feedbacks(
    *paths,
    method=None,
    series=None,
    control=None,
    response=None,
    nodrift=False,
    printer=None,
    source=None,
    testing=False,
    **constraints
):
    """
    Generate feedback deompositions using the output of `process_files`.

    Parameters
    ----------
    *paths : path-like, optional
        Location(s) of the climate and series data.
    method : {'response', 'control', 'anomalies'}
        The type of feedback to compute.
    series : 2-tuple of int, default: (0, 150)
        The year range for time series data.
    control : 2-tuple of int, default: (0, 150)
        The year range for control climate data.
    response : 2-tuple of int, default: (120, 150)
        The year range for response climate data.
    nodrift : bool, default: False
        Whether to use drift-corrected data.
    printer : callable, default: `print`
        The print function.
    source : str, default: 'eraint'
        The source for the kernels.
    testing : bool, optional
        Whether testing is enabled.
    **kwargs
        Passed to `compute_feedbacks`.
    **constraints
        Passed to `_parse_constraints`.
    """
    # Find files and restrict to unique constraints
    # NOTE: This requires flagship translation or else models with different control
    # and abrupt runs are not grouped together. Not sure how to handle e.g. non-flagship
    # abrupt runs from flagship control runs but cross that bridge when we come to it.
    kwargs = {
        key: constraints.pop(key)
        for key in ('source', 'kernels', 'output', 'overwrite')
        if key in constraints
    }
    anomalies = method == 'anomalies'  # passed to compute_feedbacks
    testing = 'test' if testing else ''
    nodrift = 'nodrift' if nodrift else ''
    method = method or 'response'
    source = source or 'eraint'
    print = printer or FacetPrinter(
        'summary', 'feedbacks', source, method, nodrift, testing, project=constraints.get('project')  # noqa: E501
    )
    print('Generating database.')
    files, _ = _glob_files(*paths, project=constraints.get('project'))
    facets = ('project', 'model', 'ensemble', 'grid')
    series = series or (0, 150)
    control = control or (0, 150)
    response = response or (120, 150)
    constraints['variable'] = sorted(
        set(_ for variables in VARIABLE_DEPENDENCIES.values() for _ in variables)
    )
    database = FacetDatabase(files, facets, **constraints)

    # Determine the file suffixes to use based on inputs
    # NOTE: Paradigm is to use climate monthly mean surface pressure when interpolating
    # to model levels and keep surface pressure time series when getting feedback
    # kernel integrals. Helps improve accuracy since so much stuff depends on kernels.
    abrupt = 'abrupt4xCO2' if database.project == 'CMIP5' else 'abrupt-4xCO2'
    nodrift = nodrift and '-' + nodrift
    series_dates = '-'.join(format(n, '04d') for n in series)
    control_dates = '-'.join(format(n, '04d') for n in control)
    response_dates = '-'.join(format(n, '04d') for n in response)
    control_suffix = f'{control_dates}-climate{nodrift}'
    control_experiment = 'piControl'
    if method == 'response':
        response_suffix = f'{series_dates}-series{nodrift}'
        response_experiment = abrupt
    elif method == 'control':
        response_suffix = f'{series_dates}-series{nodrift}'
        response_experiment = 'piControl'
    elif method == 'anomalies':
        response_suffix = f'{response_dates}-climate{nodrift}'
        response_experiment = abrupt
    else:
        raise ValueError(f'Invalid {method=}. Options are control, response, ratio.')

    # Calculate clear and full-sky feedbacks surface and TOA files
    # NOTE: Unlike the method that uses a fixed SST experiment to deduce forcing and
    # takes the ratio of local radiative flux change to global temperature change as
    # the feedback, the average of a regression of radiative flux change against global
    # temperature change will not necessarily equal the regression of average radiative
    # flux change. Therefore compute both and compare to get idea of the residual. Also
    # consider local temperature change for point of comparison (Hedemman et al. 2022).
    outs = {}
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
        print(f'Computing {method} feedbacks:')
        print(', '.join(f'{key}: {value}' for key, value in group.items()))
        files = _file_pairs(
            data,
            control_experiment,
            response_experiment,
            control_suffix,
            response_suffix,
            printer=print,
        )
        samples = tuple(pair[1] for pair in files.values())
        ensemble = samples and _item_parts['ensemble'](samples[0]) or None
        try:
            paths = compute_feedbacks(
                project=group['project'],
                model=group['model'],
                ensemble=ensemble,
                experiment=response_experiment,
                anomalies=anomalies,
                nodrift=nodrift,
                printer=print,
                testing=testing,
                source=source,
                **kwargs,
                **files,
            )
        except Exception as error:
            raise error
            _print_error(error)
            print('Warning: Failed to compute feedbacks.')
            continue
        outs[tuple(group.values())] = paths

    return outs
