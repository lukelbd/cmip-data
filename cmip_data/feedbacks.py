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
from numpy import ma  # noqa: F401

from .facets import (
    Database,
    Printer,
    glob_files,
    _item_dates,
    _item_facets,
    _item_label,
)
from .utils import assign_dates, average_regions, load_file

__all__ = [
    'get_feedbacks',
    'process_feedbacks',
]


# Feedback constants
# WARNING: Previously skipped rsdt since it is 'constant' but found with CERES data
# that since seasonal cycle so much larger than global trends (plus possibly due to
# solar trends) it can then show up in monthly regressions. Critical to include.
# NOTE: Here 'n' stands for net (follows Angie convention). These variables are built
# from the 'u' and 'd' components of cmip fluxes and to prefix the kernel-implied net
# fluxes 'alb', 'pl', 'lr', 'hu', 'cl', and 'resid' in _fluxes_from_anomalies. Also
# use convention that 'positive' is into the atmosphere, negative out of atmosphere.
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
    'ts': ('ts',),  # anomaly data for kernel fluxes
    'ta': ('ta',),  # anomaly data for kernel fluxes
    'hus': ('hus',),  # anomaly data for kernel fluxes
    'alb': ('rsds', 'rsus'),  # evaluates to rsus / rsds (up over down)
    'rlnt': ('rlut',),  # evaluates to -rlut (minus out)
    'rsnt': ('rsut', 'rsdt'),  # evaluates to rsdt - rsut (in minus out)
    'rlns': ('rlds', 'rlus'),  # evaluates to rlus - rlds (in minus out)
    'rsns': ('rsds', 'rsus'),  # evaluates to rsus - rsds (in minus out)
    'rlntcs': ('rlutcs',),  # evaluates to -rlutcs (minus out)
    'rsntcs': ('rsutcs', 'rsdt'),  # evaluates to rsdt - rsutcs (in minus out)
    'rlnscs': ('rldscs', 'rlus'),  # evaluates to rlus - rldscs (in minus out)
    'rsnscs': ('rsdscs', 'rsuscs'),  # evaluates to rsuscs - rsdscs (in minus out)
}


def _regress_monthly(denom, numer, proj=False):
    """
    Return a monthly-style weighted feedback regression.

    Parameters
    ----------
    denom, numer : xarray.DataArray
        The denominator and numerator for the regression.
    proj : bool, optional
        Whether to return the slope scaled by standard deviation or the y-intercept.
    """
    # NOTE: Critical to use actual variance and covariance here (scaled by sum of
    # weights) instead of raw sums so that sqrt(var) represents the actual standard
    # deviation. Otherwise the 'projection' is scaled by erroneous sqrt(wgts) term.
    days = denom.time.dt.days_in_month
    wgts = days / days.sum('time')
    navg = (wgts * numer).sum('time', skipna=False)
    davg = (wgts * denom).sum('time', skipna=False)
    covar = wgts * (denom - davg) * (numer - navg)
    dvar = wgts * (denom - davg) ** 2
    covar = covar.sum('time', skipna=False)  # sum scale is sum(wgts) == 1
    dvar = dvar.sum('time', skipna=False)  # sum scale is sum(wgts) == 1
    slope = covar / dvar
    if proj:  # scaled pattern
        extra = covar / np.sqrt(dvar)
    else:  # regression intercept
        extra = navg - slope * davg  # still linear since 'davg' is annual averages
    return slope, extra


def _regress_annual(denom, numer, proj=False):
    """
    Return an annual average-style feedback regression.

    Parameters
    ----------
    denom, numer : xarray.DataArray
        The denominator and numerator for the regression.
    proj : bool, optional
        Whether to return the slope scaled by standard deviation or the y-intercept.
    """
    # NOTE: The denominator should have 'time' coordinate with each slot filled
    # by annual average. This makes both the slope estimator and y intercept linearly
    # additive so that averages can be obtained afterward. See _feedbacks_from_fluxes()
    # NOTE: Assign a standardized time array so that subsequently loading into
    # ensembles with results.py can concatenate more easily. Pick 1800 because it
    # has 28 days on February (for weighted average) and is sort of pre-industrial.
    # Tested calendars in 'cmip-fluxes' and most are 'noleap' or 'standard'/'gregorian'
    # however HadGEM and UK-ESM are '360_day' so there may be slight errors there.
    # Try: for f in *.nc; do echo "$f: $(ncvarinfo time $f | grep calendar)"; done
    navg = numer.groupby('time.month').mean('time', skipna=False)  # 'month' coordinates
    davg = denom.groupby('time.month').mean('time', skipna=False)
    danom = denom.groupby('time.month') - davg  # each 'month' coordinate
    nanom = numer.groupby('time.month') - navg
    covar = (danom * nanom).groupby('time.month').mean('time', skipna=False)
    dvar = (danom ** 2).groupby('time.month').mean('time', skipna=False)
    slope = covar / dvar
    if proj:  # scaled pattern
        extra = covar / np.sqrt(dvar)
    else:  # regression intercept
        extra = navg - slope * davg  # still linear since 'davg' is annual averages
    slope = assign_dates(slope, year=1800)
    extra = assign_dates(extra, year=1800)
    return slope, extra


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
    # from metpy.units import units as mreg  # noqa: F401
    # from metpy import calc  # noqa: F401
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


def _anomalies_from_files(  # test
    project=None, select=None, model=None, dryrun=False, printer=None, **inputs,
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
    model : str, optional
        The input model. Used for NESM3 upper level issues.
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
                if model == 'NESM3' and 'plev' in data.sizes:  # picontrol feedbacks bug
                    data = data.isel(plev=slice(None, 17))
                data.name = variable
                datas.append(data)

        # Calculate variable anomalies using dependencies
        # NOTE: Here subtraction from 'groupby' objects can only be used if the array
        # has a coordinate with the corresponding indices, i.e. in this case 'month'
        # instead of 'time'. Use .groupby('time.month').mean() to turn time indices
        # into month indices (works for both time series and pre-averaged data). Then
        # can easily store monthly climate data and time series in same array since
        # they no longer have same 'time' coordinate. Use this for integration bounds.
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
            elif variable == 'alb':  # ratio of upward to downward shortwave flux
                (control_down, control_up), (response_down, response_up) = controls, responses  # noqa: E501
                control = 100 * (control_up / control_down).clip(0, 1)
                response = 100 * (response_up / response_down).clip(0, 1)
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
        # WARNING: Weird error can happen where NESM3 time coords disagree between
        # variables (single-level data starts at year 500, press-level data starts at
        # year 700) so that assigning to existing dataset just sets all values to
        # nan. Tested output and 1000hPa temperature seems to correspond to surface
        # temperature over oceans, i.e. times are actually correct, so just overwrite.
        with xr.set_options(keep_attrs=True):
            if control is None:
                data = response
            elif response is None:
                data = control.groupby('time.month').mean()  # convert to 'month'
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
        if (  # primarily for FIO-ESM-2-0 data
            all('time' in source.sizes for source in (output, data))
            and data.sizes['time'] == output.sizes['time']  # i.e. intersect not done
            and np.any(data.time.values != output.time.values)
        ):
            print(
                f'Warning: Data times {data.time.values[0]} to {data.time.values[-1]} '
                f'do not match existing times {output.time.values[0]} to {output.time.values[-1]}.'  # noqa: E501
            )
        data.attrs['long_name'] = long_name
        output, data = xr.align(output, data, join='inner')  # avoid nan time slices
        output[variable] = data
        print(f'  {variable} ({long_name}):', data.sizes.get('time', 12))

    if 'time' in output.sizes and output.sizes['time'] == 0:
        raise RuntimeError(
            'Output anomaly data has length-zero time dimension. Probably '
            'result of intersection of time series from different dates.'
        )
    return output


def _fluxes_from_anomalies(
    anoms, kernels, components=None, boundaries=None, verbose=True, printer=None,
):
    """
    Return a dataset containing the actual radiative flux responses and the radiative
    flux responses implied by the radiative kernels and input anomaly data.

    Parameters
    ----------
    anoms : xarray.Dataset
        The climate anomaly data.
    kernels : xarray.Dataset
        The radiative kernel kernels.
    components : tuple, optional
        The components to generate.
    boundaries : tuple, optional
        The boundaries to include.
    verbose : bool, optional
        Whether to print summary statistics.
    printer : callable, default: `print`
        The print function.
    """
    # Prepare kernel dataset before combining with anomaly dataset
    # WARNING: Had unbelievably frustrating issue where pressure level coordinates
    # on kernels and anomalies would be identical (same type, value, etc.) but then
    # anoms.plev == kernels.plev returned empty array and assigning time-varying cell
    # heights to data before integration in Planck feedback block created array of
    # all NaNs. Still not sure why this happened but assign_coords seems to fix it.
    from idealized import physics  # noqa: F401
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

    # Get cell height measures and standardize dataset
    # NOTE: Heights will be derived from 'air_temperature and 'surface_pressure'
    # variables stored in dataset (see VARIABLE_DEPENDENCIES). If from pre-industrial
    # then these are time-average control data (dimension 'month') added onto dataset
    # otherwise containing time series. Keeps things much simpler generally.
    # NOTE: Initially used response time series for tropopause consistent with Zelinka
    # stated methodology of 'time-varying tropopause' but got bizarre issue with Planck
    # feedback biased positive for certain models (seemingly due to higher surface
    # temperature associated with higher tropopause and more negative fluxes... so
    # should be negative bias... but whatever). This also made little sense because it
    # corrected for changing troposphere depth without correcting for reduced strength
    # of kernels under increased depth. Now try with control data average, and below
    # code works with both averages and response time series (see top of file).
    # WARNING: Using assign_coords(cell_height=height) with height 'month' coordinate
    # will cause 'month' to remain as a coordinate on the dataset, but not on other
    # relevant data arrays (tapi, pbot, ptop). Can only fix this by manually assigning
    # the 'month' coord *within the same assign_coords call*. Version: xarray 0.21.1.
    print('Calculating vertical cell measures and Clausius-Clapeyron scaling.')
    anoms = anoms.climo.add_cell_measures(surface=True, tropopause=True)
    anoms = anoms.reset_coords(('plev_bot', 'plev_top'))  # critical (see below)
    height = anoms.cell_height.climo.quantify()
    scalar = ureg.Quantity(0, 'Pa')
    array = anoms['ta' if 'ta' in anoms else 'hus' if 'hus' in anoms else 'ts']
    array = ureg.Pa * xr.zeros_like(array)  # for matching time coordinates
    base = array.groupby('time.month') if 'month' in height.dims else scalar
    with xr.set_options(keep_attrs=True):  # WARNING: critical for 'base' to be second
        height = (const.g * height).climo.to_units('Pa') + base
    if 'plev' in array.dims:
        array = array.isel(plev=0, drop=True)
    coords = {'month': anoms.month, 'cell_height': height.climo.dequantify()}
    anoms = anoms.assign_coords(coords)
    anoms = anoms.climo.quantify()
    output = {'ts': anoms.ts.climo.dequantify()}
    if verbose:
        min_, max_, mean = height.min().item(), height.max().item(), height.mean().item()  # noqa: 501
        print(f'Cell height range: min {min_:.0f} max {max_:.0f} mean {mean:.0f}')
    del height

    # Get Clausius-Clapeyron adjustments and pressure variables
    # TODO: Increase efficiency of tropopause calculation now that it is merely
    # control data repeated onto response data series. Possibly consider assigning
    # cell measures to kernels and use groupby, or build cell height from a sample
    # selection of 12 timesteps then assign repetition to anomaly data.
    # WARNING: Currently cell_height() in physics.py automatically adds surface bounds,
    # tropopause bounds, and cell heights as *coordinates* (for consistency with xarray
    # objects). We must promote to variable before working with other arrays above
    # to avoid confusing conflicts during reassignment.
    pbot = anoms.plev_bot.climo.quantify()
    base = array.groupby('time.month') if 'month' in pbot.dims else scalar
    pbot = pbot.climo.to_units('Pa') + base  # possibly expand along response times
    pbot.attrs['long_name'] = 'surface pressure'
    output['pbot'] = pbot.climo.dequantify()
    if verbose:
        min_, max_, mean = pbot.min().item(), pbot.max().item(), pbot.mean().item()
        print(f'Surface pressure range: min {min_:.0f} max {max_:.0f} mean {mean:.0f}')
    ptop = anoms.plev_top.climo.quantify().climo.to_units('Pa')
    base = array.groupby('time.month') if 'month' in ptop.dims else scalar
    ptop = ptop.climo.to_units('Pa') + base  # possibly expand along response times
    ptop.attrs['long_name'] = 'tropopause pressure'
    output['ptop'] = ptop.climo.dequantify()
    if verbose:
        min_, max_, mean = ptop.min().item(), ptop.max().item(), ptop.mean().item()
        print(f'Tropopause pressure range: min {min_:.0f} max {max_:.0f} mean {mean:.0f}')  # noqa: E501
    # pa = anoms.climo.coords.plev.climo.dequantify()
    ta = anoms.tapi.climo.dequantify()
    scale = _get_clausius_scaling(ta)  # _get_clausius_scaling(ta, pa)
    if verbose:
        min_, max_, mean = scale.min().item(), scale.max().item(), scale.mean().item()
        print(f'Clausius-Clapeyron range: min {min_:.2f} max {max_:.2f} mean {mean:.2f}')  # noqa: E501
    del ta, pbot, ptop

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
        if boundaries and boundary[0].lower() not in map(str.lower, boundaries):
            continue
        if component == '':
            print(f'Calculating {wavelength} {boundary} fluxes using kernel method.')
        if components and component not in components:
            continue
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
        anom = mask = kern = None  # ensure exists
        if component == '' or component == 'cs':  # net flux
            data = 1.0 * anoms[rad]
        elif component == 'resid':  # residual flux
            data = anoms[rad] - running
        elif component == 'alb':  # albedo flux (longwave skipped above)
            kern = kernels[f'alb_{rad}']
            anom = anoms['alb']
            data = kern * anom.groupby('time.month')
        elif component[:2] == 'pl':  # planck flux (shortwave skipped for 'pl')
            kern = 0.0 * ureg('W m^-2 Pa^-1 K^-1')
            if wavelength[0] == 'l':
                kern = kern + kernels[f'ta_{rad}']
            if component == 'pl*':
                kern = kern + kernels[f'hus_{rad}']
            anom = anoms['ts']  # has no cell height
            data = kern * anom.groupby('time.month')
            data = data.assign_coords(cell_height=anoms.cell_height)
            data = data.climo.integral('plev')
            if wavelength[0] == 'l':
                kern = kernels[f'ts_{rad}']
                data = data + kern * anom.groupby('time.month')
        elif component[:2] == 'lr':  # lapse rate flux (shortwave skipped for 'lr')
            kern = 0.0 * ureg('W m^-2 Pa^-1 K^-1')
            if wavelength[0] == 'l':
                kern = kern + kernels[f'ta_{rad}']
            if component == 'lr*':
                kern = kern + kernels[f'hus_{rad}']
            anom = anoms['ta'] - anoms['ts']
            data = kern * anom.groupby('time.month')
            data = data.climo.integral('plev')
        elif component[:2] == 'hu':  # specific and relative humidity fluxes
            kern = kernels[f'hus_{rad}']
            anom = scale * anoms['hus'].groupby('time.month')
            if component == 'hur':
                anom = anom - anoms['ta']
            data = kern * anom.groupby('time.month')
            data = data.climo.integral('plev')
        elif component == 'cl':  # shortwave and longwave cloud fluxes
            data = anoms[rad] - anoms[f'{rad}cs']
            for var in variables:  # relevant longwave or shortwave variables
                kern = kernels[f'{var}_{rad}'] - kernels[f'{var}_{rad}cs']
                mask = kern * anoms[var].groupby('time.month')  # mask adjustment
                if var == 'hus':
                    mask = scale * mask.groupby('time.month')
                if var in ('ta', 'hus'):
                    mask = mask.climo.integral('plev')
                data = data - mask.climo.to_units('W m^-2')  # apply adjustment
        else:
            raise RuntimeError(f'Invalid flux component {component!r}.')
        del anom, mask, kern
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
        if verbose:
            min_, max_, mean = data.min().item(), data.max().item(), data.mean().item()
            print(format(f'  {component} flux:', ' <15s'), end=' ')
            print(f'min {min_: <+7.2f} max {max_: <+7.2f} mean {mean: <+7.2f} ({descrip})')  # noqa: E501
        del data
        gc.collect()

    # Construct dataset and return
    output = xr.Dataset(output)
    return output


def _feedbacks_from_fluxes(
    fluxes, style=None, forcing=None,
    pattern=True, components=None, boundaries=None, verbose=True, printer=None, **kwargs  # noqa: E501
):
    """
    Return a dataset with feedbacks calculated along different surface temperature
    averaging conventions (see `average_regions` for details).

    Parameters
    ----------
    fluxes : xarray.Dataset
        The source for the flux anomaly data. Should have been generated
        in `_fluxes_from_anomalies` using standardized radiative kernel data.
    forcing : path-like, optional
        The explicit forcing data. Required for ratio-style feedbacks.
    style : {'monthly', 'annual', 'ratio'}, optional
        The type of feedback calcluation to perform.
    pattern : bool, optional
        Whether to include the local temperature pattern term in the output dataset.
    components : tuple, optional
        The components to include.
    boundaries : tuple, optional
        The boundaries to include.
    verbose : bool, optional
        Whether to print summary statistics.
    printer : callable, default: `print`
        The print function.
    **kwargs
        Passed to `average_regions`.
    """
    # Load data and perform averages
    # NOTE: Need to extract 'plev' top and bottom because for some reason they get
    # dropped from 'input' since they are defined on the coordinate dictionary.
    # NOTE: New idea is to retrieve global feedbacks from average of regressions
    # on global temperature and annual feedbacks from average of regressions on
    # annual temperature. So remove 'average_periods' and replace with simpler
    # function that gets annual temperature averages, e.g. from observed.py
    style = style or 'monthly'
    print = printer or builtins.print
    print(f'Calculating {style} climate feedbacks.')
    print('Getting average spatial regions.')
    fluxes = fluxes.climo.add_cell_measures()
    if style == 'monthly':
        temp = fluxes.ts
    elif style == 'annual' or style == 'ratio':
        days = fluxes.time.dt.days_in_month.astype(fluxes.ts.dtype)
        wgts = days.groupby('time.year') / days.groupby('time.year').sum()
        with xr.set_options(keep_attrs=True):
            temp = (fluxes.ts * wgts).groupby('time.year').sum('time', skipna=False)
            temp = xr.zeros_like(fluxes.ts).groupby('time.year') + temp
    else:
        raise ValueError(f"Invalid {style=}. Expected 'monthly', 'annual', or 'ratio'.")
    if style == 'ratio' and fluxes.sizes['time'] > 12:
        raise ValueError('Fluxes have too many time coordinates for ratio-style feedback.')  # noqa: E501
    if style == 'ratio' and forcing is None:
        raise ValueError('Forcing estimate required for ratio-style feedbacks.')
    temp = average_regions(temp, **kwargs)  # add region coordinate
    temp = temp.climo.quantify()
    temp.attrs['long_name'] = 'temperature averaging region'
    fluxes = fluxes.climo.quantify()

    # Iterate over flux components
    # NOTE: Operating one region at a time was necessary on previous server with
    # smaller cache. On new server operate on every region at once (more efficient).
    # NOTE: Since forcing is constant, the cloud masking adjustment has no effect on
    # feedback estimates, but does affect the 'cl_erf' cloud adjustment forcing. Also
    # shortwave clear-sky effects are just albedo and small water vapor effect, so very
    # small, but longwave component is always very significant.
    output = {}  # save time by writing dataset afterward
    for boundary, wavelength, (component, descrip) in itertools.product(
        ('TOA', 'surface'), ('longwave', 'shortwave'), FEEDBACK_DESCRIPTIONS.items(),
    ):
        # Get the flux components
        # TODO: Update 'flux' files and remove code that translates e.g. 'alb_rfnt'
        # arrays to 'alb_rsnt'. Should write variable renaming script.
        # NOTE: Previously computed and stored 'full' components here with appropriate
        # renames but no longer do that. Instead process.py get_data() automatically
        # combines shortwave and longwave components as needed.
        rad = f'r{wavelength[0]}n{boundary[0].lower()}'  # this wavelength
        full = f'{component}_rfn{boundary[0].lower()}'  # full wavelength
        count = sum(map(bool, FEEDBACK_DEPENDENCIES[component].values()))
        if boundaries and boundary[0].lower() not in map(str.lower, boundaries):
            continue
        if component == '':  # print header
            print(f'Calculating {boundary} {wavelength} forcing and feedback.')
        if components and component not in components:
            continue
        if component in ('', 'cs'):
            name = f'{rad}{component}'
        elif FEEDBACK_DEPENDENCIES[component][wavelength]:
            name = f'{component}_{rad}'
        else:  # skips e.g. shortwave planck or longwave albedo
            continue
        if name in fluxes:
            flux = fluxes[name]  # pointwise radiative flux
        elif full in fluxes and count == 1:
            flux = fluxes[full]  # outdated flux data with 's' and 'l' renamed to 'f'
        else:
            print(
                'Warning: Input dataset is missing the feedback '
                f'dependency {name!r}. Skipping calculation.'
            )
            continue

        # Possibly adjust for forcing masking
        # NOTE: Should have no effect if forcing is constant. Also this requires
        # calculating the net flux feedbacks before the cloud and residual components.
        # NOTE: Soden et al. (2008) used standard horizontally uniform value of 15%
        # the full forcing but no rcason not to use regressed forcing estimates.
        erfs = tuple(f'{rad}{sky}_erf' for sky in ('', 'cs'))
        erfs = erfs if component in ('cl', 'resid') else ()
        if message := ', '.join(repr(erf) for erf in erfs if erf not in output):
            print(
                'Warning: Output dataset is missing cloud-masking forcing '
                f'adjustment variable(s) {message}. Cannot make adjustment.'
            )
            continue
        if erfs:  # already ensured in output
            diff = output[erfs[0]] - output[erfs[1]]  # all-sky minus clear-sky
            if wavelength == 'longwave' and component == 'cl':
                min_, max_, mean = diff.min().item(), diff.max().item(), diff.mean().item()  # noqa: E501
                print(format('  masking:', ' <12s'), end=' ')
                print(f'min {min_: <+7.2f} max {max_: <+7.2f} mean {mean: <+7.2f}')
            if style == 'annual' or style == 'ratio':  # convert 12 times to 'month'
                flux = flux.groupby('time.month')
                diff = diff.groupby('time.month').mean()  # no-op
            scale = -1 if component == 'cl' else 1  # add or remove
            scale = ureg.Quantity(scale, 'W m^-2')
            with xr.set_options(keep_attrs=True):
                flux = flux + scale * diff  # add monthly masking
            del diff
            gc.collect()

        # Perform the regression or division
        # See: https://en.wikipedia.org/wiki/Simple_linear_regression
        # NOTE: When forcing is provided time-varying input will produce time-varying
        # feedbacks similar to Armour 2015 or just simple scalar values if data is
        # already time-averaged. When not provided time coordinate is required.
        # NOTE: Previously did separate regressions with and without intercept...
        # but piControl regressions are *always* centered on origin because they
        # are deviations from the average by *construction*. So unnecessary.
        if style == 'monthly':  # simple weighted monthly regressions
            lam, erf = _regress_monthly(temp, flux)
            erf = 0.5 * erf  # double CO2
        elif style == 'annual':  # denominator filled with annual averages
            lam, erf = _regress_annual(temp, flux)
            erf = 0.5 * erf  # double CO2
        else:
            erf = forcing.climo.get(f'{name}_erf', quantify=True).assign_coords(time=temp.time)  # noqa: E501
            lam = (flux - 2.0 * erf) / temp
        del flux
        gc.collect()

        # Standardize the results
        # NOTE: Always keep non-net forcing estimates as these represent rapid
        # adjustments. Also previously also recored equilibrium climate sensitivity
        # but this is always nonsense on local kernels, so now compute a posterior only.
        component = 'net' if component == '' else component
        descrip = 'net' if component == '' else descrip
        lam = lam.climo.to_units('W m^-2 K^-1')
        lam = lam.climo.dequantify()
        lam.name = f'{name}_lam'
        lam.attrs['long_name'] = f'{boundary} {wavelength} {descrip} feedback parameter'
        output[lam.name] = lam
        if verbose:
            min_, max_, mean = lam.min().item(), lam.max().item(), lam.mean().item()
            print(format(f'  {component} lam:', ' <12s'), end=' ')
            print(f'min {min_: <+7.2f} max {max_: <+7.2f} mean {mean: <+7.2f} ({descrip})')  # noqa: E501
        erf = erf.climo.to_units('W m^-2')
        erf = erf.climo.dequantify()
        erf.name = f'{name}_erf'  # halved from quadrupled co2
        erf.attrs['long_name'] = f'{boundary} {wavelength} {descrip} effective forcing'
        output[erf.name] = erf
        if verbose:
            min_, max_, mean = erf.min().item(), erf.max().item(), erf.mean().item()
            print(format(f'  {component} erf:', ' <12s'), end=' ')
            print(f'min {min_: <+7.2f} max {max_: <+7.2f} mean {mean: <+7.2f} ({descrip})')  # noqa: E501
        del erf, lam
        gc.collect()

    # Add final pattern effect and pressure terms
    # NOTE: For weighting see https://stats.stackexchange.com/a/489949/156605
    # NOTE: Region coordinate is necessary for pattern effect so it can be shown
    # for different feedback versions. Non-globe values get auto-filled with nan.
    print('Adding pattern and pressure terms.')
    output = xr.Dataset(output)
    output = output.transpose('region', ..., 'lat', 'lon')
    output.attrs['title'] = f'{style.title()} forcing-feedback decompositions'
    for key in ('pbot', 'ptop'):
        if key not in fluxes:
            continue
        data = fluxes[key]
        if 'time' in data.dims:  # i.e. seasonal averages as function of year
            data = data.mean(dim='time', keep_attrs=True)
        output[key] = data.climo.dequantify()
    if pattern:
        denom = temp.sel(region='globe')  # annual-averaged data
        numer = temp.sel(region='point')  # original monthly data
        if style == 'monthly':
            slope, proj = _regress_monthly(denom, numer, proj=True)
        elif style == 'annual':
            slope, proj = _regress_annual(denom, numer, proj=True)
        else:  # ratio-feedback 'slope' is ratio of changes, 'proj' is absolute change
            slope, proj = numer / denom, numer
        proj = proj.climo.dequantify()
        proj = proj.assign_coords(region='globe').expand_dims('region')
        proj.attrs.update(units='K', long_name='regional warming')
        output['tstd'] = proj  # other regions auto-filled with np.nan
        slope = slope.climo.dequantify()
        slope = slope.assign_coords(region='globe').expand_dims('region')
        slope.attrs.update(units='K / K', long_name='relative warming')
        output['tpat'] = slope  # other regions auto-filled with np.nan
    return output


def get_feedbacks(
    feedbacks='~/data',
    fluxes='~/data',
    kernels='~/data/cmip-kernels',
    select=None,
    entire=None,
    early=None,
    source=None,
    style=None,
    project=None,
    experiment=None,
    ensemble=None,
    table=None,
    model=None,
    nodrift=False,
    overwrite=False,
    recompute=False,
    printer=None,
    dryrun=False,
    noload=False,
    **inputs
):
    """
    Calculate the net surface and top-of-atmosphere feedback components with a variety
    of averaging conventions and save the result in an automatically-named file. Also
    save the flux component calculations in an intermediate location.

    Parameters
    ----------
    feedbacks : path-like, optional
        The output feedback directory.
        Subfolder ``{project}-{experiment}-feedbacks`` is used.
    fluxes : path-like, optional
        The input/output flux directory. Used for repeated feedback calculations.
        Subfolder ``{project}-{experiment}-fluxes`` is used.
    kernels : path-like, optional
        The input kernel scale directory.
        Default folder is ``~/cmip-kernels``.
    select : 2-tuple of int, optional
        The start and stop years. Used in the output file name. If relevant the anomaly
        data is filtered to these years using ``.isel(slice(12 * start, 12 * stop))``.
    entire : 2-tuple of int, optional
        The full start and stop years. Used to load flux data containing a superset
        of `select` times. Ignored if creating new flux data.
    early : 2-tuple of int, optional
        The early start and stop years. Used to load forcing data for calculating
        ratio-style feedbacks. Ignored for regression style feedbacks.
    source : str, default: 'eraint'
        The source for the kernel data (e.g. ``'eraint'``). This searches for files
        in the `kernels` directory formatted as ``kernels_{source}.nc``.
    style : {'monthly', 'annual', 'ratio'}, optional
        Whether to compute feedbacks with annual anomalies, monthly anomalies, or a
        ratio instead of a regression (latter only possible with abrupt experiments).
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
        Whether to overwrite existing feedback files. Turned on if flux file is missing.
    recompute : bool, default: False
        Whether to recompute existing flux files. Less commonly used than `overwrite`.
    printer : callable, default: `print`
        The print function.
    dryrun : bool, optional
        Whether to run with only three years of data.
    noload : bool, optional
        Whether to speed things up by skipping loading if the file exists.

    Returns
    -------
    path : Path
        The path of the output file, formatted according to cmip conventions as
        ``feedbacks_{table}_{model}_{experiment}_{ensemble}_{options}.nc`` where
        `table`, `model`, `experiment`, and `ensemble` are applied only if passed
        by the user and `options` is set to ``{source}-{style}-{nodrift}``,
        where `source` is the kernel source; `style` is one of ``annual``, ``monthly``
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
    ``style='ratio'`` this function looks for ``'annual'`` files to retrieve the
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
    entire = entire or (0, 150)
    early = early or (0, 20)
    source = source or 'eraint'
    style = style or 'monthly'
    series = 'climate' if style == 'ratio' else 'series'
    nodrift = nodrift and 'nodrift' or ''
    outputs = []
    subfolder = _item_label((project, experiment, table))
    tuples = (  # try to load from parent flux file if possible
        (fluxes, 'fluxes', series, select),  # original record
        (fluxes, 'fluxes', series, entire),  # subselection on response period
        (feedbacks, 'feedbacks', style, select),  # feedbacks file
        (feedbacks, 'feedbacks', style, early),  # ratio forcing file
    )
    for folder, prefix, suffix, times in tuples:
        version = (*(format(int(t), '04d') for t in times), source, suffix, nodrift)
        parts = (prefix, table, model, experiment, ensemble, version)
        file = _item_label(*parts, modify=False) + '.nc'
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
    # NOTE: Always overwrite if at least flux data is missing so that we never have
    # feedback data inconsistent with the flux data on storage.
    # NOTE: Try to load from same fluxes files for feedbacks estimated from subset
    # of full time series to avoid duplicating expensive calculations. Requires
    # writing full perturbed feedbacks before 'early' and 'late' feedbacks.
    # NOTE: Even for kernel-derived flux responses, rapid adjustments and associated
    # pattern effects may make the effective radiative forcing estimate non-zero (see
    # Andrews et al.) so we always save the regression intercept data.
    *fluxes, feedbacks, forcing = outputs
    fluxes_exist = tuple(flux.is_file() and flux.stat().st_size > 0 for flux in fluxes)
    feedbacks_exist = feedbacks.is_file() and feedbacks.stat().st_size > 0
    overwrite = overwrite or feedbacks_exist and not any(fluxes_exist)
    if not dryrun and not recompute and any(fluxes_exist):
        output = fluxes[0] if fluxes_exist[0] else fluxes[1]
        if noload and feedbacks_exist and not overwrite:
            print('Skipping loading flux data.')
            fluxes = output
        else:
            print(f'Loading flux data from file: {output.name}')
            fluxes = load_file(output, validate=False)
        if isinstance(fluxes, xr.Dataset) and not fluxes_exist[0]:
            init, start, stop = *entire[:1], *select  # subselect from perturbed data
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
        kwargs = dict(select=select, project=project, model=model, dryrun=dryrun)
        anoms = _anomalies_from_files(printer=print, **kwargs, **inputs)
        fluxes = _fluxes_from_anomalies(anoms, kernels, printer=print)
        for dataset in (anoms, kernels):
            dataset.close()
        if not dryrun:  # save after compressing repeated values
            encoding = {key: {'zlib': True, 'complevel': 5} for key in fluxes.data_vars}
            output.unlink(missing_ok=True)
            fluxes.to_netcdf(output, engine='netcdf4', encoding=encoding)
            print(f'Created output file: {output.name}')
        print('Skipping loading flux data.')
    if not dryrun and not overwrite and feedbacks_exist:
        output = feedbacks
        if noload:
            print('Skipping loading feedback data.')
            feedbacks = output
        else:
            print(f'Loading feedback data from file: {output.name}')
            feedbacks = load_file(output, validate=False)
    else:
        output = feedbacks  # use the name 'feedbacks' for dataset
        print(f'Output feedback file: {output.name}')
        if style == 'ratio':
            forcing = forcing.parent / forcing.name.replace('ratio', 'annual')
            print(f'Loading forcing data from file: {forcing.name}')
            forcing = load_file(forcing, project=project, validate=False)
        elif style == 'monthly' or style == 'annual':
            forcing = None
        else:
            raise ValueError(f"Invalid {style=}. Expected 'monthly', 'annual', or 'ratio'.")  # noqa: E501
        feedbacks = _feedbacks_from_fluxes(fluxes, style=style, forcing=forcing, printer=print)  # noqa: E501
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
    style=None,
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
    style : {'monthly', 'annual', 'ratio'}, optional
        The type of feedback calcluation to perform.
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
        Whether to log the printed output.
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
    style = style or 'monthly'
    logging = logging and not dryrun
    experiment = experiment or 'abrupt4xCO2'
    source = source or 'eraint'
    nodrift = 'nodrift' if nodrift else ''
    suffix = nodrift and '-' + nodrift
    control = control or (0, 150)
    if style == 'monthly' or style == 'annual':
        response = response or (0, 150)
        control_suffix = f'{control[0]:04d}-{control[1]:04d}-climate{suffix}'
        response_suffix = f'{control[0]:04d}-{control[1]:04d}-series{suffix}'
    elif style == 'ratio':  # ratio-type
        response = response or (120, 150)
        control_suffix = f'{control[0]:04d}-{control[1]:04d}-climate{suffix}'
        response_suffix = f'{response[0]:04d}-{response[1]:04d}-climate{suffix}'
    else:
        raise ValueError(f"Invalid {style=}. Expected 'monthly', 'annual', or 'ratio'.")
    select, entire = response, control
    suffixes = (*(format(int(t), '04d') for t in select), source, style, nodrift)
    constraints = {
        'variable': sorted(set(k for d in VARIABLE_DEPENDENCIES.values() for k in d)),
        'experiment': ['piControl', experiment],
        'table': 'Amon',
    }
    constraints.update({
        key: kwargs.pop(key) for key in tuple(kwargs)
        if any(s in key for s in ('model', 'flagship', 'ensemble'))
    })
    if logging:
        print = Printer('feedbacks', *suffixes, project=project, **constraints)
    else:
        print = builtins.print
    print()  # before getting logger
    print('Generating database.')
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
        print(f'Computing {style} feedbacks:')
        print(', '.join(f'{key}: {value}' for key, value in group.items()))
        files = _get_file_pairs(
            data,
            control_experiment,
            control_suffix,
            response_experiment,
            response_suffix,
            printer=print,
        )
        item = tuple(files.values())[0][1]
        try:
            datasets = get_feedbacks(
                project=group['project'],
                experiment=_item_facets['experiment'](item),
                ensemble=_item_facets['ensemble'](item),
                table=_item_facets['table'](item),
                model=group['model'],
                select=select,
                entire=entire,
                style=style,
                source=source,
                nodrift=nodrift,
                printer=print,
                dryrun=dryrun,
                **kwargs,
                **files,
            )
            for dataset in datasets:
                if isinstance(dataset, xr.Dataset):
                    dataset.close()
        except Exception as error:
            if dryrun or nowarn:
                raise error
            else:
                _print_error(error)
            print('Warning: Failed to compute feedbacks.')
            continue
