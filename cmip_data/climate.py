#!/usr/bin/env python3
"""
Create files containing combined time-mean and time-variance quantities.
"""
# TODO: Add coupled/climate.py utilities here and support feedback-style
# regressions of circulation variables against surface temperature.
import builtins
import itertools
import traceback
import re

import climopy as climo  # noqa: F401  # add accessor
import numpy as np
import xarray as xr
from icecream import ic  # noqa: F401
from climopy import const, diff, ureg, vreg  # noqa: F401
from metpy import calc as mcalc
from metpy import units as munits

from .facets import Database, Printer, glob_files, _item_dates, _item_facets
from .utils import assign_dates, load_file

__all__ = [
    'get_climate',
    'process_climate',
]

# Regular expressions
# NOTE: Period keys are outdated, currently only used with previous 'slope'
# feedback files but for climate data simply load monthly data, average later.
REGEX_DIRECTION = re.compile(r'(upwelling|downwelling|outgoing|incident)')
REGEX_EXPONENT = re.compile(r'([a-df-zA-DF-Z]+)([-+]?[0-9]+)')

# Energetics constants
# NOTE: Here the flux components table was copied from feedbacks.py. Critical
# to get net fluxes and execute renames before running transport derivations.
# NOTE: Here the %s is filled in with water, liquid, or ice depending on the
# component of the particular variable category.
# NOTE: Rename native surface budget terms to look more like cloud water and
# ice terms. Then use 'prl'/'prp' and 'evl'/'evp' for ice components.
HYDRO_DEPENDENCIES = {
    'hur': ('plev', 'ta', 'hus'),
    'hurs': ('ps', 'ts', 'huss'),
}
ENERGY_RENAMES = {
    'prsn': 'pri',
    'evspsbl': 'ev',
    'sbl': 'evi',
}
HYDRO_COMPONENTS = [
    ('ev', 'evl', 'evi', 'evp', '%s evaporation'),
    ('pr', 'prl', 'pri', 'prp', '%s precipitation'),
    ('clw', 'cll', 'cli', 'clp', 'mass fraction cloud %s'),
    ('clwvi', 'cllvi', 'clivi', 'clpvi', 'condensed %s water path'),
]
ENERGY_COMPONENTS = {
    'albedo': ('rsds', 'rsus'),  # evaluates to rsus / rsds (out over in)
    'rlnt': ('rlut',),  # evaluates to -rlut (minus out)
    'rsnt': ('rsut', 'rsdt'),  # evaluates to rsdt - rsut (in minus out)
    'rlns': ('rlds', 'rlus'),  # evaluates to rlus - rlds (in minus out)
    'rsns': ('rsds', 'rsus'),  # evaluates to rsus - rsds (in minus out)
    'rlntcs': ('rlutcs',),  # evaluates to -rlutcs (minus out)
    'rsntcs': ('rsutcs', 'rsdt'),  # evaluates to rsdt - rsutcs (in minus out)
    'rlnscs': ('rldscs', 'rlus'),  # evaluates to rlus - rldscs (in minus out)
    'rsnscs': ('rsdscs', 'rsuscs'),  # evaluates to rsuscs - rsdscs (in minus out)
}

# Transport constants
# NOTE: See Donohoe et al. (2020) for details on transport terms. Precipitation appears
# in the dry static energy formula because unlike surface evaporation, it deposits heat
# inside the atmosphere, i.e. it remains after subtracting surface and TOA loss terms.
# NOTE: Duffy et al. (2018) and Mayer et al. (2020) suggest a snow correction of
# energy budget is necessary, and Armour et al. (2019) suggests correcting for
# "latent heat associated with falling snow", but this is relative to estimate of
# hfls from idealized expression for *evaporation over ocean* based on temperature
# and humidity differences between surface and boundary layer. Since model output
# hfls = Lv * evsp + Ls * sbl exactly (compare below terms), where the sbl term is
# equivalent to adding the latent heat of fusion required to melt snow before a
# liquid-vapor transition, an additional correction is not needed here.
TRANSPORT_DESCRIPTIONS = {
    'gse': 'potential static',
    'hse': 'sensible static',
    'dse': 'dry static',
    'lse': 'latent static',
    'ocean': 'storage + ocean',
}
TRANSPORT_SCALES = {  # scaling prior to implicit transport calculations
    'pr': const.Lv,
    'prl': const.Lv,
    'pri': const.Ls - const.Lv,  # remove the 'pri * Lv' implied inside 'pr' term
    'ev': const.Lv,
    'evl': const.Lv,
    'evi': const.Ls - const.Lv,  # remove the 'evi * Lv' implied inside 'pr' term
}
TRANSPORT_EXPLICIT = {
    'dse': (1, 'intuadse', 'intvadse'),
    'lse': (const.Lv, 'intuaw', 'intvaw'),
}
TRANSPORT_IMPLICIT = {
    'dse': (('hfss', 'rlns', 'rsns', 'rlnt', 'rsnt', 'pr', 'pri'), ()),
    'lse': (('hfls',), ('pr', 'pri')),
    'ocean': ((), ('hfss', 'hfls', 'rlns', 'rsns')),  # flux into atmosphere
}
TRANSPORT_INDIVIDUAL = {
    'gse': ('zg', const.g, 'dam m s^-1'),
    'hse': ('ta', const.cp, 'K m s^-1'),
    'lse': ('hus', const.Lv, 'g kg^-1 m s^-1'),
}


def _file_pairs(data, *args, printer=None):
    """
    Return a dictionary mapping of variables to ``(climate, series)``
    file pairs from a database of files with ``(..., variable)`` keys.

    Parameters
    ----------
    data : dict
        The database of file lists.
    *args : str
        The suffixes to use.
    printer : callable, default: `print`
        The print function.
    """
    print = printer or builtins.print
    pairs = []
    for suffix in args:
        paths = {
            var: [file for file in files if _item_dates(file) in args]
            for (*_, var), files in data.items()
        }
        for variable, files in tuple(paths.items()):
            names = ', '.join(file.name for file in files)
            print(f'  {suffix} {variable} files ({len(files)}):', names)
            if message := not files and 'Missing' or len(files) > 1 and 'Ambiguous':
                print(f'Warning: {message} files (expected 1, found {len(files)}).')
                del paths[variable]
            else:
                (file,) = files
                paths[variable] = file
        pairs.append(paths)
    if message := ', '.join(pairs[0].keys() - pairs[1].keys()):
        pass  # ignore common situation of more climate files than series files
    if message := ', '.join(pairs[1].keys() - pairs[0].keys()):
        print(f'Warning: Missing climate files for series data {message}.')
    pairs = {
        variable: (pairs[0].get(variable, None), pairs[1].get(variable, None))
        for variable in sorted(pairs[0].keys() | pairs[1].keys())
    }
    return pairs


def _add_energetics(
    dataset, drop_clear=False, drop_directions=True, correct_solar=True,
):
    """
    Add albedo and net fluxes from upwelling and downwelling components and remove
    the original directional variables.

    Parameters
    ----------
    dataset : xarray.Dataset
        The dataset.
    drop_clear : bool, default: False
        Whether to drop the clear-sky components.
    drop_directions : bool, optional
        Whether to drop the directional components
    correct_solar : bool, optional
        Whether to correct zonal variations in insolation by replacing with averages.
    """
    # NOTE: Here also add 'albedo' term and generate long name by replacing directional
    # terms in existing long name. Remove the directional components when finished.
    for key, name in ENERGY_RENAMES.items():
        if key in dataset:
            dataset = dataset.rename({key: name})
    keys_directions = set()
    if correct_solar and 'rsdt' in dataset:
        data = dataset.rsdt.mean(dim='lon', keepdims=True)
        dataset.rsdt[:] = data  # automatically broadcasts
    for name, keys in ENERGY_COMPONENTS.items():
        keys_directions.update(keys)
        if any(key not in dataset for key in keys):  # skip partial data
            continue
        if drop_clear and name[-2:] == 'cs':
            continue
        if name == 'albedo':
            dataset[name] = 100 * dataset[keys[1]] / dataset[keys[0]]
            long_name = 'surface albedo'
            unit = '%'
        elif len(keys) == 2:
            dataset[name] = dataset[keys[1]] - dataset[keys[0]]
            long_name = dataset[keys[1]].attrs['long_name']
            unit = 'W m^-2'
        else:
            dataset[name] = -1 * dataset[keys[0]]
            long_name = dataset[keys[0]].attrs['long_name']
            unit = 'W m^-2'
        long_name = long_name.replace('radiation', 'flux')
        long_name = REGEX_DIRECTION.sub('net', long_name)
        dataset[name].attrs.update({'units': unit, 'long_name': long_name})
    if drop_directions:
        dataset = dataset.drop_vars(keys_directions & dataset.data_vars.keys())
    return dataset


def _add_hydrology(
    dataset, parts_cloud=True, parts_precip=False, parts_phase=False,
):
    """
    Add relative humidity and ice and liquid water terms and standardize
    the insertion order for the resulting dataset variables.

    Parameters
    ----------
    dataset : xarray.Dataset
        The dataset.
    parts_cloud : bool, optional
        Whether to keep cloud columnwise and layerwise ice components.
    parts_precip : bool, optional
        Whether to keep evaporation and precipitation ice components.
    parts_phase : bool, default: False
        Whether to keep separate liquid and ice parts in addition to ice ratio.
    """
    # Add humidity terms
    # NOTE: Generally forego downloading relative humidity variables... true that
    # operation is non-linear so relative humidity of climate is not climate of
    # relative humiditiy, but we already use this approach with feedback kernels.
    for name, keys in HYDRO_DEPENDENCIES.items():
        descrip = 'near-surface ' if name[-1] == 's' else ''
        long_name = f'{descrip}relative humidity'
        if all(key in dataset for key in keys):
            datas = []
            for key, unit in zip(keys, ('Pa', 'K', '')):
                src = dataset.climo.coords if key == 'plev' else dataset.climo.vars
                data = src[key].climo.to_units(unit)  # climopy units
                data.data = data.data.magnitude * munits.units(unit)  # metpy units
                datas.append(data)
            data = mcalc.relative_humidity_from_specific_humidity(*datas)
            data = data.climo.dequantify()  # works with metpy registry
            data = data.climo.to_units('%').clip(0, 100)
            data.attrs['long_name'] = long_name
            dataset[name] = data

    # Add cloud terms
    # NOTE: Unlike related variables (including, confusingly, clwvi), 'clw' includes
    # only liquid component rather than combined liquid plus ice. Adjust it to match
    # convention from other variables and add other component terms.
    if 'clw' in dataset:
        dataset = dataset.rename_vars(clw='cll')
        if 'cli' in dataset:
            with xr.set_options(keep_attrs=True):
                dataset['clw'] = dataset['cli'] + dataset['cll']
    for name, lname, iname, rname, descrip in HYDRO_COMPONENTS:
        skip_parts = not parts_cloud and name in ('clw', 'clwvi')
        skip_parts = skip_parts or not parts_precip and name in ('ev', 'pr')
        if not skip_parts and name in dataset and iname in dataset:
            da = (100 * dataset[iname] / dataset[name]).clip(0, 100)
            da.attrs = {'units': '%', 'long_name': descrip % 'ice' + ' ratio'}
            dataset[rname] = da
        if not skip_parts and parts_phase and name in dataset and iname in dataset:
            da = dataset[name] - dataset[iname]
            da.attrs = {'units': dataset[name].units, 'long_name': descrip % 'liquid'}
            dataset[lname] = da
        if skip_parts:  # remove everything except total
            drop = dataset.data_vars.keys() & {lname, iname, rname}
            dataset = dataset.drop_vars(drop)
        if not parts_phase:  # remove everything except ratio and total
            drop = dataset.data_vars.keys() & {lname, iname}
            dataset = dataset.drop_vars(drop)
    for name, lname, iname, _, descrip in HYDRO_COMPONENTS:
        names = (name, lname, iname)
        strings = ('water', 'liquid', 'ice')
        for name, string in zip(names, strings):
            if name not in dataset:
                continue
            data = dataset[name]
            if string not in data.long_name:
                data.attrs['long_name'] = descrip % string
    return dataset


def _add_transport(
    dataset,
    implicit=True, alternative=False, explicit=False, drop_implicit=False, drop_explicit=True,  # noqa: E501
    parts_local=True, parts_static=False, parts_eddies=False, parts_fluxes=False,
):
    """
    Add local transport convergence and zonal-mean meridional transport including
    sensible-geopotential-latent and mean-stationary-transient breakdowns.

    Parameters
    ----------
    dataset : xarray.Dataset
        The dataset.
    implicit : bool, optional
        Whether to include implicit transport estimates.
    alternative : bool, optional
        Whether to include alternative transport estimates.
    explicit : bool, optional
        Whether to include explicit transport estimates.
    drop_implicit : bool, optional
        Whether to drop energetic components used for implicit transport.
    drop_explicit : bool, optional
        Whether to drop flux components used for explicit transport.
    parts_local : bool, optional
        Whether to include local transport components.
    parts_eddies : bool, optional
        Whether to include zonal-mean stationary and transient components.
    parts_static : bool, optional
        Whether to include separate sensible and geopotential components.
    parts_fluxes : bool, default: False
        Whether to compute additional flux estimates.
    """
    # Get implicit ocean, dry, and latent transport
    # WARNING: This must come after _add_energetics introduces 'net'
    # components. Also previously removed contributors to implicit calculations
    # but not anymore... might consider adding back.
    idxs, ends = (0,), ('',)
    if alternative:
        idxs, ends = (0, 1), ('', '_alt')
    keys_implicit = set()  # components should be dropped
    for (name, pair), idx in itertools.product(TRANSPORT_IMPLICIT.items(), idxs):
        end, constants = ends[idx], {}
        for c, keys in zip((1, -1), map(list, pair)):
            if end and 'hfls' in keys:
                keys[(idx := keys.index('hfls')):idx + 1] = ('ev', 'evi')
            keys_implicit.update(keys)
            constants.update({k: c * TRANSPORT_SCALES.get(k, 1) for k in keys})
        if implicit and (not end or 'evi' in constants):  # avoid redundancy
            descrip = TRANSPORT_DESCRIPTIONS[name]
            kwargs = {'descrip': descrip, 'prefix': 'alternative' if end else ''}
            if all(key in dataset for key in constants):
                data = sum(c * dataset.climo.vars[k] for k, c in constants.items())
                cdata, rdata, tdata = _infer_transport(data, **kwargs)
                dataset[f'{name}t{end}'] = tdata  # zonal-mean transport
                if parts_local:  # include local convergence
                    dataset[f'{name}c{end}'] = cdata
                if parts_eddies:  # include residual estimate
                    dataset[f'{name}r{end}'] = rdata
    drop = keys_implicit & dataset.data_vars.keys()
    if drop_implicit:
        dataset = dataset.drop_vars(drop)

    # Get explicit dry, latent, and moist transport
    # NOTE: The below regex prefixes exponents expressed by numbers adjacent to units
    # with the carat ^, but ignores the scientific notation 1.e6 in scaling factors,
    # so dry static energy convergence units can be parsed as quantities.
    keys_explicit = set()
    if explicit:  # later processing
        ends += ('_exp',)
    for name, (scale, ukey, vkey) in TRANSPORT_EXPLICIT.items():
        keys_explicit.update((ukey, vkey))
        descrip = TRANSPORT_DESCRIPTIONS[name]
        kwargs = {'descrip': descrip, 'prefix': 'explicit'}
        if explicit and ukey in dataset and vkey in dataset:
            udata = dataset[ukey].copy(deep=False)
            vdata = dataset[vkey].copy(deep=False)
            udata *= scale * ureg(REGEX_EXPONENT.sub(r'\1^\2', udata.attrs.pop('units')))  # noqa: E501
            vdata *= scale * ureg(REGEX_EXPONENT.sub(r'\1^\2', vdata.attrs.pop('units')))  # noqa: E501
            qdata = ureg.dimensionless * xr.ones_like(udata)  # placeholder
            cdata, _, tdata = _process_transport(udata, vdata, qdata, **kwargs)
            dataset[f'{name}t_exp'] = tdata
            if parts_local:  # include local convergence
                dataset[f'{name}c_exp'] = cdata
    drop = keys_explicit & dataset.data_vars.keys()
    if drop_explicit:
        dataset = dataset.drop_vars(drop)

    # Get mean transport, stationary transport, and stationary convergence
    # TODO: Stop storing these. Instead implement in loading func or as climopy
    # derivations. Currently get simple summation, product, and difference terms
    # on-the-fly (e.g. mse and total transport) but need to support more complex stuff.
    iter_ = tuple(TRANSPORT_INDIVIDUAL.items())
    for name, (quant, scale, _) in iter_:
        descrip = TRANSPORT_DESCRIPTIONS[name]
        if (parts_eddies or parts_static) and (  # TODO: then remove after adding?
            'ps' in dataset and 'va' in dataset and quant in dataset
        ):
            qdata = scale * dataset.climo.vars[quant]
            udata, vdata = dataset.climo.vars['ua'], dataset.climo.vars['va']
            cdata, sdata, mdata = _process_transport(udata, vdata, qdata, descrip=descrip)  # noqa: E501
            dataset[f's{name}c'] = cdata  # stationary compoennt of convergence
            dataset[f's{name}t'] = sdata  # stationary component of meridional transport
            dataset[f'm{name}t'] = mdata  # mean component of meridional transport

    # Get missing transient components and total sensible and geopotential terms
    # NOTE: Transient sensible transport is calculated from the residual of the dry
    # static energy minus both the sensible and geopotential stationary components. The
    # all-zero transient geopotential is stored for consistency if sensible is present.
    iter_ = itertools.product(TRANSPORT_INDIVIDUAL, ('cs', 'tsm'), ends)
    for name, (suffix, *prefixes), end in iter_:
        ref = f'{prefixes[0]}{name}{suffix}'  # reference component
        total = 'lse' if name == 'lse' else 'dse'
        total = f'{total}{suffix}{end}'
        parts = [f'{prefix}{name}{suffix}' for prefix in prefixes]
        others = ('lse',) if name == 'lse' else ('gse', 'hse')
        others = [f'{prefix}{other}{suffix}' for prefix in prefixes for other in others]
        if (parts_eddies or parts_static) and (
            total in dataset and all(other in dataset for other in others)
        ):
            data = xr.zeros_like(dataset[ref])
            if name != 'gse':  # total transient is zero (aliased into sensible)
                with xr.set_options(keep_attrs=True):
                    data += dataset[total] - sum(dataset[other] for other in others)
            data.attrs['long_name'] = data.long_name.replace('stationary', 'transient')
            dataset[f't{name}{suffix}{end}'] = data
            if name != 'lse':  # total sensible and total geopotential
                with xr.set_options(keep_attrs=True):
                    data = data + sum(dataset[part] for part in parts)
                data.attrs['long_name'] = data.long_name.replace('transient ', '')
                dataset[f'{name}{suffix}{end}'] = data

    # Get average flux terms from the integrated terms
    # NOTE: Flux values will have units K/s, m2/s2, and g/kg m/s and are more relevant
    # to local conditions on a given latitude band. Could get vertically resolved values
    # for stationary components, but impossible for residual transient component, so
    # decided to only store this 'vertical average' for consistency.
    prefixes = ('', 'm', 's', 't')  # transport components
    names = ('hse', 'gse', 'lse')  # flux components
    iter_ = itertools.product(prefixes, names, ends)
    if parts_fluxes:
        for prefix, name, end in iter_:
            _, scale, unit = TRANSPORT_INDIVIDUAL[name]
            variable = f'{prefix}{name}t{end}'
            if variable not in dataset:
                denom = 2 * np.pi * np.cos(dataset.climo.coords.lat) * const.a
                denom = denom * dataset.climo.vars.ps.climo.average('lon') / const.g
                data = dataset[variable].climo.quantify()
                data = (data / denom / scale).climo.to_units(unit)
                data.attrs['long_name'] = dataset[variable].long_name.replace('transport', 'flux')  # noqa: E501
                dataset[f'{prefix}{name}f{end}'] = data.climo.dequantify()

    # Get dry static energy components from sensible and geopotential components
    # NOTE: Here a residual between total and storage + ocean would also suffice
    # but this also gets total transient and stationary static energy terms. Also
    # have support for adding geopotential plus
    prefixes = ('', 'm', 's', 't')  # total, zonal-mean, stationary, transient
    suffixes = ('t', 'c', 'r')  # transport, convergence, residual
    iter_ = itertools.product(prefixes, suffixes, ends)
    if not parts_static:  # only wanted combined dse eddies not components
        for prefix, suffix, end in iter_:
            variable = f'{prefix}{name}{suffix}{end}'
            parts = [variable.replace(name, part) for part in ('hse', 'gse')]
            if variable not in dataset and all(part in dataset for part in parts):
                with xr.set_options(keep_attrs=True):
                    data = sum(dataset[part] for part in parts)
                data.attrs['long_name'] = data.long_name.replace('sensible', 'dry')
                dataset[variable] = data
            drop = [part for part in parts if part in dataset]
            dataset = dataset.drop_vars(drop)
    return dataset


def _infer_transport(data, descrip=None, prefix=None, residual=True):
    """
    Convert implicit flux residuals to convergence and transport terms.

    Parameters
    ----------
    data : xarray.DataArray
        The data array.
    descrip : str, optional
        The long name description.
    prefix : str, optional
        The optional description prefix.
    residual : bool, optional
        Whether to subtract the global-average residual.

    Returns
    -------
    cdata : xarray.DataArray
        The convergence.
    rdata : xarray.DataArray
        The global-average residual.
    tdata : xarray.DataArray
        The meridional transport.
    """
    # Get convergence and residual
    # NOTE: This requires cell measures are already present for consistency with the
    # explicit transport function, which requires a surface pressure dependence.
    data = data.climo.quantify()
    descrip = f'{descrip} ' if descrip else ''
    prefix = f'{prefix} ' if prefix else ''
    cdata = -1 * data.climo.to_units('W m^-2')  # convergence equals negative residual
    cdata.attrs['long_name'] = f'{prefix}{descrip}energy convergence'
    rdata = cdata.climo.average('area').drop_vars(('lon', 'lat'))
    rdata.attrs['long_name'] = f'{prefix}{descrip}energy residual'
    # Get meridional transport
    # WARNING: Cumulative integration in forward or reverse direction will produce
    # estimates respectively offset-by-one, so compensate by taking average of both.
    tdata = cdata - rdata if residual else cdata
    tdata = tdata.climo.integral('lon')
    tdata = 0.5 * (
        -1 * tdata.climo.cumintegral('lat', reverse=False)
        + tdata.climo.cumintegral('lat', reverse=True)
    )
    tdata = tdata.climo.to_units('PW')
    tdata = tdata.drop_vars(tdata.coords.keys() - tdata.sizes.keys())
    tdata.attrs['long_name'] = f'{prefix}{descrip}energy transport'
    tdata.attrs['standard_units'] = 'PW'  # prevent cfvariable auto-inference
    return cdata.climo.dequantify(), rdata.climo.dequantify(), tdata.climo.dequantify()


def _process_transport(udata, vdata, qdata, descrip=None, prefix=None):
    """
    Convert explicit advection to convergence and transport terms.

    Parameters
    ----------
    udata : xarray.DataArray
        The zonal wind.
    vdata : xarray.DataArray
        The meridional wind.
    qdata : xarray.DataArray
        The advected quantity.
    descrip : str, optional
        The long name description.
    prefix : str, optional
        The optional description prefix.

    Returns
    -------
    cdata : xarray.DataArray
        The stationary convergence.
    sdata : xarray.DataArray
        The stationary eddy transport.
    mdata : xarray.DataArray
        The zonal-mean transport.
    """
    # Get convergence
    # NOTE: This requires cell measures are already present rather than auto-adding
    # them, since we want to include surface pressure dependence supplied by dataset.
    descrip = descrip and f'{descrip} ' or ''
    qdata = qdata.climo.quantify()
    udata, vdata = udata.climo.quantify(), vdata.climo.quantify()
    lon, lat = qdata.climo.coords['lon'], qdata.climo.coords['lat']
    x, y = (const.a * lon).climo.to_units('m'), (const.a * lat).climo.to_units('m')
    udiff = diff.deriv_even(x, udata * qdata, cyclic=True) / np.cos(lat)
    vdiff = diff.deriv_even(y, np.cos(lat) * vdata * qdata, keepedges=True) / np.cos(lat)  # noqa: E501
    cdata = -1 * (udiff + vdiff)  # convergence i.e. negative divergence
    cdata = cdata.assign_coords(qdata.coords)  # re-apply missing cell measures
    if 'plev' in cdata.dims:
        cdata = cdata.climo.integral('plev')
    string = f'{prefix} ' if prefix else 'stationary '
    cdata = cdata.climo.to_units('W m^-2')
    cdata.attrs['long_name'] = f'{string}{descrip}energy convergence'
    cdata.attrs['standard_units'] = 'W m^-2'  # prevent cfvariable auto-inference
    # Get transport components
    # NOTE: This depends on the implicit weight cell_height getting converted to its
    # longitude-average value during the longitude-integral (see _integral_or_average).
    vmean, qmean = vdata.climo.average('lon'), qdata.climo.average('lon')
    sdata = (vdata - vmean) * (qdata - qmean)  # width and height measures removed
    sdata = sdata.assign_coords(qdata.coords)  # reassign measures lost to conflicts
    sdata = sdata.climo.integral('lon')
    if 'plev' in sdata.dims:
        sdata = sdata.climo.integral('plev')
    string = f'{prefix} ' if prefix else 'stationary '
    sdata = sdata.climo.to_units('PW')
    sdata.attrs['long_name'] = f'{string}{descrip}energy transport'
    sdata.attrs['standard_units'] = 'PW'  # prevent cfvariable auto-inference
    mdata = vmean * qmean
    mdata = 2 * np.pi * np.cos(mdata.climo.coords.lat) * const.a * mdata
    # mdata = mdata.climo.integral('lon')  # integrate scalar coordinate
    if 'plev' in mdata.dims:
        mdata = mdata.climo.integral('plev')
    string = f'{prefix} ' if prefix else 'zonal-mean '
    mdata = mdata.climo.to_units('PW')
    mdata.attrs['long_name'] = f'{string}{descrip}energy transport'
    mdata.attrs['standard_units'] = 'PW'  # prevent cfvariable auto-inference
    return cdata.climo.dequantify(), sdata.climo.dequantify(), mdata.climo.dequantify()


def get_climate(output=None, project=None, **inputs):
    """
    Calculate climate variables for a given model time series and climate data.

    Parameters
    ----------
    output : path-like
        The output file.
    project : str, optional
        The project.
    **inputs : tuple of path-like lists
        Tuples of ``(climate_inputs, series_inputs)`` for the variables
        required to be added to the combined file, passed as keyword arguments for
        each variable. The variables computed will depend on the variables passed.
    """
    # Load the data
    # NOTE: Critical to overwrite the time coordinates after loading or else xarray
    # coordinate matching will apply all-NaN values for climatolgoies with different
    # base years (e.g. due to control data availability or response calendar diffs).
    climate, series = {}, {}
    for variable, paths in inputs.items():
        for path, output in zip(paths, (climate, series)):
            if path is None:  # missing climate or series
                continue
            array = load_file(path, variable, project=project)
            descrip = array.attrs.pop('title', variable)  # in case long_name missing
            descrip = array.attrs.pop('long_name', descrip)
            descrip = ' '.join(s if s == 'TOA' else s.lower() for s in descrip.split())
            array.attrs['long_name'] = descrip
            output[variable] = array

    # Standardize the data
    # NOTE: Empirical testing revealed limiting integration to troposphere
    # often prevented strong transient heat transport showing up in overturning
    # cells due to aliasing of overemphasized stratospheric geopotential transport.
    # WARNING: Critical to place time averaging after transport calculations so that
    # time-covariance of surface pressure and near-surface flux terms is factored
    # in (otherwise would need to include cell heights before averaging).
    from coupled.climate import _add_energetics, _add_hydrology, _add_transport
    dataset = xr.Dataset(output)
    if 'ps' not in dataset:
        print('Warning: Surface pressure is unavailable.', end=' ')
    dataset = dataset.climo.add_cell_measures(surface=('ps' in dataset))
    dataset = _add_energetics(dataset)  # must come before transport
    dataset = _add_transport(dataset)
    dataset = _add_hydrology(dataset)
    if 'plev' in dataset:
        slice_ = slice(None, 7000)
        dataset = dataset.sel(plev=slice_)
    if 'time' in dataset:  # possibly no-op average?
        dataset = dataset.groupby('time.month').mean('time', skipna=True)
        dataset = assign_dates(dataset, year=1800)  # see also _regress_annual
    drop = ['cell_', '_bot', '_top']
    drop = [key for key in dataset.coords if any(o in key for o in drop)]
    dataset = dataset.drop_vars(drop)
    dataset = dataset.squeeze()
    return dataset


def process_climate(
    *paths,
    series=None,
    climate=None,
    experiment=None,
    project=None,
    nodrift=False,
    logging=False,
    dryrun=False,
    nowarn=False,
    **kwargs
):
    """
    Generate combined climate files using the output of `process_files`.

    Parameters
    ----------
    *paths : path-like, optional
        Location(s) for the climate and series data.
    series : 2-tuple of int, default: (0, 150)
        The year range for the series data.
    climate : 2-tuple of int, default: (0, 150)
        The year range for the climate data.
    experiment : str, optional
        The experiment to use.
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
    experiment = experiment or 'piControl'
    series = series or (0, 150)
    climate = climate or (0, 150)
    nodrift = 'nodrift' if nodrift else ''
    suffix = nodrift and '-' + nodrift
    series_suffix = f'{series[0]:04d}-{series[1]:04d}-series{suffix}'
    climate_suffix = f'{climate[0]:04d}-{climate[1]:04d}-climate{suffix}'
    suffixes = (*(format(int(t), '04d') for t in climate), 'standard', nodrift)
    if logging:
        print = Printer('climate', *suffixes, project=project, experiment=experiment, table='Amon')  # noqa: E501
    else:
        print = builtins.print
    print('Generating database.')
    constraints = {'project': project, 'experiment': experiment}
    constraints.update({
        key: kwargs.pop(key) for key in tuple(kwargs)
        if any(s in key for s in ('model', 'flagship', 'ensemble'))
    })
    files, *_ = glob_files(*paths, project=project)
    facets = ('project', 'model', 'experiment', 'ensemble', 'grid')
    database = Database(files, facets, **constraints)

    # Calculate clear and all-sky feedbacks surface and TOA files
    # NOTE: Unlike the method that uses a fixed SST experiment to deduce forcing and
    # takes the ratio of local radiative flux change to global temperature change as
    # the feedback, the average of a regression of radiative flux change against global
    # temperature change will not necessarily equal the regression of average radiative
    # flux change. Therefore compute both and compare to get idea of the residual. Also
    # consider local temperature change for point of comparison (Hedemman et al. 2022).
    print(f'Input files ({len(database)}):')
    print(*(f'{key}: ' + ' '.join(opts) for key, opts in database.constraints.items()), sep='\n')  # noqa: E501
    print(f'Climate data: experiment {experiment} suffix {climate_suffix}')
    print(f'Series data: experiment {experiment} suffix {series_suffix}')
    _print_error = lambda error: print(
        ' '.join(traceback.format_exception(None, error, error.__traceback__))
    )
    for group, data in database.items():
        group = dict(zip(database.group, group))
        print()
        print('Computing climate variables:')
        print(', '.join(f'{key}: {value}' for key, value in group.items()))
        files = _file_pairs(
            data,
            climate_suffix,
            series_suffix,
            printer=print,
        )
        try:
            datasets = get_climate(
                project=group['project'],
                experiment=_item_facets['experiment'](tuple(files.values())[0][1]),
                ensemble=_item_facets['ensemble'](tuple(files.values())[0][1]),
                table=_item_facets['table'](tuple(files.values())[0][1]),
                model=group['model'],
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
