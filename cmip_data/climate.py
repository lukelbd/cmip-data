#!/usr/bin/env python3
"""
Create files containing combined time-mean and time-variance quantities.
"""
import builtins
import itertools
import traceback
import re

import climopy as climo  # noqa: F401  # add accessor
import numpy as np
import xarray as xr
from climopy import diff, const, ureg
from icecream import ic  # noqa: F401
from metpy import calc, units

from .utils import average_periods, open_file
from .internals import Database, Logger, glob_files, _item_dates, _item_parts

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
TRANSPORT_DESCRIPS = {
    'gse': 'potential static',
    'hse': 'sensible static',
    'dse': 'dry static',
    'lse': 'latent static',
    'mse': 'moist static',
    'ocean': 'storage + ocean',
    'total': 'total',
}
TRANSPORT_INTEGRATED = {
    'dse': (1, 'intuadse', 'intvadse'),
    'lse': (const.Lv, 'intuaw', 'intvaw'),
}
TRANSPORT_INDIVIDUAL = {
    'gse': ('zg', const.g, 'dam m s^-1'),
    'hse': ('ta', const.cp, 'K m s^-1'),
    'lse': ('hus', const.Lv, 'g kg^-1 m s^-1'),
}
TRANSPORT_IMPLICIT = {
    'dse': (('hfss', 'rlns', 'rsns', 'rlnt', 'rsnt', 'pr', 'prsn'), ()),
    'lse': (('hfls',), ('pr', 'prsn')),
    'ocean': ((), ('hfss', 'hfls', 'rlns', 'rsns')),  # flux into atmosphere
    'total': (('rlnt', 'rsnt'), ()),
}
TRANSPORT_RADIATION = {
    'rlnt': ('rlut',),  # out of the atmosphere
    'rsnt': ('rsut', 'rsdt'),  # out of the atmosphere (include constant rsdt)
    'rlntcs': ('rlutcs',),  # out of the atmosphere
    'rsntcs': ('rsutcs', 'rsdt'),  # out of the atmosphere (include constant rsdt)
    'rlns': ('rlds', 'rlus'),  # out of and into the atmosphere
    'rsns': ('rsds', 'rsus'),  # out of and into the atmosphere
    'rlnscs': ('rldscs', 'rlus'),  # out of and into the atmosphere
    'rsnscs': ('rsdscs', 'rsuscs'),  # out of and into the atmosphere
    'albedo': ('rsds', 'rsus'),  # full name to differentiate from 'alb' feedback
}

# Water cycle constants
# NOTE: Here the %s is filled in with water, liquid, or ice depending on the
# component of the particular variable category.
WATER_DEPENDENCIES = {
    'hur': ('plev', 'ta', 'hus'),
    'hurs': ('ps', 'ts', 'huss'),
}
WATER_COMPONENTS = [
    ('clw', 'cll', 'cli', 'clp', 'mass fraction cloud %s'),
    ('clwvi', 'cllvi', 'clivi', 'clpvi', 'condensed %s water path'),
    ('pr', 'prra', 'prsn', 'prp', '%s precipitation'),
    ('evspsbl', 'evsp', 'sbl', 'sblp', '%s evaporation'),
]

# Transformation constants
# NOTE: For now use the standard 1e3 kg/m3 water density (i.e. snow and ice terms
# represent melted equivalent depth) but could also use 1e2 kg/m3 snow density where
# relevant. See: https://www.sciencelearn.org.nz/resources/1391-snow-and-ice-density
SCALES_IMPLICIT = {  # scaling prior to implicit transport calculations
    'pr': const.Lv,
    'prra': const.Lv,
    'prsn': const.Ls - const.Lv,  # remove the 'prsn * Lv' implied inside 'pr' term
    'evspsbl': const.Lv,
    'evsp': const.Lv,
    'sbl': const.Ls - const.Lv,  # remove the 'evspsbl * Lv' implied inside 'pr' term
}


def _file_pairs(data, *args, printer=None):
    """
    Return a dictionary mapping of variables to ``(climate, series)`` file
    pairs from a database of files with ``(..., variable)`` keys.

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


def _transport_implicit(data, descrip=None, prefix=None, adjust=True):
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
    adjust : bool, optional
        Whether to adjust with global-average residual.

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
    tdata = cdata - rdata if adjust else cdata
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


def _transport_explicit(udata, vdata, qdata, descrip=None, prefix=None):
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
    if 'plev' in cdata.sizes:
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
    if 'plev' in sdata.sizes:
        sdata = sdata.climo.integral('plev')
    string = f'{prefix} ' if prefix else 'stationary '
    sdata = sdata.climo.to_units('PW')
    sdata.attrs['long_name'] = f'{string}{descrip}energy transport'
    sdata.attrs['standard_units'] = 'PW'  # prevent cfvariable auto-inference
    mdata = vmean * qmean
    mdata = 2 * np.pi * np.cos(mdata.climo.coords.lat) * const.a * mdata
    # mdata = mdata.climo.integral('lon')  # integrate scalar coordinate
    if 'plev' in mdata.sizes:
        mdata = mdata.climo.integral('plev')
    string = f'{prefix} ' if prefix else 'zonal-mean '
    mdata = mdata.climo.to_units('PW')
    mdata.attrs['long_name'] = f'{string}{descrip}energy transport'
    mdata.attrs['standard_units'] = 'PW'  # prevent cfvariable auto-inference
    return cdata.climo.dequantify(), sdata.climo.dequantify(), mdata.climo.dequantify()


def _update_climate_radiation(dataset):
    """
    Add albedo and net fluxes from upwelling and downwelling components and remove
    the original directional variables.

    Parameters
    ----------
    dataset : xarray.Dataset
        The dataset.
    """
    # NOTE: Here also add 'albedo' term and generate long name by replacing directional
    # terms in existing long name. Remove the directional components when finished.
    regex = re.compile(r'(upwelling|downwelling|outgoing|incident)')
    for name, keys in TRANSPORT_RADIATION.items():
        if any(key not in dataset for key in keys):  # skip partial data
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
        long_name = regex.sub('net', long_name)
        dataset[name].attrs.update({'units': unit, 'long_name': long_name})
    drop = set(key for keys in TRANSPORT_RADIATION.values() for key in keys)
    dataset = dataset.drop_vars(drop & dataset.data_vars.keys())
    return dataset


def _update_climate_transport(dataset):
    """
    Add zonally-integrated meridional transport and pointwise transport convergence
    to the dataset with dry-moist and transient-stationary-mean breakdowns.

    Parameters
    ----------
    dataset : xarray.Dataset
        The dataset.
    explicit : bool, optional
        Whether to load explicit data.
    """
    # Get total, ocean, dry, and latent transport
    # WARNING: This must come after _update_climate_radiation
    # NOTE: The below regex prefixes exponents expressed by numbers adjacent to units
    # with the carat ^, but ignores the scientific notation 1.e6 in scaling factors,
    # so dry static energy convergence units can be parsed as quantities.
    ends = ('', '_alt', '_exp')  # implicit, alternative, explicit
    for (name, pair), end in itertools.product(TRANSPORT_IMPLICIT.items(), ends[:2]):
        # Implicit transport
        constants = {}  # store component data
        for c, keys in zip((1, -1), pair):
            keys = list(keys)
            if end and 'hfls' in keys:
                keys[(idx := keys.index('hfls')):idx + 1] = ('evspsbl', 'sbl')
            constants.update({k: c * SCALES_IMPLICIT.get(k, 1) for k in keys})
        if end and 'sbl' not in constants:  # skip redundant calculation
            continue
        descrip = TRANSPORT_DESCRIPS[name]
        kwargs = {'descrip': descrip, 'prefix': 'alternative' if end else ''}
        if all(k in dataset for k in constants):
            data = sum(c * dataset.climo.vars[k] for k, c in constants.items())
            cdata, rdata, tdata = _transport_implicit(data, **kwargs)
            cname, rname, tname = f'{name}c{end}', f'{name}r{end}', f'{name}t{end}'
            dataset.update({cname: cdata, rname: rdata, tname: tdata})
        # Explicit transport
        if end and True:  # only run explicit calcs after first run
            continue
        scale, ukey, vkey = TRANSPORT_INTEGRATED.get(name, (None, None, None))
        kwargs = {'descrip': descrip, 'prefix': 'explicit'}
        regex = re.compile(r'([a-df-zA-DF-Z]+)([-+]?[0-9]+)')
        if ukey and vkey and ukey in dataset and vkey in dataset:
            udata = dataset[ukey].copy(deep=False)
            vdata = dataset[vkey].copy(deep=False)
            udata *= scale * ureg(regex.sub(r'\1^\2', udata.attrs.pop('units')))
            vdata *= scale * ureg(regex.sub(r'\1^\2', vdata.attrs.pop('units')))
            qdata = ureg.dimensionless * xr.ones_like(udata)  # placeholder
            cdata, _, tdata = _transport_explicit(udata, vdata, qdata, **kwargs)
            cname, tname = f'{name}c_exp', f'{name}t_exp'
            dataset.update({cname: cdata, tname: tdata})
        dataset = dataset.drop_vars({ukey, vkey} & dataset.data_vars.keys())

    # Get mean transport, stationary transport, and stationary convergence
    # NOTE: Here convergence can be broken down into just two components: a
    # stationary term and a transient term.
    for name, (quant, scale, _) in TRANSPORT_INDIVIDUAL.items():
        if 'ps' not in dataset or 'va' not in dataset or quant not in dataset:
            continue
        descrip = TRANSPORT_DESCRIPS[name]
        qdata = scale * dataset.climo.vars[quant]
        udata, vdata = dataset.climo.vars['ua'], dataset.climo.vars['va']
        cdata, sdata, mdata = _transport_explicit(udata, vdata, qdata, descrip=descrip)
        cname, sname, mname = f's{name}c', f's{name}t', f'm{name}t'
        dataset.update({cname: cdata, sname: sdata, mname: mdata})

    # Get missing transient components and total sensible and geopotential terms
    # NOTE: Here transient sensible transport is calculated from the residual of the dry
    # static energy minus both the sensible and geopotential stationary components. The
    # all-zero transient geopotential is stored for consistency if sensible is present.
    iter_ = itertools.product(TRANSPORT_INDIVIDUAL, ('cs', 'tsm'), ends)
    for name, (suffix, *prefixes), end in iter_:
        ref = f'{prefixes[0]}{name}{suffix}'  # reference component
        total = 'lse' if name == 'lse' else 'dse'
        total = f'{total}{suffix}{end}'
        parts = [f'{prefix}{name}{suffix}' for prefix in prefixes]
        resids = ('lse',) if name == 'lse' else ('gse', 'hse')
        resids = [f'{prefix}{resid}{suffix}' for prefix in prefixes for resid in resids]
        if total not in dataset or any(resid not in dataset for resid in resids):
            continue
        data = xr.zeros_like(dataset[ref])
        if name != 'gse':
            with xr.set_options(keep_attrs=True):
                data += dataset[total] - sum(dataset[resid] for resid in resids)
        data.attrs['long_name'] = data.long_name.replace('stationary', 'transient')
        dataset[f't{name}{suffix}{end}'] = data
        if name != 'lse':  # total is already present
            with xr.set_options(keep_attrs=True):  # add non-transient components
                data = data + sum(dataset[part] for part in parts)
            data.attrs['long_name'] = data.long_name.replace('transient ', '')
            dataset[f'{name}{suffix}{end}'] = data

    # Get dry and moist static energy from component transport and convergence terms
    # NOTE: Here a residual between total and storage + ocean would also suffice
    # but this also gets total transient and stationary static energy terms.
    replacements = {'dse': ('sensible', 'dry'), 'mse': ('dry', 'moist')}
    dependencies = {'dse': ('hse', 'gse'), 'mse': ('dse', 'lse')}
    prefixes = ('', 'm', 's', 't')  # total, zonal-mean, stationary, transient
    suffixes = ('t', 'c', 'r')  # convergence, residual, transport
    iter_ = itertools.product(('dse', 'mse'), prefixes, suffixes, ends)
    for name, prefix, suffix, end in iter_:
        variable = f'{prefix}{name}{suffix}{end}'
        parts = [variable.replace(name, part) for part in dependencies[name]]
        if variable in dataset or any(part not in dataset for part in parts):
            continue
        with xr.set_options(keep_attrs=True):
            data = sum(dataset[part] for part in parts)
        data.attrs['long_name'] = data.long_name.replace(*replacements[name])
        dataset[variable] = data

    # Get average flux terms from the integrated terms
    # NOTE: These values will have units K/s, m2/s2, and g/kg m/s and are more
    # relevant to local conditions on a given latitude band. Could get vertically
    # resolved values for stationary components, but impossible for residual transient
    # component, so decided to only store this 'vertical average' for consistency.
    iter_ = itertools.product(('hse', 'gse', 'lse'), prefixes, ends)
    for name, prefix, end in iter_:
        variable = f'{prefix}{name}t{end}'
        if variable not in dataset:
            continue
        _, scale, unit = TRANSPORT_INDIVIDUAL[name]
        denom = 2 * np.pi * np.cos(dataset.climo.coords.lat) * const.a
        denom = denom * dataset.climo.vars.ps.climo.average('lon') / const.g
        data = dataset[variable].climo.quantify()
        data = (data / denom / scale).climo.to_units(unit)
        data.attrs['long_name'] = dataset[variable].long_name.replace('transport', 'flux')  # noqa: E501
        data = data.climo.dequantify()
        dataset[f'{prefix}{name}f{end}'] = data
    return dataset


def _update_climate_water(dataset):
    """
    Add relative humidity and ice and liquid water terms and standardize
    the insertion order for the resulting dataset variables.

    Parameters
    ----------
    dataset : xarray.Dataset
        The dataset.
    """
    # Add humidity terms
    # NOTE: Generally forego downloading relative humidity variables... true that
    # operation is non-linear so relative humidity of climate is not climate of
    # relative humiditiy, but we already use this approach with feedback kernels.
    for name, keys in WATER_DEPENDENCIES.items():
        if any(key not in dataset for key in keys):
            continue
        datas = []
        for key, unit in zip(keys, ('Pa', 'K', '')):
            src = dataset.climo.coords if key == 'plev' else dataset.climo.vars
            data = src[key].climo.to_units(unit)  # climopy units
            data.data = data.data.magnitude * units.units(unit)  # metpy units
            datas.append(data)
        descrip = 'near-surface ' if name[-1] == 's' else ''
        data = calc.relative_humidity_from_specific_humidity(*datas)
        data = data.climo.dequantify()  # works with metpy registry
        data = data.climo.to_units('%').clip(0, 100)
        data.attrs['long_name'] = f'{descrip}relative humidity'
        dataset[name] = data

    # Add cloud terms
    # NOTE: Unlike related variables (including, confusingly, clwvi), 'clw' includes
    # only liquid component rather than combined liquid plus ice. Adjust it to match
    # convention from other variables and add other component terms.
    if 'clw' in dataset:
        if 'cli' not in dataset:
            dataset = dataset.rename_vars(clw='cll')
        else:
            with xr.set_options(keep_attrs=True):
                dataset['clw'] = dataset['cli'] + dataset['clw']
    for both, liquid, ice, ratio, descrip in WATER_COMPONENTS:
        if both in dataset and ice in dataset:  # note the clw variables include ice
            da = dataset[both] - dataset[ice]
            da.attrs = {'units': dataset[both].units, 'long_name': descrip % 'liquid'}
            dataset[liquid] = da
            da = (100 * dataset[ice] / dataset[both]).clip(0, 100)
            da.attrs = {'units': '%', 'long_name': descrip % 'ice' + ' ratio'}
            dataset[ratio] = da
    for both, liquid, ice, _, descrip in WATER_COMPONENTS:
        for name, string in zip((ice, both, liquid), ('ice', 'water', 'liquid')):
            if name in dataset:
                data = dataset[name]
            else:
                continue
            if string not in data.long_name:
                data.attrs['long_name'] = descrip % string
    return dataset


def compute_climate(climate, series=None, output=None, **inputs):
    """
    Compute climate variables for a given model time series and climate data.

    Parameters
    ----------
    output : path-like
        The output file.
    **inputs : tuple of path-like lists
        Tuples of ``(climate_inputs, series_inputs)`` for the variables
        required to be added to the combined file, passed as keyword arguments for
        each variable. The variables computed will depend on the variables passed.
    """
    # Initial stuff
    # NOTE: Critical to overwrite the time coordinates after loading or else xarray
    # coordinate matching will apply all-NaN values for climatolgoies with different
    # base years (e.g. due to control data availability or response calendar diffs).

    # Load the data
    # NOTE: Here open_file automatically populates the mapping MODELS_INSTITUTIONS
    att = {'axis': 'T', 'standard_name': 'time'}
    time = pd.date_range('2000-01-01', '2000-12-01', freq='MS')
    time = xr.DataArray(time, name='time', dims='time', attrs=att)
    output = {}
    for variable, files in inputs.items():
        variable = key[database.key.index('variable')]
        paths = [path for path in paths if _item_dates(path) == dates]
        if not paths:
            continue
        if len(paths) > 1:
            print(f'Warning: Skipping ambiguous duplicate paths {list(map(str, paths))}.', end=' ')  # noqa: E501
            continue
        array = open_file(paths[0], variable, project=database.project)
        if array.time.size != 12:
            print(f'Warning: Skipping path {paths[0]} with time length {array.time.size}.', end=' ')  # noqa: E501
            continue
        months = array.time.dt.month
        if sorted(months.values) != sorted(range(1, 13)):
            print(f'Warning: Skipping path {paths[0]} with month values {months.values}.', end=' ')  # noqa: E501
            continue
        array = array.assign_coords(time=time)
        descrip = array.attrs.pop('title', variable)  # in case long_name missing
        descrip = array.attrs.pop('long_name', descrip)
        descrip = ' '.join(s if s == 'TOA' else s.lower() for s in descrip.split())
        array.attrs['long_name'] = descrip
        dataset[variable] = array

    # Standardize the data
    # NOTE: Empirical testing revealed limiting integration to troposphere
    # often prevented strong transient heat transport showing up in overturning
    # cells due to aliasing of overemphasized stratospheric geopotential transport.
    # WARNING: Critical to place average_periods after adjustments so that
    # time-covariance of surface pressure and near-surface flux terms is
    # effectively factored in (since average_periods only includes explicit
    # month-length weights and ignores implicit cell height weights).
    if 'ps' not in dataset:
        print('Warning: Surface pressure is unavailable.', end=' ')
    dataset = dataset.climo.add_cell_measures(surface=('ps' in dataset))
    dataset = _update_climate_radiation(dataset)  # must come before transport
    dataset = _update_climate_transport(dataset)
    dataset = _update_climate_water(dataset)
    if 'time' in dataset:
        dataset = average_periods(dataset, **kw_times)
    if 'plev' in dataset:
        dataset = dataset.sel(plev=slice(None, 7000))
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
    experiment = experiment or 'piControl'
    series = series or (0, 150)
    climate = climate or (0, 150)
    nodrift = 'nodrift' if nodrift else ''
    suffix = nodrift and '-' + nodrift
    parts = ('climate', experiment, 'derived', nodrift)
    print = Logger('summary', *parts, project=project) if logging else builtins.print
    series_suffix = f'{series[0]:04d}-{series[1]:04d}-series{suffix}'
    climate_suffix = f'{climate[0]:04d}-{climate[1]:04d}-climate{suffix}'
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
            datasets = compute_climate(
                project=group['project'],
                experiment=_item_parts['experiment'](tuple(files.values())[0][1]),
                ensemble=_item_parts['ensemble'](tuple(files.values())[0][1]),
                table=_item_parts['table'](tuple(files.values())[0][1]),
                model=group['model'],
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