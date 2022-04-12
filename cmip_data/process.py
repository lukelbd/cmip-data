#!/usr/bin/env python3
"""
Process datasets downloaded with the ESGF python API.
"""
import os
import shutil
import warnings
from pathlib import Path

import numpy as np
from cdo import Cdo

from .download import _line_parts, _parse_constraints, _generate_printer

__all__ = [
    'process_files',
    'standardize_time',
    'standardize_horizontal',
    'standardize_vertical',
]


# Horizontal grid constants
# NOTE: Lowest resolution in CMIP5 models is 64 latitudes 128 longitudes (also still
# used by BCC-ESM1 in CMIP6) and grids are mix of regular lon-lat and guassian (assume
# that former was interpolated). Select regular grid spacing with similar resolution but
# not coarser than necessary. See Horizontal grids->Grid description->Predefined grids
# for details. Also use the following bash code to summarize resolutions:
# unset models; for f in ts_Amon*; do
#   model=$(echo $f | cut -d_ -f3)
#   [[ " ${models[*]} " =~ " $model " ]] && continue || models+=("$model")
#   echo "$model:" && ncdims $f | grep -E 'lat|lon' | tr -s ' ' | xargs
# done
GRID_NAME = 'r180x90'  # two degree resolution

# Vertical grid constants
# NOTE: To determine whether pressure level interpolation is needed we directly
# compare levels. Some grids use floats and have slight offsets while some use
# exact integers so important to use np.isclose() rather than exact comparison.
# NOTE: Some models have longname 'atmospheric model level' but zaxistype 'hybrid'
# so should still work with cdo ml2pl operator. Found only these files:
# cmip6-picontrol-amon/cl_Amon_GFDL-CM4_piControl_r1i1p1f1_gr1_015101-025012.nc
# cmip6-picontrol-amon/cl_Amon_GFDL-ESM4_piControl_r1i1p1f1_gr1_000101-010012.nc
# cmip6-picontrol-amon/cl_Amon_CNRM-CM6-1_piControl_r1i1p1f2_gr_185001-194912.nc
# cmip6-picontrol-amon/cl_Amon_CNRM-ESM2-1_piControl_r1i1p1f2_gr_185001-194912.nc
# Similarly models with longname 'sigma coordinate' or 'hybrid height coordinate'
# have zaxistype 'generic'. Found only a couple sigma coordinate models:
# cmip5-picontrol-amon/cl_Amon_FGOALS-g2_piControl_r1i1p1_020101-021012.nc
# cmip6-picontrol-amon/cl_Amon_FGOALS-g3_piControl_r1i1p1f1_gn_020001-020912.nc
# NOTE: Some models are natively run on hybrid height levels. Found these files:
# files in ~/scratch5/cmip6-picontrol-amon/ and ~/scratch2/cmip5-picontrol/amon/:
# cmip5-picontrol-amon/cl_Amon_ACCESS1-0_piControl_r1i1p1_030001-032412.nc
# cmip5-picontrol-amon/cl_Amon_ACCESS1-3_piControl_r1i1p1_025001-027412.nc
# cmip5-picontrol-amon/cl_Amon_HadGEM2-ES_piControl_r1i1p1_185912-188411.nc
# cmip6-picontrol-amon/cl_Amon_ACCESS-CM2_piControl_r1i1p1f1_gn_095001-096912.nc
# cmip6-picontrol-amon/cl_Amon_ACCESS-ESM1-5_piControl_r1i1p1f1_gn_010101-012012.nc
# cmip6-picontrol-amon/cl_Amon_HadGEM3-GC31-MM_piControl_r1i1p1f1_gn_185001-185912.nc
# cmip6-picontrol-amon/cl_Amon_HadGEM3-GC31-LL_piControl_r1i1p1f1_gn_185001-189912.nc
# cmip6-picontrol-amon/cl_Amon_KACE-1-0-G_piControl_r1i1p1f1_gr_200001-209912.nc
# cmip6-picontrol-amon/cl_Amon_UKESM1-0-LL_piControl_r1i1p1f2_gn_196001-199912.nc
# NOTE: Some CESM2 model data is erroneously interpreted as using pressure levels due
# to missing CF standard name and formula terms. Only CESM2-WACCM-FV2 has the required
# 'lev' attributes so can add kludge to copy those onto first three files:
# cmip6-picontrol-amon/cl_Amon_CESM2_piControl_r1i1p1f1_gn_000101-009912.nc
# cmip6-picontrol-amon/cl_Amon_CESM2-FV2_piControl_r1i1p1f1_gn_000101-005012.nc
# cmip6-picontrol-amon/cl_Amon_CESM2-WACCM_piControl_r1i1p1f1_gn_000101-009912.nc
# cmip6-picontrol-amon/cl_Amon_CESM2-WACCM-FV2_piControl_r1i1p1f1_gn_000101-004912.nc
# However for IPSL data level detection is completely messed up so forget it.
# cmip6-picontrol-amon/clw_Amon_IPSL-CM5A2-INCA_piControl_r1i1p1f1_gr_185001-209912.nc
# cmip6-picontrol-amon/cl_Amon_IPSL-CM6A-LR_piControl_r1i1p1f1_gr_185001-234912.nc
# Only the following models actually provide data on standard 19 pressure levels:
# cmip6-picontrol-amon/cl_Amon_IITM-ESM_piControl_r1i1p1f1_gn_192601-193512.nc
# cmip6-picontrol-amon/cl_Amon_MCM-UA-1-0_piControl_r1i1p1f1_gn_000101-010012.nc
VERT_LEVS = 100 * np.array(
    [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 70, 50, 30, 20, 10, 5, 1]  # noqa: E501
)


def _default_cdo():
    """
    Return a default `cdo.Cdo` binding instance.
    """
    # NOTE: This causes weird issues where cdo captures keyboard interrupt signals
    # even when executing python code so try to minimize its use.
    print('Generating default cdo instance.')
    cdo = Cdo(
        env={**os.environ, 'CDO_TIMESTAT_DATE': 'first'},
        options=['-s', '-O'],  # silent and overwrite
    )
    return cdo


def _enforce_exists(path):
    """
    Verify that cdo successfully crated the output file.

    Parameters
    ----------
    path : path-like
        The netcdf path.
    """
    if not path.is_file() or path.stat().st_size == 0:
        if path.is_file():
            path.unlink()
        raise RuntimeError(f'Failed to create output file: {path}.')
    print(f'Successfully created output file {path}.')


def _parse_path(path, stem=None, suffix='copy', parent='~/data.nc'):
    """
    Generate a path from an input path.

    Parameters
    ----------
    path : path-like
        The path to format.
    stem, suffix : str optional
        The path name or suffix.
    parent : path-like, optional
        The path to use for generating a default value.
    """
    parent = Path(parent).expanduser()
    suffix = suffix or 'copy'
    path = Path(path).expanduser() if path else parent.parent
    if path.is_dir():
        stem = stem or parent.stem + '_' + suffix + '.nc'
        path = path / stem
    return path


def _parse_time(climate=True, numyears=150, endyears=False, detrend=False, **kwargs):
    """
    Parse variables related to time standardization and create a file name suffix.

    Parameters
    ----------
    **kwargs
        See `standardize_time` for details.
    """
    # NOTE: File format is e.g. 'file_climate-first100-notrend.nc'. Each modeling
    # center uses different date schemes for experiments so years must be relative.
    # Similar to 'file_standard-grid.nc' and 'file_standard-levs.nc' used below.
    name = '-'.join((
        ('series', 'climate')[climate],
        ('first', 'last')[endyears] + str(numyears),
        *(('notrend',) if detrend else ()),
    )) + '.nc'
    return name, climate, numyears, endyears, detrend, kwargs


def _parse_tables(lines):
    """
    Convert the cdo horizontal or vertical grid description tables into dictionaries.

    Parameters
    ----------
    lines : list
        The lines returned by cdo commands like 'griddes' and 'zaxisdes'.
    """
    result = []
    results = [result]
    for line in lines:
        if line and line[0] != '#':
            if '=' in line:
                result.append(line)
            elif result:
                result[-1] += ' ' + line
            else:
                warnings.warn(f'Ignoring unexpected line {line!r}.')
        elif result:  # separate block
            results.append(result := [])  # redefine list
    tables = []
    for result in results:
        table = {}
        for string in result:
            try:
                key, value = (s.strip() for s in string.split('='))
            except ValueError:
                warnings.warn(f'Ignoring unexpected line(s) {string!r}.')
            else:
                if "'" in value or '"' in value:
                    value = [eval(value)]  # quoted string
                else:
                    value = [s.strip() for s in value.split(' ')]
                for dtype in (int, float, str):
                    try:
                        value = np.array(value, dtype=dtype)
                    except ValueError:
                        pass
                    else:
                        break
                if isinstance(value, list) or key in table:
                    warnings.warn(f'Ignoring unexpected line(s) {string!r}.')
                else:
                    table[key] = value.item() if value.size == 1 else value
        if not table:
            raise ValueError(f'Failed to read table {result!r}.')
        else:
            tables.append(table)
    return tables


def compare_tables(path='~/data', variable='ta'):
    """
    Print descriptions of horizontal grids and vertical levels
    for cmip files in the specified paths.

    Parameters
    ----------
    path : str, optional
        The path.
    variable : str, optional
        The variable to use for file globbing.
    """
    from cmip_data import process
    cdo = process._default_cdo()
    path = Path(path).expanduser()
    grids, zaxes = {}, {}
    for file in sorted(path.glob(variable + '_*.nc')):
        key = '_'.join(file.name.split('_')[:5])
        file = str(file)
        if key in grids or key in zaxes:
            continue
        grid, zaxis = cdo.griddes(input=str(file)), cdo.zaxisdes(input=str(file))
        grid, zaxis = map(process._parse_tables, (grid, zaxis))
        grids[key], zaxes[key] = grid, zaxis
        print(f'\n{file}:')
        for table in (*grid, *zaxis):
            print(', '.join(f'{k}: {v}' for k, v in table.items() if not isinstance(v, np.ndarray)))  # noqa: E501


def process_files(
    path='~/data', dest=None, overwrite=False, dryrun=False, cdo=None, **kwargs,
):
    """
    Average and standardize the files downloaded with a wget script.

    Parameters
    ----------
    path : path-like
        The input path for the raw data.
    dest : path-like, optional
        The output directory for the averaged and standardized data.
    overwrite : bool, optional
        Whether to overwrite existing files or skip them.
    dryrun : bool, optional
        Whether to only print time information and exit.
    cdo : `cdo.Cdo`, optional
        The cdo instance.
    **kwargs
        Passed to `standardize_time`.
    **constraints
        Passed to `_parse_constraints`.
    """
    # Find files and restrict to unique constraints
    # NOTE: Here we only constrain search to the project, which is otherwise not
    # indicated in native netcdf filenames. Similar approach to filter_script().
    suffix, climate, numyears, endyears, detrend, constraints = _parse_time(**kwargs)
    project, constraints = _parse_constraints(reverse=True, **constraints)
    print = _generate_printer('process', **constraints)
    path = Path(path).expanduser()
    dest = Path(dest).expanduser() if dest else path
    files = list(path.glob(f'{project}*/*.nc'))
    if constraints.keys() - _line_parts.keys() != set(('project',)):
        raise ValueError(f'Input constraints {constraints.keys()} must be subset of: {_line_parts.keys()}')  # noqa: E501
    if not files:
        raise FileNotFoundError(f'Pattern {project}*/*.nc in directory {path} returned no netcdf files.')  # noqa: E501
    database = {}
    for file in files:
        opts = {facet: func(file) for facet, func in _line_parts.items()}
        if any(opt not in constraints.get(facet, (opt,)) for facet, opt in opts.items()):  # noqa: E501
            continue
        if file.stat().st_size == 0:
            print(f'Warning: Removing empty input file {file.name}.')
            file.unlink()
            continue
        key = tuple(opts.keys())
        database.setdefault(key, []).append(file)

    # Initialize cdo and process files. Use 'conda install cdo' for the command-line
    # tool and 'conda install python-cdo' or 'pip install cdo' for the python binding.
    # NOTE: Here perform horizontal interpolation before vertical interpolation because
    # latter adds NaNs and extrapolated data so should delay that step until very end.
    cdo = cdo or _default_cdo()
    outputs = []
    for key, files in database.items():
        parts = dict(zip(_line_parts, key))
        name = '_'.join((*files[0].name.split('_')[:6], suffix))
        output = dest / files[0].parent.name / name
        exists = output.is_file() and output.stat().st_size > 0
        outputs.append(output)
        print('Parts:', ', '.join(parts))
        print('Output:', '/'.join((output.parent.name, output.name)))
        if exists and not overwrite:
            print('Skipping (output exists)...')
            continue
        tmp1 = dest / 'tmp_time.nc'
        standardize_time(*files, output=tmp1, dryrun=dryrun, cdo=cdo, **kwargs)
        print('Standardized time steps.')
        if dryrun:
            print('Skipping (dry run)...')
            continue
        tmp2 = tmp3 = dest / 'tmp_horiz.nc'
        standardize_horizontal(tmp1, output=tmp2, cdo=cdo)
        print('Standardized horizontal grid.')
        if name not in ('ps', 'pfull'):
            tmp3 = dest / 'tmp_vert.nc'
            dependencies = {}
            for name in ('ps', 'pfull'):
                kw = {**parts, 'variable': name, 'overwrite': False, 'cdo': cdo}
                if name == 'pfull' and project == 'CMIP5':
                    kw['experiment'] = 'piControl'  # only control tables are published
                try:
                    dep, = process_files(**kw)
                except ValueError:
                    print(f'Variable {name!r} returned ambiuguous files for {parts!r}')  # noqa: E501
                except FileNotFoundError:
                    print(f'Variable {name!r} not available for {parts!r}')
                else:
                    print(f'Variable {name!r} processed for {parts!r}')
                    dependencies[name] = dep
            standardize_vertical(tmp2, output=tmp3, cdo=cdo, **dependencies)
            print('Standardized vertical grid.')
        tmp3.rename(output)
        tmp2.unlink(missing_ok=True)
        tmp1.unlink(missing_ok=True)
        print()

    return outputs


def standardize_time(
    *paths, output=None, variable=None, dryrun=False, cdo=None, **kwargs
):
    """
    Create a standardized monthly climatology or annual mean time series
    from the input files.

    Parameters
    ----------
    *paths : path-like
        The input files.
    output : path-like, optional
        The output name and/or location.
    variable : str, optional
        The variable name to optionally select.
    dryrun : bool, optional
        Whether to only print time information and exit.
    cdo : `cdo.Cdo`, optional
        The cdo instance.
    climate : bool, optional
        Whether to create a monthly-mean climatology or an annual-mean time series.
    numyears : int, default: 50
        The number of years used in the annual time series or monthly climatology.
    endyears : bool, default: False
        Whether to use the start or end of the available times.
    detrend : bool, default: False
        Whether to detrend input data. Used with feedback and budget calculations.
    """
    # Parse input arguments
    # NOTE: This can process data provided only as monthly climate means rather than
    # time series (e.g. 'pfull' model level pressure). In this case 'ymonmean' will
    # be a harmless no-op and 'yearmonmean' will simply average the months together.
    name, climate, numyears, endyears, detrend, kwargs = _parse_time(**kwargs)
    paths = [Path(file).expanduser() for file in paths]
    output = _parse_path(output, stem=name, parent=paths[0])
    if not paths:
        raise TypeError('Input files passed as positional arguments are required.')
    if kwargs:
        raise TypeError(f'Unexpected keyword argument(s): {kwargs}')
    dates = tuple(map(_line_parts['years'], map(str, paths)))
    ymin, ymax = min(tup[0] for tup in dates), max(tup[1] for tup in dates)
    print(f'Unfiltered: {ymin}-{ymax} ({len(paths)} files)')
    paths, dates = zip((f, d) for f, d in zip(paths, dates) if d[0] < ymin + numyears - 1)  # noqa: E501
    ymin, ymax = min(tup[0] for tup in dates), max(tup[1] for tup in dates)
    print(f'Filtered: {ymin}-{ymax} ({len(paths)} files)')
    if dryrun:
        return  # only print time information

    # Merge files into single time series
    # NOTE: Here use 'mergetime' because it is more explicit and possibly safer but
    # cdo docs also mention that 'cat' and 'copy' can do the same thing.
    merge = output.parent / (output.stem + '_raw-merge.nc')
    if variable is not None:
        input = ' '.join(f'-selname,{variable} {file}' for file in paths)
    else:
        input = ' '.join(str(file) for file in paths)
    cdo = cdo or _default_cdo()
    cdo.mergetime(input=str(input), output=str(merge))
    _enforce_exists(merge)

    # Take time averages
    # NOTE: Here the radiative flux data can be detrended to improve sensitive
    # residual energy budget and feedback calculations. This uses a lot of memory
    # so should not bother with 3D constraint and circulation fields.
    numsteps = numyears * 12  # TODO: adapt for other time intervals
    filesteps = int(cdo.ntime(input=str(merge))[0])
    if numsteps > filesteps:
        print(f'Warning: Requested {numsteps} steps but only {filesteps} available')  # noqa: E501
    if endyears:
        t1, t2 = filesteps - numsteps + 1, filesteps  # endpoint inclusive
    else:
        t1, t2 = 1, numsteps
    print(f'Timesteps: {t1}-{t2} ({(t2 - t1 + 1) / 12} years)')
    input = f'-seltimestep,{t1},{t2} {merge}'
    if detrend:
        input = f'-detrend {input}'
    if climate:  # monthly-mean climatology (use 'mean' not 'avg' to ignore NaNs)
        cdo.ymonmean(input=str(input), output=str(output))
    else:  # annual-mean time series
        cdo.yearmonmean(input=str(input), output=str(output))
    _enforce_exists(output)
    return output


def standardize_horizontal(path, output=None, weights=None, reweight=False, cdo=None):
    """
    Create a file on a standardized horizontal grid from the input file
    with arbitrary native grid.

    Parameters
    ----------
    path : path-like
        The input file.
    output : path-like, optional
        The output name and/or location.
    weights : path-like, optional
        The location or name for weights data.
    reweight : bool, optional
        Whether to regenerate the weights.
    cdo : `cdo.Cdo`, optional
        The cdo instance.
    """
    # Parse input horizontal grid
    # NOTE: Sometimes cdo detects dummy 'generic' grids of size 1 or 2 indicating
    # scalar quantities or bounds. Try to ignore these grids.
    path = Path(path).expanduser()
    weights = _parse_path(weights, suffix='weights', parent=path)
    output = _parse_path(output, suffix='standard-grid', parent=path)
    cdo = cdo or _default_cdo()
    result = cdo.griddes(input=str(path))
    tables = [kw for kw in _parse_tables(result) if kw['gridsize'] > 2]
    if not tables:
        raise NotImplementedError(f'Missing horizontal grid for {path}.')
    if len(tables) > 1:
        raise NotImplementedError(f'Ambiguous horizontal grids for {path}: ', ', '.join(table['gridtype'] for table in tables))  # noqa: E501
    table = tables[0]
    string = ', '.join(f'{k}: {v}' for k, v in table.items() if not isinstance(v, np.ndarray))  # noqa: E501
    print('Current horizontal grid:', string)
    print('Destination horizontal grid:', GRID_NAME)

    # Generate weights and interpolate
    opts = ('gridtype', 'xfirst', 'yfirst', 'xsize', 'ysize')
    key_current = tuple(table.get(key, None) for key in opts)
    key_destination = ('lonlat', 0, -90, *map(int, GRID_NAME[1:].split('x')))
    if key_current == key_destination:
        print('Data is already on destination grid.')
        shutil.copy(path, output)
        return output
    if reweight or not weights.is_file():
        print('Generating destination grid weights.')
        cdo.gencon([GRID_NAME], input=str(path), output=str(weights))
        _enforce_exists(weights)
    print('Interpolating to destination grid.')
    cdo.remap([GRID_NAME, str(weights)], input=str(path), output=str(output))
    _enforce_exists(output)
    return output


def standardize_vertical(path, output=None, ps=None, pfull=None, cdo=None):
    """
    Create a file on standardized pressure levels from the input file with
    arbitrary vertical axis.

    Parameters
    ----------
    path : path-like
        The input file.
    output : path-like, optional
        The output name and/or location.
    psfc : path-like, optional
        The path to surface pressure data. Required for hybrid sigma pressure coords.
    pfull : path-like, optional
        The path to model level pressure data. Required for hybrid sigma height coords.
    cdo : `cdo.Cdo`, optional
        The cdo instance.
    """
    # Parse input vertical levels
    # NOTE: The height sigma coordinate variables include orography in the
    # individual files (will show up as a zaxis grid). However don't need that
    path = Path(path).expanduser()
    output = _parse_path(output, suffix='standard-vertical', parent=path)
    cdo = cdo or _default_cdo()
    result = cdo.zaxisdes(input=str(path))
    tables = [kw for kw in _parse_tables(result) if kw['longname'] != 'surface']
    if not tables:
        raise NotImplementedError(f'Missing vertical levels for {path!r}.')
    if len(tables) > 1:
        raise NotImplementedError(f'Ambiguous vertical levels for {path}: ', ', '.join(table['longname'] for table in tables))  # noqa: E501
    table = tables[0]
    string = ', '.join(f'{k}: {v}' for k, v in table.items() if not isinstance(v, np.ndarray))  # noqa: E501
    print('Current vertical levels:', string)
    print('Destination vertical levels:', VERT_LEVS)

    if not tables:
        print('Warning: ')
        return result
    if len(tables) == 1:
        table = tables[0]
        name = table['longname']
    else:
        table = {}
        name = None
        if len(tables) > 1:
            print('Warning: Ambiguous zaxis types ' + ', '.join(table['longname'] for table in tables))  # noqa: E501
    info = ', '.join(
        f'{key}: {value}' for key, value in tables[0].items()
        if not isinstance(value, np.ndarray)  # print scalar information
    )
    print('Current vertical levels:')
    print(f'Running pressure-level vertical interpolation from')

    # Interpolate to pressure levels with cdo
    # NOTE: Currently only some obscure extended monthly data and the core cloud
    # variables are output on model levels. Otherwise data is on pressure levels.
    # Avoid interpolation by selecting same pressure levels as standard output.
    if name == 'pressure':
        pass
    elif name == 'hybrid sigma pressure coordinate':
        if ps is None:
            raise ValueError('Surface pressure data required for hybrid pressure coordinates.')  # noqa: E501
    elif name == 'hybrid height coordinate':
        if pfull is None:
            raise ValueError('Model level pressure data required for hybrid height coords.')  # noqa: E501
    else:
        print(f'Warning: Unexpected vertical axis {name!r}')
    return output
