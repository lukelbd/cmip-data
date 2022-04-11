#!/usr/bin/env python3
"""
Process datasets downloaded with the ESGF python API.
"""
import os
import warnings
from pathlib import Path

import numpy as np
from cdo import Cdo

from .download import _line_parts, _make_printer, _parse_constraints

__all__ = [
    'process_files',
    'standardize_time',
    'standardize_horizontal',
    'standardize_vertical',
]


def _check_output(path):
    """
    Verify that cdo successfully crated the output file.

    Parameters
    ----------
    path : path-like
        The netcdf path.
    """
    if not path.is_file() or path.stat().st_size == 0:
        if path.is_file():
            os.remove(path)
        raise RuntimeError(f'Failed to create output file: {path}.')


def _default_cdo():
    """
    Return a default `cdo.Cdo` binding instance.
    """
    # NOTE: This causes weird issues with caught exceptions so try to minimize use.
    print('Generating default cdo instance.')
    cdo = Cdo(
        env={**os.environ, 'CDO_TIMESTAT_DATE': 'first'},
        options=['-s', '-O'],  # silent and overwrite
    )
    return cdo


def _parse_climate(climate=True, numyears=150, endyears=False, detrend=False, **kwargs):
    """
    Parse variables related to climate averages.

    Parameters
    ----------
    **kwargs
        See `standardize_time` for details.
    """
    name = '_'.join(
        ('timeseries', 'climate')[climate],
        '-'.join(str(numyears), 'year', ('start', 'end')[endyears], ('raw', 'detrend')[detrend]),  # noqa: E501
    ) + '.nc'
    return name, climate, numyears, endyears, detrend, kwargs


def _parse_description(lines):
    """
    Convert the cdo horizontal or vertical grid description into a dictionary.

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
    descrips = []
    for result in results:
        descrip = {}
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
                if isinstance(value, list) or key in descrip:
                    warnings.warn(f'Ignoring unexpected line(s) {string!r}.')
                else:
                    descrip[key] = value.item() if value.size == 1 else value
        if not descrip:
            raise ValueError(f'Failed to read table {result!r}.')
        else:
            descrips.append(descrip)
    return descrips


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
    suffix, climate, numyears, endyears, detrend, constraints = _parse_climate(**kwargs)
    project, constraints = _parse_constraints(reverse=True, **constraints)
    print = _make_printer('process', **constraints)
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
            os.remove(file)
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
            for key, name in (('psfc', 'ps'), ('plev', 'pfull')):
                try:
                    dep, = process_files(**dict(parts, variable=name, overwrite=False, cdo=cdo))  # noqa: E501
                except ValueError:
                    print(f'Warning: Variable {name!r} returned ambiuguous files for {parts!r}')  # noqa: E501
                except FileNotFoundError:
                    print(f'Warning: Variable {name!r} not available for {parts!r}')
                else:
                    dependencies[key] = dep
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
    if not paths:
        raise TypeError('Input files passed as positional arguments are required.')
    name, climate, numyears, endyears, detrend, kwargs = _parse_climate(**kwargs)
    if kwargs:
        raise TypeError(f'Unexpected keyword argument(s): {kwargs}')
    paths = [Path(file).expanduser() for file in paths]
    if not output:
        output = paths[0].parent
    else:
        output = Path(output).expanduser()
    if output.is_dir():
        output = output / name
    tmp = output.parent / 'tmp.nc'
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
    if variable is not None:
        input = ' '.join(f'-selname,{variable} {file}' for file in paths)
    else:
        input = ' '.join(str(file) for file in paths)
    cdo = cdo or _default_cdo()
    cdo.mergetime(input=input, output=str(tmp))
    _check_output(tmp)

    # Take time averages
    # NOTE: Here the radiative flux data can be detrended to improve sensitive
    # residual energy budget and feedback calculations. This uses a lot of memory
    # so should not bother with 3D constraint and circulation fields.
    numsteps = numyears * 12  # TODO: adapt for other time intervals
    filesteps = int(cdo.ntime(input=str(tmp))[0])
    if numsteps > filesteps:
        print(f'Warning: Requested {numsteps} steps but only {filesteps} available')  # noqa: E501
    if endyears:
        t1, t2 = filesteps - numsteps + 1, filesteps  # endpoint inclusive
    else:
        t1, t2 = 1, numsteps
    print(f'Timesteps: {t1}-{t2} ({(t2 - t1 + 1) / 12} years)')
    input = f'-seltimestep,{t1},{t2} {tmp}'
    if detrend:
        input = f'-detrend {input}'
    if climate:  # monthly-mean climatology (use 'mean' not 'avg' to ignore NaNs)
        cdo.ymonmean(input=input, output=str(output))
    else:  # annual-mean time series
        cdo.yearmonmean(input=input, output=str(output))
    _check_output(output)
    return output


def standardize_horizontal(path, output=None, cdo=None):
    """
    Create a file on a standardized horizontal grid from the input file
    with arbitrary native grid.

    Parameters
    ----------
    path : path-like
        The input file.
    output : path-like, optional
        The output name and/or location.
    cdo : `cdo.Cdo`, optional
        The cdo instance.
    """
    # Parse input arguments
    path = Path(path).expanduser()
    if not output:
        output = path.parent
    else:
        output = Path(output).expanduser()
    if output.is_dir():
        output = output / path.stem + '_standard-horizontal.nc'

    # Standardize onto regular longitude-latitude grid
    # NOTE: Sometimes cdo detects dummy 'generic' grids of size 1 or 2 indicating
    # scalar quantities or bounds. Try to ignore these grids.
    cdo = cdo or _default_cdo()
    result = cdo.griddes(input=path)
    tables = [kw for kw in _parse_description(result) if kw['gridsize'] > 2]
    if len(tables) == 1:
        table = tables[0]
        gridtype = tables['gridtype']
    else:
        table = {}
        gridtype = None
        if len(tables) > 1:
            print('Warning: Ambiguous horizontal grids ', ', '.join(table['gridtype'] for table in tables))
    _check_output(output)


def standardize_vertical(path, output=None, psfc=None, plev=None, cdo=None):
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
        The path to model level pressure data. Required for hybrid height coords.
    cdo : `cdo.Cdo`, optional
        The cdo instance.
    """
    # Parse input arguments
    # NOTE: The height sigma coordinate variables include orography in the
    # individual files (will show up as a zaxis grid). However don't need that
    path = Path(path).expanduser()
    if not output:
        output = path.parent
    else:
        output = Path(output).expanduser()
    if output.is_dir():
        output = output / path.stem + '_standard-vertical.nc'

    # # Interpolate to pressure levels with cdo
    # NOTE: Currently only some obscure extended monthly data and the core cloud
    # variables are output on model levels. Otherwise data is on pressure levels.
    # Avoid interpolation by selecting same pressure levels as standard output.
    cdo = cdo or _default_cdo()
    result = cdo.zaxisdes(input=path)
    tables = [kw for kw in _parse_description(result) if kw['longname'] != 'surface']
    if len(tables) == 1:
        table = tables[0]
        name = table['longname']
    else:
        table = {}
        name = None
        if len(tables) > 1:
            print('Warning: Ambiguous zaxis types ' + ', '.join(table['longname'] for table in tables))  # noqa: E501
    if name == 'pressure':
        pass
    elif name == 'hybrid sigma pressure coordinate':
        if psfc is None:
            raise ValueError('Surface pressure data required for hybrid pressure coordinates.')  # noqa: E501
    elif name == 'hybrid height coordinate':
        if plev is None:
            raise ValueError('Model level pressure data required for hybrid height coords.')  # noqa: E501
        pass
    else:
        print(f'Warning: Unexpected vertical axis {name!r}')
