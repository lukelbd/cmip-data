#!/usr/bin/env python3
"""
Process groups of files downloaded from the ESGF.
"""
import builtins
import copy
import re
import shutil
import traceback
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr
from icecream import ic  # noqa: F401

from . import Atted, CDOException, Rename, cdo, nco
from .facets import (
    KEYS_STORAGE,
    KEYS_SUMMARIZE,
    STANDARD_GRIDSPEC,
    STANDARD_LEVELS_CMIP5,
    STANDARD_LEVELS_CMIP6,
    Database,
    Printer,
    glob_files,
    _item_dates,
    _item_facets,
    _item_label,
    _item_years,
    _validate_ranges,
)

__all__ = [
    'process_files',
    'repair_files',
    'standardize_horizontal',
    'standardize_vertical',
    'standardize_time',
    'summarize_processed',
    'summarize_descrips',
    'summarize_ranges',
]


def _parse_time(
    years=150, climate=True, nodrift=False, skipna=False, **constraints
):
    """
    Parse variables related to time standardization and create a file suffix.

    Parameters
    ----------
    constraints : bool, optional
        Whether to allow constraint keyword arguments.
    **kwargs
        See `standardize_time` for details.
    """
    # NOTE: File format is e.g. 'file_0000-1000-climate-nodrift.nc'. Each modeling
    # center uses different date schemes for experiments so years must be relative.
    # Similar to 'file_standard-grid.nc' and 'file_standard-levs.nc' used below.
    if not np.iterable(years):
        years = (0, years)
    elif len(years) == 1:
        years = (years, years)
    year1, year2 = years = np.array(years)
    suffix = (
        format(year1, '04d'),
        format(year2, '04d'),
        ('series', 'climate')[climate],
        *(('nodrift',) if nodrift else ('nodrift',) if nodrift else ()),
    )
    kwargs = {
        'years': years,
        'climate': climate,
        'nodrift': nodrift,
        'skipna': skipna,
    }
    if constraints.pop('constraints', False):
        return suffix, kwargs, constraints
    elif not constraints:
        return suffix, kwargs
    else:
        raise TypeError(f'Unexpected keyword argument(s): {kwargs}')


def _parse_dump(path):
    """
    Parse the ncdump header into dimension sizes and variable attributes.

    Parameters
    ----------
    path : path-like
        The input path.
    """
    # NOTE: This approach is recommended by nco manual (ncks --cdl -m is ncdump -h
    # without global attributes, and ncks --cdl -M is ncdump -h without variable
    # attributes). Use this instead of e.g. netCDF4 for consistency with the rest
    # of workflow. See: http://nco.sourceforge.net/nco.html#Filters-for-ncks
    # NOTE: Here use unique type identifiers to consistently identify variables.
    # Otherwise common to match e.g. variable appearences in 'formula' attributes.
    # Also try to match e.g. 'time = UNLIMITED // (100 currently)' dimension sizes
    # and trim e.g. '1.e+20f' numeric type indicator suffixes in attribute values.
    # See: https://docs.unidata.ucar.edu/nug/current/_c_d_l.html#cdl_data_types
    path = Path(path).expanduser()
    info = nco.ncks(input=str(path), options=['--cdl', '-m', '-M']).decode()
    regex_dim = re.compile(
        r'\n\s*(\w+) = [^0-9]*([0-9]+)'
    )
    regex_var = re.compile(
        r'\n\s*(?:char|byte|short|int|long|float|real|double) (\w+)\s*(?:\((.*)\))?\s*;'
    )
    regex_att = re.compile(
        r'([-+]?[0-9._]+(?:[eE][-+]?[0-9_]+)?(?=[cbsilfrd])?|".*?(?<!\\)")'
    )
    sizes = {m.group(1): m.group(2) for m in regex_dim.finditer(info)}
    names = {m.group(1): (*s.split(','),) if (s := m.group(2)) else () for m in regex_var.finditer(info)}  # noqa: E501
    attrs = {}
    for name in ('', *names):  # include global attributes with empty variable prefix
        regex_key = re.compile(rf'\n\s*{name}:(\w+) = (.*?)\s*;(?=\n)')
        attrs[name] = {
            m.group(1): tup[0] if len(tup) == 1 else tup
            for m in regex_key.finditer(info)
            if (tup := tuple(eval(m.group(1)) for m in regex_att.finditer(m.group(2))))  # noqa: E501
        }
    return sizes, names, attrs


def _parse_descrips(data):
    """
    Convert the cdo grid or zaxis description(s) into a dictionary.

    Parameters
    ----------
    data : str or list
        The cdo grid or zaxis data file (or result of `cdo.griddes` or `cdo.zaxisdes`).
    """
    # NOTE: All cdo descriptions are formatted with block comment headers i.e.
    # '#\n# [Axis|Grid] 1\n#\n...', then scalar items appearing after the equal sign
    # on the same line and only arrays formatted on multiple lines.
    grids = []
    string = data if isinstance(data, str) else '\n'.join(data)
    regex_tables = re.compile(r'(?:\A|\n)(?:#.*\n)?([^#][\s\S]*?)(?:\Z|\n#)')
    regex_items = re.compile(r'(?:\A|\n)(\w*)\s*=\s*([\s\S]*?(?=\Z|\n.*=))')
    regex_str = re.compile(r'\A".*?"\Z')  # note strings sometimes have no quotations
    for m in regex_tables.finditer(string):
        content = m.group(1)
        grids.append(grid := {})
        for m in regex_items.finditer(content):
            key, value = m.groups()
            if regex_str.match(value):
                grid[key] = eval(value)
            else:
                value = [s.strip() for s in value.split()]
                for dtype in (int, float, str):
                    try:
                        value = np.array(value, dtype=dtype)
                    except ValueError:
                        continue
                    else:
                        grid[key] = value.item() if value.size == 1 else value
                    break
                else:
                    warnings.warn(f'Ignoring unexpected data pair {key} = {value!r}.')
    return grids


def _output_check(path, printer=None):
    """
    Check that the output file exists and contains valid data.

    Parameters
    ----------
    path : path-like
        The netcdf path.
    printer : callable, optional
        The printer.
    nodrift : bool, optional
        Whether to check for truncation.
    """
    print = printer or builtins.print
    path = Path(path).expanduser()
    message = f'Failed to create output file: {path}.'
    if not path.is_file() or path.stat().st_size == 0:
        path.unlink(missing_ok=True)
        message += ' File is missing or size zero.'
        raise RuntimeError(message)
    sizes, names, attrs = _parse_dump(path)
    if sizes.get('time', None) == 0:
        path.unlink(mising_ok=True)
        message += ' Time axis has zero length.'
        raise RuntimeError(message)
    tmps = sorted(path.parent.glob(path.name + '.pid*.nc*.tmp'))  # nco commands
    print(f'Created output file {path.name!r}.')
    if tmps:
        for tmp in tmps:
            tmp.unlink(missing_ok=True)
        print(f'Removed {len(tmps)} temporary nco files.')


def _output_path(path=None, *parts):
    """
    Generate an output path from an input path.

    Parameters
    ----------
    path : path-like
        The input path.
    *parts : str optional
        The underscore-separated components for the default path.
    """
    if path:
        path = Path(path).expanduser()
    else:
        path = Path()
    if path.is_dir():
        name = _item_label(*parts, modify=False)  # e.g. ACCESS1-0_0000-0150-climate
        if name:
            path = path / f'{name}.nc'
        else:
            raise ValueError('Path was not provided and default parts are missing.')
    return path


def process_files(
    *paths,
    output='~/data',
    constants='~/data/cmip-constants',
    facets=None,
    vertical=True,
    horizontal=True,
    overwrite=False,
    logging=False,
    dryrun=False,
    nowarn=False,
    **kwargs,
):
    """
    Average and standardize files downloaded with a wget script.

    Parameters
    ----------
    *paths : path-like
        The input path(s) for the raw data.
    output : path-like, optional
        The output directory. Subfolder ``{project}-{experiment}-{table}`` is used.
    constants : path-like, optional
        The constants directory. Default folder is ``~/cmip-constants``.
    facets : str, optional
        The facets for grouping into output folders.
    dependencies : bool, optional
        Whether to automatically add pressure dependencies.
    vertical : bool, optional
        Whether to standardize vertical levels.
    horizontal : bool, optional
        Whether to standardize horizontal grid.
    overwrite : bool, default: True
        Whether to overwrite existing files or skip them.
    logging : bool, optional
        Whether to log the printed output.
    dryrun : bool, optional
        Whether to only print time information and exit.
    nowarn : bool, optional
        Whether to always raise errors instead of warnings.
    **kwargs
        Passed to `standardize_time`, `standardize_vertical`, `standardize_horizontal`.
    **constraints
        Passed to `Printer` and `Database`.
    """
    # Find files and restrict to unique constraints
    # NOTE: Here we only constrain search to the project, which is otherwise not
    # indicated in native netcdf filenames. Similar approach to filter_script().
    logging = logging and not dryrun
    levels, search = kwargs.pop('levels', None), kwargs.pop('search', None)
    gridspec, method = kwargs.pop('gridspec', None), kwargs.pop('method', None)
    searches = (search,) if isinstance(search, (str, Path)) else tuple(search or ())
    dates, kwargs, constraints = _parse_time(constraints=True, **kwargs)
    print = Printer('process', *dates, **constraints) if logging else builtins.print
    files, *_ = glob_files(*paths, project=constraints.get('project'))
    facets = facets or KEYS_STORAGE
    database = Database(files, facets, **constraints)
    constants = Path(constants).expanduser()
    output = Path(output).expanduser()

    # Initialize cdo and process files.
    # NOTE: Here perform horizontal interpolation before vertical interpolation because
    # latter adds NaNs and extrapolated data so should delay that step until very end.
    # NOTE: Currently 'pfull' data is stored in separate 'data-dependencies' folder
    # but 'ps' data is stored in normal folders since it is used for time-varying
    # vertical integration of kernels and is always available.
    kw = {'overwrite': overwrite, 'printer': print}
    outs = []
    print(f'Input files ({len(database)}):')
    print(*(f'{key}: ' + ' '.join(opts) for key, opts in database.constraints.items()), sep='\n')  # noqa: E501
    print()
    _print_error = lambda error: print(
        ' '.join(traceback.format_exception(None, error, error.__traceback__))
    )
    for files in database:
        # Initial stuff
        folder = output / files[0].parent.name  # TODO: use explicit components?
        folder.mkdir(exist_ok=True)
        model = _item_facets['model'](files[0])
        variable = _item_facets['variable'](files[0])
        search = (*searches, files[0].parent.parent)  # use the parent cmip folder
        if variable == 'pfull':
            continue
        stem = '_'.join(files[0].name.split('_')[:-1])
        time = _output_path(folder, stem, 'standard-time')
        vert = _output_path(folder, stem, 'standard-vertical')
        hori = _output_path(folder, stem, 'standard-horizontal')
        out = _output_path(folder, stem, dates)
        outs.append(out)
        kwtime = {
            'offset': constants, 'slope': constants,
            'variable': variable, 'model': model, **kwargs
        }
        kwvert = {
            'levels': levels, 'search': search, 'project': database.project,
            'pfull': constants, 'ps': constants, 'model': model,
        }
        kwhori = {
            'gridspec': gridspec, 'method': method,
            'weights': constants, 'model': model,
        }
        print('Output:', '/'.join((out.parent.name, out.name)))

        # Repair files and standardize time
        _remove_temps = lambda: tuple(
            path.unlink(missing_ok=True) for path in (time, vert, hori)
        )
        try:
            repair_files(*files, dryrun=dryrun, printer=print)
        except Exception as error:
            if nowarn:
                raise error
            else:
                _print_error(error)
            print('WARNING: Failed to standardize attributes.\n')
            continue
        updated = not overwrite and out.is_file() and (
            all(file.stat().st_mtime <= out.stat().st_mtime for file in files)
        )
        try:
            standardize_time(*files, output=time, dryrun=dryrun or updated, **kwtime, **kw)  # noqa: E501
        except Exception as error:
            if nowarn:
                raise error
            else:
                _print_error(error)
            print('WARNING: Failed to standardize temporally.\n')
            continue
        if updated:
            print('Skipping (up to date)...\n')
            continue
        if dryrun:
            print('Skipping (dry run)...\n')
            continue

        # Standardize horizontal grid and vertical levels
        sizes, names, attrs = _parse_dump(files[0])
        constants.mkdir(exist_ok=True)
        if not vertical or 'lev' not in attrs:
            time.replace(vert)
        else:
            try:
                standardize_vertical(time, output=vert, **kwvert, **kw)
            except Exception as error:
                if nowarn:
                    raise error
                else:
                    _print_error(error)
                print('WARNING: Failed to standardize vertically.\n')
                continue
        if not horizontal:
            vert.replace(hori)
        else:
            try:
                standardize_horizontal(vert, output=hori, **kwhori, **kw)
            except Exception as error:
                if nowarn:
                    raise error
                else:
                    _print_error(error)
                print('WARNING: Failed to standardize horizontally.\n')
                continue
        hori.replace(out)
        _remove_temps()
        _output_check(out, print)
        today = datetime.today().strftime('%Y-%m-%dT%H:%M:%S')  # date +%FT%T
        print(f'Processing completed: {today}')
        print('Removed temporary files.\n')
    return outs


def repair_files(*paths, dryrun=False, printer=None):
    """
    Repair the metadata on the input files in-place so they are compatible with the
    `cdo` subcommands used by `standardize_vertical` and `standardize_horizontal`.

    Parameters
    ----------
    *paths : path-like
        The input path(s).
    dryrun : bool, default: False
        Whether to only print command information and exit.
    printer : callable, default: `print`
        The print function.
    """
    # Declare attribute edits for converting pure sigma coordinates to hybrid and for
    # handling non-standard hybrid pressure coordinates using 'ap' instead of 'a * p0'.
    # NOTE: For cdo to successfully parse hybrid coordinates need so-called "vertices".
    # So critical to include 'lev:bounds = "lev_bnds"' and have "lev_bnds" attrs
    # that point to the "a_bnds" and "b_bnds" hybrid level interfaces.
    print = printer or builtins.print
    as_str = lambda att: opt if isinstance(opt := att.prn_option(), str) else ' '.join(opt)  # noqa: E501
    pure_sigma = 'atmosphere_sigma_coordinate'
    hybrid_pressure = 'atmosphere_hybrid_sigma_pressure_coordinate'
    hybrid_pressure_atted = []
    for name in ('lev', 'lev_bnds'):
        end = '_bnds' if '_' in name else ''
        bnd = ' bounds' if '_' in name else ''
        idx = '+1/2' if '_' in name else ''
        hybrid_pressure_atted.extend(
            as_str(att) for att in (
                Atted('overwrite', 'long_name', name, f'atmospheric model level{bnd}'),
                Atted('overwrite', 'standard_name', name, hybrid_pressure),
                Atted('overwrite', 'formula', name, 'p = ap + b*ps'),
                Atted('overwrite', 'formula_terms', name, f'ap: ap{end} b: b{end} ps: ps'),  # noqa: E501
            )
        )
        hybrid_pressure_atted.extend(
            as_str(att).replace('" "', '') for term in ('b', 'ap') for att in (  # noqa: E501
                Atted('delete', ' ', f'{term}{end}'),  # remove all attributes
                Atted('overwrite', 'long_name', f'{term}{end}', f'vertical coordinate formula term: {term}(k{idx})'),  # noqa: E501
            )
        )

    # Declare attribute removals for interpreting hybrid height coordinates as a
    # generalized height coordinate in preparation for cdo ap2pl pressure interpolation.
    # NOTE: The cdo ap2pl operator also requires long_name="generalized height" but
    # this causes issues when applying setzaxis so currently delay this attribute
    # edit until standardize_vertical. See: https://code.mpimet.mpg.de/issues/10692
    # NOTE: Using -mergetime with files with different time 'bounds' or 'climatology'
    # can create duplicate 'time_bnds' and 'climatolog_bnds' variables, and using
    # -merge can trigger bizarre incorrect 'reserved variable zaxistype' errors. So
    # delete these attributes when detected on candidate 'pfull' data.
    hybrid_height = 'atmosphere_hybrid_height_coordinate'
    hybrid_height_atted = [
        as_str(att) for name in ('lev', 'lev_bnds') for att in (
            Atted('overwrite', 'standard_name', name, 'height'),
            Atted('delete', 'long_name', name),
            Atted('delete', 'units', name),
            Atted('delete', 'formula', name),
            Atted('delete', 'formula_terms', name),
        )
    ]
    hybrid_height_atted_extra = [
        as_str(att) for att in (
            Atted('delete', 'climatology', 'time'),
            Atted('delete', 'bounds', 'time'),  # prevents clean merge
        )
    ]

    print(f'Checking {len(paths)} files for required repairs.')
    for path in paths:
        # Make the simplest generalized attribute and variable repairs
        # NOTE: The leading 'dot' tells ncrename to look for attributes on
        # all variables and only apply the rename if the attribute is found.
        # NOTE: Missing vertical level axis and formula terms attributes cause cdo to
        # detect vertical coordinates as separate grids, resulting in nebulous sounding
        # error message before vertical interpolation methods can get farther along.
        path = Path(path).expanduser()
        sizes, names, attrs = _parse_dump(path)
        lev_std = attrs.get('lev', {}).get('standard_name', '')
        lev_bnds_key = attrs.get('lev', {}).get('bounds', 'lev_bnds')
        lev_bnds_std = attrs.get(lev_bnds_key, {}).get('standard_name', '')
        lev_bnds_formula = attrs.get(lev_bnds_key, {}).get('formula_terms', '')
        var_extra = ('average_T1', 'average_T2', 'average_DT')
        if any(s in attrs for s in var_extra):
            print(
                'WARNING: Repairing GFDL-like inclusion of unnecessary '
                f'variable(s) for {path.name!r}:', *var_extra
            )
            if not dryrun:
                remove = ['-x', '-v', ','.join(var_extra)]
                nco.ncks(input=str(path), output=str(path), options=remove)
        if any('formula_term' in kw for kw in attrs.values()):
            rename = [as_str(Rename('a', {'.formula_term': 'formula_terms'}))]
            print(
                'WARNING: Repairing GFDL-like misspelling of formula_terms '
                + f'attribute(s) for {path.name!r}:', *rename
            )
            if not dryrun:
                nco.ncrename(input=str(path), output=str(path), options=rename)
        if 'presnivs' in sizes:
            rename = [
                as_str(Rename(c, {'presnivs': 'lev'}))
                for c, src in (('d', sizes), ('v', attrs)) if 'presnivs' in src
            ]
            print(
                'WARNING: Repairing IPSL-like non-standard vertical axis name '
                + f'presnivs for {path.name!r}:', *rename
            )
            for dict_ in (sizes, names, attrs):
                dict_['lev'] = dict_.pop('presnivs')
            if not dryrun:
                nco.ncrename(input=str(path), output=str(path), options=rename)
        if 'lev' in attrs and attrs['lev'].get('axis', '') != 'Z':
            atted = [as_str(Atted('overwrite', 'axis', 'lev', 'Z'))]
            print(
                'WARNING: Repairing IPSL-like missing level variable axis '
                + f'attribute for {path.name!r}:', *atted
            )
            if not dryrun:
                nco.ncatted(input=str(path), output=str(path), options=atted)

        # Handle IPSL models with weird extra dimension on level coordinates. This has
        # to come first or else the CNRM block will be triggered for IPSL files.
        # NOTE: Here only the hybrid coordinates used 'klevp1' so can delete it. Also
        # probably would cause issues with remapping functions. For manual alternative
        # to make_bounds see: https://stackoverflow.com/a/36680196/4970632
        # NOTE: Critical that lev_bnds bounds dimension order matches hybrid vertical
        # coordinate bounds order (although dummy bounds variable names can differ).
        # Verified this is only an issue for IPSL-CM6A using following:
        # for f in cmip[56]-picontrol-amon/cl_Amon*; do m=$(echo $f | cut -d_ -f3);
        # [[ " ${models[*]} " =~ " $m "  ]] && continue || models+=("$m"); echo "$m:";
        # ncinfo "$f" | grep -E '(ap_bnds|a_bnds|b_bnds|lev_bnds)\('; done; unset models
        if 'klevp1' in attrs:  # IPSL handling
            atted = hybrid_pressure_atted
            math = repr(
                'b[$lev] = 0.5 * (b(1:) + b(:-2)); '
                'ap[$lev] = 0.5 * (ap(1:) + ap(:-2)); '
                '*b_tmp = b_bnds.permute($klevp1, $bnds); '
                '*ap_tmp = ap_bnds.permute($klevp1, $bnds); '
                'b_bnds[$lev, $bnds] = 0.5 * (b_tmp(1:, :) + b_tmp(:-2, :)); '
                'ap_bnds[$lev, $bnds] = 0.5 * (ap_tmp(1:, :) + ap_tmp(:-2, :)); '
                'lev_bnds = make_bounds(lev, $bnds, "lev_bnds"); '
            )
            print(
                'WARNING: Repairing IPSL-like hybrid coordinates with unusual missing '
                + f'attributes for {path.name!r}:', math, ' '.join(atted)
            )
            if not dryrun:
                nco.ncap2(input=str(path), output=str(path), options=['-s', math])
                nco.ncatted(input=str(path), output=str(path), options=atted)
                nco.ncks(input=str(path), output=str(path), options=['-x', '-v', 'klevp1'])  # noqa: E501

        # Handle FGOALS models with pure sigma coordinates. Currently cdo cannot
        # handle them so translate attributes so they are recognized as hybrid.
        # See: https://code.mpimet.mpg.de/boards/1/topics/167
        if 'ptop' in attrs or any(s == pure_sigma for s in (lev_std, lev_bnds_std)):
            atted = hybrid_pressure_atted
            math = repr(
                'b[$lev] = lev; '
                'ap[$lev] = (1 - lev) * ptop; '
                'b_bnds[$lev, $bnds] = lev_bnds; '
                'ap_bnds[$lev, $bnds] = (1 - lev_bnds) * ptop; '
            )
            print(
                'WARNING: Converting FGOALS-like sigma coordinates to hybrid sigma '
                f'pressure coordinates for {path.name!r}:', math, ',', ' '.join(atted)
            )
            if not dryrun:
                nco.ncap2(input=str(path), output=str(path), options=['-s', math])
                nco.ncatted(input=str(path), output=str(path), options=atted)
                nco.ncks(input=str(path), output=str(path), options=['-x', '-v', 'ptop'])  # noqa: E501

        # Handle CMIP5 FGOALS models with incorrect coordinate encoding
        # NOTE: These show 'a' normalized by 'p0' in formula terms but the
        # values are obviously absolute pressure. So manually normalize.
        # NOTE: If reading into cdo using e.g. 'seltimestep' it will automatically
        # rename 'a' to 'ap' and update formula_terms but keep 'formula' the same
        # i.e. it sort of seems to correct the normalization but it does not.
        if (
            '_FGOALS-s2' in path.stem and 'a' in attrs and 'a_bnds' in attrs
            and not any('normalize_status' in attrs[s] for s in ('a', 'a_bnds'))
        ):
            math = repr(
                'a = a / p0; '
                'a_bnds = a_bnds / p0; '
                'a@normalize_status = "repaired"; '
                'a_bnds@normalize_status = "repaired"; '
            )
            print(
                'WARNING: Repairing FGOALS-like hybrid coordinate pressure component '
                f'with unnormalized values for {path.name!r}:', math,
            )
            if not dryrun:
                nco.ncap2(input=str(path), output=str(path), options=['-s', math])

        # Handle CMIP6 FGOALS models with messed up hybrid coordinates.
        # NOTE: These models have a/b coordinate bounds of e.g. 1e+23, 1.6, etc. that
        # don't correspond to centers so we totally disregard them with a half-way
        # ncap2 estimate (with adjustments to satisfy boundary conditions).
        if (
            '_FGOALS-f3-L' in path.stem and 'a' in attrs and 'b' in attrs
            and not any('bounds_status' in attrs[s] for s in ('a', 'b'))
        ):
            math = repr(
                'a_bnds = make_bounds(a, $bnds, "a_bnds"); '
                'b_bnds = make_bounds(b, $bnds, "b_bnds"); '
                'a_bnds(0, 0) = 0; a_bnds(-1, 1) = 0; '
                'b_bnds(0, 0) = 1; b_bnds(-1, 1) = 0; '
                'a@bounds_status = "repaired"; '
                'b@bounds_status = "repaired"; '
            )
            print(
                'WARNING: Repairing FGOALS-like hybrid coordinate bounds with '
                f'messed up values for {path.name!r}:', math,
            )
            if not dryrun:
                nco.ncap2(input=str(path), output=str(path), options=['-s', math])

        # Handle CNRM models with hybrid coordinate boundaries permuted and lev_bounds
        # formula_terms attribute pointing incorrectly to the actual variables.
        # NOTE: Previously tried ncpdq for permutations but was insanely slow
        # and would eventualy fail due to some memory allocation error. For
        # some reason ncap2 permute (while slow) does not implode this way.
        if all(
            s in attrs and s not in lev_bnds_formula for s in ('ap_bnds', 'b_bnds')
        ):
            math = repr(
                'b_bnds = b_bnds.permute($lev, $bnds); '
                'ap_bnds = ap_bnds.permute($lev, $bnds); '
                f'{lev_bnds_key}@formula_terms = "ap: ap_bnds b: b_bnds ps: ps"; '
            )
            print(
                'WARNING: Repairing CNRM-like incorrect formula terms attribute '
                f'for {path.name!r}: ', math,
            )
            if not dryrun:
                nco.ncap2(input=str(path), output=str(path), options=['-s', math])

        # Handle CESM models with necessary attributes on only
        # coordinate bounds variables rather than coordinate centers.
        # NOTE: These models also sometimes have issues where levels are
        # inverted. That is handled with an ncap2 block below.
        if lev_std != hybrid_pressure and lev_bnds_std == hybrid_pressure:
            atted = [
                as_str(Atted('overwrite', key, 'lev', new))
                for key, old in attrs[lev_bnds_key].items()
                if (new := old.replace(' bounds', '').replace('_bnds', ''))
            ]
            print(
                'WARNING: Repairing CESM-like hybrid coordinates with missing level '
                + f'center attributes for {path.name!r}:', ' '.join(atted)
            )
            if not dryrun:
                nco.ncatted(input=str(path), output=str(path), options=atted)

        # Handle CESM files that have negative levels
        # NOTE: Testing showed that level bounds in non-WACCM files need to be
        # inverted whenever level centers are inverted, but level bounds in WAACM
        # files are always correct (also hybrid coordinate bounds always seem to
        # follow level bounds). Carefully account for this below. Also enforce that
        # level bounds decrease along 'bnds' dimension along with the 'lev' dimension,
        # although this is not critical (cdo zaxisdes shows that the detected 'vct'
        # hybrid coordinate bounds are not sensitive to the bounds order).
        options = [
            ('lev', 'invert_status'),
            ('lev_bnds', 'invert_bnds_status'),
            ('lev_bnds', 'invert_levs_status'),
        ]
        if (
            '_CESM2' in path.stem and 'lev' in attrs and 'lev_bnds' in attrs
            and not any(attr in attrs[var] for var, attr in options)
        ):
            bnds = names.get(lev_bnds_key, ())
            bnds = tuple(dim for dim in bnds if 'bnd' in dim or 'bound' in dim)
            table = tuple(tab for tab in _parse_descrips(cdo.zaxisdes(input=str(path))))  # noqa: E501
            table = tuple(tab for tab in table if all(s in tab for s in ('levels', 'lbounds', 'ubounds')))  # noqa: E501
            if len(bnds) != 1 or len(table) != 1:
                continue
            (bnds,), (table,) = bnds, table
            invert_levs = np.any(table['levels']) < 0
            invert_bnds_bnds = table['lbounds'][0] < table['ubounds'][0]
            invert_bnds_levs = table['lbounds'][0] < table['lbounds'][1]
            if not any((invert_levs, invert_bnds_bnds, invert_bnds_levs)):
                continue
            maths = []
            if invert_levs:
                maths.append('lev = -1 * lev')
                maths.append('lev@invert_status = "repaired"')
                maths.extend(
                    f'{name} = {name}.reverse($lev)' for name, dims in names.items()
                    if 'lev' in dims and bnds not in dims
                )
            if invert_bnds_bnds:
                maths.append('lev_bnds@invert_bnds_status = "repaired"')
                maths.extend(
                    f'{name} = {name}.reverse(${bnds})' for name, dims in names.items()
                    if 'lev' in dims and bnds in dims
                )
            if invert_bnds_levs:
                maths.append('lev_bnds@invert_levs_status = "repaired"')
                maths.extend(
                    f'{name} = {name}.reverse($lev)' for name, dims in names.items()
                    if 'lev' in dims and bnds in dims
                )
            print(
                'WARNING: Repairing CESM-like negative level axis coordinates and/or '
                'incorrectly ordered bounds:', math := repr('; '.join(maths))
            )
            if not dryrun:
                nco.ncap2(input=str(path), output=str(path), options=['-s', math])

        # Handle hybrid height coordinates for cdo compatibility. Must remove
        # coordinate information and rely on pressure levels intsead.
        # NOTE: For consistency, could stop deleting orography and delay until after
        # interpolating as with hybrid pressure file surface pressure. ...or maybe this
        # makes sense as way to signify that these files are modified and intended
        # for interpolation to pressure levels rather than height levels.
        if 'orog' in attrs or any(s == hybrid_height for s in (lev_std, lev_bnds_std)):
            atted = hybrid_height_atted
            atted += hybrid_height_atted_extra if 'pfull' in attrs else []
            print(
                'WARNING: Converting ACCESS-like hybrid height coordinates to '
                f'generalized height coordinates for {path.name!r}:', ' '.join(atted)
            )
            if not dryrun:
                var_extra = ('a', 'b', 'a_bnds', 'b_bnds', 'orog')
                var_extra += ('time_bnds', 'climatology_bnds') if 'pfull' in attrs else ()  # noqa: E501
                remove = ['-x', '-v', ','.join(var_extra)]
                nco.ncatted(input=str(path), output=str(path), options=atted)
                nco.ncks(input=str(path), output=str(path), options=remove)

        # Handle MCM-UA-1-0 and FIO-ESM-2-0 models with messed up longitude/latitude
        # bounds that have non-global coverage and cause error when calling 'cdo gencon'
        # Also remove unneeded 'areacella' that most likely contains incorrect weights
        # (seems all other files reference this in cell_measures but exclude variable).
        # This is the only horizontal interpolation-related correction required.
        # NOTE: Verified only MCM-UA uses 'longitude' and 'latitude' (although can
        # use either full name or abbreviated for matching bounds). Try the following:
        # for f in cmip[56]-picontrol-amon/cl_Amon*; do m=$(echo $f | cut -d_ -f3);
        # [[ " ${models[*]} " =~ " $m "  ]] && continue || models+=("$m");
        # echo "$m: "$(ncdimlist "$f" | tail -n +2 | xargs); done; unset models
        lon = 'longitude' if 'longitude' in attrs else 'lon'
        lat = 'latitude' if 'latitude' in attrs else 'lat'
        models = ['_MCM-UA-1-0', '_FIO-ESM-2-0']
        bounds = [attrs.get(key, {}).get('bounds', None) for key in (lon, lat)]
        if (
            any(bound is not None for bound in bounds)
            and any(model in path.stem for model in models)
        ):
            atted = [
                as_str(Atted('delete', 'bounds', key))
                for key in (lon, lat)
            ]
            print(
                'WARNING: Converting MCM-UA-like issue with non-global longitude and '
                f'latitude bounds for {path.name!r}:', ' '.join(atted)
            )
            if not dryrun:  # first flag required because is member of cell_measures
                remove = ['-x', '-v', ','.join((*bounds, 'areacella'))]
                nco.ncatted(input=str(path), output=str(path), options=atted)
                nco.ncks(input=str(path), output=str(path), options=remove)

        # Handle ACCESS1-0 and ACCESS1-3 pfull files that all have 12 timesteps
        # but have the same time value (obviously intended to be a monthly series).
        # This is the only time coordinate related correction required.
        # NOTE: The year in these files is 444 which is a leap year in proleptic
        # gregorian calendar, so use 29 days for offsetting month of february.
        if (
            'pfull' in attrs and sizes.get('time', 1) == '12'
            and cdo.nmon(input=str(path))[0] == '1'
        ):
            math = repr(
                '*days_per_month[$time] = {31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 30}; '  # noqa: E501
                '*offsets_per_month[$time] = 0; '
                'for (*idx = 1; idx < $time.size; idx++) '
                'offsets_per_month(idx) = days_per_month(0:idx-1).total(); '
                'time += offsets_per_month; '
            )
            print(
                'WARNING: Repairing ACCESS-like 12 timesteps with identical months by '
                'adding one month to each step:', math
            )
            if not dryrun:
                nco.ncap2(input=str(path), output=str(path), options=['-s', math])

        # Handle GISS-E2-1-G and GISS-E2-1-H hus files that have missing value set
        # to 1e+20 but then impossible values of -1.000000062271131e+27 coinciding
        # with topography that are obviously meant to be missing.
        # NOTE: Cannot write to same file with cdo so have to work with a temporary
        # file. Use cdo rather than nco for speed since have to scan all values.
        # NOTE: Initially worried that this value might vary between models or
        # experiments but checked across files and it is always the same... weird.
        if (
            ('_GISS-E2-1' in path.stem or '_GISS-E2-2' in path.stem)  # avoid cmip5
            and 'hus' in attrs and 'missing_status' not in attrs['hus']
        ):
            atted = 'hus@missing_status="repaired"'
            input = f'-setctomiss,-1.000000062271131e+27 {path}'
            print(
                'WARNING: Repairing GISS-like issue where impossible float value '
                'is not correctly detected as a missing value:', atted, input
            )
            if not dryrun:
                temp = path.parent / f'{path.stem}_tmp{path.suffix}'
                cdo.setattribute(atted, input=input, output=str(temp))
                temp.replace(path)  # see https://bugs.python.org/issue27886


def standardize_time(
    *paths,
    output=None,
    dryrun=False,
    offset=None,
    slope=None,
    variable=None,
    model=None,
    rebuild=False,
    allow_incomplete=True,
    overwrite=False,
    printer=None,
    **kwargs
):
    """
    Create a standardized monthly climatology or annual mean time series from
    the input files. Uses `cdo.mergetime` and optionally `cdo.ymonmean`.

    Note
    ----
    This requires an `offset` and `slope` file when drift correcting the data.
    If they do not exist, they are created from the current file.

    Parameters
    ----------
    *paths : path-like
        The input files.
    output : path-like, optional
        The output name and/or location.
    years : int or 2-tuple, default: 150
        The years to use relative to the start of the simulation.
    climate : bool, default: True
        Whether to create a monthly-mean climatology or a time series.
    nodrift : bool, default: False
        Whether to drift correct the time series using the control data.
    offset, slope : path-like, optional
        The location or name for the offset and slope data needed for `nodrift`.
    variable, model : str, optional
        The variable and model to use in the default offset and slope names.
    rebuild : path-like, default: False
        Whether to rebuild the offset and slope files.
    skipna : bool, default: False
        Whether to ignore NaNs.
    dryrun : bool, default: False
        Whether to only print time information and exit.
    allow_incomplete : default: True
        Whether to allow incomplete time periods.
    overwrite : bool, default: False
        Whether to overwrite existing files or skip them.
    printer : callable, default: `print`
        The print function.
    """
    # Initial stuff
    # NOTE: Some models use e.g. file_200012-201011.nc instead of file_200001-200912.nc
    # so 'selyear' could end up removing a year... but not worth worrying about. There
    # is also endpoint inclusive file_200001-201001.nc but seems only for HadGEM3 pfull.
    print = printer or builtins.print
    dates, kwtime = _parse_time(**kwargs)
    paths = [Path(file).expanduser() for file in paths]
    output = _output_path(output or paths[0].parent, paths[0].stem, dates)
    if not paths:
        raise TypeError('Input files passed as positional arguments are required.')
    if not overwrite and output.is_file() and output.stat().st_size > 0:
        print(f'Output file already exists: {output.name!r}.')
        return output

    # Filter input files for requested year range
    # NOTE: Previously included kludge here for CESM2 files with incorrect (negative)
    # levels. Without kludge cdo will average upper with lower levels. However have
    # moved this to repair_files. Super slow but needs to be done that way.
    years = tuple(map(_item_years, paths))
    ymin, ymax = min(ys[0] for ys in years), max(ys[1] for ys in years)
    print(f'Initial year range: {ymin}-{ymax} ({len(years)} files)')
    rmin, rmax = kwtime.pop('years')  # relative years
    ymin, ymax = ymin + rmin, ymin + rmax - 1  # e.g. (0, 50) is up to year 49
    inputs = {}
    for ys, path in zip(years, paths):
        if ys[0] > ymax or ys[1] < ymin:
            continue
        y0, y1 = max((ys[0], ymin)), min((ys[1], ymax))
        sel = '' if ys[0] >= ymin and ys[1] <= ymax else f'-selyear,{y0}/{y1} '
        inputs[f'{sel}{path}'] = (ys[0], ys[1])  # test file years later
    print(f'Filtered year range: {ymin}-{ymax} ({len(inputs)} files)')
    if not inputs:
        message = f'No files found within requested range: {ymin}-{ymax}'
        if dryrun:
            print(f'WARNING: {message}')
        else:
            raise RuntimeError(message)

    # Fitler intersecting ranges and print warning for incomplete range
    # NOTE: FIO-ESM-2-0 control psl returned 300-399 and 400-499 in addition to 301-400
    # and 401-500, CAS-ESM2-0 rsus returned 001-600 control and 001-167 abrupt in
    # addition to standard 001-550 and 001-150, and GFDL-CM4 abrupt returned 000-010
    # in addition to 000-100. The below should handle these situations... although
    # ideally should never download these files anymore (see download_script function).
    ranges = []
    for input, (y0, y1) in tuple(inputs.items()):
        for other, (z0, z1) in tuple(inputs.items()):
            if input == other or input not in inputs:
                continue  # e.g. already deleted
            message = f'Skipping file years {y0}-{y1} in presence of %s file years {z0}-{z1}.'  # noqa: E501
            if z0 <= y0 <= y1 <= z1:
                print('WARNING: ' + message % 'superset')
                del inputs[input]
            if y1 - y0 > 1 and y0 == z0 + 1 and y1 == z1 + 1:
                print('WARNING: ' + message % 'offset-by-one')
                del inputs[input]
    for y0, y1 in inputs.values():
        for range_ in ranges:
            if range_[0] in (y1, y1 + 1):
                range_[0] = y0  # prepend to existing distinct range
                break
            if range_[1] in (y0, y0 - 1):
                range_[1] = y1  # append to existing distinct time range
                break
        else:
            ranges.append([y0, y1])
    if len(ranges) != 1 or ranges[0][0] > ymin or ranges[0][1] < ymax:
        ranges = ', '.join('-'.join(map(str, range_)) for range_ in ranges)
        message = f'Full year range not available from file years: {ranges}'
        if dryrun or allow_incomplete:
            print(f'WARNING: {message}')
        else:
            raise RuntimeError(message)
    if dryrun:
        return  # only print time information

    # Generate trend files for drift correction
    # NOTE: Important to use (1) annual averages rather than monthly averages to help
    # reduce probability of spurious trends (e.g. starting in december and ending in
    # november might trigger a warming signal), (2) local values rather than global
    # values to prevent imposing large trends associated with large climatological
    # values on regions with small trends (e.g. Gupta 2013 and others also use local
    # trends), and (3) ignore fields where a single timestep is NaN (can alias strong
    # seasonal cycle elements in -- noticed huge trends in air temperature near surface
    # on first run). Slopes are then scaled by 12 (see python notebook; note monthly
    # trends are also different from annual trends in general).
    # NOTE: Cannot simply use 'detrend' for drift corrections because that also
    # removes mean and we want to *add* trends from the starting point (i.e. instead
    # of data(i) - (a + b * t(i)) want data(i) - ((a + b * t(i)) - (a + b * t(0)).
    # Testing revealed that cdo uses time step indices rather than actual time
    # values, so we simply need the detrended data plus the offset parameter. Also
    # important to add missing slope rise due to e.g. removing subtracting trend from
    # an abrupt climatology starting after the control run branch time at year 0. And
    # finally considered a multiplicative adjustment of data(i) * ((a + b * t(0))
    # / (a + b * t(i))) but can cause problems with zero-valued data that outweigh
    # benefits of fixing impossible negatives. In practice the latter is acceptable.
    arg = None
    input = '[ ' + ' '.join(input for input in inputs) + ' ]'
    average = 'ymonmean' if kwtime.pop('skipna') else 'ymonavg'
    if not kwtime.pop('nodrift'):
        if kwtime.pop('climate'):
            method = average
            prefix, suffix = '-mergetime ', ''
        elif kwtime.pop('seasonal', None):  # TODO: remove?
            method = 'timstd'
            prefix, suffix = f'{average} -mergetime ', ''
        elif kwtime.pop('interannual', None):  # TODO: remove?
            method = 'timstd'
            prefix, suffix = '-ymonsub -mergetime ', f' {average} -mergetime {input}'
        else:
            method = 'mergetime'
            prefix = suffix = ''
    else:
        sel = ''  # previously included -enlarge,{paths[0]} for global trends
        equal = 'equal=false'  # account for different numbers of days per month
        offset = _output_path(offset or paths[0].parent, variable, model, 'offset')
        slope = _output_path(slope or paths[0].parent, variable, model, 'slope')
        if 'NESM3' in paths[0].name and 'abrupt-4xCO2' in paths[0].name:
            sel = '-sellevidx,1/17 '  # abrupt-4xCO2 has fewer levels than piControl
        prefix = f'-add -add {sel}{offset} -mulc,{rmin} {sel}{slope} -mergetime '
        suffix = f' {sel}{offset} -divc,12 {sel}{slope}'
        if kwtime.pop('climate'):
            method = average
            prefix = f'-subtrend,{equal} {prefix}'
        else:
            arg = equal
            method = 'subtrend'
        if rebuild or any(  # previously included -fldmean before -yearmonmean
            not file.is_file() or file.stat().st_size == 0 for file in (offset, slope)
        ):
            trend = '[ ' + ' '.join(f'-yearmonmean {input}' for input in inputs) + ' ]'
            if 'piControl' not in paths[0].name:
                raise ValueError('Cannot rebuild drift correction files from paths.')
            trend = f'-mergetime {trend}'
            descrip = f'-mergetime {trend}'.replace(f'{paths[0].parent}/', '')
            print(f'Generating trend files with -trend {descrip}')
            cdo.trend(equal, input=f'-mergetime {trend}', output=f'{offset} {slope}')
            _output_check(offset, print)
            _output_check(slope, print)

    # Consolidate times and apply drift corrections
    # NOTE: Remove only the source file directory components (i.e. not trend files)
    # so that if something goes wrong can try things out by navigating to directory
    # and pasting the code. Could keep everything but that makes enormous log files.
    descrip = f'{method} {prefix}{input}{suffix}'.replace(f'{paths[0].parent}/', '')
    print(f'Merging {len(inputs)} files with {descrip}.')
    args = (arg,) if arg else ()
    getattr(cdo, method)(*args, input=f'{prefix}{input}{suffix}', output=str(output))
    _output_check(output, print)
    return output


def standardize_vertical(
    path,
    output=None,
    levels=None,
    ps=None,
    pfull=None,
    model=None,
    search=None,
    project=None,
    rebuild=None,
    overwrite=False,
    printer=None
):
    """
    Create a file on standardized pressure levels from the input file with
    arbitrary vertical axis. Uses  `cdo.intlevel`, `cdo.ml2pl`, or `cdo.ap2pl`.

    Parameters
    ----------
    path : path-like
        The input file.
    output : path-like, optional
        The output name and/or location.
    levels : array-like, optional
        The output levels. If not passed then the standard cmip levels are used.
    ps, pfull : path-like, optional
        The reference surface and model level pressure paths.
    model : str, optional
        The model to use in the default reference pressure paths and search pattern.
    search : path-like or sequence, optional
        The path(s) to pass to `glob_files` when searching for reference pressure data.
    project : str, optional
        The project to pass to `glob_files` when searching for reference pressure data.
    rebuild : path-like, default: False
        Whether to rebuild the dependency files.
    overwrite : bool, default: True
        Whether to overwrite existing files or skip them.
    printer : callable, default: `print`
        The print function.

    Important
    ---------
    The input file requires surface pressure to interpolate sigma and hybrid pressure
    coordinates and model level pressure to interpolate hybrid height coordinates.
    """
    # Initial stuff
    print = printer or builtins.print
    path = Path(path).expanduser()
    project = (project or 'cmip6').lower()
    output = _output_path(output or path.parent, path.stem, 'standard-vertical')
    search = search or '~/data'
    search = (search,) if isinstance(search, (str, Path)) else tuple(search)
    deps = output.parent / (output.stem + '-deps' + output.suffix)
    if levels is not None:
        levels = np.array(levels)
    elif project == 'cmip5':
        levels = STANDARD_LEVELS_CMIP5
    elif project == 'cmip6':
        levels = STANDARD_LEVELS_CMIP6
    else:
        raise ValueError(f'Invalid {project=} for determining levels.')
    if not overwrite and output.is_file() and output.stat().st_size > 0:
        print(f'Output file already exists: {output.name!r}.')
        return output

    # Parse input vertical levels
    # NOTE: Using 'generic' for height levels triggers lots of false positives for
    # files with incorrectly formatted level attributes but required since we cannot
    # apply 'generalized height' long name without setzaxis erroring out.
    # NOTE: For some reason long_name='generalized height' required for ap2pl detection
    # but fails with setzaxis and trying to modify the zaxistype causes subsequent
    # merges to fail. See: https://code.mpimet.mpg.de/issues/10692
    # NOTE: For some reason ap2pl requires top-to-bottom (i.e. low-to-high) 3D pressure
    # array orientation, so prepend command with -invertlev. This weird limitation is
    # undocumented and error is silent. See: https://code.mpimet.mpg.de/issues/10692
    # NOTE: Had all-NaN upper levels 'cli' and 'clw' KACE files, and NaNs are not
    # supported by ap2pl (possibly also for ml2pl but have not yet encountered error).
    # Checked pfull files and occurs around 40hPa for 'cli' and 20hPa for 'clw' (max
    # values in level right above drop off), also have lots of center levels with NaNs
    # (fewest NaNs for 'cli' occur around 310hPa where there is lots of ice, and fewest
    # NaNs for 'clw' occur around 900hPa where there is lots of water), thus NaNs seem
    # to correspond to where concentrations are close to zero... so decided to simply
    # set missing values to NaN, should not create artifacts when making constraints.
    ps = _output_path(ps or path.parent, 'ps', model, 'climate')
    pfull = _output_path(pfull or path.parent, 'pfull', model, 'climate')
    ap2pl = '-ap2pl,%s -invertlev -setmisstoc,0.0'  # too-small cloud mass
    options = {
        'pressure': ('-intlevel,%s', {}),
        'hybrid': ('-ml2pl,%s', {'ps': ps}),
        'generic': (ap2pl, {'ps': ps, 'pfull': pfull}),
        'generalized_height': (ap2pl, {'ps': ps, 'pfull': pfull}),
    }
    grids = _parse_descrips(cdo.zaxisdes(input=str(path)))
    grids = [kw for kw in grids if kw['zaxistype'] != 'surface']
    axis = ', '.join(grid['zaxistype'] for grid in grids)
    if len(grids) != 1:
        raise NotImplementedError(f'Missing or ambiguous vertical axis {axis!r} for {path.name!r}.')  # noqa: E501
    if axis not in options:
        raise NotImplementedError(f'Cannot interpolate vertical axis {axis!r} for {path.name!r}.')  # noqa: E501
    string = ', '.join(f'{k}: {v}' for k, v in grids[0].items() if not isinstance(v, np.ndarray))  # noqa: E501
    print('Current vertical levels:', string)
    print('Destination vertical levels:', ', '.join(map(str, levels.flat)))

    # Generate the dependencies
    # NOTE: Here -setzaxis is required to prevent merge conflicts due to 'lev' differing
    # between files (it is calculated as average level pressures at given timesteps).
    # Note the -merge or e.g. -ap2pl commands could fail when trying to combine climate
    # pfull data with time series of model level data... but so far no need for that.
    # NOTE: This adds 'ps' and 'pfull' to hybrid height coordinate variables and adds
    # 'ps' to hybrid sigma pressure coordinate variables that don't already include
    # surface pressure data in the files (so far this is just GFDL-ESM4). When searching
    # for data we ignore table, experiment, and ensemble because we only download
    # arbitrary control-like experiments with monthly time frequency due to limited
    # availability (e.g. some cmip6 models have only AERmon instead of Amon and
    # control-1950 instead of piControl, and all cmip5 models plus some cmip6 models
    # have only piControl data instead of abrupt4xCO2). See top of file for details.
    timesteps = cdo.ntime(input=str(path))[0]
    variables = ' '.join(cdo.showname(input=str(path))).split()
    method, dependencies = options[axis]
    atted, merge = [], ''
    for variable, dependency in dependencies.items():
        if variable in variables:  # already present in file
            continue
        if timesteps != '12':
            raise NotImplementedError(
                'Auto-adding dependencies is only supported for monthly climate data. '
                f'Instead input file {path.name!r} has {timesteps} timesteps.'
            )
        text = dependency.parent / f'{dependency.stem}.txt'
        merge = f'{merge} {dependency}'
        if rebuild or not dependency.is_file() or dependency.stat().st_size == 0:
            print(f'Generating pressure dependency file {dependency.name!r}.')
            pattern = f'{variable}_*{model}_*'  # search pattern
            if not model:
                raise ValueError(f'Model is required when searching for dependency {variable!r}.')  # noqa: E501
            files, *_ = glob_files(*search, pattern=pattern, project=project)
            if not files:
                raise ValueError(f'Glob {pattern!r} returned no results for path(s) {search}.')  # noqa: E501
            input = '[ ' + ' '.join(f'-selname,{variable} {file}' for file in files) + ' ]'  # noqa: E501
            repair_files(*files, printer=print)
            descrip = f'-ymonavg -mergetime {input}'.replace(f'{files[0].parent}/', '')
            print(f'Averaging {len(files)} files with {descrip}.')
            cdo.ymonavg(input=f'-mergetime {input}', output=str(dependency))
            _output_check(dependency, print)
        if variable == 'pfull':
            data = '\n'.join(cdo.zaxisdes(input=str(dependency)))
            open(text, 'w').write(data)
            merge = f'{merge} -setzaxis,{text}'  # applied to original file
            atted = [format(Atted('overwrite', 'long_name', 'lev', 'generalized height'))]  # noqa: E501

    # Add dependencies and interpolate
    # NOTE: Removing unneeded variables by leading chain with -delname,ps,orog,pfull
    # seemed to cause cdo > 1.9.1 and cdo < 2.0.0 to hang indefinitely. Not sure
    # why and not worth issuing bug report but make sure cdo is up to date.
    # NOTE: Here 'long_name="generalized height"' is required for ap2pl to detect
    # variables, but this triggers error when applying -setzaxis using the resulting
    # zaxisdes because 'generalized_height' is an invalid type (note applying
    # -setzaxis,axis to pfull instead of the data variable, whether or not the
    # zaxistype was changed, also triggers duplicate level issues). Solution is to
    # update long_name *after* the merge. See: https://code.mpimet.mpg.de/issues/10692
    # Also tried applying as part of chain but seems axis is determined before stepping
    # through the chain so this still fails to detect data variables.
    levels_equal = lambda axis, array: (
        axis == 'pressure' and array.size == levels.size
        and np.all(np.isclose(array, levels))
    )
    if not merge:
        print(f'File {path.name!r} already has required dependencies.')
        shutil.copy(path, deps)
    else:
        print(f'Adding pressure dependencies with -merge {merge} {path}.')
        cdo.merge(input=f'{merge} {path}', output=str(deps))
        if atted:
            print('Preparing attributes for interpolation with:', *atted)
            nco.ncatted(input=str(deps), output=str(deps), options=atted)  # noqa: E501
        _output_check(deps, print)
    if levels_equal(axis, grids[0]['levels']):
        print(f'File {path.name!r} is already on standard vertical levels.')
        deps.rename(output)
    else:
        method = method % ','.join(map(str, np.atleast_1d(levels)))
        print(f'Vertically interpolating with -delname,ps,orog,pfull {method} {deps}.')
        cdo.delname(  # missing variables causes warning not error
            'ps,orog,pfull',
            input=f'{method} {deps}',
            output=str(output),
            options='-P 8',
        )
        _output_check(output, print)

    # Verify that interpolation was successful
    # NOTE: Some models use floats and have slight offsets for standard pressure levels
    # while others use integer, so important to use np.isclose() when comparing. Note
    # all pressure level models use Pa and are ordered with decreasing pressure.
    grids = _parse_descrips(cdo.zaxisdes(input=str(output)))
    grids = [kw for kw in grids if kw['zaxistype'] != 'surface']
    axis = ', '.join(grid['zaxistype'] for grid in grids)
    if levels_equal(axis, grids[0]['levels']):
        print('Verified correct output file levels.')
        deps.unlink(missing_ok=True)  # remove deps file
    else:
        result = ', '.join(map(str, np.atleast_1d(grids[0]['levels'])))
        output.unlink(missing_ok=True)  # retain deps file for debugging
        raise RuntimeError(f'Incorrect output axis {axis!r} or levels {result}.')
    return output


def standardize_horizontal(
    path,
    output=None,
    gridspec=None,
    method=None,
    weights=None,
    model=None,
    rebuild=False,
    overwrite=False,
    printer=None
):
    """
    Create a file on a standardized horizontal grid from the input file
    with arbitrary native grid. Uses `cdo.genmethod` and `cdo.remap`.

    Parameters
    ----------
    path : path-like
        The input file.
    output : path-like, optional
        The output name and/or location.
    gridspec : str, default: 'r72x36'
        The standard gridspec. Default is uniform 5 degree resolution.
    method : str, default: 'con'
        The `cdo` remapping command suffix.
    weights : path-like, optional
        The location or name for weights data.
    model : str, optional
        The model to use in the default weights name.
    rebuild : bool, optional
        Whether to rebuild the weight files.
    overwrite : bool, default: True
        Whether to overwrite existing files or skip them.
    printer : callable, default: `print`
        The print function.

    Note
    ----
    For details see the SCRIPS repository and Jones 1999, Monthly Weather Review.
    Code: https://github.com/SCRIP-Project/SCRIP
    Paper: https://doi.org/10.1175/1520-0493(1999)127%3C2204:FASOCR%3E2.0.CO;2
    """
    # Initial stuff
    print = printer or builtins.print
    path = Path(path).expanduser()
    output = _output_path(output or path.parent, path.stem, 'standard-horizontal')
    method = method or 'con'
    method = method if method[:3] == 'gen' else 'gen' + method
    gridspec = STANDARD_GRIDSPEC if gridspec is None else gridspec
    if not overwrite and output.is_file() and output.stat().st_size > 0:
        print(f'Output file already exists: {output.name!r}.')
        return output

    # Parse input horizontal grid
    # NOTE: Sometimes cdo detects dummy 'generic' grids of size 1 or 2 indicating
    # scalar quantities or bounds. Try to ignore these grids.
    result = cdo.griddes(input=str(path))
    grids = [kw for kw in _parse_descrips(result) if kw['gridsize'] > 2]
    if not grids:
        raise NotImplementedError(f'Missing horizontal grid for {path.name!r}.')
    if len(grids) > 1:
        raise NotImplementedError(f'Ambiguous horizontal grids for {path.name!r}: ', ', '.join(grid['gridtype'] for grid in grids))  # noqa: E501
    grid = grids[0]
    string = ', '.join(f'{k}: {v}' for k, v in grid.items() if not isinstance(v, np.ndarray))  # noqa: E501
    print('Current horizontal grid:', string)
    print('Destination horizontal grid:', gridspec)

    # Generate weights
    # NOTE: Grids for same model but different variables are sometimes different
    # most likely due to underlying staggered grids. Account for this by including
    # the grid spec of the source file in the default weight file name.
    grid_current = grid.get('gridtype', 'unknown')[:1] or 'u'
    if 'xsize' in grid and 'ysize' in grid:
        grid_current += 'x'.join(str(grid[s]) for s in ('xsize', 'ysize'))
    else:
        grid_current += str(grid.get('gridsize', 'XXX'))
    weights = _output_path(weights or path.parent, method, model, grid_current)
    if rebuild or not weights.is_file() or weights.stat().st_size == 0:
        print(f'Generating destination grid weights {weights.name!r}.')
        getattr(cdo, method)(
            gridspec,
            input=str(path),
            output=str(weights),
            options='-P 8',
        )
        _output_check(weights, print)

    # Interpolate to horizontal grid
    # NOTE: Unlike vertical standardization there are fewer pitfalls here (e.g.
    # variables getting silently skipped). So testing output griddes not necessary.
    opts = ('gridtype', 'xstart', 'ystart', 'xsize', 'ysize')
    key_current = tuple(grid.get(key, None) for key in opts)
    key_destination = ('lonlat', 0, -90, *map(int, gridspec[1:].split('x')))
    if key_current == key_destination:
        print(f'File {path.name!r} is already on destination grid.')
        shutil.copy(path, output)
    else:
        print(f'Horizontally interpolating with -remap,{gridspec},{weights} {path}.')
        cdo.remap(
            f'{gridspec},{weights}',
            input=str(path),
            output=str(output),
            options='-P 8',
        )
        _output_check(output, print)
    return output


def summarize_descrips(*paths, facets=None, **constraints):
    """
    Print descriptions of horizontal grids and vertical levels for input files.

    Parameters
    ----------
    paths : path-like, optional
        The folder(s).
    facets : str, optional
        The facets to group by.
    **constraints
        Passed to `Printer` and `Database`.
    """
    facets = facets or KEYS_SUMMARIZE
    print = Printer('summary', 'descrips')
    print('Generating database.')
    files, *_ = glob_files(*paths, project=constraints.get('project'))
    database = Database(files, facets, **constraints)
    grids, zaxes = {}, {}
    for file, *_ in database:  # select first file from every file list
        key = '_'.join(file.name.split('_')[:5])
        try:
            grid, zaxis = cdo.griddes(input=str(file)), cdo.zaxisdes(input=str(file))  # noqa: E501
        except CDOException:
            print(f'WARNING: Failed to read file {file}.')  # message printed
            continue
        grid, zaxis = map(_parse_descrips, (grid, zaxis))
        grids[key], zaxes[key] = grid, zaxis
        print(f'\n{file.name}:')
        for kw in (*grid, *zaxis):
            string = ', '.join(
                f'{k}: {v}' for k, v in kw.items()
                if not isinstance(v, np.ndarray)
            )
            print(string, end='\n\n')
            if kw in zaxis and 'vctsize' not in kw and kw['size'] > 1:
                warnings.warn('Vertical axis does not have vertices.')
    return grids, zaxes


def summarize_processed(*paths, facets=None, **constraints):
    """
    Compare the output netcdf files in the folder(s) ending with ``'climate'``
    or ``'series'`` to the input files in the same folder(s).

    Parameters
    ----------
    paths : path-like, optional
        The folder(s) containing input and output files.
    facets : str, optional
        The facets to group by in the database.
    **constraints
        Passed to `Printer` and `Database`.
    """
    # Iterate over dates
    facets = facets or KEYS_SUMMARIZE
    print = Printer('summary', 'processed', **constraints)
    glob, *_ = glob_files(*paths, project=constraints.get('project'))
    key = lambda pair: ('4xCO2' in pair[0], 'series' in pair[1], pair[1])
    opts = ('climate', 'series')  # recognized output file suffixes
    pairs = set((_item_facets['experiment'](file), _item_dates(file)) for file in glob)
    pairs = sorted((pair for pair in pairs if any(o in pair[1] for o in opts)), key=key)
    interval = 500
    database = Database(glob, facets, **constraints)
    for experiment, date in pairs:
        # Print the raw summary files
        constraints['experiment'] = experiment
        print(f'Finished output for {experiment} {date}.')
        database_date = copy.deepcopy(database)
        for i, files in enumerate(database_date):
            i % interval or print(f'Files: {i} out of {len(database_date)}')
            files[:] = [file for file in files if _item_dates(file) == date]
        database_date.summarize(missing=True, printer=print)

        # Print the files with missing inputs
        # NOTE: Skip groups where we did not attempt to get output e.g.
        # time series of non-feedback variables.
        print(f'Missing inputs for {experiment} {date}.')
        missing_inputs = copy.deepcopy(database)
        for i, files in enumerate(missing_inputs):
            i % interval or print(f'Files: {i} out of {len(missing_inputs)}')
            if not any(_item_dates(file) == date for file in files):
                files.clear()  # current date is not present (covered below)
            elif not all(any(o in _item_dates(file) for o in opts) for file in files):
                files.clear()  # inputs are present (i.e. not missing)
        for group, data in tuple(missing_inputs.items()):
            if all(files for files in data.values()):
                del missing_inputs.database[group]
        missing_inputs.summarize(missing=False, printer=print)

        # Print the files with missing outputs
        # NOTE: Skip groups where we did not attempt to get output e.g.
        # time series of non-feedback variables.
        print(f'Missing output for {experiment} {date}.')
        missing_outputs = copy.deepcopy(database)
        for i, files in enumerate(missing_outputs):
            i % interval or print(f'Files: {i} out of {len(missing_outputs)}')
            if any(_item_dates(file) == date for file in files):
                files.clear()  # current date is present (i.e. not missing)
            elif all(any(o in _item_dates(file) for o in opts) for file in files):
                files.clear()  # inputs are not present (covered above)
        for group, data in tuple(missing_outputs.items()):
            if all(files for files in data.values()):
                del missing_outputs.database[group]
        missing_outputs.summarize(missing=False, printer=print)


def summarize_ranges(*paths, facets=None, **constraints):
    """
    Print minimum and maximum ranges for the processed climate and time series
    files. Files that fail the tests usually justify investigating the data
    downloaded from esgf and adding them to the `facets.py` corrupt files list.

    Parameters
    ----------
    paths : path-like, optional
        The folder(s).
    facets : str, optional
        The facets to group by.
    **constraints
        Passed to `Printer` and `Database`.
    """
    facets = facets or KEYS_SUMMARIZE
    print = Printer('summary', 'ranges', **constraints)
    print('Generating database.')
    files, *_ = glob_files(*paths, project=constraints.get('project'))
    database = Database(files, facets, **constraints)
    keys = (path.name.split('_')[:2] for paths in database for path in paths)
    ranges = {key: _validate_ranges(*key) for key in sorted(set(map(tuple, keys)))}
    for path in (path for paths in database for path in paths):
        # Load test data
        print(f'\n{path}:')
        invalids = {'identical': None, 'pointwise': None, 'averages': None}
        variable, table, *_ = path.name.split('_')
        data = xr.open_dataset(path, use_cftime=True)
        data = data.get(variable, None)
        if data is None:
            print(f'WARNING: Variable {variable!r} is missing from file.')
            print('Identical check: MISSING')
            print('Pointwise check: MISSING')
            print('Average check: MISSING')
            continue
        test = data.mean('time') if 'time' in data.sizes else data
        min_, max_ = test.min().item(), test.max().item()
        pmin, pmax, amin, amax = ranges[variable, table]
        skip_range = variable == 'rsut' and 'MCM-UA-1-0' in path.stem
        skip_identical = variable == 'rsdt' and 'MCM-UA-1-0' in path.stem

        # Pointwise check
        # NOTE: Previously screwed up the drift correction step so files had uniform
        # values everywhere. However some downloaded files also have this issue.
        if not skip_identical:
            if b := test.size > 1 and np.isclose(min_, max_) and variable != 'rsdt':
                print(
                    f'WARNING: Variable {variable!r} has the identical value {min_} '
                    'across entire domain.'
                )
            invalids['identical'] = b
        if not skip_range and (pmin is not None or pmax is not None):
            b = False  # declare PASSED for negative values (we simply multiply by -1)
            if pmin and pmax and np.sign(pmin) == np.sign(pmax) and min_ >= -pmax and max_ <= -pmin:  # noqa: E501
                print(
                    f'WARNING: Pointwise {variable!r} range ({min_}, {max_}) '
                    f'is inside negative of valid cmip range ({pmin}, {pmax}).'
                )
            elif b := pmin is not None and min_ < pmin or pmax is not None and max_ > pmax:  # noqa: E501
                print(
                    f'WARNING: Pointwise {variable!r} range ({min_}, {max_}) '
                    f'is outside valid cmip range ({pmin}, {pmax}).'
                )
            invalids['pointwise'] = b

        # Absolute global average check
        # NOTE: Sometimes get temperature outside of this range for models with
        # different stratospheres so use annual mean for more conservative result.
        if not skip_range and (amin is not None or amax is not None):
            mean = test.mean('lon')  # zonal mean data
            mask = ~mean.isnull()  # non-nan data
            clat = np.cos(test.lat * np.pi / 180)
            mean = (mean * clat).sum('lat', min_count=1) / (mask * clat).sum('lat')
            min_, max_ = np.abs(mean.min().item()), np.abs(mean.max().item())
            b = False  # declare PASSED for negative values (we simply multiply by -1)
            if amin and amax and np.sign(amin) == np.sign(amax) and min_ >= -amax and max_ <= -amin:  # noqa: E501
                print(
                    f'WARNING: Global average {variable!r} range ({min_}, {max_}) '
                    f'is inside negative of valid cmip range ({pmin}, {pmax}).'
                )
            elif b := amin is not None and min_ < amin or amax is not None and max_ > amax:  # noqa: E501
                print(
                    f'WARNING: Global average {variable!r} range ({min_}, {max_}) '
                    f'is outside valid cmip range ({amin}, {amax}).'
                )
            invalids['averages'] = b

        # Print test results
        messages = {False: 'PASSED', True: 'FAILED', None: 'SKIPPED'}
        print('Identical check:', messages[invalids['identical']])
        print('Pointwise check:', messages[invalids['pointwise']])
        print('Average check:', messages[invalids['averages']])
