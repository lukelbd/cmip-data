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

from . import cdo, nco, Atted, Rename, CDOException
from .facets import (
    _file_parts,
    _get_ranges,
    _glob_files,
    _item_dates,
    _item_years,
    FacetPrinter,
    FacetDatabase,
    GRID_CMIP,
    LEVELS_CMIP5,
    LEVELS_CMIP6,
    FACETS_FOLDER,
    FACETS_SUMMARY,
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
        tmps = sorted(path.parent.glob(path.name + '.pid*.nc*.tmp'))  # nco commands
        for tmp in tmps:
            tmp.unlink(missing_ok=True)
        message += f'Removed {len(tmps)} temporary nco files.' if tmps else ''
        raise RuntimeError(message)
    print(f'Created output file {path.name!r}.')


def _output_path(path=None, *parts):
    """
    Generate an output path from an input path.

    Parameters
    ----------
    path : path-like
        The input path.
    *parts : str optional
        The underscore-separated components to use for the default path.
    """
    if path:
        path = Path(path).expanduser()
    else:
        path = Path()
    if path.is_dir():
        parts = tuple(filter(None, parts))  # delete empty components
        if parts:
            path = path / ('_'.join(parts) + '.nc')
        else:
            raise ValueError('Path was not provided and default parts are missing.')
    return path


def _parse_time(
    years=150, climate=True, nodrift=False, skipna=False, **constraints
):
    """
    Parse variables related to time standardization and create a file name suffix.

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
    name = '-'.join((
        f'{year1:04d}-{year2:04d}',
        ('series', 'climate')[climate],
        *(('nodrift',) if nodrift else ('nodrift',) if nodrift else ()),
    ))
    kwargs = {
        'years': years,
        'climate': climate,
        'nodrift': nodrift,
        'skipna': skipna,
    }
    if constraints.pop('constraints', False):
        return name, kwargs, constraints
    elif not constraints:
        return name, kwargs
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
    # without globals). Use this instead of nc4 for consistency with the rest of
    # workflow. See: http://nco.sourceforge.net/nco.html#Filters-for-ncks
    # NOTE: Here use unique type identifiers to consistently identify variables.
    # Otherwise common to match e.g. variable appearences in 'formula' attributes.
    # Also try to match e.g. 'time = UNLIMITED // (100 currently)' dimension sizes
    # and trim e.g. '1.e+20f' numeric type indicator suffixes in attribute values.
    # See: https://docs.unidata.ucar.edu/nug/current/_c_d_l.html#cdl_data_types
    path = Path(path).expanduser()
    info = nco.ncks(input=str(path), options=['--cdl', '-m']).decode()
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
    return info, sizes, names, attrs


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


def process_files(
    *paths,
    output='~/data',
    constants='~/data/cmip-constants',
    facets=None,
    vertical=True,
    horizontal=True,
    overwrite=False,
    printer=None,
    dryrun=False,
    **kwargs,
):
    """
    Average and standardize files downloaded with a wget script.

    Parameters
    ----------
    *paths : path-like
        The input path(s) for the raw data.
    output : path-like, optional
        The output directory for processed data. Subfolders are created here.
    constants : path-like, optional
        The output directory for constants.
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
    printer : callable, default: `print`
        The print function.
    dryrun : bool, optional
        Whether to only print time information and exit.
    **kwargs
        Passed to `standardize_time`, `standardize_vertical`, `standardize_horizontal`.
    **constraints
        Passed to `FacetPrinter` and `FacetDatabase`.
    """
    # Find files and restrict to unique constraints
    # NOTE: Here we only constrain search to the project, which is otherwise not
    # indicated in native netcdf filenames. Similar approach to filter_script().
    levels, search = kwargs.pop('levels', None), kwargs.pop('search', None)
    gridspec, method = kwargs.pop('gridspec', None), kwargs.pop('method', None)
    searches = (search,) if isinstance(search, (str, Path)) else tuple(search or ())
    dates, kwargs, constraints = _parse_time(constraints=True, **kwargs)
    print = printer or FacetPrinter('process', dates, **constraints)
    files, _ = _glob_files(*paths, project=constraints.get('project'))
    facets = facets or FACETS_FOLDER
    database = FacetDatabase(files, facets, **constraints)
    constants = Path(constants).expanduser()
    output = Path(output).expanduser()

    # Initialize cdo and process files.
    # NOTE: Here perform horizontal interpolation before vertical interpolation because
    # latter adds NaNs and extrapolated data so should delay that step until very end.
    # NOTE: Currently 'pfull' data is stored in separate 'data-dependencies' folder
    # but 'ps' data is stored in normal folders since it is used for time-varying
    # vertical integration of kernels and is always available.
    print(f'Input files ({len(database)}):')
    print(*(f'{key}: ' + ' '.join(opts) for key, opts in database.constraints.items()), sep='\n')  # noqa: E501
    print()
    outs = []
    for files in database:
        # Initial stuff
        folder = output / files[0].parent.name
        folder.mkdir(exist_ok=True)
        model = _file_parts['model'](files[0])
        variable = _file_parts['variable'](files[0])
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
        kw = {'overwrite': overwrite, 'printer': print}
        print('Output:', '/'.join((out.parent.name, out.name)))

        # Repair files and standardize time
        _print_error = lambda error: print(
            ' '.join(traceback.format_exception(None, error, error.__traceback__))
        )
        _remove_temps = lambda: tuple(
            path.unlink(missing_ok=True) for path in (time, vert, hori)
        )
        try:
            repair_files(*files, dryrun=dryrun, printer=print)
        except Exception as error:
            _print_error(error)
            print('Warning: Failed to standardize attributes.\n')
            continue
        updated = not overwrite and out.is_file() and (
            all(file.stat().st_mtime <= out.stat().st_mtime for file in files)
        )
        try:
            standardize_time(*files, output=time, dryrun=dryrun or updated, **kwtime, **kw)  # noqa: E501
        except Exception as error:
            _print_error(error)
            print('Warning: Failed to standardize temporally.\n')
            continue
        if updated:
            print('Skipping (up to date)...\n')
            continue
        if dryrun:
            print('Skipping (dry run)...\n')
            continue

        # Standardize horizontal grid and vertical levels
        info, sizes, names, attrs = _parse_dump(files[0])
        constants.mkdir(exist_ok=True)
        if not vertical or 'lev' not in attrs:
            time.replace(vert)
        else:
            try:
                standardize_vertical(time, output=vert, **kwvert, **kw)
            except Exception as error:
                _print_error(error)
                print('Warning: Failed to standardize vertically.\n')
                continue
        if not horizontal:
            vert.replace(hori)
        else:
            try:
                standardize_horizontal(vert, output=hori, **kwhori, **kw)
            except Exception as error:
                _print_error(error)
                print('Warning: Failed to standardize horizontally.\n')
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
    # NOTE: For cdo to successfully parse hybrid coordinates need vertices. So
    # critical to include 'lev:bounds = "lev_bnds"' and have "lev_bnds" attrs
    # that point to the "a_bnds", "b_bnds" hybrid level interfaces.
    print = printer or builtins.print
    pure_sigma = 'atmosphere_sigma_coordinate'
    hybrid_pressure = 'atmosphere_hybrid_sigma_pressure_coordinate'
    hybrid_pressure_atted = []
    for name in ('lev', 'lev_bnds'):
        end = '_bnds' if '_' in name else ''
        bnd = ' bounds' if '_' in name else ''
        idx = '+1/2' if '_' in name else ''
        hybrid_pressure_atted.extend(
            att.prn_option() for att in (
                Atted('overwrite', 'long_name', name, f'atmospheric model level{bnd}'),  # noqa: E501
                Atted('overwrite', 'standard_name', name, hybrid_pressure),
                Atted('overwrite', 'formula', name, 'p = ap + b*ps'),
                Atted('overwrite', 'formula_terms', name, f'ap: ap{end} b: b{end} ps: ps'),  # noqa: E501
            )
        )
        hybrid_pressure_atted.extend(
            att.prn_option().replace('" "', '') for term in ('b', 'ap') for att in (
                Atted('delete', ' ', f'{term}{end}'),  # remove all attributes
                Atted('overwrite', 'long_name', f'{term}{end}', f'vertical coordinate formula term: {term}(k{idx})'),  # noqa: E501
            )
        )

    # Declare attribute removals for interpreting hybrid height coordinates as a
    # generalized height coordinate in preparation for cdo ap2pl pressure interpolation.
    # NOTE: The cdo ap2pl operator standard also requires long_name="generalized height"
    # but this causes issues when applying setzaxis so currently delay this attribute
    # edit until standardize_vertical. See: https://code.mpimet.mpg.de/issues/10692
    # NOTE: Using -mergetime with files with different time 'bounds' or 'climatology'
    # can create duplicate 'time_bnds' and 'climatolog_bnds' variables, and using
    # -merge can trigger bizarre incorrect 'reserved variable zaxistype' errors. So
    # delete these attributes when detected on candidate 'pfull' data.
    hybrid_height = 'atmosphere_hybrid_height_coordinate'
    hybrid_height_atted = [
        att.prn_option() for name in ('lev', 'lev_bnds') for att in (
            Atted('overwrite', 'standard_name', name, 'height'),
            Atted('delete', 'long_name', name),
            Atted('delete', 'units', name),
            Atted('delete', 'formula', name),
            Atted('delete', 'formula_terms', name),
        )
    ]
    hybrid_height_atted_extra = [
        att.prn_option() for att in (
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
        info, sizes, names, attrs = _parse_dump(path)
        lev_std = attrs.get('lev', {}).get('standard_name', '')
        lev_bnds_key = attrs.get('lev', {}).get('bounds', 'lev_bnds')
        lev_bnds_std = attrs.get(lev_bnds_key, {}).get('standard_name', '')
        lev_bnds_formula = attrs.get(lev_bnds_key, {}).get('formula_terms', '')
        var_extra = ('average_T1', 'average_T2', 'average_DT')
        if any(s in attrs for s in var_extra):
            print(
                'Warning: Repairing GFDL-like inclusion of unnecessary '
                f'variable(s) for {path.name!r}:', *var_extra
            )
            if not dryrun:
                remove = ['-x', '-v', ','.join(var_extra)]
                nco.ncks(input=str(path), output=str(path), options=remove)
        if any('formula_term' in kw for kw in attrs.values()):
            rename = [Rename('a', {'.formula_term': 'formula_terms'}).prn_option()]
            print(
                'Warning: Repairing GFDL-like misspelling of formula_terms '
                + f'attribute(s) for {path.name!r}:', *rename
            )
            if not dryrun:
                nco.ncrename(input=str(path), output=str(path), options=rename)
        if 'presnivs' in sizes:
            rename = [
                Rename(c, {'presnivs': 'lev'}).prn_option()
                for c, src in (('d', sizes), ('v', attrs)) if 'presnivs' in src
            ]
            print(
                'Warning: Repairing IPSL-like non-standard vertical axis name '
                + f'presnivs for {path.name!r}:', *rename
            )
            if not dryrun:
                nco.ncrename(input=str(path), output=str(path), options=rename)
        if 'lev' in attrs and attrs['lev'].get('axis', '') != 'Z':
            atted = [Atted('overwrite', 'axis', 'lev', 'Z').prn_option()]
            print(
                'Warning: Repairing CNRM-like missing level variable axis '
                + f'attribute for {path.name!r}:', *atted
            )
            if not dryrun:
                nco.ncatted(input=str(path), output=str(path), options=atted)

        # Handle IPSL models with weird extra dimension on coordinates. This has
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
                'Warning: Repairing IPSL-like hybrid coordinates with unusual missing '
                + f'attributes for {path.name!r}:', math, ' '.join(atted)
            )
            if not dryrun:
                nco.ncap2(input=str(path), output=str(path), options=['-s', math])
                nco.ncatted(input=str(path), output=str(path), options=atted)
                nco.ncks(input=str(path), output=str(path), options=['-x', '-v', 'klevp1'])  # noqa: E501

        # Handle FGOALS models with pure sigma coordinates. Currently cdo cannot
        # handle them so translate attributes so they are recdgnized as hybrid.
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
                'Warning: Converting FGOALS-like sigma coordinates to hybrid sigma '
                f'pressure coordinates for {path.name!r}:', math, ',', ' '.join(atted)
            )
            if not dryrun:
                nco.ncap2(input=str(path), output=str(path), options=['-s', math])
                nco.ncatted(input=str(path), output=str(path), options=atted)
                nco.ncks(input=str(path), output=str(path), options=['-x', '-v', 'ptop'])  # noqa: E501

        # Handle CNRM models with hybrid coordinate boundaries permuted and a
        # lev_bounds formula_terms pointing incorrectly to the actual variables.
        # NOTE: Previously tried ncpdq for permutations but was insanely slow
        # and would eventualy fail due to some memory allocation error. For
        # some reason ncap2 permute (while slow) does not implode this way.
        if all(s in attrs and s not in lev_bnds_formula for s in ('ap_bnds', 'b_bnds')):
            math = repr(
                'b_bnds = b_bnds.permute($lev, $bnds); '
                'ap_bnds = ap_bnds.permute($lev, $bnds); '
                f'{lev_bnds_key}@formula_terms = "ap: ap_bnds b: b_bnds ps: ps"; '
            )
            print(
                'Warning: Repairing CNRM-like incorrect formula terms attribute '
                f'for {path.name!r}: ', math,
            )
            if not dryrun:
                nco.ncap2(input=str(path), output=str(path), options=['-s', math])

        # Handle CESM2 models with necessary attributes on only
        # coordinate bounds variables rather than coordinate centers.
        # NOTE: These models also sometimes have issues where levels are inverted
        # but that is handled inside standardize_time() below.
        if lev_std != hybrid_pressure and lev_bnds_std == hybrid_pressure:
            atted = [
                Atted('overwrite', key, 'lev', new).prn_option()
                for key, old in attrs[lev_bnds_key].items()
                if (new := old.replace(' bounds', '').replace('_bnds', ''))
            ]
            print(
                'Warning: Repairing CESM-like hybrid coordinates with missing level '
                + f'center attributes for {path.name!r}:', ' '.join(atted)
            )
            if not dryrun:
                nco.ncatted(input=str(path), output=str(path), options=atted)

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
                'Warning: Converting ACCESS-like hybrid height coordinates to '
                f'generalized height coordinates for {path.name!r}:', ' '.join(atted)
            )
            if not dryrun:
                var_extra = ('a', 'b', 'a_bnds', 'b_bnds', 'orog')
                var_extra += ('time_bnds', 'climatology_bnds') if 'pfull' in attrs else ()  # noqa: E501
                remove = ['-x', '-v', ','.join(var_extra)]
                nco.ncatted(input=str(path), output=str(path), options=atted)
                nco.ncks(input=str(path), output=str(path), options=remove)

        # Handle MCM-UA-1-0 models with messed up longitude/latitude bounds that have
        # non-global coverage and cause error when generating weights with 'cdo gencon'.
        # Also remove unneeded 'areacella' that most likely contains incorrect weights
        # (seems all other files reference this in cell_measures but excludes variable).
        # This is the only horitontal interpolation-related correction required.
        # NOTE: Verified only this model uses 'longitude' and 'latitude' (although can
        # use either full name or abbreviated for matching bounds). Try the following:
        # for f in cmip[56]-picontrol-amon/cl_Amon*; do m=$(echo $f | cut -d_ -f3);
        # [[ " ${models[*]} " =~ " $m "  ]] && continue || models+=("$m");
        # echo "$m: "$(ncdimlist "$f" | tail -n +2 | xargs); done; unset models
        lon_bnds = attrs.get('longitude', {}).get('bounds', None)
        lat_bnds = attrs.get('latitude', {}).get('bounds', None)
        if (
            lon_bnds and lat_bnds and 'longitude' in sizes and 'latitude' in sizes
            and any('bounds_status' not in attrs[s] for s in ('longitude', 'latitude'))
        ):
            math = repr(
                f'{lon_bnds}(-1, 1) = {lon_bnds}(0, 0) + 360.0; '
                f'{lat_bnds}(0, 0) = -90.0; '
                f'{lat_bnds}(-1, 1) = 90.0; '
                f'longitude@bounds_status = "repaired"; '
                f'latitude@bounds_status = "repaired"; '
            )
            print(
                'Warning: Converting MCM-UA-like issue with non-global longitude and '
                f'latitude bounds for {path.name!r}:', math
            )
            if not dryrun:  # first flag required because is member of cell_measures
                remove = ['-C', '-x', '-v', 'areacella']
                nco.ncap2(input=str(path), output=str(path), options=['-s', math])
                nco.ncks(input=str(path), output=str(path), options=remove)

        # Handle ACCESS1-0 and ACCESS1-3 pfull files that all have 12 timesteps
        # but have the same time value (obviously intended to be a monthly series).
        # This is the only time coordinate related correction required.
        # NOTE: The year in these files is 444 which is a leap year in proleptic
        # gregorian calendar, so use 29 days for month of february.
        if (
            'pfull' in names and sizes.get('time', 1) == '12'
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
                'Warning: Repairing ACCESS-like 12 timesteps with identical months by '
                'adding one month to each step:', math
            )
            if not dryrun:
                nco.ncap2(input=str(path), output=str(path), options=['-s', math])


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
    # NOTE: This includes kludge for individual CESM2-WACCM-FV2 files with incorrect
    # (negative) levels. Must invert or else cdo averages upper with lower levels.
    # After invert cdo raises a level mismatch warning but results will be correct.
    years = tuple(map(_item_years, paths))
    ymin, ymax = min(ys[0] for ys in years), max(ys[1] for ys in years)
    print(f'Initial year range: {ymin}-{ymax} ({len(years)} files)')
    rmin, rmax = kwtime.pop('years')  # relative years
    ymin, ymax = ymin + rmin, ymin + rmax - 1  # e.g. (0, 50) is 49 years
    ignore = lambda line: not re.search('(warning|error)', line)  # ignore messages
    inputs = {}
    for ys, path in zip(years, paths):
        levs = []
        if 'CESM' in path.name:  # slow so only use where this is known problem
            levs = ' '.join(filter(ignore, cdo.showlevel(input=str(path)))).split()
        if invert := ('-invertlev ' if np.any(np.array(levs, dtype=float) < 0) else ''):
            dryrun or print(f'Warning: Inverting level axis for {path.name!r}.')
        if ys[0] > ymax or ys[1] < ymin:
            continue
        y0, y1 = max((ys[0], ymin)), min((ys[1], ymax))
        sel = '' if ys[0] >= ymin and ys[1] <= ymax else f'-selyear,{y0}/{y1} '
        inputs[f'{sel}{invert}{path}'] = (ys[0], ys[1])  # test file years later
    print(f'Filtered year range: {ymin}-{ymax} ({len(inputs)} files)')
    if not inputs:
        message = f'No files found within requested range: {ymin}-{ymax}'
        if dryrun:
            print(f'Warning: {message}')
        else:
            raise RuntimeError(message)

    # Fitler intersecting ranges and print warning for incomplete range
    # NOTE: FIO-ESM-2-0 control psl returned 300-399 and 400-499 in addition to 301-400
    # and 401-500, CAS-ESM2-0 rsus returned 001-600 control and 001-167 abrupt in
    # addition to standard 001-550 and 001-150, and GFDL-CM4 abrupt returned 000-010
    # in addition to 000-100. The below block should handle these situations.
    ranges = []
    for input, (y0, y1) in tuple(inputs.items()):
        for other, (z0, z1) in tuple(inputs.items()):
            if input == other or input not in inputs:
                continue  # e.g. already deleted
            message = f'Skipping file years {y0}-{y1} in presence of %s file years {z0}-{z1}.'  # noqa: E501
            if z0 <= y0 <= y1 <= z1:
                print('Warning: ' + message % 'superset')
                del inputs[input]
            if y1 - y0 > 1 and y0 == z0 + 1 and y1 == z1 + 1:
                print('Warning: ' + message % 'offset-by-one')
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
            print(f'Warning: {message}')
        else:
            raise RuntimeError(message)
    if dryrun:
        return  # only print time information

    # Generate trend files for drift correction
    # NOTE: Important to use annual averages rather than monthly averages to help
    # reduce probability of spurious trends and for consistency with prevailing
    # literature recommendation. Then simply scale slopes by 12 (see py notebook).
    arg = None
    input = '[ ' + ' '.join(input for input in inputs) + ' ]'
    equal = 'equal=false'  # account for different days in month
    enlarge = f'-enlarge,{paths[0]}'
    average = 'ymonmean' if kwtime.pop('skipna') else 'ymonavg'
    if not kwtime.pop('nodrift'):
        if kwtime.pop('climate'):
            method = average
            prefix, suffix = '-mergetime ', ''
        else:
            method = 'mergetime'
            prefix = suffix = ''
    else:
        offset = _output_path(offset or paths[0].parent, variable, model, 'offset')
        slope = _output_path(slope or paths[0].parent, variable, model, 'slope')
        prefix = f'-add {enlarge} [ -add {offset} -mulc,{rmin} {slope} ] -mergetime '
        suffix = f' {enlarge} {offset} {enlarge} -divc,12 {slope}'
        if kwtime.pop('climate'):
            method = average
            prefix = f'-subtrend,{equal} {prefix}'
        else:
            arg = equal
            method = 'subtrend'
        if rebuild or any(
            not file.is_file() or file.stat().st_size == 0
            for file in (offset, slope)
        ):
            print(f'Generating trend files {offset.name!r} and {slope.name!r}.')
            names = tuple(path.name for path in paths)
            trend = '[ ' + ' '.join(f'-fldmean -yearmonmean {input}' for input in inputs) + ' ]'  # noqa: E501
            if any('control' not in name.lower() for name in names):
                raise ValueError('Cannot rebuild drift correction files from paths.')
            descrip = re.sub(r'/\S*/', '', f'-mergetime {trend}')
            print(f'Calling trend with {descrip}.')
            cdo.trend(equal, input=f'-mergetime {trend}', output=f'{offset} {slope}')
            _output_check(offset, print)
            _output_check(slope, print)

    # Consolidate times and apply drift corrections
    # NOTE: Cannot simply use 'detrend' for drift corrections because that also
    # removes mean and we want to *add* trends from the starting point (i.e. instead
    # of data(i) - (a + b * t(i)) want data(i) - ((a + b * t(i)) - (a + b * t(0)).
    # Testing revealed that cdo uses time step indices rather than actual time
    # values, so we simply need the detrended data plus the offset parameter. Also
    # important to add missing slope rise due to e.g. removing subtracting trend from
    # an abrupt climatology starting after the control run branch time at year 0.
    descrip = re.sub(r'/\S*/', '', f'{method} {prefix}{input}{suffix}')
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
        The path(s) to pass to `_glob_files` when searching for reference pressure data.
    project : str, optional
        The project to pass to `_glob_files` when searching for reference pressure data.
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
        pass
    elif project == 'cmip5':
        levels = LEVELS_CMIP5
    else:
        levels = LEVELS_CMIP6
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
    # NOTE: For some reason height must be oriented top-to-bottom (i.e. pressures
    # oriented low-to-height) before interpolating to pressure levels with ap2pl. This
    # is undocumented and error is silent. See: https://code.mpimet.mpg.de/issues/10692
    ps = _output_path(ps or path.parent, 'ps', model, 'climate')
    pfull = _output_path(pfull or path.parent, 'pfull', model, 'climate')
    options = {
        'pressure': ('-intlevel,%s', {}),
        'hybrid': ('-ml2pl,%s', {'ps': ps}),
        'generic': (opts := ('-ap2pl,%s -invertlev', {'ps': ps, 'pfull': pfull})),
        'generalized_height': opts,
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
            files, _ = _glob_files(*search, pattern=pattern, project=project)
            if not files:
                raise ValueError(f'Glob {pattern!r} returned no results for path(s) {search}.')  # noqa: E501
            input = '[ ' + ' '.join(f'-selname,{variable} {file}' for file in files) + ' ]'  # noqa: E501
            repair_files(*files, printer=print)
            descrip = re.sub(r'/\S*/', '', f'-ymonavg -mergetime {input}')
            print(f'Averaging {len(files)} files with {descrip}.')
            cdo.ymonavg(input=f'-mergetime {input}', output=str(dependency))
            _output_check(dependency, print)
        if variable == 'pfull':
            data = '\n'.join(cdo.zaxisdes(input=str(dependency)))
            open(text, 'w').write(data)
            merge = f'{merge} -setzaxis,{text}'  # applied to original file
            atted = [Atted('overwrite', 'long_name', 'lev', 'generalized height').prn_option()]  # noqa: E501

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
    # Also tried applying as aprt of chain but seems axis is determined before stepping
    # through the chain so this still fails to detect data variables.
    levels_equal = lambda axis, array: (
        axis == 'pressure' and array.size == levels.size
        and np.all(np.isclose(array, levels))
    )
    if not merge:
        print(f'File {path.name!r} already has required dependencies.')
        shutil.copy(path, deps)
    else:
        descrip = re.sub(r'/\S*/', '', f'-merge {merge}')
        print(f'Adding pressure dependencies with {descrip}.')
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
        print(f'Vertically interpolating with method {method}.')
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
    gridspec = GRID_CMIP if gridspec is None else gridspec
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
        print(f'Horizontally interpolating with grid weights {weights.name!r}.')
        cdo.remap(
            f'{gridspec},{weights}',
            input=str(path),
            output=str(output),
            options='-P 8',
        )
        _output_check(output, print)
    return output


def summarize_processed(*paths, facets=None, **constraints):
    """
    Print summary of the processed climate and time series files.

    Parameters
    ----------
    paths : path-like, optional
        The folder(s).
    facets : str, optional
        The facets to group by.
    **constraints
        Passed to `FacetPrinter` and `FacetDatabase`.
    """
    facets = facets or FACETS_SUMMARY
    print = FacetPrinter('summary', 'processed', **constraints)
    print('Generating database.')
    files, _ = _glob_files(*paths, project=constraints.get('project'))
    dates = sorted(set(map(_item_dates, files)))
    interval = 500
    database = FacetDatabase(files, facets, **constraints)
    for date in dates:
        if not date:  # e.g. temporary files with suffix like 'standard-horizontal'
            continue
        print(f'Partitioning outputs {date}.')
        database_date = copy.deepcopy(database)
        for i, files in enumerate(database_date):
            i % interval or print(f'Files: {i} out of {len(database_date)}')
            files[:] = [file for file in files if _item_dates(file) == date]
        database.summarize(message=f'Output {date}', printer=print)


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
        Passed to `FacetPrinter` and `FacetDatabase`.
    """
    facets = facets or FACETS_SUMMARY
    print = FacetPrinter('summary', 'descrips')
    print('Generating database.')
    files, _ = _glob_files(*paths, project=constraints.get('project'))
    database = FacetDatabase(files, facets, **constraints)
    grids, zaxes = {}, {}
    for file, *_ in database:  # select first file from every file list
        key = '_'.join(file.name.split('_')[:5])
        try:
            grid, zaxis = cdo.griddes(input=str(file)), cdo.zaxisdes(input=str(file))  # noqa: E501
        except CDOException:
            print(f'Warning: Failed to read file {file}.')  # message printed
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
        Passed to `FacetPrinter` and `FacetDatabase`.
    """
    facets = facets or FACETS_SUMMARY
    print = FacetPrinter('summary', 'ranges', **constraints)
    print('Generating database.')
    files, _ = _glob_files(*paths, project=constraints.get('project'))
    database = FacetDatabase(files, facets, **constraints)
    points, averages = _get_ranges()
    if not points or not averages:
        raise RuntimeError('Failed to load ranges from RANGES.txt')
    paths = (path for paths in database for path in paths)
    for path in paths:
        # Load test data
        print(f'\n{path}:')
        invalids = {'identical': None, 'pointwise': None, 'averages': None}
        variable, table, *_ = path.name.split('_')
        dataset = xr.open_dataset(path, use_cftime=True)
        if variable not in dataset:
            print(f'Warning: Variable {variable!r} is missing from file.')
            print('Identical check: MISSING')
            print('Pointwise check: MISSING')
            print('Average check: MISSING')
            continue
        test = data = dataset[variable]
        if 'time' in data.sizes:  # use more liberal test
            try:
                test = data.resample(time='AS').mean('time')
            except ValueError:  # e.g. non-monotonic
                pass

        # Pointwise check
        # NOTE: Previously screwed up the drift correction step so files had uniform
        # values everywhere. If uncorrected files trigger this block cloud be a
        # weird issue with downloaded data.
        min_, max_ = test.min().item(), test.max().item()
        if True:  # always run
            if b := test.size > 1 and np.isclose(min_, max_):
                print(
                    f'Warning: Variable {variable!r} has the identical value {min_} '
                    'across entire domain.'
                )
            invalids['identical'] = b
        pmin, pmax = points.get((variable, table), (None, None))
        if pmin is not None or pmax is not None:
            if b := pmin is not None and min_ < pmin or pmax is not None and max_ > pmax:  # noqa: E501
                print(
                    f'Warning: Pointwise {variable!r} range ({min_}, {max_}) '
                    f'is outside the valid cmip range ({pmin}, {pmax}).'
                )
            invalids['pointwise'] = b

        # Absolute global average check
        # NOTE: Sometimes get temperature outside of this range for models with
        # different stratospheres so use annual mean for more conservative result.
        amin, amax = points.get((variable, table), (None, None))
        if amin is not None or amax is not None:
            mean = test.mean('lon')  # zonal mean data
            mask = ~mean.isnull()  # non-nan data
            clat = np.cos(test.lat * np.pi / 180)
            mean = (mean * clat).sum('lat', min_count=1) / (mask * clat).sum('lat')
            min_, max_ = np.abs(mean.min().item()), np.abs(mean.max().item())
            if b := amin is not None and min_ < amin or amax is not None and max_ > amax:  # noqa: E501
                print(
                    f'Warning: Global average {variable!r} range ({min_}, {max_}) '
                    f'is outside the valid cmip range ({amin}, {amax}).'
                )
            invalids['averages'] = b

        # Print test results
        messages = {False: 'PASSED', True: 'FAILED', None: 'SKIPPED'}
        print('Identical check:', messages[invalids['identical']])
        print('Pointwise check:', messages[invalids['pointwise']])
        print('Average check:', messages[invalids['averages']])
