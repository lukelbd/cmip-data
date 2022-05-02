#!/usr/bin/env python3
"""
Process and standardize datasets downloaded from ESGF.
"""
import builtins
import os
import re
import shutil
import warnings
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
from cdo import Cdo, CDOException  # noqa: F401
from nco import Nco, NCOException  # noqa: F401
from nco.custom import Atted, Rename

from .download import (
    _glob_files,
    _init_printer,
    _line_parts,
    _line_years,
    _parse_constraints,
)

__all__ = [
    'process_files',
    'repair_files',
    'standardize_dependencies',
    'standardize_time',
    'standardize_horizontal',
    'standardize_vertical',
]

# Initialize cdo and nco bindings
# NOTE: Use 'conda install cdo' and 'conda install nco' for the command-line
# tools. Use 'conda install python-cdo' and 'conda install pynco' (or 'pip
# install cdo' and 'pip install pynco') for only the python bindings. Note for
# some reason cdo returns lists while nco returns strings with newlines, and for
# the signal catch fix must install from github with subdirectory using pip install
# 'git+https://github.com/lukelbd/cdo-bindings@fix-signals#subdirectory=python'.
# Also should use 'pip install git+https://github.com/nco/pynco.git' to fix Rename
# issue (although currently use prn_option() for debugging help anyway).
# NOTE: This requires cdo > 2.0.0 or cdo <= 1.9.1 (had trouble installing recent
# cdo in base environments so try uninstalling libgfortran4, libgfortran5, and
# libgfortran-ng then re-installing cdo and the other packages removed by that
# action). In cdo 1.9.9 ran into weird bug where the delname line before ap2pl caused
# infinite hang for ap2pl. Also again note cdo is faster and easier to use than nco.
# Compare 'time ncremap -a conserve -G latlon=36,72 tmp.nc tmp_nco.nc' to 'time cdo
# remapcon,r72x36 tmp.nc tmp_cdo.nc'. Also can use '-t 8' for nco and '-P 8' for cdo
# for parallelization but still makes no difference (note real time speedup is
# marginal and user time should increase significantly, consistent with ncparallel
# results). The only exception seems to be attribute and name management.
cdo = Cdo(env={**os.environ, 'CDO_TIMESTAT_DATE': 'first'}, options=['-s'])
nco = Nco()  # overwrite is default, and see https://github.com/nco/pynco/issues/56

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
GRID_SPEC = 'r360x180'  # 1.0 resolution
GRID_SPEC = 'r180x90'  # 2.0 resolution
GRID_SPEC = 'r144x72'  # 2.5 resolution
GRID_SPEC = 'r72x36'  # 5.0 resolution

# Vertical grid constants
# NOTE: Some models use floats and ahve slight offsets for standard pressure levels
# while others use integer, so important to use np.isclose() when comparing. Note all
# pressure level models use Pa and are ordered with decreasing pressure.
# NOTE: Some models have missing 'bounds' attribute which causes 'cdo ngrids' to
# report two grids rather than one (single point grid for 'a_bnds' and 'b_bnds'
# variables) so must add attribute before standardization.
# NOTE: Some models use hybrid coordinates but different names. The GFDL and CNRM
# models have zaxistype 'hybrid' and longname 'atmospheric model level' instead of the
# 'hybrid sigma pressure coordinate' (seems due to using an 'ap' pressure term instead
# of an 'a' coefficient times a 'p0' reference), and the FGOALS models have zaxistype
# 'generic' and longname 'sigma coordinate' instead of zaxistype 'hybrid'. However
# 'cdo ml2pl' is general so works fine (seems to parse 'formula' term). Relevant files:
# cmip6-picontrol-amon/cl_Amon_GFDL-CM4_piControl_r1i1p1f1_gr1_015101-025012.nc
# cmip6-picontrol-amon/cl_Amon_GFDL-ESM4_piControl_r1i1p1f1_gr1_000101-010012.nc
# cmip6-picontrol-amon/cl_Amon_CNRM-CM6-1_piControl_r1i1p1f2_gr_185001-194912.nc
# cmip6-picontrol-amon/cl_Amon_CNRM-ESM2-1_piControl_r1i1p1f2_gr_185001-194912.nc
# cmip5-picontrol-amon/cl_Amon_FGOALS-g2_piControl_r1i1p1_020101-021012.nc
# cmip6-picontrol-amon/cl_Amon_FGOALS-g3_piControl_r1i1p1f1_gn_020001-020912.nc
# NOTE: Some models have zaxistype 'generic' and longname 'hybrid height coordinates'
# and require 'pfull' for vertical interpolation. This consists of ACCESS, KACE, UKESM,
# and HadGEM. Annoyingly, the CMIP5 versions only provide 'pfull' for picontrol not
# abrupt-4xco2, so have to manually copy to folder. Similarly, HadGEM only provides
# 'pfull' for control-1950 and UKESM only provides 'pfull' for AERmon instead of Amon
# so we manually obtained wget scripts using the online interface. Also have to manually
# remove 'pfull' for sigma coordinate FGOALS-g3 which is not needed. Relevant files:
# cmip5-picontrol-amon/cl_Amon_ACCESS1-0_piControl_r1i1p1_030001-032412.nc
# cmip5-picontrol-amon/cl_Amon_ACCESS1-3_piControl_r1i1p1_025001-027412.nc
# cmip6-picontrol-amon/cl_Amon_ACCESS-CM2_piControl_r1i1p1f1_gn_095001-096912.nc
# cmip6-picontrol-amon/cl_Amon_ACCESS-ESM1-5_piControl_r1i1p1f1_gn_010101-012012.nc
# cmip5-picontrol-amon/cl_Amon_HadGEM2-ES_piControl_r1i1p1_185912-188411.nc
# cmip6-picontrol-amon/cl_Amon_HadGEM3-GC31-MM_piControl_r1i1p1f1_gn_185001-185912.nc
# cmip6-picontrol-amon/cl_Amon_HadGEM3-GC31-LL_piControl_r1i1p1f1_gn_185001-189912.nc
# cmip6-picontrol-amon/cl_Amon_KACE-1-0-G_piControl_r1i1p1f1_gr_200001-209912.nc
# cmip6-picontrol-amon/cl_Amon_UKESM1-0-LL_piControl_r1i1p1f2_gn_196001-199912.nc
# NOTE: Some model data is erroneously interpreted as having pressure levels due to
# missing CF standard name and formula terms. This includes all CESM2 models except
# CESM2-WACCM-FV2, so can infer correct attributes for this (note the latter is also
# messed up, since some have levels with positive descending magnitude and others with
# negative ascending magnitude, so must manually flip axis before merging and let cdo
# issue a level mismatch warning). It also includes IPSL models but no point of
# reference, have to adjust manually. Finally, IITM-ESM and MCM-UA *actually do* provide
# standard pressure level data when not required by the table protocol. Relevant files:
# cmip6-picontrol-amon/cl_Amon_CESM2_piControl_r1i1p1f1_gn_000101-009912.nc
# cmip6-picontrol-amon/cl_Amon_CESM2-FV2_piControl_r1i1p1f1_gn_000101-005012.nc
# cmip6-picontrol-amon/cl_Amon_CESM2-WACCM_piControl_r1i1p1f1_gn_000101-009912.nc
# cmip6-picontrol-amon/cl_Amon_CESM2-WACCM-FV2_piControl_r1i1p1f1_gn_000101-004912.nc
# cmip6-picontrol-amon/cl_Amon_IPSL-CM6A-LR_piControl_r1i1p1f1_gr_185001-234912.nc
# cmip6-picontrol-amon/clw_Amon_IPSL-CM5A2-INCA_piControl_r1i1p1f1_gr_185001-209912.nc
# cmip6-picontrol-amon/cl_Amon_IITM-ESM_piControl_r1i1p1f1_gn_192601-193512.nc
# cmip6-picontrol-amon/cl_Amon_MCM-UA-1-0_piControl_r1i1p1f1_gn_000101-010012.nc
VERT_LEVS = 100 * np.array(
    [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 70, 50, 30, 20, 10, 5, 1]  # noqa: E501
)


def _parse_args(years=150, climate=True, detrend=False, skipna=False, **kwargs):
    """
    Parse variables related to time standardization and create a file name suffix.

    Parameters
    ----------
    **kwargs
        See `standardize_time` for details.
    """
    # NOTE: File format is e.g. 'file_0000-1000-climate-notrend.nc'. Each modeling
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
        *(('notrend',) if detrend else ()),
    ))
    kw = {
        'years': years,
        'climate': climate,
        'detrend': detrend,
        'skipna': skipna,
    }
    return name, kw, kwargs


def _parse_desc(data):
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
    tables = []
    string = data if isinstance(data, str) else '\n'.join(data)
    regex_tables = re.compile(r'(?:\A|\n)(?:#.*\n)?([^#][\s\S]*?)(?:\Z|\n#)')
    regex_items = re.compile(r'(?:\A|\n)(\w*)\s*=\s*([\s\S]*?(?=\Z|\n.*=))')
    regex_str = re.compile(r'\A".*?"\Z')  # note strings sometimes have no quotations
    for m in regex_tables.finditer(string):
        content = m.group(1)
        tables.append(table := {})
        for m in regex_items.finditer(content):
            key, value = m.groups()
            if regex_str.match(value):
                table[key] = eval(value)
            else:
                value = [s.strip() for s in value.split()]
                for dtype in (int, float, str):
                    try:
                        value = np.array(value, dtype=dtype)
                    except ValueError:
                        continue
                    else:
                        table[key] = value.item() if value.size == 1 else value
                    break
                else:
                    warnings.warn(f'Ignoring unexpected data pair {key} = {value!r}.')
    return tables


def _parse_ncdump(path):
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


def _output_check(path, printer):
    """
    Check that the output file exists.

    Parameters
    ----------
    path : path-like
        The netcdf path.
    """
    print = printer or builtins.print
    if not path.is_file() or path.stat().st_size == 0:
        message = f'Failed to create output file: {path}.'
        path.unlink(missing_ok=True)
        tmps = sorted(path.parent.glob(path.name + '.pid*.nc*.tmp'))  # nco commands
        message += f' Removed {len(tmps)} temporary nco files.' if tmps else ''
        for tmp in tmps:
            tmp.unlink(missing_ok=True)
        raise RuntimeError(message)
    print(f'Created output file {path.name!r}.')


def _output_name(path=None, stem=None, suffix=None, parent='~/data.nc'):
    """
    Generate a path from an input path.

    Parameters
    ----------
    path : path-like
        The input path.
    stem, suffix : str optional
        The stem and suffix to use for default values.
    parent : path-like, optional
        The reference file to use for default values.
    """
    parent = Path(parent).expanduser()
    if path:
        path = Path(path).expanduser()
    else:
        path = parent.parent
    if path.is_dir():
        suffix = suffix or 'copy'
        stem = stem or parent.stem
        name = stem + '_' + suffix + '.nc'
        path = path / name
    return path


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
    path = Path(path).expanduser()
    grids, zaxes = {}, {}
    files, _ = _glob_files(path, variable + '_*')
    for file in files:
        key = '_'.join(file.name.split('_')[:5])
        if key in grids or key in zaxes:
            continue
        try:
            grid, zaxis = cdo.griddes(input=str(file)), cdo.zaxisdes(input=str(file))
        except CDOException:
            print(f'Warning: Failed to read file {file}.')  # message already printed
            continue
        grid, zaxis = map(_parse_desc, (grid, zaxis))
        grids[key], zaxes[key] = grid, zaxis
        print(f'\n{file.name}:')
        for table in (*grid, *zaxis):
            print(', '.join(f'{k}: {v}' for k, v in table.items() if not isinstance(v, np.ndarray)))  # noqa: E501
            if table in zaxis and 'vctsize' not in table and table['size'] > 1:
                warnings.warn('Vertical axis does not have vertices.')


def process_files(
    path='~/data',
    dest='~/data',
    dependencies=True,
    vertical=True,
    horizontal=True,
    overwrite=False,
    printer=None,
    dryrun=False,
    **kwargs,
):
    """
    Average and standardize the files downloaded with a wget script.

    Parameters
    ----------
    path : path-like
        The input path for the raw data.
    dest : path-like, optional
        The output directory for the averaged and standardized data.
    dependencies : bool, optional
        Whether to automatically add pressure dependencies.
    vertical : bool, optional
        Whether to standardize vertical levels.
    horizontal : bool, optional
        Whether to standardize horizontal grid.
    overwrite : bool, optional
        Whether to overwrite existing files or skip them.
    printer : callable, optional
        The print function.
    dryrun : bool, optional
        Whether to only print time information and exit.
    **kwargs
        Passed to `standardize_time`, `standardize_vertical`, `standardize_horizontal`.
    **constraints
        Passed to `_parse_constraints`.
    """
    # Find files and restrict to unique constraints
    # NOTE: Here we only constrain search to the project, which is otherwise not
    # indicated in native netcdf filenames. Similar approach to filter_script().
    suffix, kwtime, constraints = _parse_args(**kwargs)
    project, constraints = _parse_constraints(reverse=True, **constraints)
    project = project.lower()
    print = printer or _init_printer('process', **constraints)
    path = Path(path).expanduser()
    dest = Path(dest).expanduser()
    files, _ = _glob_files(path, project + '*/*')
    if constraints.keys() - _line_parts.keys() != set(('project',)):
        raise ValueError(f'Input constraints {constraints.keys()} must be subset of: {_line_parts.keys()}')  # noqa: E501
    if not files:
        raise FileNotFoundError(f'Pattern {project}*/*.nc in directory {path} returned no netcdf files.')  # noqa: E501
    database = {}
    for file in files:
        opts = {facet: func(file.name) for facet, func in _line_parts.items()}
        if opts['variable'] == 'pfull':  # only ever used for interpolation
            continue
        if any(opt not in constraints.get(facet, (opt,)) for facet, opt in opts.items()):  # noqa: E501
            continue
        if file.stat().st_size == 0:
            print(f'Warning: Removing empty input file {file.name!r}.')
            file.unlink()
            continue
        key = tuple(opts.values())
        database.setdefault(key, []).append(file)

    # Initialize cdo and process files.
    # NOTE: Here perform horizontal interpolation before vertical interpolation because
    # latter adds NaNs and extrapolated data so should delay that step until very end.
    print('Input files:', sum(len(files) for files in database.values()), end=', ')
    print(', '.join(f'{key}: ' + ' '.join(opts) for key, opts in constraints.items()))
    print()
    outputs = []
    constants = dest / 'cmip-constants'
    constants.mkdir(exist_ok=True)
    kwargs = {'overwrite': overwrite, 'printer': print}
    for key, files in database.items():
        # Standardize time and dependencies
        folder = dest / files[0].parent.name
        folder.mkdir(exist_ok=True)
        parts = dict(zip(_line_parts, key))
        stem = '_'.join(files[0].name.split('_')[:-1])
        time = _output_name(folder, stem, 'standard-time')
        deps = _output_name(folder, stem, 'standard-deps')
        vert = _output_name(folder, stem, 'standard-vertical')
        hori = _output_name(folder, stem, 'standard-horizontal')
        output = _output_name(folder, stem, suffix)
        outputs.append(output)
        print('Output:', '/'.join((output.parent.name, output.name)))
        exception = lambda error: (
            ' '.join(traceback.format_exception(None, error, error.__traceback__))
        )
        try:
            repair_files(*files, dryrun=dryrun, printer=print)
        except Exception as error:
            print(exception(error))
            print('Warning: Failed to standardize attributes.\n')
            continue
        updated = not overwrite and output.is_file() and (
            all(file.stat().st_mtime <= output.stat().st_mtime for file in files)
        )
        dryrun = dryrun or updated  # always want time information for e.g. warnings
        try:
            standardize_time(*files, dryrun=dryrun, output=time, **kwtime, **kwargs)
        except Exception as error:
            print(exception(error))
            print('Warning: Failed to standardize temporally.\n')
            continue
        if updated:
            print('Skipping (up to date)...\n')
            continue
        if dryrun:
            print('Skipping (dry run)...\n')
            continue

        # Standardize horizontal grid and vertical levels
        info, sizes, names, attrs = _parse_ncdump(files[0])
        ps = _output_name(constants, 'ps', parts['model'])
        pfull = _output_name(constants, 'pfull', parts['model'])
        weights = _output_name(constants, 'gencon', parts['model'])
        kwdeps = {'ps': ps, 'pfull': pfull, 'model': parts['model'], 'search': files[0].parent}  # noqa: E501
        kwhori = {'weights': weights}
        kwvert = {}
        if not dependencies or 'lev' not in attrs:
            time.replace(deps)
        else:
            try:
                standardize_dependencies(time, deps, **kwdeps, **kwargs)
            except Exception as error:
                print(exception(error))
                print('Warning: Failed to standardize dependencies.\n')
                continue
        if not vertical or 'lev' not in attrs:
            deps.replace(vert)
        else:
            try:
                standardize_vertical(deps, vert, **kwvert, **kwargs)
            except Exception as error:
                print(exception(error))
                print('Warning: Failed to standardize vertically.\n')
                continue
        if not horizontal:
            vert.replace(hori)
        else:
            try:
                standardize_horizontal(vert, hori, **kwhori, **kwargs)
            except Exception as error:
                print(exception(error))
                print('Warning: Failed to standardize horizontally.\n')
                continue
        hori.replace(output)
        _output_check(output, print)
        hori.unlink(missing_ok=True)
        vert.unlink(missing_ok=True)
        deps.unlink(missing_ok=True)
        time.unlink(missing_ok=True)
        today = datetime.today().strftime('%Y-%m-%dT%H:%M:%S')  # date +%FT%T
        print(f'Processing completed: {today}')
        print('Removed temporary files.\n')
    return outputs


def repair_files(*paths, dryrun=False, printer=None):
    """
    Repair the metadata on the input files in preparation for vertical
    and horizontal standardization using `cdo`.

    Parameters
    ----------
    *paths : path-like
        The input path(s).
    dryrun : bool, default: False
        Whether to only print command information and exit.
    printer : callable, optional
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
    # but this causes issues when applying setzaxis so delay this attribute edit until
    # standardize_dependencies. See: https://code.mpimet.mpg.de/issues/10692
    # NOTE: Using -ensavg with pfull files with disagreeing time_bnds or
    # climatology_bnds can create duplicate bounds variables, and -merge can cause
    # issues merging with parent files (incorrect 'reserved variable zaxistype' error).
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
        info, sizes, names, attrs = _parse_ncdump(path)
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
                f'pressure coordinates for {path.name!r}:', math, ',', ', '.join(atted)
            )
            if not dryrun:
                nco.ncap2(input=str(path), output=str(path), options=['-s', math])
                nco.ncatted(input=str(path), output=str(path), options=atted)
                nco.ncks(input=str(path), output=str(path), options=['-x', '-v', 'ptop'])  # noqa: E501

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
                + f'center attributes for {path.name!r}:', ', '.join(atted)
            )
            if not dryrun:
                nco.ncatted(input=str(path), output=str(path), options=atted)

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

        # Handle IPSL models with weird extra dimension on coordinates.
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
                + f'attributes for {path.name!r}:', math, ', '.join(atted)
            )
            if not dryrun:
                nco.ncap2(input=str(path), output=str(path), options=['-s', math])
                nco.ncatted(input=str(path), output=str(path), options=atted)
                nco.ncks(input=str(path), output=str(path), options=['-x', '-v', 'klevp1'])  # noqa: E501

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
                f'generalized height coordinates for {path.name!r}:', ', '.join(atted)
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
                f'latitude bounds for {path.name!r}: ', math
            )
            if not dryrun:  # first flag required because is member of cell_measures
                remove = ['-C', '-x', '-v', 'areacella']
                nco.ncap2(input=str(path), output=str(path), options=['-s', math])
                nco.ncks(input=str(path), output=str(path), options=remove)


def standardize_time(
    *paths,
    output=None,
    overwrite=False,
    printer=None,
    dryrun=False,
    **kwargs
):
    """
    Create a standardized monthly climatology or annual mean time series from
    the input files. Uses `cdo.mergetime` and optionally `cdo.ymonmean`.

    Parameters
    ----------
    *paths : path-like
        The input files.
    output : path-like, optional
        The output name and/or location.
    years : int or 2-tuple, default: 150
        The years to use relative to the start of the simulation.
    climate : bool, optional
        Whether to create a monthly-mean climatology or a time series.
    detrend : bool, default: False
        Whether to detrend input data. Used with feedback and budget calculations.
    skipna : bool, default: False
        Whether to ignore NaNs.
    overwrite : bool, optional
        Whether to overwrite existing files or skip them.
    printer : callable, optional
        The print function.
    dryrun : bool, default: False
        Whether to only print time information and exit.
    """
    # Initial stuff
    # NOTE: Some models use e.g. file_200012-200911.nc instead of file_200001-200912.nc
    # so 'selyear' could add a few extra months... but not worth worrying about. There
    # is also endpoint inclusive file_200001-201001.nc but seems only for HadGEM3 pfull.
    print = printer or builtins.print
    suffix, kwtime, unknown = _parse_args(**kwargs)
    paths = [Path(file).expanduser() for file in paths]
    output = _output_name(output, suffix=suffix, parent=paths[0])
    if not paths:
        raise TypeError('Input files passed as positional arguments are required.')
    if unknown:
        raise TypeError(f'Unexpected keyword argument(s): {unknown}')
    if not overwrite and output.is_file() and output.stat().st_size > 0:
        print(f'Output file already exists: {output.name!r}.')
        return output

    # Filter input times
    # NOTE: This includes kludge for individual CESM2-WACCM-FV2 files with incorrect
    # (negative) levels. Must invert or else cdo averages upper with lower levels.
    # After invert cdo raises a level mismatch warning but results will be correct.
    years = tuple(_line_years(path.name) for path in paths)
    ymin, ymax = min(ys[0] for ys in years), max(ys[1] for ys in years)
    print(f'Initial year range: {ymin}-{ymax} ({len(years)} files)')
    yrel = kwtime.pop('years')
    ymin, ymax = ymin + yrel[0], ymin + yrel[1] - 1  # e.g. (0, 50) is 49 years
    filt = lambda line: not re.search('(warning|error)', line)  # ignore messages
    inputs = []
    for ys, path in zip(years, paths):
        levs = []
        if 'CESM' in path.name:  # slow so only use where this is known problem
            levs = ' '.join(filter(filt, cdo.showlevel(input=str(path)))).split()
        if invert := ('-invertlev' if np.any(np.array(levs, dtype=float) < 0) else ''):
            print(f'Warning: Found negative levels for {path.name!r}. Inverting axis.')
        if ys[0] > ymax or ys[1] < ymin:
            continue
        y0, y1 = min((ys[1], ymax)), max((ys[0], ymin))
        selyear = '' if (y0, y1) == (ymin, ymax) else f'-selyear,{y0}/{y1}'
        inputs.append(f'{selyear} {invert} {path}')
    print(f'Filtered year range: {ymin}-{ymax} ({len(inputs)} files)')
    if dryrun:
        return  # only print time information
    if not inputs:
        raise ValueError(f'No files found within requested range: {ymin}-{ymax}')
    input = ' '.join(inputs)

    # Take time averages
    # NOTE: Here data can be detrended to improve sensitive residual energy budget and
    # feedback calculations. This uses a lot of memory so don't bother with 3D fields.
    detrend = '-detrend ' if kwtime.pop('detrend') else ''
    method = 'ymonmean' if kwtime.pop('skipna') else 'ymonavg'
    method = method if kwtime.pop('climate') else 'copy'
    input = f'{detrend}-mergetime [ {input} ]'
    print(f'Merging {len(inputs)} files with {method} {detrend}-mergetime.')
    getattr(cdo, method)(input=str(input), output=str(output))
    _output_check(output, print)
    return output


def standardize_dependencies(
    path,
    output=None,
    ps=None,
    pfull=None,
    model=None,
    rebuild=None,
    search=None,
    overwrite=False,
    printer=None,
):
    """
    Create a file with surface pressure and model level pressure appended for
    subsequent invocation of `cdo.ap2pl` and save the average pressure data.

    Parameters
    ----------
    path : path-like
        The input file.
    output : path-like, optional
        The output file.
    ps, pfull : path-like, optional
        The reference surface and model level pressure names.
    model : str, optional
        The model to use when searching for pressure data.
    search : path-like, optional
        The reference path to use when searching for pressure data.
    rebuild : path-like, optional
        Whether to rebuild the dependency files.
    overwrite : bool, optional
        Whether to overwrite existing files or skip them.
    printer : callable, optional
        The print function.
    """
    # Initial stuff
    # NOTE: Data passed to this function should already have had its attributes
    # modified using repair_files(). This just adds or removes dependencies.
    print = printer or builtins.print
    ps = _output_name(ps, suffix='ps', parent=path)
    pfull = _output_name(pfull, suffix='pfull', parent=path)
    output = _output_name(output, suffix='standard-dependencies', parent=path)
    path = Path(path).expanduser()
    search = Path(search).expanduser() if search else path.parent
    if not overwrite and output.is_file() and output.stat().st_size > 0:
        print(f'Output file already exists: {output.name!r}.')
        return output

    # Parse input dependencies
    # NOTE: The height interpolation scheme needs surface pressure not just model level
    # pressure, and then must use setzaxis to prevent disagreement due to decimal lev
    # differences between different model level files (see below).
    dependencies = {
        'hybrid': {'ps': ps},
        'generic': {'ps': ps, 'pfull': pfull},
        'generalized_height': {'ps': ps, 'pfull': pfull},
        'pressure': {},
    }
    tables = _parse_desc(cdo.zaxisdes(input=str(path)))
    tables = [kw for kw in tables if kw['zaxistype'] != 'surface']
    names = ' '.join(cdo.showname(input=str(path))).split()
    axis = ', '.join(table['zaxistype'] for table in tables)
    if len(tables) != 1:
        raise NotImplementedError(f'Missing or ambiguous vertical axis {axis!r} for {path.name!r}.')  # noqa: E501
    if axis not in dependencies:
        raise NotImplementedError(f'Cannot manage vertical axis dependencies {axis!r} for {path.name!r}.')  # noqa: E501

    # Generate the dependencies
    # NOTE: Even though 'pfull' has 'clim' suffix some modeling centers provide time
    # averages and should have no time dimension. So must use timavg as well. And ignore
    # table, experiment, and enesemble due to limited availability (see top of file).
    # NOTE: Even though 'long_name="generalized height"' required for ap2pl it triggers
    # error when naively applying the resulting zaxisdes to another file using setzaxis,
    # so tried manually changing the zaxistype from 'generalized_height' to 'height'.
    # However this triggered another issue where 'cdo merge' produced duplicate levels
    # (note applying -setzaxis,axis to pfull, whether or not the zaxistype coordinate
    # was changed, also triggers duplicate level issue). Solution is to apply long_name
    # on-the-fly before ap2pl. See: https://code.mpimet.mpg.de/issues/10692
    atted, merge = [], ''
    for name, file in dependencies[axis].items():
        if name in names:
            continue
        merge = f'{merge} {file}'
        if rebuild or (not file.is_file() or file.stat().st_size == 0):
            print(f'Generating pressure dependency file {file.name!r}.')
            glob = f'{name}_*{model}_*' if model else f'{name}_*'
            files, _ = _glob_files(search, glob)
            if not files:
                raise ValueError(f'Glob pattern {glob!r} returned no results.')
            input = ' '.join(f'-ymonavg -selname,{name} {f}' for f in files)
            repair_files(*files, printer=print)
            print(f'Averaging {len(files)} files with ensavg -ymonavg.')
            cdo.ensavg(input=input, output=str(file))
            _output_check(file, print)
        if name == 'pfull':
            text = file.parent / (file.stem + '.txt')
            data = '\n'.join(cdo.zaxisdes(input=str(file)))
            open(text, 'w').write(data)
            merge = f'{merge} -setzaxis,{text}'  # applied to original file
            atted = [Atted('overwrite', 'long_name', 'lev', 'generalized height').prn_option()]  # noqa: E501

    # Merge the dependencies
    # NOTE: Tried applying long_name as part of the chain before ap2pl but seems axis
    # is deciphered before stepping through chain so this fails. Instead apply
    # attribute here. Also cdo cannot apply to 'lev_bnds' (says variable not found).
    if not merge:
        print(f'File {path.name!r} already has required dependencies.')
        shutil.copy(path, output)
    else:
        print(f'Adding pressure dependencies with -merge {merge}.')
        cdo.merge(input=f'{merge} {path}', output=str(output))
        if atted:
            print('Preparing attributes for interpolation with:', *atted)
            nco.ncatted(input=str(output), output=str(output), options=atted)
        _output_check(output, print)
    return output


def standardize_vertical(path, output=None, overwrite=False, printer=None):
    """
    Create a file on standardized pressure levels from the input file with
    arbitrary vertical axis. Uses  `cdo.intlevel`, `cdo.ml2pl`, or `cdo.ap2pl`.

    Parameters
    ----------
    path : path-like
        The input file.
    output : path-like, optional
        The output name and/or location.
    overwrite : bool, optional
        Whether to overwrite existing files or skip them.
    printer : callable, optional
        The print function.

    Important
    ---------
    The input file requires surface pressure to interpolate sigma and hybrid pressure
    coordinates and model level pressure to interpolate hybrid height coordinates.
    """
    # Initial stuff
    print = printer or builtins.print
    path = Path(path).expanduser()
    output = _output_name(output, suffix='standard-vertical', parent=path)
    if not overwrite and output.is_file() and output.stat().st_size > 0:
        print(f'Output file already exists: {output.name!r}.')
        return output

    # Parse input vertical levels
    # NOTE: For some reason height must be oriented top-to-bottom (i.e. pressures
    # oriented low-to-height) before interpolating to pressure levels with ap2pl. This
    # is undocumented and error is silent. See: https://code.mpimet.mpg.de/issues/10692
    # NOTE: For some reason long_name='generalized height' required for ap2pl detection
    # but fails with setzaxis and trying to modify the zaxistype causes subsequent
    # merges to fail. See: https://code.mpimet.mpg.de/issues/10692
    methods = {
        'hybrid': '-ml2pl,%s',
        'generalized_height': '-ap2pl,%s -invertlev',
        'pressure': '-intlevel,%s',
    }
    tables = _parse_desc(cdo.zaxisdes(input=str(path)))
    tables = [kw for kw in tables if kw['zaxistype'] != 'surface']
    axis = ', '.join(table['zaxistype'] for table in tables)
    if len(tables) != 1:
        raise NotImplementedError(f'Missing or ambiguous vertical axis {axis!r} for {path.name!r}.')  # noqa: E501
    if axis not in methods:
        raise NotImplementedError(f'Cannot interpolate vertical axis {axis!r} for {path.name!r}.')  # noqa: E501
    string = ', '.join(f'{k}: {v}' for k, v in tables[0].items() if not isinstance(v, np.ndarray))  # noqa: E501
    print('Current vertical levels:', string)
    print('Destination vertical levels:', ', '.join(map(str, VERT_LEVS.flat)))

    # Interpolate to pressure levels with cdo
    # NOTE: removing unneeded variables by leading chain with -delname,ps,orog,pfull
    # seemed to cause cdo > 1.9.1 and cdo < 2.0.0 to hang indefinitely. Not sure
    # why and not worth issuing bug report but make sure cdo is up to date.
    # NOTE: Currently only some obscure extended monthly data and the core cloud
    # variables are output on model levels. Otherwise data is on pressure levels.
    # Avoid interpolation by selecting same pressure levels as standard output.
    # NOTE: All sigma and hybrid sigma pressure coordinate models include 'ps' in every
    # nc file, and all hybrid height coordinate models include 'orog' in every nc file.
    # In former case this is sufficient for pressure interpolation but in latter case
    # need to add separately provided 'pfull'. Rely on process_files to append this.
    if axis == 'pressure' and np.all(np.isclose(tables[0]['levels'], VERT_LEVS)):
        print(f'File {path.name!r} is already on standard vertical levels.')
        shutil.copy(path, output)
    else:
        method = methods[axis] % ','.join(map(str, VERT_LEVS.flat))
        print(f'Vertically interpolating with method {method}.')
        cdo.delname('ps,orog,pfull', input=f'{method} {path}', output=str(output))
    _output_check(output, print)
    tables = _parse_desc(cdo.zaxisdes(input=str(output)))
    axis = ', '.join(table['zaxistype'] for table in tables)
    if axis != 'pressure' or not np.all(np.isclose(tables[0]['levels'], VERT_LEVS)):
        levels = ', '.join(map(str, tables[0]['levels'].flat))
        output.unlink(missing_ok=True)
        raise RuntimeError(f'Incorrect output axis {axis!r} or levels {levels}.')
    return output


def standardize_horizontal(
    path, output=None, method=None, weights=None, rebuild=False, overwrite=False, printer=None  # noqa: E501
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
    method : str, default: 'con'
        The `cdo` remapping command suffix.
    weights : path-like, optional
        The location or name for weights data.
    rebuild : bool, optional
        Whether to rebuild the weight files.
    overwrite : bool, optional
        Whether to overwrite existing files or skip them.
    printer : callable, optional
        The print function.

    Note
    ----
    For details see the SCRIPS repository and Jones 1999, Monthly Weather Review.
    Code: https://github.com/SCRIP-Project/SCRIP
    Paper: https://doi.org/10.1175/1520-0493(1999)127%3C2204:FASOCR%3E2.0.CO;2
    """
    # Initial stuff
    method = method or 'con'
    method = method if method[:3] == 'gen' else 'gen' + method
    print = printer or builtins.print
    path = Path(path).expanduser()
    weights = _output_name(weights, suffix=method, parent=path)
    output = _output_name(output, suffix='standard-horizontal', parent=path)
    if not overwrite and output.is_file() and output.stat().st_size > 0:
        print(f'Output file already exists: {output.name!r}.')
        return output

    # Parse input horizontal grid
    # NOTE: Sometimes cdo detects dummy 'generic' grids of size 1 or 2 indicating
    # scalar quantities or bounds. Try to ignore these grids.
    result = cdo.griddes(input=str(path))
    tables = [kw for kw in _parse_desc(result) if kw['gridsize'] > 2]
    if not tables:
        raise NotImplementedError(f'Missing horizontal grid for {path.name!r}.')
    if len(tables) > 1:
        raise NotImplementedError(f'Ambiguous horizontal grids for {path.name!r}: ', ', '.join(table['gridtype'] for table in tables))  # noqa: E501
    table = tables[0]
    string = ', '.join(f'{k}: {v}' for k, v in table.items() if not isinstance(v, np.ndarray))  # noqa: E501
    print('Current horizontal grid:', string)
    print('Destination horizontal grid:', GRID_SPEC)

    # Generate weights and interpolate
    # NOTE: Unlike vertical standardization there are fewer pitfalls here (e.g.
    # variables getting silently skipped). So testing output griddes not necessary.
    opts = ('gridtype', 'xfirst', 'yfirst', 'xsize', 'ysize')
    key_current = tuple(table.get(key, None) for key in opts)
    key_destination = ('lonlat', 0, -90, *map(int, GRID_SPEC[1:].split('x')))
    if key_current == key_destination:
        print(f'File {path.name!r} is already on destination grid.')
        shutil.copy(path, output)
    else:
        if rebuild or not weights.is_file() or weights.stat().st_size == 0:
            print(f'Generating destination grid weights {weights.name!r}.')
            getattr(cdo, method)(GRID_SPEC, input=str(path), output=str(weights))
            _output_check(weights, print)
        print(f'Horizontally interpolating with grid weights {weights.name!r}.')
        cdo.remap(f'{GRID_SPEC},{weights}', input=str(path), output=str(output))
        _output_check(output, print)
    return output
