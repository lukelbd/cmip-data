#!/usr/bin/env python3
"""
Convenient wrappers for python APIs to download CMIP data.
"""
import itertools
import os
import sys
from pathlib import Path

import numpy as np
from cdo import Cdo
from pyesgf.logon import LogonManager
from pyesgf.search import SearchConnection


# Global functions
__all__ = [
    'process_files',
    'download_wgets',
    'filter_wgets',
]

# Constants
# TODO: Make these configurable? Add conditionals for other stations?
DATA = Path.home() / 'data'
if sys.platform == 'darwin':
    ROOT = Path.home() / 'data'
else:
    ROOT = Path('/mdata5') / 'ldavis'

# CMIP nodes for OpenID logon
# Nodes are listed here: https://esgf.llnl.gov/nodes.html
CMIP_NODES = {
    'LLNL': 'https://esgf-node.llnl.gov/',
    'CEDA': 'https://esgf-index1.ceda.ac.uk/',
    'DKRZ': 'https://esgf-data.dkrz.de/',
    'GFDL': 'https://esgdata.gfdl.noaa.gov/',
    'IPSL': 'https://esgf-node.ipsl.upmc.fr/',
    'JPL': 'https://esgf-node.jpl.nasa.gov/',
    'LIU': 'https://esg-dn1.nsc.liu.se/',
    'NCI': 'https://esgf.nci.org.au/',
    'NCCS': 'https://esgf.nccs.nasa.gov/',
}
# CMIP models and nodes to ignore by default
# These can be adjusted by experience
CMIP_MODELS_BAD = (
    'EC-Earth3-CC',
)
CMIP_NODES_BAD = (
    # 'ceda.ac.uk',
    'nird.sigma2.no',
    'nmlab.snu.ac.kr',
    'esg.lasg.ac.cn',
    'cmip.fio.org.cn',
    'vesg.ipsl.upmc.fr',
)

# CMIP constants obtained from get_facet_options() for SearchContext(project='CMIP5')
# and SearchContext(project='CMIP6') using https://esgf-node.llnl.gov/esg-search
# for the SearchConnection URL. Conventions changed between projects so e.g.
# 'experiment', 'ensemble', 'cmor_table', and 'time_frequency' in CMIP5 must be
# changed to 'experiment_id', 'variant_label', 'table_id', and 'frequency' in CMIP6.
# Note 'member_id' is equivalent to 'variant_label' if 'sub_experiment_id' is unset
# and for some reason 'variable' and 'variable_id' are kept as synonyms in CMIP6.
# URL https://esgf-node.llnl.gov/esg-search:     11900116 hits for CMIP6 (use this one!)
# URL https://esgf-data.dkrz.de/esg-search:      01009809 hits for CMIP6
# URL https://esgf-node.ipsl.upmc.fr/esg-search: 01452125 hits for CMIP6
CMIP5_FACETS = [
    'access', 'cera_acronym', 'cf_standard_name', 'cmor_table', 'data_node',
    'ensemble', 'experiment', 'experiment_family', 'forcing', 'format',
    'index_node', 'institute', 'model', 'product', 'realm', 'time_frequency',
    'variable', 'variable_long_name', 'version'
]
CMIP6_FACETS = [
    'access', 'activity_drs', 'activity_id', 'branch_method', 'creation_date',
    'cf_standard_name', 'data_node', 'data_specs_version', 'datetime_end',
    'experiment_id', 'experiment_title', 'frequency', 'grid', 'grid_label',
    'index_node', 'institution_id', 'member_id', 'nominal_resolution', 'realm',
    'short_description', 'source_id', 'source_type', 'sub_experiment_id', 'table_id',
    'variable', 'variable_id', 'variable_long_name', 'variant_label', 'version'
]

# Constants for path management
# TODO: Increase to 100 years after first step
DELIM = 'EOF--dataset.file.url.chksum_type.chksum'
OPENID = 'https://esgf-node.llnl.gov/esgf-idp/openid/lukelbd'
MAXYEARS = 50  # retain only first N years of each simulation?
ENDYEARS = False  # whether to use the end years instead of start years
# MAXYEARS = 100  # retain 100 years for very solid climatology

# Helper functions for reading lines of wget files
# NOTE: Some files have a '-clim' suffix at the end of the date range.
get_url = lambda line: line.split("'")[3].strip()
get_file = lambda line: line.split("'")[1].strip()
get_model = lambda line: line.split('_')[2]  # model id from netcdf line
get_years = lambda line: tuple(int(date[:4]) for date in line.split('.')[0].split('_')[-1].split('-')[:2])  # noqa: E501
get_var = lambda line: line.split('_')[0][1:]  # variable from netcdf line


def _wget_parse(filename, complement=False):
    """
    Return the download lines of the wget files or their complement. The latter is
    used when constructing a single wget file from many wget files.
    """
    lines = open(filename, 'r').readlines()  # list of lines
    idxs = [i for i, line in enumerate(lines) if DELIM in line]
    if not idxs:
        return []
    elif len(idxs) != 2:
        raise NotImplementedError
    if complement:
        return lines[:idxs[0] + 1], lines[idxs[1]:]  # return tuple of pairs
    else:
        return lines[idxs[0] + 1:idxs[1]]
    return lines


def _wget_files(
    project='cmip6',
    experiment='piControl',
    table='Amon',
    variables='ta',
    intersection=False,
):
    """
    Return a list of input wget files matching the criteria along with an output
    wget file to be constructed and the associated output folder for NetCDFs.

    Todo
    ----
    Add `intersection` option for getting e.g. models with both pre-industrial and
    abrupt 4xCO2 versions of the input variable. Needed for forced-response utilities.
    """
    # TODO: Permit both 'and' or 'or' logic when searching and filtering files
    # matching options. Maybe add boolean 'intersection' keywords or something.
    # TODO: Auto name wget files based on the files listed in the script. The wgets
    # generated by searches do not generally mix variables or experiments.
    parts = [
        sorted(set((part,) if isinstance(part, str) else part))
        for part in (project, experiment, table, variables)
    ]
    path = ROOT / 'wgets'
    input = []
    patterns = []
    for i, part in enumerate(itertools.product(*parts)):
        if i == 0:  # project
            part = tuple(s.lower() for s in part)
        patterns.append(pattern := 'wget_' + '[_-]*'.join(part) + '[_-]*.sh')
        input.extend(sorted(path.glob(pattern)))
    if not input:
        raise ValueError(f'No wget files found in {path} for pattern(s): {patterns!r}')
    if intersection:
        raise NotImplementedError('Cannot yet find intersection of parts.')
    path = ROOT / '-'.join(part[0] for part in parts[:3])  # TODO: improve
    if not path.is_dir():
        os.mkdir(path)
    name = 'wget_' + '_'.join('-'.join(part) for part in parts) + '.sh'
    output = path / name
    return input, output


def _wget_models(models=None, **kwargs):
    """
    Return a set of all models in the group of wget scripts, a dictionary
    with keys of (experiment, variable) pairs listing the models present,
    and a dictionary with 'experiment' keys listing the possible variables.

    Todo
    ----
    Clean this up to support either union or intersection operations.
    Sometimes acceptable to have missing variables e.g. for constraints
    and sometimes not e.g. for getting transport as energy budget residual.
    """
    # Group models available for various (experiment, variable)
    input, output = _wget_files(**kwargs)
    project, experiment, table = output.parent.name.split('-')
    frequency = table[-3:].lower()  # convert e.g. Amon to mon
    models_all = {*()}  # all models
    models_grouped = {}  # models by variable
    for file in input:
        models_file = {get_model(line) for line in _wget_parse(file) if line}
        models_all.update(models_file)
        if (experiment, frequency) in models_grouped:
            models_grouped[(experiment, frequency)].update(models_file)
        else:
            models_grouped[(experiment, frequency)] = models_file

    # Get dictionary of variables per experiment
    exps_vars = {}
    for (exp, var) in models_grouped:
        if exp not in exps_vars:
            exps_vars[exp] = {*()}
        exps_vars[exp].add(var)

    # Find models missing in some of these pairs
    exps_missing = {}
    models_download = set()
    models_ignore = set()
    for model in sorted(models_all):
        # Manual filter, e.g. filter daily data based on
        # whether they have some monthly data for same period?
        pairs_missing = []
        for pair, group in models_grouped.items():
            if model not in group:
                pairs_missing.append(pair)
        filtered = bool(models and model not in models)
        if not pairs_missing and not filtered:
            models_download.add(model)
            continue

        # Get variables missing per experiment
        models_ignore.add(model)
        exp_dict = {}
        for (exp, var) in pairs_missing:
            if exp not in exp_dict:
                exp_dict[exp] = {*()}
            exp_dict[exp].add(var)

        # Add to per-experiment dictionary that records models for
        # which certain individual variables are missing
        for exp, vars in exp_dict.items():
            if exp not in exps_missing:
                exps_missing[exp] = {}
            missing = exps_missing[exp]
            if filtered:
                # Manual filter, does not matter if any variables were missing
                if 'filtered' not in missing:
                    missing['filtered'] = []
                missing['filtered'].append(model)
            elif vars == exps_vars[exp]:  # set comparison
                # All variables missing
                if 'all' not in missing:
                    missing['all'] = []
                missing['all'].append(model)
            else:
                # Just some variables missing
                for var in vars:
                    if var not in missing:
                        missing[var] = []
                    missing[var].append(model)

    # Print message
    print()
    print(f'Input models ({len(models_download)}):')
    print(', '.join(sorted(models_download)))
    for exp, exps_missing in exps_missing.items():
        vars = sorted(exps_missing)
        print()
        print(f'{exp}:')
        for var in vars:
            missing = exps_missing[var]
            print(f'Missing {var} ({len(missing)}): {", ".join(sorted(missing))}')
    print()
    return models_download


def download_wgets(node=None, **kwargs):
    """
    Download CMIP wget files.

    Parameters
    ----------
    url : str, default: 'LLNL'
        The ESGF node to use for searching.
    **kwargs
        Passed to `~pyesgf.search.SearchContext`.
    """
    # Log on and initalize connection
    lm = LogonManager()
    host = 'esgf-node.llnl.gov'
    if not lm.is_logged_on():  # surface orography
        lm.logon(username='lukelbd', password=None, hostname=host)
    node = node or 'LLNL'
    if node in CMIP_NODES:
        url = CMIP_NODES[node]
    else:
        raise ValueError(f'Invalid node {node!r}.')
    conn = SearchConnection(url, distrib=True)

    # Create contexts with default facets
    facets = kwargs.pop('facets', None)
    facets = facets or ','.join(kwargs)  # default to the search keys
    ctx = conn.new_context(facets=facets, **kwargs)

    # Create the wgets
    # TODO: Check that looking up 'facets' works instead of checking lists
    print('Context:', ctx, ctx.facet_constraints)
    print('Hit count:', ctx.hit_count)
    parts = []
    for facet in ctx.facets:  # facet search
        opts = ctx.facet_constraints.getall(facet)
        part = '-'.join(opt.replace('-', '') for opt in sorted(set(opts)))
        if facet in ('table_id', 'cmor_table'):
            part = part or 'Amon'  # TODO: remove this kludge?
        if facet == 'project':
            part = part.lower()
        parts.append(part)
    # Write wget file
    for j, ds in enumerate(ctx.search()):
        print(f'Dataset {j}:', ds)
        fc = ds.file_context()
        fc.facets = ctx.facets  # TODO: report bug and remove?
        name = 'wget_' + '_'.join((*parts, format(j, '05d'))) + '.sh'
        path = Path(ROOT, 'wgets', name)
        if path.is_file() and False:
            print('Skipping script:', name)
            continue
        try:
            wget = fc.get_download_script()
        except Exception:
            print('Download failed:', name)
        else:
            print('Creating script:', name)
            with open(path, 'w') as f:
                f.write(wget)


def filter_wgets(
    models=None, variables=None, maxyears=None, endyears=None,
    badnodes=None, badmodels=None, duplicate=False, overwrite=False,
    **kwargs
):
    """
    Filter the input wget files to within some input climatology.

    Todo
    ----
    Automatically invoke this from `download_wgets` or better to download
    cache of files then filter later?
    """
    # Get all lines for download
    # Also manually replace the open ID
    kwargs['variables'] = variables
    input, output = _wget_files(**kwargs)
    prefix, suffix = _wget_parse(input[0], complement=True)
    for i, line in enumerate(prefix):
        if line == 'openId=\n':
            prefix[i] = 'openId=' + OPENID + '\n'
            break

    # Collect all download lines for files
    files = set()  # unique file tracking (ignore same file from multiple nodes)
    lines = []  # final lines
    lines_input = [line for file in input for line in _wget_parse(file) if line]

    # Iterate by *model*, filter to date range for which *all* variables are available!
    # So far just issue for GISS-E2-R runs but important to make this explicit!
    if models is None:
        models = _wget_models(**kwargs)
    if badmodels is None:
        badmodels = CMIP_MODELS_BAD
    if badnodes is None:
        badnodes = CMIP_NODES_BAD
    if maxyears is None:
        maxyears = MAXYEARS
    if endyears is None:
        endyears = ENDYEARS
    if isinstance(badmodels, str):
        badmodels = (badmodels,)
    if isinstance(badnodes, str):
        badnodes = (badnodes,)
    if isinstance(models, str):
        models = (models,)
    if isinstance(variables, str):
        variables = (variables,)
    for model in sorted(models):
        # Find minimimum date range for which all variables are available
        # NOTE: Use maxyears - 1 or else e.g. 50 years with 190001-194912 will not
        # "satisfy" the range and result in the next file downloaded.
        # NOTE: Ensures variables required for same analysis are present over
        # matching period. Data availability or table differences could cause bugs.
        year_ranges = []
        lines_model = [line for line in lines_input if f'_{model}_' in line]
        for var in set(map(get_var, lines_model)):
            if variables and var not in variables:
                continue
            try:
                years = [get_years(line) for line in lines_model if f'{var}_' in line]
            except ValueError:  # helpful message
                for line in lines_model:
                    get_years(line)
            year_ranges.append((min(y for y, _ in years), max(y for _, y in years)))
        if not year_ranges:
            continue
        year_range = (max(p[0] for p in year_ranges), min(p[1] for p in year_ranges))
        print('Model:', model)
        print('Initial years:', year_range)
        if endyears:
            year_range = (max(year_range[0], year_range[1] - maxyears + 1), year_range[1])  # noqa: E501
        else:
            year_range = (year_range[0], min(year_range[1], year_range[0] + maxyears - 1))  # noqa: E501
        print('Final years:', year_range)

        # Add lines within the date range that were not downloaded already
        # WARNING: Only exclude files that are *wholly* outside range. Allow
        # intersection of date ranges. Important for cfDay IPSL data.
        for line in lines_model:
            url = get_url(line)
            if badnodes and any(node in url for node in badnodes):
                continue
            file = get_file(line)
            if not duplicate and file in files:
                continue
            model = get_model(line)
            if badmodels and any(model == m for m in badmodels):
                continue
            var = get_var(line)
            if variables and var not in variables:
                continue
            years = get_years(line)
            if years[1] < year_range[0] or years[0] > year_range[1]:
                continue
            dest = output.parent / file
            if dest.is_file() and dest.stat().st_size == 0:
                os.remove(dest)  # remove empty files caused by download errors
            if not overwrite and dest.is_file():
                continue  # skip if destination exists
            files.add(file)
            lines.append(line)

    # Save bulk download file
    if output.is_file():
        os.remove(output)
    print('File:', output)
    with open(output, 'w') as file:
        file.write(''.join(prefix + lines + suffix))
    os.chmod(output, 0o755)
    print(f'Output file ({len(lines)} files): {output}')

    return output, models


def process_files(
    project, experiment, table, *vars,
    wget=False, dryrun=False, overwrite=False,
    nyears=50, nchunks=10, endyears=False, chunk=False,
):
    """
    Average and standardize the raw files downloaded with the wget script.
    """
    # Initialize cdo (a bit sloow)
    # NOTE: conda cdo is the command-line tool and pip cdo is the python binding
    opts = '-s -O'  # universal
    cdo = Cdo()
    if nyears % nchunks:
        raise ValueError(f'Chunk years {nchunks} to not divide total years {nyears}.')

    # Find available models and variables
    # NOTE: Have to emit warning if files from multiple tables are found.
    # TODO: This should list the files we failed to download.
    pattern = 'Amon' if 'Emon' in table else table
    string = '-'.join(project, experiment, pattern)
    files_all = _wget_files(project, experiment, table, vars)
    if wget:  # search files in wget script
        input = ROOT / string
        output = DATA / string
        if not input.is_dir():
            raise RuntimeError(f'Input directory {input!r} not found.')
        if not output.is_dir():
            raise RuntimeError(f'Output directory {output!r} not found.')
        files_all = input.glob('*.nc')

    # Iterate through models then variables
    # NOTE: The quotes around files are important! contains newlines!
    models = sorted(set(map(get_model, map(str, files_all))))
    if not vars:
        vars = sorted(set(map(get_var, map(str, files_all))))
    print()
    print(f'Table {table}, experiment {experiment}: {len(models)} models found')
    for model, var in itertools.product(models, vars):
        # List files and remove empty files
        files = [file for file in files_all if f'{var}_{table}_{model}_' in file]
        if var == vars[0]:
            print()
            print('Input: {input.name}, Model: {model}')
        for file in files:
            if file.stat().st_size == 0:
                print(f'Warning: Deleting empty input file {file.name}.')
                os.remove(file)
        dates = tuple(map(get_years, map(str, files)))
        prefix = format(f'{var} (unfiltered):', ' <20s')
        if files:
            ymin, ymax = min(tup[0] for tup in dates), max(tup[1] for tup in dates)
            print(f'{prefix} {ymin}-{ymax} ({len(files)} files)')
        else:
            print(f'{prefix} not found')
            continue

        # Select files for averaging
        # TODO: Add year suffix? Then need to auto-remove files with different range.
        # NOTE: Simulations should already be spun up, see parent_experiment_id.
        mfiles = []
        mdates = []
        for file, date in zip(files, dates):
            if date[0] < ymin + nyears - 1:
                mfiles.append(file)
                mdates.append(date)
        tmp = output / 'tmp.nc'
        ymin, ymax = min(tup[0] for tup in mdates), max(tup[1] for tup in mdates)
        out = output / f'{var}_{table}_{model}_{experiment}_{project}.nc'
        exists = out.is_file() and out.stat().st_size > 0
        prefix = format(f'{var} (filtered):', ' <20s')
        print(f'{prefix} {ymin}-{ymax} ({len(mfiles)} files)')
        if exists and not overwrite:
            print('Skipping (output exists)...')
            continue
        if wget:
            print('Skipping (wget only)...')
            continue
        if dryrun:
            print('Skipping (dry run)...')
            continue

        # Take time averages
        # NOTE: Can use mergetime as dedicated func with probably more sanity checks
        # but cdo docs also mention that cat and copy should work.
        # NOTE: This returns either chunk-yearly averages or a blanket time average,
        # optionally uses initial or final years, and ignores NaNs when averaging.
        # TODO: Also detrend data before averaging using simple cdo command? And also
        # interpolate horizontally and vertically to standard grid?
        # cmds = [f'-zonmean -selname,{var} {file}' for file in mfiles]
        if tmp.is_file():
            os.remove(tmp)
        cmd = ' '.join(f'-selname,{var} {file}' for file in mfiles)
        cdo.mergetime(input=cmd, output=str(tmp), options=opts)
        if not tmp.is_file() or tmp.stat().st_size == 0:
            if tmp.is_file():
                os.remove(tmp)
            raise RuntimeError(f'Failed to merge {tuple(f.name for f in mfiles)}.')
        ntime = cdo.ntime(input=str(tmp), options=opts)
        ntime = int(ntime[0])  # returns singleton list of string
        ntime_years = nyears * 12  # TODO: adapt for other time intervals
        ntime_chunks = nchunks * 12
        prefix = format(f'{var} (timesteps):', ' <20s')
        if chunk:
            cmds = []
            for itime in range(1, ntime_years + 1, ntime_chunks):  # endpoint exclusive
                if itime > ntime:
                    print(f'Warning: Requested {ntime_years} steps but file has only {ntime} steps.')  # noqa: E501
                    break
                if itime + ntime_chunks - 1 > ntime:
                    print(f'Warning: Averaging chunk size {ntime_chunks} incompatible with file with {ntime} steps.')  # noqa: E501
                    break
                if endyears:  # note this is endpoint inclusive
                    time1, time2 = ntime - itime - ntime_chunks, ntime - itime + 1
                else:
                    time1, time2 = itime, itime + ntime_chunks - 1
                time1, time2 = np.clip((time1, time2), 1, ntime)
                print(f'{prefix} {time1}-{time2} ({(time2 - time1 + 1) / 12} years)')
                cmds.append(f'-timmean -seltimestep,{time1}/{time2} {tmp}')
            cmd = ' '.join(cmds)  # single command
            cdo.mergetime(input=cmd, output=str(out), options=opts)
        else:
            if ntime_years > ntime:
                print(f'Warning: Requested {ntime_years} but file has only {ntime} steps.')  # noqa: E501
            if endyears:
                time1, = ntime - ntime_years + 1, ntime  # endpoint inclusive
            else:
                time1, time2 = 1, ntime_years
            print(f'{prefix} {time1}-{time2} ({(time2 - time1 + 1) / 12} years)')
            cmd = f'-seltimestep,{time1},{time2} {tmp}'
            cdo.ymonmean(input=cmd, output=str(out), options=opts)
        if out.is_file() and out.stat().st_size > 0:
            print(f'Output: {output.name}/{out.name}')
        else:
            if out.is_file():  # if zero-length
                os.remove(out)
            raise RuntimeError(f'Failed to create output file {out.name}.')
