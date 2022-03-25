#!/usr/bin/env python3
"""
Convenient wrappers for python APIs to download CMIP data.
"""
import itertools
import os
import re
import sys
from pathlib import Path

import numpy as np
from cdo import Cdo
from pyesgf.logon import LogonManager
from pyesgf.search import SearchConnection


# Global functions
__all__ = [
    'process_files',
    'download_wget',
    'filter_wget',
]

# Constants
# TODO: Make these configurable? Add conditionals for other stations?
DATA = Path.home() / 'data'
if sys.platform == 'darwin':
    ROOT = Path.home() / 'data'
else:
    ROOT = Path('/mdata5') / 'ldavis'

# ESGF hosts for OpenID logon and pyesgf usage
# Hosts are listed here: https://esgf.llnl.gov/nodes.html
HOST_URLS = {
    'llnl': 'esgf-node.llnl.gov',
    'ceda': 'esgf-index1.ceda.ac.uk',
    'dkrz': 'esgf-data.dkrz.de',
    'gfdl': 'esgdata.gfdl.noaa.gov',
    'ipsl': 'esgf-node.ipsl.upmc.fr',
    'jpl': 'esgf-node.jpl.nasa.gov',
    'liu': 'esg-dn1.nsc.liu.se',
    'nci': 'esgf.nci.org.au',
    'nccs': 'esgf.nccs.nasa.gov',
}

# ESGF data nodes for downloading sorted by priority
# Nodes and statuses are listed here: https://esgf-node.llnl.gov/status/
# NOTE: Periodically update these lists as participating institutes change (ignore
# esgf node and university department prefixes during sort). Last updated 2022-03-24.
NODE_URLS = [
    'aims3.llnl.gov',
    'cmip.bcc.cma.cn',
    'cmip.dess.tsinghua.edu.cn',
    'cmip.fio.org.cn',
    'cordexesg.dmi.dk',
    'crd-esgf-drc.ec.gc.ca',
    'data.meteo.unican.es',
    'dataserver.nccs.nasa.gov',
    'dist.nmlab.snu.ac.kr',
    'dpesgf03.nccs.nasa.gov',
    'eridanus.eoc.dlr.de',
    'esg-cccr.tropmet.res.in',
    'esg-dn1.nsc.liu.se',
    'esg-dn1.ru.ac.th',
    'esg-dn2.nsc.liu.se',
    'esg.camscma.cn',
    'esg.lasg.ac.cn',
    'esg.pik-potsdam.de',
    'esg1.umr-cnrm.fr',
    'esg2.umr-cnrm.fr',
    'esgdata.gfdl.noaa.gov',
    'esgf-cnr.hpc.cineca.it',
    'esgf-data.csc.fi',
    'esgf-data.ucar.edu',
    'esgf-data1.ceda.ac.uk',
    'esgf-data1.diasjp.net',
    'esgf-data1.llnl.gov',
    'esgf-data2.ceda.ac.uk',
    'esgf-data2.diasjp.net',
    'esgf-data2.llnl.gov',
    'esgf-data3.ceda.ac.uk',
    'esgf-data3.diasjp.net',
    'esgf-ictp.hpc.cineca.it',
    'esgf-nimscmip6.apcc21.org',
    'esgf-node.cmcc.it',
    'esgf-node2.cmcc.it',
    'esgf.anl.gov',
    'esgf.apcc21.org',
    'esgf.bsc.es',
    'esgf.dwd.de',
    'esgf.ichec.ie',
    'esgf.nccs.nasa.gov',
    'esgf.nci.org.au',
    'esgf.rcec.sinica.edu.tw',
    'esgf1.dkrz.de',
    'esgf2.dkrz.de',
    'esgf3.dkrz.de',
    'noresg.nird.sigma2.no',
    'polaris.pknu.ac.kr',
    'vesg.ipsl.upmc.fr'
]

# ESGF data node domains sorted by order of download preference. This reduces download
# time without inadvertantly missing files by restricting search to particular nodes.
# NOTE: Previously we removed bad or down nodes but this is unreliable and has
# to be updated regularly. Instead now prioritize local nodes over distant nodes and
# then wget script automatically skips duplicate files after successful download.
NODE_ORDER = [
    'ucar.edu',  # colorado
    'anl.gov',  # illinois
    'llnl.gov',  # california
    'nccs.nasa.gov',  # maryland
    'gfdl.noaa.gov',  # new jersey
    'ec.gc.ca',  # canada
    'ceda.ac.uk',  # uk
    'ichec.ie',  # ireland
    'ipsl.upmc.fr',  # france
    'umr-cnrm.fr',  # france
    'dkrz.de',  # germany
    'dlr.de',  # germany
    'dwd.de',  # germany
    'pik-potsdam.de',  # germany
    'cineca.it',  # italy
    'cmcc.it',  # italy
    'bsc.es',  # spain
    'unican.es',  # spain
    'dmi.dk',  # denmark
    'nird.sigma2.no',  # norway
    'liu.se',  # sweden
    'csc.fi',  # finland
    'nci.org.au',  # australia
    'diasjp.net',  # japan
    'snu.ac.kr',  # south korea
    'pknu.ac.kr',  # south korea
    'apcc21.org',  # south korea
    'bcc.cma.cn',  # china
    'camscma.cn',  # china
    'fio.org.cn',  # china
    'lasg.ac.cn',  # china
    'tsinghua.edu.cn',  # china
    'sinica.edu.tw',  # taiwan
    'ru.ac.th',  # thailand
    'tropmet.res.in',  # india
]

# ESGF facets obtained with get_facet_options() for SearchContext(project='CMIP5')
# then SearchContext(project='CMIP6') with host https://esgf-node.llnl.gov/esg-search.
# NOTE: Conventions changed between projects so e.g. 'experiment', 'ensemble',
# 'cmor_table', and 'time_frequency' in CMIP5 must be changed to 'experiment_id',
# 'variant_label', 'table_id', and 'frequency' in CMIP6.
# NOTE: 'member_id' is equivalent to 'variant_label' if 'sub_experiment_id' is unset
# and for some reason 'variable' and 'variable_id' are kept as synonyms in CMIP6.
# Node esgf-node.llnl.gov:     11900116 hits for CMIP6 (best!)
# Node esgf-data.dkrz.de:      01009809 hits for CMIP6
# Node esgf-node.ipsl.upmc.fr: 01452125 hits for CMIP6
FACET_CMIP5 = [
    'access', 'cera_acronym', 'cf_standard_name', 'cmor_table', 'data_node',
    'ensemble', 'experiment', 'experiment_family', 'forcing', 'format',
    'index_node', 'institute', 'model', 'product', 'realm', 'time_frequency',
    'variable', 'variable_long_name', 'version'
]
FACET_CMIP6 = [
    'access', 'activity_drs', 'activity_id', 'branch_method', 'creation_date',
    'cf_standard_name', 'data_node', 'data_specs_version', 'datetime_end',
    'experiment_id', 'experiment_title', 'frequency', 'grid', 'grid_label',
    'index_node', 'institution_id', 'member_id', 'nominal_resolution', 'realm',
    'short_description', 'source_id', 'source_type', 'sub_experiment_id', 'table_id',
    'variable', 'variable_id', 'variable_long_name', 'variant_label', 'version'
]
FACET_ORDER = [
    'project', 'model', 'source_id', 'experiment', 'experiment_id',
    'ensemble', 'variant_label', 'cmor_table', 'table_id', 'variable', 'variable_id',
]


# Helper functions for reading lines of wget files
# NOTE: Some files have a '-clim' suffix at the end of the date range.
_get_experiment = lambda line: line.split('_')[3]  # netcdf experiment id
_get_file = lambda line: line.split("'")[1].strip()  # netcdf string name
_get_model = lambda line: line.split('_')[2]  # netcdf model id
_get_node = lambda line: line.split("'")[3].strip()  # node string name
_get_order = lambda line: min((i for i, node in enumerate(NODE_ORDER) if node in _get_node(line)), default=len(NODE_ORDER))  # noqa: E501
_get_table = lambda line: line.split('_')[1]  # netcdf table id
_get_variable = lambda line: line.split('_')[0][1:]  # netcdf variable id (omit quote)
_get_years = lambda line: tuple(int(date[:4]) for date in line.split('.')[0].split('_')[-1].split('-')[:2])  # noqa: E501


def _wget_parse(arg, complement=False):
    """
    Return the download lines of the wget files or their complement. The latter is
    used when constructing a single files file from many wget files.

    Parameters
    ----------
    arg : str or path-like
        The string content or `pathlib.Path` location.
    complement : bool, default: False
        Whether to return the filename lines or the preceding and succeeding lines.
    """
    if isinstance(arg, Path):
        lines = open(arg, 'r').readlines()
    else:
        lines = [_ + '\n' for _ in arg.split('\n')]
    eof = 'EOF--dataset.file.url.chksum_type.chksum'  # esgf filename marker
    idxs = [i for i, line in enumerate(lines) if eof in line]
    if idxs and len(idxs) != 2:
        raise NotImplementedError
    if complement:
        if not idxs:
            raise NotImplementedError
        return lines[:idxs[0] + 1], lines[idxs[1]:]  # return tuple of pairs
    else:
        if idxs:
            lines = lines[idxs[0] + 1:idxs[1]]
        return lines


def _wget_read(
    path='~/data',
    project='cmip6',
    experiment='piControl',
    table='Amon',
    variable='ta',
):
    """
    Return a list of input wget files matching the criteria along with an output
    wget file to be constructed and the associated output folder for NetCDFs.

    Parameters
    ----------
    path : path-like
        The location of the files.
    project, experiment, table, variable : str, optional
        The facets to search for and use in the output folder.
    """
    # TODO: Permit retrieving and naming based on arbitrary facets rather
    # than selecting only four facets. Example: large CESM ensemble.
    # TODO: Auto name wget files based on the files listed in the script. The wgets
    # generated by searches do not generally mix variables or experiments.
    path = Path(path).expanduser()
    parts = [project, experiment, table, variable]
    parts = [sorted(set((part,) if isinstance(part, str) else part)) for part in parts]
    parts[0] = sorted(map(str.lower, parts[0]))  # lowercase project
    source = path / 'unfiltered'
    files = []
    patterns = []
    for i, items in enumerate(itertools.product(*parts)):
        items = list(items)
        items[0] = items[0].lower()  # project is lowercase
        pattern = 'wget_' + '[_-]*'.join(items) + '[_-]*.sh'
        patterns.append(pattern)
        for file in source.glob(pattern):
            if '_' + '-'.join(parts[1]) + '_' not in str(file):
                pass  # enforce an exact match for the experiment?
            files.append(file)
    if not files:
        raise ValueError(f'No wget files found in {path} for pattern(s): {patterns!r}')
    files = sorted(set(files))
    for file in files:
        print(f'Input script: {file}')
    name = 'wget_' + '_'.join('-'.join(part) for part in parts) + '.sh'
    pieces = list(parts)  # group like table ids into the same folder
    pieces[2] = sorted(set('Amon' if s == 'Emon' else s for s in pieces[2]))
    dest = path / '-'.join(piece[0] for piece in pieces[:3]) / name
    return files, dest, *parts


def _wget_write(path, prefix, center, suffix, openid=None):
    """
    Write the wget file to the specified path. Lines are sorted first by model,
    second by variable, and third by node priority.

    Parameters
    ----------
    path : path-like
        The script path.
    prefix, center, suffix : list
        The script components.
    openid : str, optional
        The openid to be hardcoded in the file.
    """
    # NOTE: Previously we detected and removed identical files available from
    # multiple nodes but this was terrible idea. The wget script auto-skip files
    # that both exist and are recorded in the .wget cache so as it loops over files
    # it autoskips entries already downloaded. This lets us maximize probability
    # of retrieving file without having to account for intermittently bad nodes.
    path = Path(path).expanduser()
    sorter = lambda line: (
        _get_experiment(line),
        _get_model(line),
        _get_variable(line),
        _get_table(line),
        _get_order(line),
        line  # last resort is to sort by line (e.g. date range)
    )
    center = sorted(center, key=sorter)
    script = ''.join((*prefix, *center, *suffix))
    if openid is not None:
        script = re.sub('openId=\n', f'openId={openid!r}\n', script)
    if not path.parent.is_dir():
        os.mkdir(path.parent)
    with open(path, 'w') as f:
        f.write(script)
    os.chmod(path, 0o755)
    print(f'Output script ({len(center)} files): {path}\n')
    return path


def download_wget(path='~/data', node='llnl', openid=None, **kwargs):
    """
    Download a wget file using `pyesgf`. The resulting file can be subsequently
    filtered to particular years or models using `filter_wget`.

    Parameters
    ----------
    path : path-like, default: '~/data'
        The output path for the resulting wget file.
    node : str, default: 'llnl'
        The ESGF node to use for searching.
    openid : str, optional
        The openid to hardcode into the resulting wget file.
    **kwargs
        Passed to `~pyesgf.search.SearchContext`.
    """
    # Log on and initalize the connection using requested node
    node = node.lower()
    if node in HOST_URLS:
        host = HOST_URLS[node]
    elif node in HOST_URLS.values():
        host = node
    else:
        raise ValueError(f'Invalid node {node!r}.')
    lm = LogonManager()
    if not lm.is_logged_on():  # surface orography
        lm.logon(username='lukelbd', password=None, hostname=host)
    url = f'https://{host}/esg-search'  # suffix is critical
    conn = SearchConnection(url, distrib=True)

    # Create context with input facets
    # NOTE: Since project is always all-upercase we convert to lowercase
    # for convenience. Otherwise case of variables is preserved.
    facets = kwargs.pop('facets', None)
    facets = facets or list(kwargs)  # default to the search keys
    if isinstance(facets, str):
        facets = facets.split(',')
    ctx = conn.new_context(facets=facets, **kwargs)
    print('Context:', ctx, ctx.facet_constraints)
    print('Hit count:', ctx.hit_count)
    parts = []
    facets = (
        *(facet for facet in FACET_ORDER if facet in ctx.facets),
        *sorted(facet for facet in ctx.facets if facet not in FACET_ORDER)
    )
    for facet in facets:  # facet search
        opts = ctx.facet_constraints.getall(facet)
        part = '-'.join(opt.replace('-', '') for opt in sorted(set(opts)))
        if facet == 'project':
            part = part.lower()
        parts.append(part)
    name = '_'.join(('wget', *parts)) + '.sh'
    path = Path(path).expanduser() / 'unfiltered'

    # Create the wget file
    # NOTE: Thousands of these files can take up significant space... so
    # instead just save into a single script.
    center = []
    prefix = suffix = None
    for j, ds in enumerate(ctx.search()):
        print(f'Dataset {j}:', ds)
        fc = ds.file_context()
        fc.facets = ctx.facets  # TODO: report bug and remove?
        try:
            script = fc.get_download_script()
        except Exception:  # download failed
            print(f'Download {j} failed.')
            continue
        if script.count('\n') <= 1:  # just an error message
            print(f'Download {j} is empty.')
            continue
        lines = _wget_parse(script, complement=False)
        if not prefix and not suffix:
            prefix, suffix = _wget_parse(script, complement=True)
        center.extend(lines)

    # Return filename
    return _wget_write(path / name, prefix, center, suffix, openid=openid)


def filter_wget(
    path='~/data', maxyears=50, endyears=False, intersect=False, overwrite=False,
    models_include=None, models_exclude=None, **kwargs
):
    """
    Filter the input wget files to within some input climatology.

    Parameters
    ----------
    path : path-like, default: '~/data'
        The output path for the resulting data subfolder and wget file.
    maxyears : int, default: 50
        The number of years required for downloading.
    endyears : bool, default: False
        Whether to download from the start or end of the available times.
    intersect : bool, optional
        Whether to enforce intersection over the variable time ranges.
    overwrite : bool, optional
        Whether to overwrite the resulting datasets.
    models_include : str or sequence, optional
        If passed, files not belonging to these models are excluded.
    models_exclude : str or sequence, optional
        If passed, files belonging to thse models are excluded.
    **kwargs
        Passed to `_wget_read`.

    Todo
    ----
    Automatically invoke this from `download_wget` or better to download
    cache of files then filter later?
    """
    # Collect wget files and restrict the models
    # NOTE: To overwrite previous files this script can simply be called with
    # overwrite=True. Then the file is removed so the wget script can be called
    # without -U and thus still skip duplicate files from the same node. Note
    # that if a file is not in the esgf wget script cache then the wget command
    # itself will skip files that already exist unless ESGF_WGET_OPTS includes -O.
    files, dest, _, experiments, tables, variables = _wget_read(path, **kwargs)
    models = set(_get_model(line) for file in files for line in _wget_parse(file))
    if isinstance(models_include, str):
        models_include = (models_include,)
    if isinstance(models_exclude, str):
        models_exclude = (models_exclude,)
    if models_include is not None:
        models &= set(models_include)
    if models_exclude is not None:
        models -= set(models_exclude)
    lines_all = [
        line for file in files for line in _wget_parse(file) if any(
            '_'.join((var, table, model, experiment)).replace('-', '')
            in line.replace('-', '')  # match 'abrupt4xCO2' to 'abrupt-4xCO2'
            for var, table, model, experiment
            in itertools.product(variables, tables, models, experiments)
        )
    ]

    # Iterate over models
    center = []
    prefix, suffix = _wget_parse(files[0], complement=True)
    for model in sorted(models):
        # Get date ranges for each variable, possibly enforcing intersection (transport
        # project) or inspecting each range individually (constraints project).
        # NOTE: Must use maxyears - 1 or else e.g. 50 years with 190001-194912 will
        # not "satisfy" the range and result in the next file downloaded.
        print('Model:', model)
        lines = [line for line in lines_all if f'_{model}_' in line]
        ranges = {}  # year ranges for each variable
        min_, max_ = 10000, -10000  # dummy range when files not found
        for var in sorted(variables):
            years = [_get_years(line) for line in lines if f'{var}_' in line]
            years = (
                min((y for y, _ in years), default=min_),
                max((y for _, y in years), default=max_),
            )
            ranges[var] = years
        print('Initial years: ', end='')
        print(', '.join(k + ' ' + '-'.join(map(str, v)) for k, v in ranges.items()))
        if intersect:
            for var in tuple(ranges):
                ranges[var] = (
                    max((p[0] for p in ranges.values()), default=min_),
                    min((p[1] for p in ranges.values()), default=max_),
                )
        for var in tuple(ranges):
            years = ranges[var]
            if endyears:
                years = (int(max(years[0], years[1] - maxyears + 1)), years[1])
            else:
                years = (years[0], int(min(years[1], years[0] + maxyears - 1)))
            ranges[var] = years
        print('Final years: ', end='')
        print(', '.join(k + ' ' + '-'.join(map(str, v)) for k, v in ranges.items()))

        # Restrict output to within date range. If overwriting then remove old files
        # now so the script can be called without -U and still skip duplicates.
        include = set()
        for line in lines:
            ys = _get_years(line)
            var = _get_variable(line)
            years = ranges[var]
            if ys[1] < years[0] or ys[0] > years[1]:
                continue
            output = dest.parent / _get_file(line)
            if not output.is_file():
                pass
            elif overwrite or output.stat().st_size == 0:  # test if empty due to error
                os.remove(output)
            else:  # could continue but instead rely on caching
                pass
            include.add(line)  # duplicates can be caused by multiple source scripts
        if include:
            center.extend(include)
        else:
            print('No intersecting model data found.')
            models.remove(model)

    # Return filename and model list
    dest = _wget_write(dest, prefix, center, suffix)
    return dest, models


def process_files(
    path='~/data', nyears=50, nchunks=1, endyears=False, chunks=False,
    overwrite=False, dryrun=False, wget=False, **kwargs,
):
    """
    Average and standardize the files downloaded with a wget script.

    Parameters
    ----------
    path : path-like
        The input path for the raw data.
    project, experiment, table, *variables
        The facets to select when searching files.
    nyears : int, default: 50
        The number of years for the resulting climatology.
    nchunks : int, default: 1
        The number of years in each chunk for the resulting times eries.
    endyears : bool, default: False
        Whether to average from the start or end of the available times.
    chunks : bool, optional
        Whether to create an `nyears` long time series of successive `nchunks` long
        time averages or simply average `nyears` of data.
    output : path-like, optional
        The output path for the raw data.
    overwrite : bool, optional
        Whether to overwrite existing files or skip them.
    dryrun : bool, optional
        Whether to only print time information and exit.
    wget : bool, optional
        Whether to print time information about wget files rather than downloaded files.
    **kwargs
        Passed to `_wget_read`.
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
    path = Path(path).expanduser()
    files_all, dest, projects, experiments, tables, variables = _wget_read(**kwargs)
    if not dest.is_dir():
        raise RuntimeError(f'Output directory {dest!r} not found.')
    if not wget:  # search actual files
        path = path / dest.name  # match subfolder name
        if not path.is_dir():
            os.mkdir(path)
        files_all = dest.glob('*.nc')

    # Iterate through models then variables
    # TODO: Support different variants e.g. ensemble membes.
    models = sorted(set(map(_get_model, map(str, files_all))))
    suffix = (
        ('climate', f'timeseries-{nchunks}yr')[chunks]
        + f'-{nyears}yr-'
        + ('start', 'end')[endyears]
    )
    print()
    print(f'Input {dest.name}: {len(models)} found for variables {", ".join(variables)}.')  # noqa: E501
    for project, experiment, model, table, variable in itertools.product(
        projects, experiments, models, tables, variables
    ):
        # List the files and remove empty files
        files = [f for f in files_all if f'{variable}_{table}_{model}_{experiment}' in f]  # noqa: E501
        output = dest / f'{variable}_{table}_{model}_{experiment}_{project}_{suffix}.nc'
        exists = output.is_file() and output.stat().st_size > 0
        print(f'Output: {output.parent.name}/{output.name}')
        for file in files:
            if file.stat().st_size == 0:
                print(f'Warning: Removing empty input file {file.name}.')
                os.remove(file)
        dates = tuple(map(_get_years, map(str, files)))
        prefix = format(f'{variable} (unfiltered):', ' <20s')
        if files:
            ymin, ymax = min(tup[0] for tup in dates), max(tup[1] for tup in dates)
            print(f'{prefix} {ymin}-{ymax} ({len(files)} files)')
        else:
            print(f'{prefix} not found')
            continue

        # Merge files into a single time series
        # TODO: Add year suffix? Then need to auto-remove files with different range.
        # NOTE: Can use mergetime as dedicated func with probably more sanity checks
        # but cdo docs also mention that cat and copy should work.
        mfiles = []
        mdates = []
        for file, date in zip(files, dates):
            if date[0] < ymin + nyears - 1:
                mfiles.append(file)
                mdates.append(date)
        tmp = dest / 'tmp.nc'
        ymin, ymax = min(tup[0] for tup in mdates), max(tup[1] for tup in mdates)
        prefix = format(f'{variable} (filtered):', ' <20s')
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
        if tmp.is_file():
            os.remove(tmp)
        cmd = ' '.join(f'-selname,{variable} {file}' for file in mfiles)
        cdo.mergetime(input=cmd, output=str(tmp), options=opts)
        if not tmp.is_file() or tmp.stat().st_size == 0:
            if tmp.is_file():
                os.remove(tmp)
            raise RuntimeError(f'Failed to merge {tuple(f.name for f in mfiles)}.')

        # Take time averages
        # NOTE: This returns either chunk-yearly averages or a blanket time average,
        # optionally uses initial or final years, and ignores NaNs when averaging.
        # TODO: Also detrend data before averaging using simple cdo command? And also
        # interpolate horizontally and vertically to standard grid?
        # cmds = [f'-zonmean -selname,{var} {file}' for file in mfiles]
        ntime = cdo.ntime(input=str(tmp), options=opts)
        ntime = int(ntime[0])  # returns singleton list of string
        ntime_years = nyears * 12  # TODO: adapt for other time intervals
        ntime_chunks = nchunks * 12
        prefix = format(f'{variable} (timesteps):', ' <20s')
        if chunks:
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
            cdo.mergetime(input=cmd, output=str(output), options=opts)
        else:
            if ntime_years > ntime:
                print(f'Warning: Requested {ntime_years} but file has only {ntime} steps.')  # noqa: E501
            if endyears:
                time1, = ntime - ntime_years + 1, ntime  # endpoint inclusive
            else:
                time1, time2 = 1, ntime_years
            print(f'{prefix} {time1}-{time2} ({(time2 - time1 + 1) / 12} years)')
            cmd = f'-seltimestep,{time1},{time2} {tmp}'
            cdo.ymonmean(input=cmd, output=str(output), options=opts)
        if output.is_file() and output.stat().st_size > 0:
            if output.is_file():  # if zero-length
                os.remove(output)
            raise RuntimeError(f'Failed to create output file {output.name}.')
