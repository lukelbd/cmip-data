#!/usr/bin/env python3
"""
Download and filter datasets using the ESGF python API.
Convenient wrappers for python APIs to download CMIP data.
"""
import os
import re
from pathlib import Path

import numpy as np
from pyesgf.logon import LogonManager
from pyesgf.search import SearchConnection

__all__ = [
    'compare_files',
    'search_connection',
    'download_script',
    'filter_script',
]

# ESGF hosts for OpenID logon and pyesgf usage
# Hosts are listed here: https://esgf.llnl.gov/nodes.html
# LLNL: 11900116 hits for CMIP6 (best!)
# DKRZ: 01009809 hits for CMIP6
# IPSL: 01452125 hits for CMIP6
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

# ESGF data nodes sorted by order of download preference. This reduces download time
# without inadvertantly missing files by restricting search to particular nodes. The
# available nodes and statuses are listed here: https://esgf-node.llnl.gov/status/
# NOTE: Previously we removed bad or down nodes but this is unreliable and has to be
# updated regularly. Instead now prioritize local nodes over distant nodes and then
# wget script automatically skips duplicate files after successful download.
NODE_PRIORITIES = [
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
    'dmi.dk',  # denmark
    'nird.sigma2.no',  # norway
    'liu.se',  # sweden
    'csc.fi',  # finland
    'cineca.it',  # italy
    'cmcc.it',  # italy
    'bsc.es',  # spain
    'unican.es',  # spain
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

# ESGF facets obtained with get_facet_options() for SearchContext(project='CMIP5')
# then SearchContext(project='CMIP6') with host https://esgf-node.llnl.gov/esg-search.
# See this page for all vocabularies: https://github.com/WCRP-CMIP/CMIP6_CVs
# Experiments: https://wcrp-cmip.github.io/CMIP6_CVs/docs/CMIP6_experiment_id.html
# Institutions: https://wcrp-cmip.github.io/CMIP6_CVs/docs/CMIP6_institution_id.html
# Models/sources: https://wcrp-cmip.github.io/CMIP6_CVs/docs/CMIP6_source_id.html
FACETS_CMIP5 = [
    'access', 'cera_acronym', 'cf_standard_name', 'cmor_table', 'data_node',
    'ensemble', 'experiment', 'experiment_family', 'forcing', 'format',
    'index_node', 'institute', 'model', 'product', 'realm', 'time_frequency',
    'variable', 'variable_long_name', 'version'
]
FACETS_CMIP6 = [
    'access', 'activity_drs', 'activity_id', 'branch_method', 'creation_date',
    'cf_standard_name', 'data_node', 'data_specs_version', 'datetime_end',
    'experiment_id', 'experiment_title', 'frequency', 'grid', 'grid_label',
    'index_node', 'institution_id', 'member_id', 'nominal_resolution', 'realm',
    'short_description', 'source_id', 'source_type', 'sub_experiment_id', 'table_id',
    'variable', 'variable_id', 'variable_long_name', 'variant_label', 'version'
]
FACETS_ORDER = [
    'project',  # same as folder
    'model', 'source_id',
    'experiment', 'experiment_id',  # same as folder
    'ensemble', 'variant_label',
    'table', 'cmor_table', 'table_id',  # same as folder
    'variable', 'variable_id',
]
FACET_ALIASES = {
    ('CMIP5', 'table'): 'cmor_table',
    ('CMIP5', 'institution'): 'institute',
    ('CMIP5', 'frequency'): 'time_frequency',
    ('CMIP6', 'variable'): 'variable_id',
    ('CMIP6', 'table'): 'table_id',
    ('CMIP6', 'model'): 'source_id',
    ('CMIP6', 'experiment'): 'experiment_id',
    ('CMIP6', 'ensemble'): 'variant_label',
    ('CMIP6', 'institution'): 'institution_id',
}
FACET_RENAMES = {
    ('CMIP5', 'hist-nat'): 'historicalNat',
    ('CMIP5', 'hist-GHG'): 'historicalGHG',
    ('CMIP5', 'esm-hist'): 'esmHistorical',
    ('CMIP5', 'esm-piControl'): 'esmControl',
    ('CMIP5', 'abrupt-4xCO2'): 'abrupt4xCO2',
}
FACET_RENAMES = {
    **FACET_RENAMES,
    **{('CMIP6', new_opt): old_opt for (_, old_opt), new_opt in FACET_RENAMES.items()},
}

# Variant labels associated with flagship versions of the pre-industrial control
# and abrupt 4xCO2 experiments. Note the CNRM, MIROC, and UKESM models run both their
# control and abrupt experiments with 'f2' forcing, HadGEM runs the abrupt experiment
# with the 'f3' forcing and the (parent) control experiment with 'f1' forcing, and
# EC-Earth3 strangely only runs the abrupt experiment with the 'r8' realization
# and the (parent) control experiment with the 'r1' control realiztion.
# NOTE: See the download_process.py script for details on available models.
FLAGSHIP_ENSEMBLES = {
    ('CMIP5', None, None): 'r1i1p1',
    ('CMIP6', None, None): 'r1i1p1f1',
    ('CMIP6', 'piControl', 'CNRM-CM6-1'): 'r1i1p1f2',
    ('CMIP6', 'piControl', 'CNRM-CM6-1-HR'): 'r1i1p1f2',
    ('CMIP6', 'piControl', 'CNRM-ESM2-1'): 'r1i1p1f2',
    ('CMIP6', 'piControl', 'MIROC-ES2L'): 'r1i1p1f2',
    ('CMIP6', 'piControl', 'MIROC-ES2H'): 'r1i1p1f2',
    ('CMIP6', 'piControl', 'UKESM1-0-LL'): 'r1i1p1f2',
    ('CMIP6', 'abrupt-4xCO2', 'CNRM-CM6-1'): 'r1i1p1f2',
    ('CMIP6', 'abrupt-4xCO2', 'CNRM-CM6-1-HR'): 'r1i1p1f2',
    ('CMIP6', 'abrupt-4xCO2', 'CNRM-ESM2-1'): 'r1i1p1f2',
    ('CMIP6', 'abrupt-4xCO2', 'MIROC-ES2L'): 'r1i1p1f2',
    ('CMIP6', 'abrupt-4xCO2', 'MIROC-ES2H'): 'r1i1p1f2',
    ('CMIP6', 'abrupt-4xCO2', 'UKESM1-0-LL'): 'r1i1p1f2',
    ('CMIP6', 'abrupt-4xCO2', 'HadGEM3-GC31-LL'): 'r1i1p1f3',
    ('CMIP6', 'abrupt-4xCO2', 'HadGEM3-GC31-MM'): 'r1i1p1f3',
    ('CMIP6', 'abrupt-4xCO2', 'EC-Earth3'): 'r8i1p1f1',
}

# Helper functions for building files
# NOTE: This is used for log files, script files, and folder names (which use
# simply <project_id>-<experiment_id>-<table_id>). Replace the dashes so that
# e.g. models and experiments with dashes are not mistaken for field delimiters.
_join_opts = lambda options: '_'.join(
    '-'.join(opt.lower().replace('-', '') for opt in opts)
    for opts in options
)

# Helper functions for parsing script lines
# NOTE: Netcdf names are: <variable_id>_<table_id>_<source_id>_<experiment_id>...
# ..._<member_id>_[<grid_label>[_<time_range>]].nc where grid labels are new in
# cmip6 (see https://github.com/WCRP-CMIP/CMIP6_CVs/blob/master/CMIP6_grid_label.json).
_line_file = lambda line: line.split()[0].strip("'")
_line_node = lambda line: line.split()[1].strip("'")
_line_date = lambda line: line.split('_')[-1].split('.')[0]
_line_range = lambda line: tuple(int(s[:4]) for s in _line_date(line).split('-')[:2])
_line_parts = {
    'model': lambda line: line.split('_')[2],
    'experiment': lambda line: line.split('_')[3],
    'ensemble': lambda line: line.split('_')[4],
    'table': lambda line: line.split('_')[1],
    'variable': lambda line: line.split('_')[0].strip("'"),
    'grid': lambda line: s if (s := line.split('_')[5].split('.')[0])[0] == 'g' else 'g'
}

# Helper functions for managing lists and databases of script lines
# NOTE: Previously we detected and removed identical files available from
# multiple nodes but this was terrible idea. The wget script auto-skip files
# that both exist and are recorded in the .wget cache so as it loops over files
# it autoskips entries already downloaded. This lets us maximize probability
# of retrieving file without having to account for intermittently bad nodes.
_sort_index = lambda line: min(
    (i for i, node in enumerate(NODE_PRIORITIES) if node in _line_node(line)),
    default=len(NODE_PRIORITIES)
)
_sort_lines = lambda lines: sorted(
    lines,
    key=lambda line: (*(func(line) for func in _line_parts.values()), _sort_index(line), _line_date(line))  # noqa: E501
)


def _parse_constraints(reverse=False, restrict=False, **constraints):
    """
    Standardize the constraints, accounting for facet aliases and option renames.

    Parameters
    ----------
    reverse : bool, optional
        Whether to reverse translate facet aliases.
    restrict : bool, optional
        Whether to restrict to facets readable in script lines.
    **constraints
        The constraints.
    """
    # NOTE: This sets a default project when called by download_script, filter_script,
    # or process_files, and enforces a standard order for file and folder naming.
    constraints = {
        facet: sorted(set(opts.split(',') if isinstance(opts, str) else opts))
        for facet, opts in constraints.items()
    }
    renames = FACET_RENAMES
    if reverse:
        aliases = {(_, facet): alias for (_, alias), facet in FACET_ALIASES.items()}
    else:
        aliases = FACET_ALIASES
    project = constraints.setdefault('project', ['CMIP6'])
    project = project[0] if len(project) == 1 else None
    facets = (
        *(facet for facet in FACETS_ORDER if facet in constraints),
        *sorted(facet for facet in constraints if facet not in FACETS_ORDER)
    )
    constraints = {
        aliases.get((project, facet), facet):
        sorted(renames.get((project, opt), opt) for opt in constraints[facet])
        for facet in facets
    }
    if not restrict:
        pass
    elif constraints.keys() - _line_parts.keys() not in (set(), set(('project',))):
        raise ValueError(f'Facets {constraints.keys()} must be subset of: {_line_parts.keys()}')  # noqa: E501
    return project, constraints


def _parse_script(arg, complement=False):
    """
    Return the download lines of the wget scripts or their complement. The latter is
    used when constructing a single script from many scripts.

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
    eof = 'EOF--dataset.file.url.chksum_type.chksum'  # marker for download lines
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


def _save_script(path, prefix, center, suffix, openid=None, **constraints):
    """
    Save the wget script to the specified path. Lines are sorted first by model,
    second by variable, and third by node priority.

    Parameters
    ----------
    script : str
        The script string.
    prefix, center, suffix : list
        The script parts. The center is sorted.
    openid : str, optional
        The openid to be hardcoded in the file.
    **constraints
        The constraints.
    """
    path = Path(path).expanduser()
    if not path.is_dir():
        os.mkdir(path)
    name = _join_opts(constraints.values())
    path = path / ('wget_' + name + '.sh')
    script = ''.join((*prefix, *_sort_lines(center), *suffix))
    if openid is not None:
        script = re.sub('openId=\n', f'openId={openid!r}\n', script)
    with open(path, 'w') as f:
        f.write(script)
    os.chmod(path, 0o755)
    ntotal, nunique = len(center), len(set(map(_line_file, center)))
    print(f'Output script ({ntotal} total files, {nunique} unique files): {path}\n')  # noqa: E501
    return path


def _generate_printer(prefix, **constraints):
    """
    Return a funcation that simultaneously prints information and records
    the result in a log file named based on the input constraints.

    Parameters
    ----------
    prefix : str
        The log file prefix.
    **constraints
        The constraints.
    """
    name = prefix + '_' + _join_opts(constraints.values()) + '.log'
    path = Path(__file__).parent.parent / 'logs' / name
    if not path.parent.is_dir():
        path.parent.mkdir()
    if path.is_file():
        print(f'Moving previous log to backup: {path}')
        path.rename(str(path) + '.bak')
    def _printer(*args, sep=' ', end='\n'):  # noqa: E306
        print(*args, sep=sep, end=end)
        with open(path, 'a') as f:
            f.write(sep.join(map(str, args)) + end)
    return _printer


def _generate_database(
    source, facets, project=None, flagship_detect=False, **constraints
):
    """
    Return a database whose keys are facet options and whose values are
    dictionaries mapping the remaining facet options to wget script lines.

    Parameters
    ----------
    source : iterable
        The input option dictionaries and corresponding values.
    facets : str or sequence
        The facets to use as group keys.
    project : str, optional
        The scalar project.
    flagship_detect : bool, optional
        Whether to categorize flagship and nonflagship.
    **constraints
        The constraints.
    """
    # NOTE: Since project is not included in script lines and file names we
    # propagate it here. Note it is always used for folders and intersect groups.
    database = {}
    project = constraints.pop('project', None)
    facets = (facets.split(',') if isinstance(facets, str) else facets)
    facets = {facet: () for facet in facets}
    _, facets = _parse_constraints(reverse=True, restrict=True, **facets)
    key_facets = (*(facet for facet in _line_parts if facet not in facets),)
    group_facets = ('project', *(facet for facet in _line_parts if facet in facets))
    for opts, values in source:
        if any(opt not in constraints.get(facet, (opt,)) for facet, opt in opts.items()):  # noqa: E501
            continue  # useful during initial consolidation of script lines
        opts = {'project': project, **opts}
        if flagship_detect and 'flagship' not in (ensemble := opts['ensemble']):
            key1, key2 = (project, opts['experiment'], opts['model']), (project, None, None)  # noqa: E501
            try:
                flagship = FLAGSHIP_ENSEMBLES.get(key1, FLAGSHIP_ENSEMBLES[key2])
            except KeyError:
                raise ValueError('Project CMIP5 or CMIP6 required for flaghip filtering.')  # noqa: E501
            opts['ensemble'] = 'flagship' if ensemble == flagship else 'nonflagship'
        group = tuple(opts[facet] for facet in group_facets)
        data = database.setdefault(group, {})
        key = tuple(opts[facet] for facet in key_facets)
        if isinstance(values, str):
            data.setdefault(key, []).append(values)
        elif key not in data:
            data[key] = list(values)
        else:
            raise RuntimeError(f'Facet data already present: {group}, {key}.')
    return database, group_facets, key_facets


def _summarize_database(database, group_facets, key_facets, message=None):
    """
    Print information about the database.

    Parameters
    ----------
    database : dict
        The database.
    group_facets : str
        The facets used as group keys.
    key_facets : str
        The remaining facets.
    message : str
        An additional message
    """
    if message:
        print(f'{message}:')
    for parent, group in database.items():
        print(', '.join(f'{facet}: {opt}' for facet, opt in zip(group_facets, parent)))
        options = tuple(sorted(set(opts)) for opts in zip(*group.keys()))
        for facet, opts in zip(key_facets, options):
            print(f'  {facet} ({len(opts)}): ' + ', '.join(map(str, opts)))
    print()


def search_connection(node='llnl', username=None, password=None):
    """
    Initialize a distributed search connection over the specified node
    with the specified user information.

    Parameters
    ----------
    node : str, default: 'llnl'
        The ESGF node to use for searching.
    username : str, optional
        The username for logging on.
    password : str, optional
        The password for logging on.
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
        lm.logon(username=username, password=password, hostname=host)
    url = f'https://{host}/esg-search'  # suffix is critical
    return SearchConnection(url, distrib=True)


def compare_files(path='~/data', remove=False):
    """
    Compare the netcdf files in the directory to the netcdf files listed
    in the wget scripts in the same directory.

    Parameters
    ----------
    path : path-like
        The path to be searched.
    remove : bool, optional
        Whether to remove detected missing files. Use this option with caution!
    """
    # NOTE: This is generally used to remove unnecessarily downloaded files as
    # users refine their filtering or downloading steps.
    path = Path(path).expanduser()
    files_downloaded = {
        file.name for file in path.glob('*.nc')
    }
    files_scripts = {
        _line_file(line) for file in path.glob('wget*.sh')
        for line in _parse_script(file, complement=False)
    }
    files_finished = sorted(files_downloaded & files_scripts)
    files_missing = sorted(files_scripts - files_downloaded)
    files_extra = sorted(files_downloaded - files_scripts)
    print('Path:', path)
    print('Finished files:', len(files_finished))
    print('Missing files:', len(files_missing))
    print('Extra files:', len(files_extra))
    if remove:
        response = input(f'Remove {len(files_extra)} files (y/[n])?').lower().strip()
        if response[:1] == 'y':
            print('Removing files...')
            for file in files_extra:
                os.remove(path / file)
    return files_finished, files_missing, files_extra


def download_script(
    path='~/data', node='llnl', username=None, password=None, openid=None,
    dataset_filter=None, flagship_filter=False, **constraints
):
    """
    Download a wget script using `pyesgf`. The resulting script can be filtered to
    particular years or intersecting constraints using `filter_script`.

    Parameters
    ----------
    path : path-like, default: '~/data'
        The output path for the resulting wget file.
    node, username, password : str, defaultflagship_filter 'llnl'
        Passed to `connect_node`.
    openid : str, optional
        The openid to hardcode into the resulting wget script.
    dataset_filter : callable, optional
        Function that returns whether to retain a dataset id.
    flagship_filter : bool, optional
        Whether to select only flagship CMIP5 and CMIP6 ensembles.
    **constraints
        Constraints passed to `~pyesgf.search.SearchContext`.
    """
    # Translate constraints and possibly apply flagship filter
    # NOTE: The datasets returned by .search() will often be 'empty' but this usually
    # means it is not available from a particular node and available at another node.
    # NOTE: The flagship filter must compare lowercase because project identifier is
    # lowercase in cmip5 dataset ids, and cannot parse dataset ids directly because
    # they are unstandardized and differ significantly between cmip5 and cmip6.
    project, constraints = _parse_constraints(**constraints)
    print = _generate_printer('download', **constraints)
    if flagship_filter:
        ensembles = [ensemble for (proj, _, _), ensemble in FLAGSHIP_ENSEMBLES.items() if proj == project]  # noqa: E501
        if ensembles:
            _, constraints = _parse_constraints(ensemble=ensembles, **constraints)
        else:
            raise ValueError(f'Invalid {project=} for {flagship_filter=}. Must be CMIP5 or CMIP6.')  # noqa: E501
        def flagship_filter(string):  # noqa: E301
            identifiers = [s.lower() for s in string.split('.')]
            for keys, ensemble in FLAGSHIP_ENSEMBLES.items():
                if all(key and key.lower() in identifiers for key in keys):
                    break
            else:
                ensemble = FLAGSHIP_ENSEMBLES[project, None, None]
            return ensemble in identifiers

    # Search and parse wget scripts
    # NOTE: Thousands of these scripts can take up significant space... so instead we
    # consolidate results into a single script. Also increase the batch size from the
    # default of only 50 results to reduce the number of http requests.
    # NOTE: This uses the 'dataset' search context to find individual simulations (i.e.
    # 'datasets' in ESGF parlance) then creates a 'file' search context within the
    # individual dataset and generates files form each list.
    # NOTE: Since cmip5 'datasets' often contain multiple variables, calling search()
    # on the DatasetSearchContext returned by new_context() then get_download_script()
    # on the resulting DatasetResult could include unwanted files. Therefore use
    # FileContext.constrain() to re-apply constraints to files within each dataset
    # (can also pass constraints to search() or get_download_script(), which both
    # call constrain() internally). This is in line with other documented approaches for
    # both the GUI and python API. See the below pages (the gitlab notebook also filters
    # out duplicate files but mentions the advantage of preserving all duplicates in
    # case one download node fails, which is what we use in our wget script workflow):
    # https://claut.gitlab.io/man_ccia/lab2.html#searching-and-parsing-the-results
    # https://esgf.github.io/esgf-user-support/user_guide.html#narrow-a-cmip5-data-search-to-just-one-variable
    center = []
    prefix = suffix = None
    facets = constraints.pop('facets', list(constraints))
    conn = search_connection(node, username, password)
    ctx = conn.new_context(facets=facets, **constraints)
    print('Context:', ctx.facet_constraints)
    print('Hit count:', ctx.hit_count)
    if ctx.hit_count == 0:
        print('Search returned no results.')
        return
    for i, ds in enumerate(ctx.search(batch_size=200)):
        fc = ds.file_context()
        fc.facets = ctx.facets  # TODO: report bug and remove?
        message = f'Dataset {i} (%s): {ds.dataset_id}'
        if dataset_filter and not dataset_filter(ds.dataset_id):
            print(message % 'skipped!!!')
            continue
        if flagship_filter and not flagship_filter(ds.dataset_id):
            print(message % 'nonflag!!!')
            continue
        try:
            script = fc.get_download_script(**constraints)
        except Exception:  # download failed
            print(message % 'failed!!!')
            continue
        if script.count('\n') <= 1:  # just an error message
            print(message % 'empty!!!')
            continue
        lines = _parse_script(script, complement=False)
        print(message % f'{len(lines)} files')
        if not prefix and not suffix:
            prefix, suffix = _parse_script(script, complement=True)
        center.extend(lines)

    # Create the script and return its path
    path = Path(path).expanduser() / 'unfiltered'
    dest = _save_script(path, prefix, center, suffix, openid=openid, **constraints)
    return dest


def filter_script(
    path='~/data', maxyears=50, endyears=False, overwrite=False,
    facets_intersect=None, facets_folder=None, flagship_filter=False, overrides=None,
    **constraints
):
    """
    Filter the wget scripts to the input number of years for intersecting
    facet constraints and group the new scripts into folders.

    Parameters
    ----------
    path : path-like, default: '~/data'
        The output path for the resulting data subfolder and wget file.
    maxyears : int, default: 50
        The number of years required for downloading.
    endyears : bool, default: False
        Whether to download from the start or end of the available times.
    overwrite : bool, optional
        Whether to overwrite the resulting datasets.
    facets_intersect : str or sequence, optional
        The facets that should be enforced to intersect across other facets.
    facets_folder : str or sequence, optional
        The facets that should be grouped into unique folders.
    flagship_filter : bool, optional
        Whether to group ensembles according to flagship or nonflagship identity.
    overrides : dict-like, optional
        The constraints to always include in the output, ignoring the intersect filter.
    **constraints
        The constraints.
    """
    # Read the file and group lines into dictionaries indexed by the facets we
    # want to intersect and whose keys indicate the remaining facets, then find the
    # intersection of these facets (e.g., 'ts_Amon_MODEL' for two experiment ids).
    # NOTE: Since _parse_constraints imposes a default project this will quietly
    # enforce that 'intersect' and 'folder' facets include a project identifier (so
    # the minimum folder indication will be e.g. 'cmip5' or 'cmip6').
    # NOTE: To overwrite previous files this script can simply be called with
    # overwrite=True. Then the file is removed so the wget script can be called
    # without -U and thus still skip duplicate files from the same node. Note
    # that if a file is not in the ESGF script cache then the wget command itself
    # will skip files that already exist unless ESGF_WGET_OPTS includes -O.
    overrides = overrides or {}
    _, overrides = _parse_constraints(reverse=True, restrict=True, **overrides)
    project, constraints = _parse_constraints(reverse=True, restrict=True, **constraints)  # noqa: E501
    print = _generate_printer('filter', **constraints)
    path = Path(path).expanduser() / 'unfiltered'
    files = list(path.glob(f'wget_{project.lower()}_*.sh'))
    print('Source file(s):')
    print('\n'.join(map(str, files)))
    print()
    prefix, suffix = _parse_script(files[0], complement=True)
    source = [
        ({facet: func(line) for facet, func in _line_parts.items()}, line)
        for file in files for line in _parse_script(file, complement=False)
    ]
    database, group_facets, key_facets = _generate_database(
        source, facets_intersect, project=project, flagship_detect=flagship_filter,
    )
    _summarize_database(database, group_facets, key_facets, message='Initial groups')
    keys = set(tuple(database.values())[0].keys())
    keys = keys.intersection(*(set(group.keys()) for group in database.values()))
    database = {
        group: {
            key: lines for key, lines in data.items() if key in keys or any(
                opt in overrides.get(facet, ()) for facet, opt in zip(key_facets, key)
            )
        }
        for group, data in database.items()
    }
    _summarize_database(database, group_facets, key_facets, message='Intersect groups')

    # Collect the facets into a dictionary whose keys are the facets unique to
    # each folder and whose values are dictionaries with keys indicating the remaining
    # facets and values containing the associated script lines for subsequent filtering.
    # NOTE: Must use maxyears - 1 or else e.g. 50 years with 190001-194912
    # will not "satisfy" the range and result in the next file downloaded.
    source = [
        (dict(zip((*group_facets, *key_facets), (*group, *key))), lines)
        for group, data in database.items() for key, lines in data.items()
    ]
    dests = []
    database, group_facets, key_facets = _generate_database(
        source, facets_folder, project=project, flagship_detect=flagship_filter,
    )
    _summarize_database(database, group_facets, key_facets, message='Folder groups')
    for group, data in database.items():
        center = []  # wget script lines
        folder = path.parent / _join_opts((group,))
        parts = {facet: (opt,) for facet, opt in zip(group_facets, group)}
        parts.update({facet: opts for facet, opts in constraints.items() if facet not in group_facets})  # noqa: E501
        _, parts = _parse_constraints(reverse=True, restrict=True, **parts)
        print('Writing script:', ', '.join(group))
        for key, lines in data.items():
            print('  ' + ', '.join(key) + ':', end=' ')
            years = (
                min((y for y, _ in map(_line_range, lines)), default=+10000),
                max((y for _, y in map(_line_range, lines)), default=-10000),
            )
            print('initial', '-'.join(map(str, years)), end=' ')
            if endyears:
                years = (int(max(years[0], years[1] - maxyears + 1)), years[1])
            else:
                years = (years[0], int(min(years[1], years[0] + maxyears - 1)))
            print('final', '-'.join(map(str, years)), end=' ')
            for line in lines:
                ys = _line_range(line)
                if ys[1] < years[0] or ys[0] > years[1]:
                    continue
                output = folder / _line_file(line)
                if output.is_file() and (overwrite or output.stat().st_size == 0):
                    print('removed {output.name}', end=' ')
                    os.remove(output)
                center.append(line)
            print()
        dest = _save_script(folder, prefix, center, suffix, **parts)
        dests.append(dest)
    return dests
