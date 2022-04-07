#!/usr/bin/env python3
"""
Convenient wrappers for python APIs to download CMIP data.
"""
import os
import re
import warnings
from pathlib import Path

import numpy as np
from cdo import Cdo
from pyesgf.logon import LogonManager
from pyesgf.search import SearchConnection

__all__ = [
    'download_script',
    'filter_script',
    'process_files',
    'search_connection',
]

# Interpolate grid constants
# NOTE: To determine whether pressure level interpolation is needed we compare zaxisdes
# to this grid. Some grids use floats and have slight offsets while some use exact ints
# so important to use np.isclose() rather than exact comparison.
GRID_HORIZONTAL = ''
GRID_VERTICAL = 100 * np.array(
    [
        1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150,
        100, 70, 50, 30, 20, 10, 5, 1,
    ]
)


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

# ESGF facets obtained with get_facet_options() for SearchContext(project='CMIP5')
# then SearchContext(project='CMIP6') with host https://esgf-node.llnl.gov/esg-search.
# See this page for all vocabularies: https://github.com/WCRP-CMIP/CMIP6_CVs
# Experiments: https://wcrp-cmip.github.io/CMIP6_CVs/docs/CMIP6_experiment_id.html
# Institutions: https://wcrp-cmip.github.io/CMIP6_CVs/docs/CMIP6_institution_id.html
# Models/sources: https://wcrp-cmip.github.io/CMIP6_CVs/docs/CMIP6_source_id.html
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
    ('CMIP5', 'abrupt-4xCO2'): 'abrupt4xCO2',
    ('CMIP5', 'esm-piControl'): 'esmControl',
    ('CMIP5', 'esm-hist'): 'esmHistorical',
    ('CMIP5', 'hist-GHG'): 'historicalGHG',
    ('CMIP5', 'hist-nat'): 'historicalNat',
}
FACET_PRIORITIES = [
    'project',  # same as folder
    'model', 'source_id',
    'experiment', 'experiment_id',  # same as folder
    'ensemble', 'variant_label',
    'table', 'cmor_table', 'table_id',  # same as folder
    'variable', 'variable_id',
]

# Variant labels associated with flagship versions of the pre-industrial control
# and abrupt 4xCO2 experiments. Note the CNRM, MIROC, and UKESM models run both their
# control and abrupt experiments with 'f2' forcing, HadGEM runs the abrupt experiment
# with the 'f3' forcing and the (parent) control experiment with 'f1' forcing, and
# EC-Earth3 strangely only runs the abrupt experiment with the 'r8' realization
# and the (parent) control experiment with the 'r1' control realiztion.
# NOTE: See the download_process.py script for details on available models.
MODEL_FLAGSHIPS = {
    'r8i1p1f1': (
        'EC-Earth3',
    ),
    'r1i1p1f3': (
        'HadGEM3-GC31-LL',
        'HadGEM3-GC31-MM'
    ),
    'r1i1p1f2': (
        'CNRM-CM6-1',
        'CNRM-CM6-1-HR',
        'CNRM-ESM2-1',
        'MIROC-ES2L',
        'MIROC-ES2H',
        'UKESM1-0-LL',
    ),
}

# Helper functions for parsing script lines
# NOTE: Netcdf names are: <variable_id>_<table_id>_<source_id>_<experiment_id>...
# ..._<member_id>_<grid_label>[_<time_range>].nc, where grid labels are described
# here: https://github.com/WCRP-CMIP/CMIP6_CVs/blob/master/CMIP6_grid_label.json.
# However data folders are sorted <project_id>-<experiment_id>-<table_id>.
_line_file = lambda line: line.split()[0].strip("'")
_line_node = lambda line: line.split()[1].strip("'")
_line_date = lambda line: line.split('_')[6].split('.')[0]
_line_range = lambda line: tuple(int(s[:4]) for s in _line_date(line).split('-')[:2])
_line_parts = {
    'model': lambda line: line.split('_')[2],
    'experiment': lambda line: line.split('_')[3],
    'ensemble': lambda line: line.split('_')[4],
    'table': lambda line: line.split('_')[1],
    'variable': lambda line: line.split('_')[0].strip("'"),
    'grid': lambda line: line.split('_')[5].split('.')[0],
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
    key=lambda line: (*(f(line) for f in _line_parts.values()), _sort_index(line), line)
)


def _files_unknown(path='~/data'):
    """
    Print and return a list of the netcdf files not present in the wget scripts for the
    specified inputs. This is printed after adding a script with `filter_wgets`.

    Parameters
    ----------
    path : path-like
        The path to be searched.
    """
    # NOTE: This is generally used to remove unnecessarily downloaded files as
    # users refine their filtering or downloading steps.
    path = Path(path).expanduser()
    files_downloaded = {
        file.name for file in path.glob('*.nc')
    }
    files_scripts = {
        _line_file(line) for file in path.glob('wget*.sh')
        for line in _script_lines(file, complement=False)
    }
    files_finished = files_downloaded & files_scripts
    files_missing = files_scripts - files_downloaded
    files_extra = files_downloaded - files_scripts
    print(f'Finished files ({len(files_finished)}): ', *sorted(files_finished))
    print(f'Missing files ({len(files_missing)}): ', *sorted(files_missing))
    print(f'Extra files ({len(files_extra)})): ', *sorted(files_extra))
    return files_finished, files_missing, files_extra


def _parse_constraints(reverse=False, **constraints):
    """
    Standardize the constraints, accounting for facet aliases and option renames.

    Parameters
    ----------
    reverse : bool, optional
        Whether to reverse translate facet aliases.
    **constraints
        The constraints.
    """
    # NOTE: This sets a default project when called by download_script, filter_script,
    # or process_files, and enforces a standard order for file and folder naming.
    constraints = {
        facet: sorted(opts.split(',') if isinstance(opts, str) else opts)
        for facet, opts in constraints.items()
    }
    if reverse:
        aliases = {(_, facet): alias for (_, alias), facet in FACET_ALIASES.items()}
    else:
        aliases = FACET_ALIASES
    renames = {
        **FACET_RENAMES,
        **{('CMIP6', new): old for (_, old), new in FACET_RENAMES.items()},
    }
    project = constraints.setdefault('project', ['CMIP6'])
    project = project[0] if len(project) == 1 else None
    facets = (
        *(facet for facet in FACET_PRIORITIES if facet in constraints),
        *sorted(facet for facet in constraints if facet not in FACET_PRIORITIES)
    )
    constraints = {
        aliases.get((project, facet), facet):
        sorted(renames.get((project, opt), opt) for opt in constraints[facet])
        for facet in facets
    }
    return project, constraints


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
                value = [s.strip().strip("'").strip('"') for s in value.split(' ')]
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


def _script_lines(arg, complement=False):
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


def _script_write(path, prefix, center, suffix, openid=None, **constraints):
    """
    Write the wget script to the specified path. Lines are sorted first by model,
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
    parts = ('-'.join(opt.lower().replace('-', '') for opt in opts) for opts in constraints.values())  # noqa: E501
    if not path.is_dir():
        os.mkdir(path)
    path = path / ('wget_' + '_'.join(parts) + '.sh')
    script = ''.join((*prefix, *_sort_lines(center), *suffix))
    if openid is not None:
        script = re.sub('openId=\n', f'openId={openid!r}\n', script)
    with open(path, 'w') as f:
        f.write(script)
    os.chmod(path, 0o755)
    ntotal, nunique = len(center), len(set(map(_line_file, center)))
    print(f'Output script ({ntotal} total files, {nunique} unique files): {path}\n')  # noqa: E501
    return path


def search_connection(node='llnl', username=None, password=None):
    """
    Initialize a distributed search connection over the specified node
    with the specified user information.
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


def download_script(
    path='~/data', node='llnl', username=None, password=None, openid=None,
    skip=None, flagship=False, **constraints
):
    """
    Download a wget file using `pyesgf`. The resulting file can be subsequently
    filtered to particular years or models using `filter_script`.

    Parameters
    ----------
    path : path-like, default: '~/data'
        The output path for the resulting wget file.
    node : str, default: 'llnl'
        The ESGF node to use for searching.
    username : str, optional
        The username for logging on.
    password : str, optional
        The password for logging on.
    openid : str, optional
        The openid to hardcode into the resulting wget script.
    skip : callable, optional
        Function that returns whether to skip a dataset id.
    flagship : bool, optional
        Whether to select variant labels corresponding to CMIP6 flagships only.
    **constraints
        Constraints passed to `~pyesgf.search.SearchContext`.
    """
    # Translate constraints and possibly apply flagship filter
    # NOTE: The flagship list should be updated as CMIP6 results filter in.
    # For example EC-Earth3 likely will publish 'r1' in near future.
    func = skip  # avoid recursion issues
    project, constraints = _parse_constraints(**constraints)
    if not flagship:
        pass
    elif project == 'CMIP5':  # sort ensemble constraint
        _, constraints = _parse_constraints(ensemble='r1i1p1', **constraints)
    elif project == 'CMIP6':  # sort ensemble constraint and define dataset filter
        _, constraints = _parse_constraints(ensemble=['r1i1p1f1', *MODEL_FLAGSHIPS], **constraints)  # noqa: E501
        func = lambda s: skip and skip(s) or any(
            f'.{variant}.' in s and not any(f'.{model}.' in s for model in models)
            for variant, models in MODEL_FLAGSHIPS.items()
        )
    else:
        raise ValueError(f'Invalid {project=} for {flagship=}. Must be CMIP[56].')

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
        if func and func(ds.dataset_id):
            print(message % 'skipped!!!')
            continue
        try:
            script = fc.get_download_script(**constraints)
        except Exception:  # download failed
            print(message % 'failed!!!')
            continue
        if script.count('\n') <= 1:  # just an error message
            print(message % 'empty!!!')
            continue
        lines = _script_lines(script, complement=False)
        print(message % f'{len(lines)} files')
        if not prefix and not suffix:
            prefix, suffix = _script_lines(script, complement=True)
        center.extend(lines)

    # Create the script and return its path
    path = Path(path).expanduser() / 'unfiltered'
    dest = _script_write(path, prefix, center, suffix, openid=openid, **constraints)
    return dest


def filter_script(
    path='~/data', maxyears=50, endyears=False, overwrite=False,
    facets_intersect=None, facets_folder=None, **constraints
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
    overwrite : bool, optional
        Whether to overwrite the resulting datasets.
    facets_intersect : str or sequence, optional
        The facets that should be enforced to intersect across other options.
    facets_folder : str or sequence, optional
        The facets that should be grouped into unique folders.
    **constraints
        The constraints.
    """
    # Helper functions for working with wget script line databases
    # NOTE: Since _parse_constraints imposes a default project this will quetly enforces
    # that 'intersect' and 'folder' facets include a project identifier (so the minimum
    # folder indication is e.g. 'cmip5' or 'cmip6'). However since project is not
    # included in script lines and file names we specially propagate it here.
    # NOTE: To overwrite previous files this script can simply be called with
    # overwrite=True. Then the file is removed so the wget script can be called
    # without -U and thus still skip duplicate files from the same node. Note
    # that if a file is not in the ESGF script cache then the wget command itself
    # will skip files that already exist unless ESGF_WGET_OPTS includes -O.
    def _standardize_facets(**facets):  # noqa: E301
        project, facets = _parse_constraints(reverse=True, **facets)
        if facets.keys() - _line_parts.keys() not in (set(), set(('project',))):
            raise ValueError(f'Facets {facets.keys()} must be subset of: {_line_parts.keys()}')  # noqa: E501
        return project, facets
    def _summarize_database(database, group_facets, key_facets, message=None):  # noqa: E301, E501
        if message:
            print(f'{message}:')
        for parent, group in database.items():
            print(', '.join(f'{facet}: {opt}' for facet, opt in zip(group_facets, parent)))  # noqa: E501
            options = tuple(sorted(set(opts)) for opts in zip(*group.keys()))
            print('\n'.join(f'  {facet} ({len(opts)}): ' + ', '.join(opts) for facet, opts in zip(key_facets, options)))  # noqa: E501
        print()
    def _generate_database(source, facets, project=None):  # noqa: E301
        database = {}
        facets = tuple(facets.split(',') if isinstance(facets, str) else facets)
        facets = {facet: () for facet in facets}
        _, facets = _standardize_facets(**facets)
        group_facets = tuple(facet for facet in _line_parts if facet in facets)
        key_facets = tuple(facet for facet in _line_parts if facet not in facets)
        if project:
            group_facets = ('project', *group_facets)
        for opts, content in source:
            if any(opt not in constraints.get(facet, (opt,)) for facet, opt in opts.items()):  # noqa: E501
                continue  # useful during initial consolidation of script lines
            if project:
                opts = {'project': project, **opts}
            group = tuple(opts[facet] for facet in group_facets)
            data = database.setdefault(group, {})
            key = tuple(opts[facet] for facet in key_facets)
            if isinstance(content, str):
                data.setdefault(key, []).append(content)
            elif key in data:
                raise RuntimeError(f'Facet data already present: {group}, {key}.')
            else:
                data[key] = list(content)
        return database, group_facets, key_facets

    # Read the file and group lines into dictionaries indexed by the facets we
    # want to intersect and whose keys indicate the remaining facets, then find the
    # intersection of these facets (e.g., 'ts_Amon_MODEL' for two experiment groups).
    # NOTE: Here we only constrain search to the project, which is otherwise not
    # indicated in native wget script lines. Similar approach to process_files().
    # NOTE: For some reason in CMIP5 have piControl EC-EARTH but abrupt4xCO2
    # EC-Earth so ignore case when comparing. Seems that Mark Zelinka failed to
    # do this! Otherwise available models are exactly the same as his results.
    project, constraints = _standardize_facets(**constraints)
    path = Path(path).expanduser() / 'unfiltered'
    files = list(path.glob(f'wget_{project}_*.sh'))
    prefix, suffix = _script_lines(files[0], complement=True)
    source = [
        ({facet: func(line) for facet, func in _line_parts.items()}, line)
        for file in files for line in _script_lines(file, complement=False)
    ]
    database, group_facets, key_facets = _generate_database(source, facets_intersect, project=project)  # noqa: E501
    keys = set(tuple(database.values())[0].keys())
    keys = keys.intersection(*(set(group.keys()) for group in database.values()))
    _summarize_database(database, group_facets, key_facets, message='Initial groups')
    database = {
        group: {key: lines for key, lines in data.items() if key in keys}
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
    unknowns = []
    database, group_facets, key_facets = _generate_database(source, facets_folder, project=project)  # noqa: E501
    _summarize_database(database, group_facets, key_facets, message='Folder groups')
    for group, data in database.items():
        center = []  # wget script lines
        folder = path.parent / '-'.join(s.lower().replace('-', '') for s in group)
        parts = {facet: (opt,) for facet, opt in zip(group_facets, group)}
        parts.update({facet: opts for facet, opts in constraints.items() if facet not in group_facets})  # noqa: E501
        _, parts = _parse_constraints(**parts)
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
        dest = _script_write(folder, prefix, center, suffix, **parts)
        unknown = _files_unknown(folder, **parts)
        dests.append(dest)
        unknowns.append(unknown)
    return dests, unknowns


def process_files(
    path='~/data', dest=None, climate=True, numyears=50, endyears=False, detrend=False,
    overwrite=False, dryrun=False, **constraints,
):
    """
    Average and standardize the files downloaded with a wget script.

    Parameters
    ----------
    path : path-like
        The input path for the raw data.
    dest : path-like, optional
        The output path for the averaged and standardized data.
    climate : bool, optional
        Whether to create a monthly-mean climatology or an annual-mean time series.
    numyears : int, default: 50
        The number of years used in the annual time series or monthly climatology.
    endyears : bool, default: False
        Whether to use the start or end of the available times.
    detrend : bool, default: False
        Whether to detrend input data. Used with feedback and budget calculations.
    output : path-like, optional
        The output path for the raw data.
    overwrite : bool, optional
        Whether to overwrite existing files or skip them.
    dryrun : bool, optional
        Whether to only print time information and exit.
    **constraints
        Passed to `_parse_constraints`.
    """
    # Find files and restrict to unique constraints
    # NOTE: Here we only constrain search to the project, which is otherwise not
    # indicated in native netcdf filenames. Similar approach to filter_script().
    opts = '-s -O'  # overwrite existing output and engage silent mode
    project, constraints = _parse_constraints(reverse=True, **constraints)
    path = Path(path).expanduser()
    dest = Path(dest).expanduser() if dest else path
    files = list(path.glob(f'{project}*/*.nc'))
    numsteps = numyears * 12  # TODO: adapt for other time intervals
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
    def _check_output(path):
        if not path.is_file() or path.stat().st_size == 0:
            if path.is_file():
                os.remove(path)
            raise RuntimeError(f'Failed to create output file: {path}.')
    print('Initializing cdo python binding.')
    cdo = Cdo()
    cdo.env['CDO_TIMESTAT_DATE'] = 'first'
    descrip = (
        ('timeseries', 'climate')[climate],
        '-'.join(str(numyears), 'year', ('start', 'end')[endyears], ('raw', 'detrend')[detrend]),  # noqa: E501
    )
    outputs = []
    for key, files in database.items():
        # Print information and merge files into single time series
        # TODO: Should also merge surface pressure for model level interpolation
        # and for graying out subsurface pressure regions in figures.
        parts = dict(zip(_line_parts, key))
        name = '_'.join((*files[0].name.split('_')[:6], *descrip))
        output = dest / files[0].parent.name / name
        exists = output.is_file() and output.stat().st_size > 0
        outputs.append(output)
        print('Parts:', ', '.join(parts))
        print('Output:', '/'.join((output.parent.name, output.name)))
        if exists and not overwrite:
            print('  skipping (output exists)...')
            continue

        # Merge files into single time series
        # NOTE: Here use 'mergetime' because it is more explicit and possibly safer but
        # cdo docs also mention that 'cat' and 'copy' can do the same thing.
        tmp_merge = dest / 'tmp_merge.nc'
        dates = tuple(map(_line_parts['years'], map(str, files)))
        ymin, ymax = min(tup[0] for tup in dates), max(tup[1] for tup in dates)
        print(f'  unfiltered: {ymin}-{ymax} ({len(files)} files)')
        files, dates = zip((f, d) for f, d in zip(files, dates) if d[0] < ymin + numyears - 1)  # noqa: E501
        ymin, ymax = min(tup[0] for tup in dates), max(tup[1] for tup in dates)
        print(f'  filtered: {ymin}-{ymax} ({len(files)} files)')
        if dryrun:
            print('  skipping (dry run)...')
            continue
        var = parts['variable']  # ignore other variables
        input = ' '.join(f'-selname,{var} {file}' for file in files)
        cdo.mergetime(input=input, output=str(tmp_merge), options=opts)
        _check_output(tmp_merge)

        # Take time averages
        # NOTE: Here the radiative flux data can be detrended to improve sensitive
        # residual energy budget and feedback calculations. This uses a lot of memory
        # so should not bother with 3D constraint and circulation fields.
        tmp_mean = dest / 'tmp_mean.nc'
        filesteps = int(cdo.ntime(input=str(tmp_merge), options=opts)[0])
        if numsteps > filesteps:
            print(f'  warning: requested {numsteps} steps but only {filesteps} available')  # noqa: E501
        if endyears:
            t1, t2 = filesteps - numsteps + 1, filesteps  # endpoint inclusive
        else:
            t1, t2 = 1, numsteps
        print(f'  timesteps: {t1}-{t2} ({(t2 - t1 + 1) / 12} years)')
        input = f'-seltimestep,{t1},{t2} {tmp_mean}'
        if detrend:
            input = f'-detrend {input}'
        if climate:  # monthly-mean climatology (use 'mean' not 'avg' to ignore NaNs)
            cdo.ymonmean(input=input, output=str(tmp_mean), options=opts)
        else:  # annual-mean time series
            cdo.yearmonmean(input=input, output=str(tmp_mean), options=opts)
        _check_output(tmp_mean)
        os.rename(tmp_mean, output)

    return outputs
