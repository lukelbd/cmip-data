#!/usr/bin/env python3
"""
Utilities related to facet handling and logging.
"""
import builtins
from pathlib import Path


__all__ = [
    'FacetDatabase',
    'FacetPrinter',
]


# ESGF hosts for logon and nodes for downloading
# Hosts are listed here: https://esgf.llnl.gov/nodes.html
# Nodes and statuses are listed here: https://esgf-node.llnl.gov/status/
# LLNL: 11900116 hits for CMIP6 (best!)
# DKRZ: 01009809 hits for CMIP6
# IPSL: 01452125 hits for CMIP6
NODES_HOSTS = {
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
NODES_DATASETS = [
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


# ESGF data nodes sorted by order of download preference. This reduces download time
# without inadvertantly missing files by restricting search to particular nodes.
# NOTE: Previously we removed bad or down nodes but this is unreliable and has to be
# updated regularly. Instead now prioritize local nodes over distant nodes and then
# wget script automatically skips duplicate files after successful download (see top).
FACETS_FOLDER = (  # default facets for folder naming
    'project', 'experiment', 'table'
)
FACETS_SUMMARY = (  # default facets for summary reports
    'project', 'experiment', 'table', 'variable'
)
FACET_ORDER = [
    'project',  # same as folder
    'model', 'source_id',
    'experiment', 'experiment_id',  # same as folder
    'ensemble', 'variant_label',
    'table', 'cmor_table', 'table_id',  # same as folder
    'variable', 'variable_id',
    'grid_label',
]
FACET_PRIORITIES = {
    'experiment': [  # control runs come first
        'control',
        'historical',
        'hist',
    ],
    'node': [  # closer nodes come first
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
    ],
}

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
    ('CMIP6', 'historicalNat'): 'hist-nat',
    ('CMIP6', 'historicalGHG'): 'hist-GHG',
    ('CMIP6', 'esmHistorical'): 'esm-hist',
    ('CMIP6', 'esmControl'): 'esm-piControl',
    ('CMIP6', 'abrupt4xCO2'): 'abrupt-4xCO2',
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
    ('CMIP6', 'piControl', 'MIROC-ES2H'): 'r1i1p4f2',
    ('CMIP6', 'piControl', 'UKESM1-0-LL'): 'r1i1p1f2',
    ('CMIP6', 'abrupt-4xCO2', 'CNRM-CM6-1'): 'r1i1p1f2',
    ('CMIP6', 'abrupt-4xCO2', 'CNRM-CM6-1-HR'): 'r1i1p1f2',
    ('CMIP6', 'abrupt-4xCO2', 'CNRM-ESM2-1'): 'r1i1p1f2',
    ('CMIP6', 'abrupt-4xCO2', 'MIROC-ES2L'): 'r1i1p1f2',
    ('CMIP6', 'abrupt-4xCO2', 'MIROC-ES2H'): 'r1i1p4f2',
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
    '-'.join(
        opt.lower().replace('-', '')
        for opt in ((opts,) if isinstance(opts, str) else tuple(opts))
    ) for opts in options
)

# Helper functions for parsing netcdf file names and wget script lines
# NOTE: Netcdf names are <variable_id>_<table_id>_<source_id>_<experiment_id>...
# ..._<member_id>[_<grid_label>[_<start_date>-<end_date>[-<climate_indicator>]]].nc
# where grid labels are only in cmip6 and climate indicator is used for e.g. pfull.
# See: https://github.com/WCRP-CMIP/CMIP6_CVs/blob/master/CMIP6_grid_label.json
_item_part = lambda file, idx: (
    getattr(file, 'name', file)  # str or Path input
    .strip("'").split('.')[0].split('_')[idx]
)
_item_file = lambda line: line.split()[0].strip("'")
_item_node = lambda line: line.split()[1].strip("'")
_item_dates = lambda file: d if (d := _item_part(file, -1))[0].isdecimal() else ''
_item_years = lambda file: tuple(int(s[:4]) for s in _item_dates(file).split('-')[:2])
_main_parts = {
    'model': lambda file: _item_part(file, 2),
    'experiment': lambda file: _item_part(file, 3),
    'ensemble': lambda file: _item_part(file, 4),
    'table': lambda file: _item_part(file, 1),
    'variable': lambda file: _item_part(file, 0),
    'grid': lambda file: g if (g := _item_part(file, -1))[0] == 'g' else 'g',
}
_file_parts = {**_main_parts, 'dates': _item_dates}
_line_parts = {**_file_parts, 'node': _item_node}

# Helper functions for sorting netcdf file names and wget script lines
# NOTE: Previously we sorted files alphabetically but this means when processing
# files for the first time we iterate over abrupt variables before the control
# variables required for drift correction are available. The generalized priority
# sorting ensures control data comes first and leaves room for additional restrictions.
# NOTE: Previously we detected and removed identical files available from
# multiple nodes but this was terrible idea. The scripts will auto-skip files
# that both exist and are recorded in the .wget cache file, so as they loop over
# files they ignore already-downloaded entries. This lets us maximize probability
# and speed of retrieving files without having to explicitly filter bad nodes.
_sort_facet = lambda items, facet: sorted(
    set(items), key=lambda item: _sort_index(item, facet)
)
_sort_facets = lambda items, facets: sorted(
    items, key=lambda item: tuple(_sort_index(item, facet) for facet in facets),
)
_sort_index = lambda item, facet: (
    (part := _line_parts[facet](item).lower() if '_' in getattr(item, 'name', item) else item)  # noqa: E501
    and (priorities := (*FACET_PRIORITIES.get(facet, ()), part))  # noqa: E501
    and min(f'{i}_{part}' for i, opt in enumerate(priorities) if opt in part)
)


def _glob_files(*paths, pattern='*', project=None):
    """
    Find netcdf files using the input pattern, ignoring duplicates with
    different file extensions.

    Parameters
    ----------
    *path : path-like
        The source path(s).
    pattern : str, default: '*'
        The glob pattern preceding the file extension.
    """
    project = (project or 'cmip6').lower()
    paths = paths or ('~/data',)
    files = _sort_facets(  # have only seen .nc4 but this is in case
        (
            file for ext in ('.nc', '.nc[0-9]') for path in paths
            for folder in Path(path).expanduser().glob(f'{project}*')
            for file in folder.glob(pattern + ext)
            if file.name.count('_') >= 4 and (  # components can be parsed
                file.stat().st_size > 0
                or print(f'Warning: Removing empty download file {file.name!r}')
                or file.unlink()
            )
        ), facets=tuple(_file_parts)
    )
    files_duplicate = []
    files_filtered = [
        file for file in files if file.suffix == '.nc'
        or file.parent / (file.stem + '.nc') not in files
        or files_duplicate.append(file)  # always false
    ]
    if files_duplicate:
        print(
            'Ignoring duplicate files with alternative nc extensions:',
            ', '.join(file.name for file in files_duplicate)
        )
    return files_filtered, files_duplicate


def _parse_constraints(reverse=True, restrict=True, **constraints):
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
    # or process_files, and enforces a standard order for file and folder naming. Also
    # enforced that control runs come before abrupt runs for consistency with database.
    constraints = {
        facet: list(set(opts.split(',') if isinstance(opts, str) else opts))
        for facet, opts in constraints.items()
    }
    renames = FACET_RENAMES
    if reverse:
        aliases = {(_, facet): alias for (_, alias), facet in FACET_ALIASES.items()}
    else:
        aliases = FACET_ALIASES
    projects = constraints.setdefault('project', ['CMIP6'])
    if len(projects) != 1:
        raise NotImplementedError('Non-scalar projects are not supported.')
    projects[:] = [project := projects[0].upper()]
    facets = (
        *(facet for facet in FACET_ORDER if facet in constraints),  # impose order
        *(facet for facet in constraints if facet not in FACET_ORDER),  # keep order
    )
    constraints = {
        aliases.get((project, facet), facet):
        _sort_facet((renames.get((project, opt), opt) for opt in constraints[facet]), facet)  # noqa: E501
        for facet in facets
    }
    if not restrict:
        pass
    elif not constraints.keys() - _main_parts.keys() <= {'project'}:
        raise ValueError(f'Facets {constraints.keys()} must be subset of: {_main_parts.keys()}')  # noqa: E501
    return project, constraints


class FacetPrinter(object):
    """
    Simultaneously print the results and print to a log file named
    automatically based on the input facet constraints.
    """
    def __init__(self, prefix, suffix=None, backup=False, **constraints):
        """
        Parameters
        ----------
        prefix : str
            The log file prefix.
        suffix : str, optional
            The optional log file suffix.
        backup : bool, optional
            Whether to backup existing files.
        **constraints
            The constraints.
        """
        suffix = suffix or ''
        suffix = suffix and '_' + suffix
        _, constraints = _parse_constraints(**constraints)
        name = prefix + '_' + _join_opts(constraints.values()) + suffix + '.log'
        path = Path(__file__).parent.parent / 'logs' / name
        path.parent.mkdir(exist_ok=True)
        if path.is_file():
            if backup:
                print(f'Moving previous log to backup: {path}')
                path.rename(str(path) + '.bak')
            else:
                print(f'Removing previous log: {path}')
                path.unlink(missing_ok=True)
        self._path = path
        self._constraints = constraints

    def __call__(self, *args, sep=' ', end='\n'):
        r"""
        Parameters
        ----------
        *args
            Objects to print.
        sep : str, default: ' '
            Separation between objects
        end : str, default: '\n'
            Ending after last printed object.
        """
        print(*args, sep=sep, end=end)
        with open(self._path, 'a') as f:
            f.write(sep.join(map(str, args)) + end)


class FacetDatabase(object):
    """
    A database whose keys are facet options and whose values are dictionaries
    mapping the remaining facet options to file names or wget script lines.
    """
    def __str__(self):
        return self.summarize(printer=lambda *args, **kwargs: None)

    def __len__(self):  # total number of files or lines
        return sum(len(data.values()) for data in self._database.values())

    def __iter__(self):  # iterate through lists of files or lines
        for data in self._database.values():
            yield from data.values()

    def __init__(self, *args, **kwargs):
        if len(args) != 2:
            raise TypeError(f'Expected 2 positional arguments. Got {len(args)}.')
        self.reset(*args, **kwargs)

    def keys(self):
        return self._database.keys()

    def values(self):
        return self._database.values()

    def items(self):
        return self._database.items()

    def filter(self, keys, always_include=None, always_exclude=None):
        """
        Filter the database keys. Compare to `.restrict`.

        Parameters
        ----------
        keys : sequence, optional
            Valid keys for the database.
        always_include : dict or sequence, optional
            Dictionary describing valid constraints, or a list thereof.
        always_exclude : dict or sequence, optional
            Dictionary describing invalid constraints, or a list thereof.
        """
        project = self._project
        overrides = []
        for override in (always_include, always_exclude):
            override = override or ()
            if not isinstance(override, (tuple, list)):
                override = (override,)
            dicts = []
            for constraints in override:
                _, constraints = _parse_constraints(project=project, **constraints)
                constraints.pop('project', None)
                dicts.append(constraints)
            overrides.append(dicts)
        always_include, always_exclude = overrides
        for group, data in self._database.items():
            for key in tuple(data):
                parts = dict(zip((*self._group, *self._key), (*group, *key)))
                if (
                    key not in (keys or data)
                    and not any(
                        all(parts[facet] in opts for facet, opts in constraints.items())
                        for constraints in always_include
                    )
                    or any(
                        all(parts[facet] in opts for facet, opts in constraints.items())
                        for constraints in always_exclude
                    )
                ):
                    del data[key]

    def reset(self, *args, flagship_translate=False, **constraints):
        """
        Reset the database.

        Parameters
        ----------
        source : sequence of str or path-like, optional
            The script lines or file paths.
        facets : sequence of str
            The facets to use as group keys.
        flagship_translate : bool, optional
            Whether to categorize flagship and nonflagship.
        **constraints : optional
            The constraints.
        """
        # NOTE: Since project is not included in script lines and file names we
        # propagate it here. Also must always be scalar since not included in
        # native file names or script lines and would be difficult to track
        # outside of simply globbing the correct scripts and files.
        if len(args) == 1:
            source, facets = [item for items in self for item in items], *args
        elif len(args) == 2:
            source, facets = args
        else:
            raise TypeError(f'Expected 1 or 2 positional arguments. Got {len(args)}.')
        facets = facets.split(',') if isinstance(facets, str) else tuple(facets)
        project, constraints = _parse_constraints(**constraints)
        facets_group = ('project', *(facet for facet in _main_parts if facet in facets))
        facets_key = tuple(facet for facet in _main_parts if facet not in facets)
        self._group = facets_group
        self._key = facets_key
        self._project = project
        self._constraints = constraints
        self._database = {}
        for value in _sort_facets(source, facets=(*facets_group[1:], *facets_key)):
            parts = {'project': project}
            parts.update({facet: func(value) for facet, func in _main_parts.items()})
            if any(opt not in constraints.get(facet, (opt,)) for facet, opt in parts.items()):  # noqa: E501
                continue
            key_flagship = (project, parts['experiment'], parts['model'])
            ens_default = FLAGSHIP_ENSEMBLES[project, None, None]
            ens_flagship = FLAGSHIP_ENSEMBLES.get(key_flagship, ens_default)
            if flagship_translate and 'flagship' not in (ens := parts['ensemble']):
                parts['ensemble'] = 'flagship' if ens == ens_flagship else 'nonflagship'
            group = tuple(parts[facet] for facet in facets_group)
            data = self._database.setdefault(group, {})
            key = tuple(parts[facet] for facet in facets_key)
            data.setdefault(key, []).append(value)

    def summarize(self, missing=True, printer=None, message=None):
        """
        Print information about the database.

        Parameters
        ----------
        missing : bool, optional
            Whether to include info about missing facets.
        printer : callable, optional
            The print function.
        message : str
            An additional message
        """
        lines = []
        print = lambda *args, **kwargs: (
            lines.append(' '.join(map(str, args)))
            or (printer or builtins.print)(*args, **kwargs)
        )
        flush = lambda data: tuple(
            print(f'  {facet}s ({len(opts)}):', ', '.join(map(str, _sort_facet(opts, facet))))  # noqa: E501
            for facet, opts in data.items() if opts
        )
        everything = {
            group: {facet: set(opts) for facet, opts in zip(self._key, zip(*data))}
            for group, data in self._database.items()
        }
        available = {
            facet: set(_ for opts in everything.values() for _ in opts.get(facet, ()))
            for facet in self._key
        }
        if message:
            print(f'{message}:')
        for group, opts in everything.items():
            header = ', '.join(
                f'{facet}: {opt}' for facet, opt in zip(self._group, group)
            )
            unavailable = {
                facet: available[facet] - set(opts[facet])
                for facet in self._key
            }
            print(header)
            if missing:
                print('unavailable:')
                flush(unavailable)
                print('available:')
            flush(opts)
        print()
        return '\n'.join(lines)
