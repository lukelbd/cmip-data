#!/usr/bin/env python3
"""
Download and filter datasets using the ESGF python API.
"""
import builtins
import copy
import re
from pathlib import Path

from pyesgf.logon import LogonManager
from pyesgf.search import SearchConnection

from .facets import (
    ENSEMBLES_FLAGSHIP,
    FACETS_STORAGE,
    KEYS_SUMMARIZE,
    URLS_HOSTS,
    Database,
    Printer,
    glob_files,
    _item_facets,
    _item_file,
    _item_label,
    _item_years,
    _parse_constraints,
    _sort_items,
)

__all__ = [
    'init_connection',
    'download_script',
    'filter_script',
    'summarize_downloads',
]


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


def _write_script(
    path, prefix, center, suffix, openid=None, printer=None, **constraints
):
    """
    Save the wget script to the specified path and sort the file list by
    . Lines are sorted first by model,
    second by variable, and third by node priority.

    Parameters
    ----------
    path : path-like
        The script location.
    prefix, center, suffix : list
        The script line parts. The center lines are sorted.
    openid : str, optional
        The openid to be hardcoded in the file.
    printer : callable, optional
        The print function.
    **constraints
        The constraints.
    """
    # NOTE: The download status logs list files only once because the script code
    # replaces existing file lines. Try: for f in ./*/.wget*; do echo "File: $f"
    # cat $f | cut -d' '| wc -l && cat $f | cut -d' ' -f1 | sort | uniq | wc -l; done
    # NOTE: This repairs the script to skip the modification time check and modify
    # the checksum comparison to auto-redownload available SHA256 files over MD5. This
    # lets us update files with ./wget.sh -U that have been filtered to latest versions
    # by download_script() while preventing oscillatory downloads between identical
    # files with different checksum type and still allowing file version changes
    # involving a switch to the newer SHA256 standard (used for all CMIP6 data).
    openid = openid or ''
    print = printer or builtins.print
    path = Path(path).expanduser()
    path.mkdir(exist_ok=True)
    _, constraints = _parse_constraints(**constraints)
    name = _item_label(*constraints.values())  # e.g. ta-ts_amon_picontrol-abrupt4xco2
    path = path / f'wget_{name}.sh'
    original = r'openId=\n'  # only replace if already unset
    replace = f'openId={openid!r}\n'
    prefix = re.sub(original, replace, ''.join(prefix))
    if openid and replace not in prefix:  # only check if input provided
        print('Warning: Failed to substitute in user openid.')
    original = '"$(get_mod_time_ $file)" == $(echo "$cached" | cut -d \' \' -f2)'
    replace = r'-r "$file"'  # do not skip checksum comparison for updated files
    suffix = re.sub(re.escape(original), replace, ''.join(suffix))
    if replace not in suffix:
        print('Warning: Failed to repair modification time comparison line.')
    original = '"$chksum" == "$(echo "$cached" | cut -d \' \' -f3)"'
    replace = original.replace('-d ', '-d')  # prevent replacing existing replacements
    replace += ' || "${#chksum}" -lt "$(echo "$cached" | cut -d\' \' -f3 | wc -c)"'
    suffix = re.sub(re.escape(original), replace, suffix)
    if replace not in suffix:
        print('Warning: Failed to repair checksum comparison line.')
    center = _sort_items(center, (*_item_facets, 'dates', 'node'))
    with open(path, 'w') as f:
        f.write(prefix + ''.join(center) + suffix)
    path.chmod(0o755)
    ntotal, nunique = len(center), len(set(map(_item_file, center)))
    print(f'Output script ({ntotal} total files, {nunique} unique files): {path}\n')
    return path


def init_connection(node='llnl', username=None, password=None):
    """
    Initialize a distributed search connection over the specified node
    with the specified user information.

    Parameters
    ----------
    node : str, default: 'llnl'
        The ESGF node to use for searching.
    username : str, optional
        The username for connection.
    password : str, optional
        The password for connection.
    """
    node = node.lower()
    urls = URLS_HOSTS
    if node in urls:
        host = urls[node]
    elif node in urls.values():
        host = node
    else:
        raise ValueError(f'Invalid node {node!r}.')
    lm = LogonManager()
    if not lm.is_logged_on():  # surface orography
        lm.logon(username=username, password=password, hostname=host)
    url = f'https://{host}/esg-search'  # suffix is critical
    return SearchConnection(url, distrib=True)


def download_script(
    path='~/data',
    openid=None,
    logging=False,
    flagship_filter=False,
    **constraints
):
    """
    Download a wget script using `pyesgf`. The resulting script can be filtered to
    particular years or intersecting constraints using `filter_script`.

    Parameters
    ----------
    path : path-like, default: '~/data'
        The output path for the resulting wget file.
    openid : str, optional
        The openid to hardcode into the resulting wget script.
    logging : bool, optional
        Whether to log the printed output.
    flagship_filter : bool, optional
        Whether to select only flagship CMIP5 and CMIP6 ensembles.
    **kwargs
        Passed to `init_connection`.
    **constraints
        Passed to `~pyesgf.search.SearchContext`.
    """
    # Translate constraints and possibly apply flagship filter
    # NOTE: The datasets returned by .search() will often be 'empty' but this usually
    # means it is not available from a particular node and available at another node.
    # NOTE: Thousands of these scripts can take up significant space... so instead we
    # consolidate results into a single script. Also increase the batch size from the
    # default of only 50 results to reduce the number of http requests.
    kw = dict(decode=False, restrict=False)
    keys = ('node', 'username', 'password')
    kwargs = {key: constraints.pop(key) for key in keys if key in constraints}
    project, constraints = _parse_constraints(**kw, **constraints)
    print = Printer('download', **constraints) if logging else builtins.print
    conn = init_connection(**kwargs)
    if flagship_filter:
        ensembles = [ens for key, ens in ENSEMBLES_FLAGSHIP.items() if key[0] == project]  # noqa: E501
        if not ensembles:
            raise ValueError(f'Invalid {project=} for {flagship_filter=}. Must be CMIP5 or CMIP6.')  # noqa: E501
        _, constraints = _parse_constraints(**kw, **constraints, ensemble=ensembles)
        def flagship_filter(value):  # noqa: E301
            identifiers = [s.lower() for s in value.split('.')]
            for keys, ensemble in ENSEMBLES_FLAGSHIP.items():
                if all(key and key.lower() in identifiers for key in (*keys, ensemble)):
                    return True
            else:
                return ENSEMBLES_FLAGSHIP[project, None, None] in identifiers

    # Search and parse wget scripts
    # NOTE: This uses the 'dataset' search context to find individual simulations
    # (i.e. 'datasets' in ESGF parlance) then creates a 'file' search context within
    # the individual dataset and generates files form each list.
    # NOTE: Since cmip5 'datasets' contain multiple variables, it is common for
    # newer versions of the same dataset to exlude variables. Therefore we create
    # pseudo-dataset ids for cmip5 data on a per-variable basis.
    # NOTE: Since cmip5 'datasets' contain multiple variables, calling search() on
    # the DatasetSearchContext returned by new_context() then get_download_script()
    # on the resulting DatasetResult could include unwanted files. Therefore use
    # FileContext.constrain() to re-apply constraints to files within each dataset
    # (can also pass constraints to search() or get_download_script(), which both call
    # constrain() internally). See the below pages (the gitlab notebook also filters
    # out duplicate files but mentions the advantage of preserving all replicas in
    # case one download node fails, which is what we use with replace=None below):
    # https://claut.gitlab.io/man_ccia/lab2.html#searching-and-parsing-the-results
    # https://esgf.github.io/esgf-user-support/user_guide.html#narrow-a-cmip5-data-search-to-just-one-variable
    facets = constraints.pop('facets', list(constraints))
    latest = None  # show all versions on each node and use custom filter
    replica = None  # show replicas across other nodes and use custom filter
    ctx = conn.new_context(facets=facets, latest=latest, replica=replica, **constraints)
    print('Context:', ctx.facet_constraints)
    print('Hit count:', ctx.hit_count)
    if ctx.hit_count == 0:
        print('Search returned no results.')
        return
    count = 0
    prefix = suffix = None
    database = {}
    for num, ds in enumerate(ctx.search(batch_size=200)):
        num += 1
        fc = ds.file_context()
        fc.facets = ctx.facets  # TODO: report bug and remove?
        message = f'Dataset {num} (%s): {ds.dataset_id}'
        dataset, node = ds.dataset_id.split('|')
        dataset, version = dataset.rsplit('.', 1)
        if flagship_filter and not flagship_filter(ds.dataset_id):
            print(message % 'non flag!!!')
            continue
        if version[0] != 'v' or not version[1:].isnumeric():
            print(message % 'bad id!!!')
            continue
        try:
            script = fc.get_download_script(**constraints)
        except Exception:  # download failed
            print(message % 'failed!!!')
            continue
        if script.count('\n') < 2:  # just an error message
            print(message % 'failed!!!')
            continue
        lines = _parse_script(script, complement=False)
        count += len(lines)
        print(message % f'{len(lines)} files')
        if not prefix and not suffix:
            prefix, suffix = _parse_script(script, complement=True)
        regex = re.compile('output[0-9]')  # cmip5 has this in place of variable
        variables = sorted(set(map(_item_facets['variable'], lines)))
        for variable in variables:
            ilines = lines
            idataset = dataset
            if len(variables) > 1:
                ilines = [line for line in lines if _item_facets['variable'](line) == variable]  # noqa: E501
                idataset = regex.sub(variable, dataset) if regex.search(dataset) else f'{variable}.{dataset}'  # noqa: E501
            if ilines:
                versions = database.setdefault(idataset, {})
                nodes = versions.setdefault(version, {})
                nodes[num, node] = ilines

    # Create the script from latest versions
    # NOTE: Can run into issues where the same version of a given file uses different
    # MD5 and SHA256 checksums on different nodes, causing differences between the
    # status log checksum and the download checksum, so the wget script thinks a file
    # has been "updated" and should be re-downloaded. Could account for this here by
    # manually syncing file checksums, but instead simply make _wget_script prefer
    # SHA256 hashes when considering whether to update files and skip otherwise.
    # NOTE: Using latest=True with replica=None seems to show older versions that were
    # replicated onto other nodes without an updated newer version. Can account for
    # this using replica=False, but this keeps us from downloading e.g. replicas of
    # a Chinese or Japanese model in North America. Instead manually filter to the
    # latest version using the indicator at the end of all CMIP5 and CMIP6 dataset
    # ids, permitting replicas from arbitrary nodes so that we can target faster and
    # physically closer servers while still automatically updated outdated data with
    # e.g. ./wget.sh -U (note some older datasets use e.g. v1, v2 version strings while
    # later uses e.g. v20190101, v20220101 version strings, support both). Also test
    # *datasets* instead of individual files for convenience (prevents us from having
    # to parse the url) and since they might use different date storage conventions.
    result = []
    skipped = {}
    for dataset, versions in database.items():
        latest = sorted(versions)[-1]
        result.extend(_ for lines in versions[latest].values() for _ in lines)
        for version in versions:
            if version == latest:
                continue
            for num, node in versions[version]:
                dataset_id = f'{dataset}.{version}|{node}'
                skipped[num, dataset_id] = latest
    print()
    print('Skipped files from the following outdated datasets:')
    print(f'Retained {num - len(skipped)} out of {num} datasets (skipped {len(skipped)}).')  # noqa: E501
    print(f'Retained {len(latest)} out of {count} files (skipped {count - len(latest)}).')  # noqa: E501
    for num, dataset_id in sorted(skipped):
        latest = skipped[num, dataset_id]
        print(f'Dataset {num} (latest {latest}): {dataset_id}')
    print()
    path = Path(path).expanduser() / 'unfiltered'
    dest = _write_script(
        path,
        prefix,
        result,
        suffix,
        openid=openid,
        printer=print,
        **constraints
    )
    return dest


def filter_script(
    path='~/data',
    openid=None,
    maxyears=50,
    endyears=False,
    logging=False,
    facets_intersect=None,
    facets_folder=None,
    always_include=None,
    always_exclude=None,
    **constraints
):
    """
    Filter the wget scripts to the input number of years for intersecting
    facet constraints and group the new scripts into folders.

    Parameters
    ----------
    path : path-like, default: '~/data'
        The output path for the resulting data subfolder and wget file.
    openid : str, optional
        The openid to hardcode into the resulting wget script.
    maxyears : int, default: 50
        The number of years required for downloading.
    endyears : bool, default: False
        Whether to download from the start or end of the available times.
    logging : bool, optional
        Whether to log the printed output.
    facets_intersect : str or sequence, optional
        The facets that should be enforced to intersect across other facets.
    facets_folder : str or sequence, optional
        The facets that should be grouped into unique folders.
    always_include : dict-like, optional
        The constraints to always include in the output, ignoring the filters.
    always_exclude : dict-like, optional
        The constraints to always exclude from the output, ignoring the filters.
    **constraints
        Passed to `Printer` and `Database`.
    """
    # Read the file and group lines into dictionaries indexed by the facets we
    # want to intersect and whose keys indicate the remaining facets, then find the
    # intersection of these facets (e.g., 'ts_Amon_MODEL' for two experiment ids).
    # NOTE: Since _parse_constraints imposes a default project this will enforce that
    # we never try to intersect projects and minimum folder id is the project name.
    path = Path(path).expanduser() / 'unfiltered'
    print = Printer('filter', **constraints) if logging else builtins.print
    project = constraints.get('project') or 'cmip6'
    files = sorted(path.glob(f'wget_{project.lower()}_*.sh'))
    print('Source file(s):')
    print('\n'.join(map(str, files)))
    print()
    prefix, suffix = _parse_script(files[0], complement=True)
    source = [line for file in files for line in _parse_script(file, complement=False)]
    facets_intersect = facets_intersect or FACETS_STORAGE
    database = Database(source, facets_intersect, **constraints)
    database.summarize(message='Initial groups', printer=print)
    groups = tuple(database.values())  # the group dictionaries
    keys = set(groups[0]).intersection(*map(set, groups)) if groups else ()  # dict keys
    database.filter(keys, always_include=always_include, always_exclude=always_exclude)
    database.summarize(message='Intersect groups', printer=print)

    # Collect the facets into a dictionary whose keys are the facets unique to each
    # folder and whose values are dictionaries of the remaining facets and script lines.
    # facets and values containing the associated script lines for subsequent filtering.
    # NOTE: Must use maxyears - 1 or else e.g. 50 years with 190001-194912
    # will not "satisfy" the range and result in the next file downloaded.
    dests = []
    facets_folder = facets_folder or FACETS_STORAGE
    source = [line for lines in database for line in lines]
    database = Database(source, facets_folder, **constraints)
    database.summarize(message='Folder groups', printer=print)
    for group, data in database.items():
        group = dict(zip(database.group, group))
        kwargs = {facet: (opt,) for facet, opt in group.items()}
        for facet, opts in constraints.items():
            kwargs.setdefault(facet, opts)
        folder = path.parent / _item_label(group.values())  # e.g. cmip6-picontrol-amon
        center = []  # wget script lines
        print('Writing download script:')
        print(', '.join(f'{key}: {value}' for key, value in group.items()))
        for key, lines in data.items():
            print('  ' + ', '.join(key) + ':', end=' ')
            items = (*key, *group.values())  # special cases
            year1 = min((y for y, _ in map(_item_years, lines)), default=+10000)
            year2 = max((y for _, y in map(_item_years, lines)), default=-10000)
            if all(item in items for item in ('FIO-ESM-2-0', 'piControl')):
                year1 = 401 if year1 in (301, 401) else 400  # not year 300 entries
            if all(item in items for item in ('NESM3', 'piControl')):
                year1 = 700  # not single-level year 500 entries
            if all(item in items for item in ('KIOST-ESM', 'piControl')):
                year1 = 3189  # not outdated year 2689 entries
            print(f'initial {year1}-{year2}', end=' ')
            if endyears:
                year1 = int(max(year1, year2 - maxyears + 1))
            else:
                year2 = int(min(year2, year1 + maxyears - 1))
            print(f'final {year1}-{year2}', end=' ')
            for line in lines:
                y1, y2 = _item_years(line)
                if y2 < year1 or y1 > year2:
                    continue
                center.append(line)
            print()
        dest = _write_script(
            folder,
            prefix,
            center,
            suffix,
            openid=openid,
            printer=print,
            **kwargs
        )
        dests.append(dest)
    return dests


def summarize_downloads(
    *paths,
    facets=None,
    remove_duplicate=False,
    remove_corrupt=False,
    remove_noscript=False,
    remove_nostatus=False,
    **constraints,
):
    """
    Compare the input netcdf files in the folder(s) to the files
    listed in the wget scripts in the same folder(s).

    Parameters
    ----------
    *paths : path-like, optional
        The folder(s) containing input files and wget scripts.
    facets : str, optional
        The facets to group by in the dataset.
    remove : bool, optional
        Whether to remove detected missing files. Use this option with caution!
    **constraints
        Passed to `Printer` and `Database`.
    """
    # Generate script and file databases
    # NOTE: This is generally used to remove unnecessarily downloaded files as
    # users refine their filtering or downloading steps.
    facets = facets or KEYS_SUMMARIZE
    print = Printer('summary', 'downloads', **constraints)
    print('Generating databases.')
    files_downloaded, files_duplicate, files_corrupt = glob_files(
        *paths, project=constraints.get('project', None),
    )
    folders = sorted(set(file.parent for file in files_downloaded))
    names_scripts = [
        _item_file(line) for folder in folders
        for file in folder.glob('wget*.sh')
        for line in _parse_script(file, complement=False)
    ]
    names_status = [
        _item_file(line) for folder in folders
        for file in folder.glob('.wget*.sh.status')
        for line in open(file).readlines()[:-1]  # last line is table key
        if line.split()[0].count('_') >= 4
    ]
    database_downloads = Database(files_downloaded, facets, **constraints)
    database_scripts = Database(names_scripts, facets, **constraints)
    database_status = Database(names_status, facets, **constraints)

    # Partition into separate databases
    # NOTE: Critical to retain only files listed in 'status' logs or else cannot
    # use wget scripts to automatically update.
    print('Finished downloads.')
    database_downloads.summarize(missing=True, rawdata=False, printer=print)
    print('Unfinished downloads.')
    missing_netcdfs = copy.deepcopy(database_scripts)
    for group, data in missing_netcdfs.items():
        for key, names in data.items():
            files = database_downloads.get(group, {}).get(key, ())
            files = [file.name for file in files]
            names[:] = [name for name in names if name not in files]
    missing_netcdfs.summarize(missing=False, rawdata=True, printer=print)
    print('Downloaded but missing from wget scripts.')
    missing_scripts = copy.deepcopy(database_downloads)
    for group, data in missing_scripts.items():
        for key, files in data.items():
            names = database_scripts.get(group, {}).get(key, ())
            files[:] = [file for file in files if file.name not in names]
    missing_scripts.summarize(missing=False, rawdata=True, printer=print)
    print('Downloaded but missing from status logs.')
    missing_status = copy.deepcopy(database_downloads)
    for group, data in missing_status.items():
        for key, files in data.items():
            names = database_status.get(group, {}).get(key, ())
            scripts = database_scripts.get(group, {}).get(key, ())
            files[:] = [file for file in files if file.name not in names and file.name in scripts]  # noqa: E501
    missing_status.summarize(missing=False, rawdata=True, printer=print)

    # Remove unknown files and return file lists
    for remove, files in (
        (remove_duplicate, files_duplicate),
        (remove_corrupt, files_corrupt),
        (remove_noscript, tuple(file for files in missing_scripts for file in files)),
        (remove_nostatus, tuple(file for files in missing_status for file in files)),
    ):
        if not remove or not files:
            continue
        files_string = ', '.join(file.name for file in files)
        response = input(f'Remove {len(files)} files: {files_string} (y/[n])?')
        if response.lower().strip()[:1] == 'y':
            for file in files:
                print(f'Removing file: {file}')
                file.unlink()
    files_downloads = sorted(file for files in database_downloads for file in files)
    return files_downloads, files_duplicate, files_corrupt
