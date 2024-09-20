#!/usr/bin/env python3
"""
Code for running various checks and post-processing on cmip data.
"""
from pathlib import Path

import cftime
from cmip_data.download import _parse_script
from cmip_data.process import _parse_dump
from cmip_data.facets import _sort_items

# Constants
base = Path(__file__).parent.parent
scratch = Path('~/scratch/cmip-downloads').expanduser()
institutes = True
starts = False
checksums = False

# Summarize institution ids
# NOTE: Hardcode and hand-sort these results in facets.py to group models by institutes
# and order from least to most complex, then run this again to re-print in nicer order.
# This will also put any newly downloaded unrecognized models at the bottom.
# NOTE: Can survey institutions with following code: { models=(); for f in ts_Amon_*;
# do model=$(echo $f | cut -d_ -f3); [[ " ${models[*]} " =~ " $model " ]] && continue;
# models+=("$model"); echo $f; ncinfo $f | grep -E 'institut(e|ion)_id'; done; } | less
if institutes:
    seen = set()
    files = [
        file for project in ('cmip5', 'cmip6')
        for file in (scratch / f'{project}-picontrol-amon').glob('ts_Amon_*')
        if (model := file.name.split('_')[2]) not in seen and not seen.add(model)
    ]
    debug = open(base / 'INSTITUTES.txt', 'w')
    debug.write('Summary of institution ids.\n')
    debug.write('This was automatically generated by manual_info.py\n')
    base = Path(f'~/{base}').expanduser()
    for file in _sort_items(files, 'model'):
        project = file.parent.name.split('-')[0]
        model = file.name.split('_')[2]
        attrs = _parse_dump(file)[2]
        info, time = attrs[''], attrs['time']
        institute_id = info.get('institute_id', 'unknown')
        institute_id = info.get('institution_id', institute_id)
        debug.write('\n')
        debug.write(f'Project: {project.upper()}\n')
        debug.write(f'Model: {model}\n')
        debug.write(f'Institute: {institute_id}\n')

# Summarize starting and ending dates of files
# TODO: Fix this... currently giving wrong answers.
# NOTE: This was written to try to figure out why Mark Zelinka omitted certain
# models and to ensure latest downloaded data represented the actual base time.
if starts:
    seen = set()
    files = [
        file for project in ('cmip5', 'cmip6')
        for experiment in ('picontrol', 'abrupt4xco2')
        for file in (scratch / f'{project}-{experiment}-amon').glob('ts_Amon_*')
        if (model := file.name.split('_')[2], experiment) not in seen
        and not seen.add((model, experiment))
    ]
    debug = open(base / 'STARTS.txt', 'w')
    debug.write('Summary of experiment start dates.\n')
    debug.write('This was automatically generated by manual_info.py\n')
    for file in _sort_items(files, ('model', 'experiment')):
        # Header information
        project = file.parent.name.split('-')[0]
        model = file.name.split('_')[2]
        experiment = file.name.split('_')[3]
        print(project, model, experiment)
        if 'control' in experiment.lower():
            debug.write('\n')
            units_child = calendar_child = start_parent = None
        debug.write(f'Project: {project.upper()} ')
        debug.write(f'Model: {model} ')
        debug.write(f'Experiment: {experiment}\n')
        attrs = _parse_dump(file)[2]
        # Time information
        info, time = attrs[''], attrs['time']
        start = file.name.split('_')[-1].split('-')[0][:6]
        start = f'{start[:4]}-{start[4:]}-01'
        debug.write(f'Base file: {start}\n')
        units_parent = info.get('parent_time_units', units_child)  # infer
        calendar_parent = info.get('parent_time_calendar', calendar_child)
        units_child = time.get('units', None)
        calendar_child = time.get('calendar', None)
        offset_parent = info.get('branch_time_in_parent')
        offset_child = info.get('branch_time', info.get('branch_time_in_child'))
        for title, offset, units, calendar in (
            ('', offset_child, units_child, calendar_child),
            ('Parent', offset_parent, units_parent, calendar_parent),
        ):
            if any(_ is None for _ in (units, calendar, offset)):
                continue
            offset = int(float(str(offset).rstrip('SLDF')))
            prefix = f'{title} t' if title else 'T'
            date = cftime.num2date(0, units, calendar)
            date = date.strftime('%Y-%m-%d')
            debug.write(f'{prefix}ase time: {date}\n')
            date = cftime.num2date(offset, units, calendar)
            date = date.strftime('%Y-%m-%d')
            debug.write(f'{prefix}ranch time: {date} ')
            debug.write(f'(offset {offset} calendar {calendar})\n')
        if start_parent is None or offset_parent is None:
            start_parent = start
            continue
        offset_parent = int(float(str(offset_parent).rstrip('SLDF')))
        branch_parent = cftime.num2date(offset_parent, units, calendar)
        branch_parent = branch_parent.strftime('%Y-%m-%d')
        if start_parent != branch_parent:
            debug.write(f'WARNING: Base file {start_parent} differs from branch date {branch_parent}.\n')  # noqa: E501
            continue

# Summarize E3SM-1-0 files with different versions
# NOTE: This block is no longer valid, since we now automatically filter datasets
# to their latest available versions inside download_script(). Was originally used
# to help show which variables had multiple checksums available (turned out to be
# just the pressure level data hus, ta, ua, va, and zg), and the older versions of
# these files were manually found to have invalid data (e.g. minimum temperatures
# close to zero on levels with sub-surface grid cells, but the minimum varied, so did
# not seem to simply be an incorrectly encoded missing value like with GISS data).
if checksums:
    print('Reading script files...')
    source = Path('~/scratch').expanduser()
    scripts = sorted(source.glob('unfiltered/wget_cmip6_*.sh'))
    lines = tuple(line for script in scripts for line in _parse_script(script))
    files = {}
    debug = open(base / 'CHECKSUMS.txt', 'w')
    debug.write('Summary of E3SM-1-0 model files with unique checksums.\n')
    debug.write('This was automatically generated by manual_info.py\n\n')
    print('Categorizing script lines...')
    for experiment in ('piControl', 'abrupt-4xCO2'):
        for line in lines:
            var, tab, mod, exp, ens, _, date, *_ = (p.strip("'") for p in line.split('_'))  # noqa: E501
            if (tab, mod, exp) != ('Amon', 'E3SM-1-0', experiment):
                continue
            _, year = (p[:4] for p in date.split('.nc')[0].split('-'))
            if int(year) > 150:
                continue
            file = line.split()[0].strip("'")
            url = line.split()[1].strip("'").split('/')[2]
            sha = line.split()[3].strip("'")
            opts = files.setdefault(file, {})
            urls = opts.setdefault(sha, [])
            urls.append(url)
        for file, opts in files.items():
            debug.write(f'File: {file}\n')
            for i, urls in enumerate(opts.values()):
                debug.write(f'Checksum {i + 1}: ' + ' '.join(urls) + '\n')
            debug.write('\n')
