#!/usr/bin/env python
"""
Filter CMIP5 and CMIP6 wget files.

1. Get models for which we can get daily temperature and flux terms for that
   relaxation timescale estimate.
2. Get models for which climate sensitivity can be calculated by regressing
   monthly surface temperature against TOA flux.
"""
# Imports
import itertools
import os
import sys
from pathlib import Path

# Constants for path management
DELIM = 'EOF--dataset.file.url.chksum_type.chksum'
OPENID = 'https://esgf-node.llnl.gov/esgf-idp/openid/lukelbd'
MAXYEARS = 50  # retain only first N years of each simulation?
ENDYEARS = False  # whether to use the end years instead of start years
# MAXYEARS = 100  # retain 100 years for very solid climatology
if sys.platform == 'darwin':
    ROOT = Path.home() / 'data'
else:  # TODO: add conditionals?
    ROOT = Path('/mdata5') / 'ldavis'

# Helper functions for reading lines of wget files
# NOTE: Some files have a '-clim' suffix at the end of the date range.
get_url = lambda line: line.split("'")[3].strip()
get_file = lambda line: line.split("'")[1].strip()
get_model = lambda line: line.split('_')[2]  # model id from netcdf line
get_years = lambda line: tuple(int(date[:4]) for date in line.split('.')[0].split('_')[-1].split('-')[:2])  # noqa: E501
get_var = lambda line: line.split('_')[0][1:]  # variable from netcdf line


def wget_lines(filename, complement=False):
    """
    Return the download lines or the lines on either side.
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


def wget_files(project='cmip6', experiment='piControl', table='Amon', variables='ta'):
    """
    Return a list of input wget files matching the criteria along with an output
    wget file to be constructed and the associated output folder for NetCDFs.
    """
    # TODO: Permit both 'and' or 'or' logic when searching and filtering files
    # matching options. Maybe add boolean 'intersection' keywords or something.
    # TODO: Auto name wget files based on the files listed in the script. The wgets
    # generated by searches do not generally mix variables or experiments.
    parts = [
        [part] if isinstance(part, str) else part
        for part in (project, experiment, table, variables)
    ]
    path = ROOT / 'wgets'
    input = []
    patterns = []
    for part in itertools.product(*parts):
        patterns.append(pattern := 'wget_' + '[_-]*'.join(part) + '[_-]*.sh')
        input.extend(sorted(path.glob(pattern)))
    if not input:
        raise ValueError(f'No wget files found in {path} for pattern(s): {patterns!r}')
    path = ROOT / '-'.join(part[0] for part in parts[:3])  # TODO: improve
    if not path.is_dir():
        os.mkdir(path)
    name = 'wget_' + '_'.join('-'.join(part) for part in parts) + '.sh'
    output = path / name
    return input, output


def wget_filter(
    models=None, variables=None, maxyears=None, endyears=None,
    badnodes=None, badmodels=None, duplicate=False, overwrite=False,
    **kwargs
):
    """
    Construct the summary wget file (optionally for only the input models).
    """
    # Get all lines for download
    # Also manually replace the open ID
    kwargs['variables'] = variables
    input, output = wget_files(**kwargs)
    prefix, suffix = wget_lines(input[0], complement=True)
    for i, line in enumerate(prefix):
        if line == 'openId=\n':
            prefix[i] = 'openId=' + OPENID + '\n'
            break

    # Collect all download lines for files
    files = set()  # unique file tracking (ignore same file from multiple nodes)
    lines = []  # final lines
    lines_input = [line for file in input for line in wget_lines(file) if line]

    # Iterate by *model*, filter to date range for which *all* variables are available!
    # So far just issue for GISS-E2-R runs but important to make this explicit!
    if models is None:
        models = wget_models(**kwargs)
    if isinstance(badmodels, str):
        badmodels = (badmodels,)
    if isinstance(badnodes, str):
        badnodes = (badnodes,)
    if isinstance(models, str):
        models = (models,)
    if isinstance(variables, str):
        variables = (variables,)
    if maxyears is None:
        maxyears = MAXYEARS
    if endyears is None:
        endyears = ENDYEARS
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
            if dest.exists() and dest.stat().st_size == 0:
                os.remove(dest)  # remove empty files caused by download errors
            if not overwrite and dest.exists():
                continue  # skip if destination exists
            files.add(file)
            lines.append(line)

    # Save bulk download file
    if output.exists():
        os.remove(output)
    print('File:', output)
    with open(output, 'w') as file:
        file.write(''.join(prefix + lines + suffix))
    os.chmod(output, 0o755)
    print(f'Output file ({len(lines)} files): {output}')

    return output, models


def wget_models(models=None, **kwargs):
    """
    Return a set of all models in the group of wget scripts, a dictionary
    with keys of (experiment, variable) pairs listing the models present,
    and a dictionary with 'experiment' keys listing the possible variables.
    """
    # Group models available for various (experiment, variable)
    input, output = wget_files(**kwargs)
    project, experiment, table = output.parent.name.split('-')
    frequency = table[-3:].lower()  # convert e.g. Amon to mon
    models_all = {*()}  # all models
    models_grouped = {}  # models by variable
    for file in input:
        models_file = {get_model(line) for line in wget_lines(file) if line}
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
    models_download = {*()}
    models_ignore = {*()}
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


# Run script
# NOTE: Have 22 daily models with *no shortwave*, but 7 cf daily models *with
# shortwave*. Probable that shortwave contribution to covariance is small, but
# better to have entire budget.
table_monthly = 'Amon'
table_daily = None
# table_daily = 'day'
# table_daily = 'cfDay'
kwargs = {
    'overwrite': False,
    'duplicate': False,
    'badmodels': 'EC-Earth3-CC',
    'badnodes': ('ceda.ac.uk', 'nird.sigma2.no', 'nmlab.snu.ac.kr', 'esg.lasg.ac.cn'),  # noqa: E501
}
if __name__ == '__main__':
    # Get monthly variable data without response
    # vars = ('ta', 'hur', 'hus', 'cl', 'clw', 'cli', 'clwvp', 'clwvi', 'clivi', 'cct')
    # # for project in ('cmip6', 'cmip5'):
    # # for project in ('cmip5',):
    # for project in ('cmip6',):
    #     file, models = wget_filter(
    #         project=project,
    #         # variables='ta',
    #         variables=vars,
    #         experiment='piControl',
    #         table=table_monthly,
    #         endyears=True,
    #         maxyears=100,
    #         **kwargs,
    #     )
    # print()
    # print(f'Models ({len(models)}):', ', '.join(models))

    # Get integrated transport in CMIP6
    vars = ('intuadse', 'intvadse', 'intuaw', 'intvaw')
    kwargs.update({'maxyears': 100, 'endyears': True})
    file, models_control = wget_filter(
        project='cmip6',
        variables=vars,
        experiment='piControl',
        table=table_monthly,
        **kwargs,
    )
    file, models_response = wget_filter(
        project='cmip6',
        variables=vars,
        experiment='abrupt4xCO2',
        table=table_monthly,
        **kwargs,
    )
    both = {*models_response} & {*models_control}
    nocontrol = {*models_response} - {*models_control}
    noresponse = {*models_control} - {*models_response}
    print()
    print(f'Both ({len(both)}):', ', '.join(both))
    print(f'No control ({len(nocontrol)}):', ', '.join(nocontrol))
    print(f'No response ({len(noresponse)}):', ', '.join(noresponse))

    # Get monthly temp data plus daily fluxes
    # Also filter to abrupt experiments for custom sensitivity assessment
    # # for project in ('cmip6', 'cmip5'):
    # # for project in ('cmip5',):
    # for project in ('cmip6',):
    #     file, models_monthly = wget_filter(
    #         project=project,
    #         experiment=('piControl', 'abrupt4xCO2'),
    #         table=table_monthly,
    #         **kwargs
    #     )
    #     file, models_daily = wget_filter(
    #         project=project,
    #         experiment='piControl',
    #         table=table_daily,
    #         models=models_monthly,
    #         **kwargs
    #     )
    #     both = {*models_monthly} & {*models_daily}
    #     nodaily = {*models_monthly} - {*models_daily}
    #     nomonthly = {*models_daily} - {*models_monthly}
    #     print()
    #     print(f'Both ({len(both)}):', ', '.join(both))
    #     print(f'No daily ({len(nodaily)}):', ', '.join(nodaily))
    #     print(f'No monthly ({len(nomonthly)}):', ', '.join(nomonthly))
