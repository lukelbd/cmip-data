#!/usr/bin/env python
#------------------------------------------------------------------------------#
# 1) Get models for which we can calculate climate sensitivity, by
# regressing temperature against TOA flux; need climatologcal temp and TOA
# budget on monthly means for control and forced run.
# 2) Get models for which we can get daily temperature and
# flux terms for that timescale estimate.
# WARNING: This was too complicated to do with bash script, need sets and
# dictionary objects! Worth the extra verbosity!
#------------------------------------------------------------------------------#
# Imports and globals
from glob import glob
import os
import re
root = '/mdata2/ldavis/cmip5'
# Parsing files
delim = 'EOF--dataset.file.url.chksum_type.chksum'
line_var = lambda line : line.split('_')[0][1:] # ignores first single quote
line_model = lambda line : line.split('_')[2]
line_years = lambda line : tuple(int(date[:4]) for date in line.split('.')[0].split('_')[-1].split('-'))

#------------------------------------------------------------------------------#
# Helper funcs
#------------------------------------------------------------------------------#
def file_downloads(filename, complement=False):
    """
    Returns download lines between the EOF thingy.
    """
    lines = open(filename, 'r').readlines()
    idxs = [i for i,line in enumerate(lines) if delim in line]
    if len(idxs)!=2:
        raise ValueError('Invalid wget script.')
    if complement:
        return lines[:idxs[0]+1], lines[idxs[1]:] # return tuple of pairs
    else:
        return lines[idxs[0]+1:idxs[1]]
    return lines

def file_output(table='Amon', exp='piControl', filter=None):
    """
    Construct summary wget file.
    """
    # Get all lines for download
    filter = filter or []
    files = glob(f'wgets/{exp}-{table}-*.sh')
    if not files:
        raise ValueError(f'No {exp}-{table}-*.sh files found.')
    prefix, suffix = file_downloads(files[0], complement=True)
    # Collect all download lines for files
    lines = [] # final lines
    input = [line for file in files for line in file_downloads(file)]
    # Iterate by *model*, filter to date range for which *all* variables
    # are available! So far just issue for GISS-E2-R runs but important to
    # make this explicit!
    for model in filter:
        mlines = [line for line in input if f'_{model}_' in line]
        # Find minimimum date range for which all variables are available
        years = []
        vars = {line_var(line) for line in mlines}
        for ivar in vars:
            iyears = [line_years(line) for line in mlines if f'{ivar}_' in line]
            years.append((min([date[0] for date in iyears]),
                          max([date[1] for date in iyears])))
        years = (max([date[0] for date in years]),
                  min([date[1] for date in years]))
        # Finally iterate back through lines and add them if they are within this date range
        ilines = []
        for line in mlines:
            nc = line.split("'")[1].strip()
            # Date range
            # WARNING: Exclude files only *wholly* outside range; allow
            # intersection of date ranges. Important for cfDay IPSL data.
            iyears = line_years(line)
            if iyears[1]<years[0] or iyears[0]>years[1]:
                print(f'Skipping {model} {line_var(line)} years {iyears}.')
                continue
            # Consider skipping existing files
            if os.path.exists(f'{root}/{exp}-{table}/{nc}'):
                # print(f'Skipping {nc} (file exists).')
                continue
            ilines.append(line)
        # Consider adding another test
        lines.extend(ilines)
    # Save bulk download file
    out = f'wgets/{exp}-{table}.sh'
    if os.path.exists(out):
        os.remove(out)
    with open(out, 'w') as file:
        file.write(''.join(prefix + lines + suffix))
    print(f'Output file ({len(lines)} files): {out}')
    return out

def wget_summary(table='Amon', exp='piControl', filter=None):
    """
    Returns a set of all models in the group of wget scripts, a dictionary
    with keys of (experiment, variable) pairs listing the models present, and
    a dictionary with 'experiment' keys listing the possible variables.
    """
    # Group models available for various (experiment, variable)
    exps = [exp] if isinstance(exp, str) else exp
    files = [file for exp in exps for file in glob(f'wgets/{exp}-{table}-*.sh')]
    if not files:
        raise ValueError(f'No {exps}-{table}-*.sh files found.')
    models_all = {*()} # all models
    models_grouped = {} # models by variable
    filter = filter or []
    for file in files:
        split = os.path.basename(file).split('-')
        exp, var = split[0], split[2][:-3]
        imodels = {line_model(line) for line in file_downloads(file)}
        models_all.update(imodels)
        models_grouped[(exp, var)] = imodels
    # Get dictionary of variables per experiment
    exps_vars = {}
    for (exp,var) in models_grouped:
        if exp not in exps_vars:
            exps_vars[exp] = {*()}
        exps_vars[exp].add(var)
    # Find models missing in some of these pairs
    exps_missing = {}
    models_download = {*()}
    models_ignore = {*()}
    for model in sorted(models_all):
        pairs_missing = []
        for pair,group in models_grouped.items():
            if model not in group:
                pairs_missing.append(pair)
        filtered = bool(filter and model not in filter) # manual filter, e.g. filter daily data based on whether have some monthly data for same period?
        if not pairs_missing and not filtered:
            models_download.add(model)
        else:
            models_ignore.add(model)
            # Get variables missing per experiment
            exp_dict = {}
            for (exp,var) in pairs_missing:
                if exp not in exp_dict:
                    exp_dict[exp] = {*()}
                exp_dict[exp].add(var)
            # Add to per-experiment dictionary that records models for
            # which certain individual variables are missing
            for exp,vars in exp_dict.items():
                if exp not in exps_missing:
                    exps_missing[exp] = {}
                missing = exps_missing[exp]
                if filtered:
                    # Manual filter, does not matter if any variables were missing
                    if 'filtered' not in missing:
                        missing['filtered'] = []
                    missing['filtered'].append(model)
                elif vars==exps_vars[exp]: # set comparison
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
    # Message
    print()
    print(f'Download ({len(models_download)}):')
    print(", ".join(sorted(models_download)))
    for exp,exps_missing in exps_missing.items():
        vars = sorted(exps_missing)
        print()
        print(f'{exp}:')
        for var in vars:
            imissing = exps_missing[var]
            print(f'Missing {var} ({len(imissing)}): {", ".join(sorted(imissing))}')
    print()
    # Return
    return models_download

#------------------------------------------------------------------------------#
# Run script
# NOTE: Have 22 daily models with *no shortwave*, but 7 cf daily models *with
# shortwave*. Probable that shortwave contribution to covariance is small, but
# better to have entire budget.
#------------------------------------------------------------------------------#
month = 'Amon'
# steps = 'day'
steps = 'cfDay'
dayfilter = True
# steps = None
if __name__ == '__main__':
    # Run for monthly data
    exps = ('piControl', 'abrupt4xCO2')
    monthly = wget_summary(month, exp=exps)
    for exp in exps:
        file = file_output(month, exp=exp, filter=monthly)
    if not steps:
        exit()
    # Run for daily data
    daily = wget_summary(steps, exp='piControl', filter=(monthly if dayfilter else None))
    file = file_output(steps, exp='piControl', filter=daily)
    # Final message
    # NOTE: The filter ensures daily is always *subset* of monthly data; only
    # want to play with data whose climate sensitivity we can diagnose.
    both = {*monthly} & {*daily}
    nodaily = {*monthly} - {*daily}
    print()
    print(f'Both ({len(both)}): {", ".join(both)}.')
    print(f'No daily ({len(nodaily)}): {", ".join(nodaily)}.')


