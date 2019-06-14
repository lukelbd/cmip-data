#!/usr/bin/env python
#------------------------------------------------------------------------------#
# Define necessary categories
# 1) Models for which we can calculate climate sensitivity, by
# regressing temperature against TOA flux; need climatologcal temp and TOA
# budget on monthly means for control and forced run.
# 2) Models for which we can get daily temperature and
# flux terms for that timescale estimate. Only do this for a few select
# models.
#------------------------------------------------------------------------------#
# Imports and helper funcs
from glob import glob
import re
regex = re.compile("'.*\.nc'")
def models(filename):
    lines = open(filename).readlines()
    lines = [line for line in lines if regex.match(line)]
    models = {line.split('_')[2] for line in lines}
    return models

# Filter 2D variable-experiment space
# NOTE: This was too complicated to do with bash script, need sets and
# dictionary objects! Worth the extra verbosity!
files = glob('wgets/wget-raw-*-Amon-*.sh')
splits = [file.split('-') for file in files]
amodels = {*()} # all models
vmodels = {} # models by variable
for pair,file in zip(pairs,files):
    split = file.split('-')
    exp, var = split[2], split[4][:-3]
    imodels = models(file)
    amodels.update(imodels)
    vmodels[var] = imodels
for model in sorted(amodels):
    missing = []
    for var,group in vmodels.items():
        if model not in group:
            missing.append(var)
    # if not missing:
    #     print('All good', model)
    # else:
    print(model, missing)
