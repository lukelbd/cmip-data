#!/usr/bin/env python3
"""
File for downloading and merging relevant data.
"""
import itertools

import cmip_data

# Toggle various activities
# NOTE: The analogy with dynamical core git packages is cmip_data:drycore for data
# generation, coupled:idealized for data processing and shared templates, and then
# timescales:constraints/transport for storing specific figure code and figure and
# manuscript output. This file is analogous to the 'post' files in 'drycore' for
# generating and merging output generated by the preliminary 'process' step.
climate = False
feedbacks = True

# Calculate the control and response climate and variance variables
# NOTE: Here time-variance quantities will only be produced for control non-cloud
# variables since most abrupt 4xCO2 experiments are not fully equilibriated. Only
# use them for year 120-150 climate estimates.
if climate:
    nodrifts = (False,)
    # nodrifts = (False, True)
    experiments = ('piControl', 'abrupt4xCO2')
    for nodrift, experiment in itertools.product(nodrifts, experiments):
        projects = ('CMIP6', 'CMIP5')
        projects = ('CMIP6',)  # TODO: remove
        # projects = ('CMIP5',)
        for project in projects:
            cmip_data.process_climate(
                # '~/scratch/cmip-processed',  # TODO: move climate here
                '~/data/cmip-climate',  # source climate location
                climate='~/data/cmip-climate',  # output feedback location
                source='eraint',
                project=project,
                experiment=experiment,
                flagship_filter=True,
                overwrite=False,
                logging=True,  # ignored if dryrun true
                dryrun=False,
                nowarn=False,
                # model=['FIO-ESM-2-0', 'NESM3'],
                # model=['CanESM5-1', 'E3SM-2-0'],
            )

# Calculate the control, response, and anomaly feedbacks (note control anomaly
# feedbacks are impossible because cannot select a period to use for anomalies).
# NOTE: The start and stop years are python-style endpoint-exclusive
# and relative to native years.
# NOTE: The residual feedback will only be calculated if all kernels
# for the associated flux are requested. Otherwise this is bypassed.
# TODO: Run new annual files. Should generate feedback files with 'time' coordinate
# of 12 months, then have results.py standardize dates before concatenating. Also
# monthly-style feedbacks will uniquely have no time coordinate.
if feedbacks:
    nodrifts = (False,)
    # nodrifts = (False, True)
    options = (
        # ('historical', 'monthly', (2000, 2023)),  # observed data
        # ('piControl', 'monthly', (0, 150)),  # monthly mean regression
        # ('abrupt4xCO2', 'monthly', (0, 150)),  # monthly mean regression
        # ('abrupt4xCO2', 'monthly', (0, 20)),  # monthly mean regression
        # ('abrupt4xCO2', 'monthly', (20, 150)),  # monthly mean regression
        # ('abrupt4xCO2', 'monthly', (0, 50)),  # monthly mean regression
        # ('abrupt4xCO2', 'monthly', (100, 150)),  # monthly mean regression
        ('piControl', 'annual', (0, 150)),  # annual mean regression
        ('abrupt4xCO2', 'annual', (0, 150)),  # annual mean regression
        ('abrupt4xCO2', 'annual', (0, 20)),  # annual mean regression
        ('abrupt4xCO2', 'annual', (20, 150)),  # annual mean regression
        ('abrupt4xCO2', 'annual', (0, 50)),  # annual mean regression
        ('abrupt4xCO2', 'annual', (100, 150)),  # annual mean regression
        ('abrupt4xCO2', 'ratio', (120, 150)),  # ratio feedbacks (requires annual)
    )
    for nodrift, (experiment, style, response) in itertools.product(nodrifts, options):
        projects = ('CMIP6', 'CMIP5')
        # projects = ('CMIP6',)  # TODO: remove
        # projects = ('CMIP5',)
        for project in projects:
            cmip_data.process_feedbacks(
                '~/data/cmip-climate',  # source climate location
                # '~/scratch/cmip-processed',  # TODO: move climate here
                '~/scratch/cmip-processed',  # source series location
                fluxes='~/scratch/cmip-fluxes',  # intermediate flux location
                kernels='~/data/cmip-kernels',  # dependency kernels location
                feedbacks='~/data/cmip-feedbacks',  # output feedback location
                source='eraint',
                style=style,
                response=response,
                project=project,
                experiment=experiment,
                flagship_filter=True,
                overwrite=True,
                logging=True,  # ignored if dryrun true
                dryrun=False,
                nowarn=True,
                # model=['FIO-ESM-2-0', 'NESM3'],
                # model=['CanESM5-1', 'E3SM-2-0'],
            )
