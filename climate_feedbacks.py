#!/usr/bin/env python3
"""
File for downloading and merging relevant data.
"""
import cmip_data
import socket

# Toggle various activities
# NOTE: The analogy with dynamical core git packages is cmip_data:drycore for data
# generation, coupled:idealized for data processing and shared templates, and then
# timescales:constraints/transport for storing specific figure code and figure and
# manuscript output. This file is analogous to the 'post' files in 'drycore' for
# generating and merging output generated by the preliminary 'process' step.
climate = False
feedbacks = False

# Calculate the control and response climate and variance variables
# NOTE: Here time-variance quantities will only be produced for control non-cloud
# variables since most abrupt 4xCO2 experiments are not fully equilibriated. Only
# use them for year 120-150 climate estimates.
if climate:
    nodrifts = (False,)
    # nodrifts = (False, True)
    for nodrift in nodrifts:
        experiments = (
            'piControl',
            'abrupt4xCO2',
        )
        for experiment in experiments:
            projects = ('CMIP6', 'CMIP5')
            # projects = ('CMIP5',)
            for project in projects:
                if 'monde' in socket.gethostname():
                    series = '~/scratch2/data-processed'
                elif 'local' in socket.gethostname():
                    series = '~/data/cmip-series'
                else:
                    raise RuntimeError
                cmip_data.process_feedbacks(
                    '~/data',  # source climate location
                    series,  # source series location
                    climate='~/data',  # output feedback location
                    source='eraint',
                    project=project,
                    experiment=experiment,
                    flagship_filter=True,
                    overwrite=False,
                    logging=True,  # ignored if dryrun true
                    dryrun=False,
                )

# Calculate the control, response, and anomaly feedbacks (note control anomaly
# feedbacks are impossible because cannot select a period to use for anomalies).
# NOTE: The residual feedback will only be calculated if all kernels
# for the associated flux are requested. Otherwise this is bypassed.
if feedbacks:
    nodrifts = (False,)
    # nodrifts = (False, True)
    for nodrift in nodrifts:
        experiments = (
            ('abrupt4xCO2', False),  # regression of series
            ('piControl', False),  # regression of series
            ('abrupt4xCO2', True),  # ratio of anomalies
        )
        for experiment, ratio in experiments:
            projects = ('CMIP6', 'CMIP5')
            # projects = ('CMIP5',)
            for project in projects:
                if 'monde' in socket.gethostname():
                    series = '~/scratch2/data-processed'
                elif 'local' in socket.gethostname():
                    series = '~/data/cmip-series'
                else:
                    raise RuntimeError
                cmip_data.process_feedbacks(
                    '~/data',  # source climate location
                    series,  # source series location
                    feedbacks='~/data',  # output feedback location
                    kernels='~/data',  # input kernels location
                    fluxes=series,  # output flux location
                    ratio=ratio,
                    source='eraint',
                    project=project,
                    experiment=experiment,
                    flagship_filter=True,
                    nodrift=nodrift,
                    overwrite=False,
                    logging=True,  # ignored if dryrun true
                    dryrun=False,
                )
