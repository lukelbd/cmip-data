#!/usr/bin/env python3
"""
Calculate feedback, forcing, and sensitivity estimates from ESGF data.
"""
import itertools

import climopy as climo  # noqa: F401  # add accessor
from icecream import ic  # noqa: F401

from .facets import (
    _glob_files,
    _main_parts,
    FacetPrinter,
    FacetDatabase,
)

__all__ = [
    'feedback_kernels',
    'feedback_parameters',
]


# Variables
# 'rsdt',  # downwelling SW TOA (identical to solar constant)
# 'rlut',  # upwelling LW TOA
# 'rsut',  # upwelling SW TOA
# 'rlds',  # downwelling LW surface
# 'rsds',  # downwelling SW surface
# 'rlus',  # upwelling LW surface
# 'rsus',  # upwelling SW surface
# 'rlutcs',  # upwelling LW TOA (clear-sky)
# 'rsutcs',  # upwelling SW TOA (clear-sky)
# 'rsuscs',  # upwelling SW surface (clear-sky) (in response to downwelling)
# 'rldscs',  # downwelling LW surface (clear-sky)
# 'rsdscs',  # downwelling SW surface (clear-sky)

def _correlation_feedback():
    """
    Calculate the feedbacks using covariance-variance ratios of deseasonalized
    month-to-month anomalies in a control climate time series.
    """


def _regression_feedback():
    """
    Calculate the Gregory et al. regression feedbacks using time series from the
    abrupt forcing experiment.
    """


def _ratio_feedback():
    """
    Calculate the feedbacks using the ratio of of abrupt to control experiments. This
    requires an effective forcing estimated from a regression calculation.
    """


def _feedback_from_kernel():
    """
    Get the radiation due to a feedback kernel by multiplying and verticaly
    integrating the response of the quantity.
    """


def feedback_kernels():
    """
    Generate contribution to radiative flux from clear-sky and all-sky
    temperature and moisture kernels.

    Parameters
    ----------
    *path : path-like, optional
        Location(s) of the climate and series data.
    """
    # TODO: This should give time series of contributions from individual radiative
    # flux components (e.g. surface all-sky longwave due to temperature changes
    # relative to the pre-industrial mean climate).


def feedback_parameters(
    *paths, mode='regression', series=None, climate=None, response=None, printer=None,
    project=None, **kwargs
):
    """
    Generate radiative feedback estimates using the input cmip files. Also
    retrieve the effective forcing and sensitivity estimates.

    Parameters
    ----------
    *path : path-like, optional
        Location(s) of the climate and series data.
    mode : {'regression', 'correlation', 'ratio'}
        The type of feedback to compute.
    series : 2-tuple of int, default: (0, 150)
        The year range for response time series data.
    climate : 2-tuple of int, default: (0, 150)
        The year range for reference climate data.
    response : 2-tuple of int, default: (120, 150)
        The year range for response climate data.
    printer : callable, default: `print`
        The print function.
    **kwargs
        Passed to the feedback function.
    """
    # Find files and restrict to unique constraints
    # NOTE: This requires flagship translation or else models with different control
    # and abrupt runs are not grouped together. Not sure how to handle e.g. non-flagship
    # abrupt runs from flagship control runs but cross that bridge when we come to it.
    project = (project or 'cmip6').upper()
    print = printer or FacetPrinter('feedbacks', mode, project=project)
    files, _ = _glob_files(*paths, project=project)
    facets = ('project', 'model', 'ensemble', 'grid')
    print('Generating database.')
    database = FacetDatabase(
        files, facets, project=project, flagship_translate=True
    )

    # Calculate clear and full-sky feedbacks surface and TOA files
    # NOTE: Unlike the method that uses a fixed SST experiment to deduce forcing and
    # takes the ratio of local radiative flux change to global temperature change as
    # the feedback, the average of a regression of radiative flux change against global
    # temperature change will not necessarily equal the regression of average radiative
    # flux change. Therefore compute both and compare to get idea of the residual. Also
    # consider local temperature change for point of comparison (Hedemman et al. 2022).
    series = series or (0, 150)
    climate = climate or (0, 150)
    response = response or (0, 150)
    for group, data in database.items():
        group = dict(zip(database._group, group))
        string = ', '.join(f'{key}: {value}' for key, value in group.items())
        print(f'\nCalculating {mode} feedbacks:', string)
        key1 = 'piControl'
        key2 = 'abrupt4xCO2' if project == 'CMIP5' else 'abrupt-4xCO2'
        for freq, direc, level, sky in itertools.product('sl', 'ud', 'st', ('', 'cs')):
            variable = f'r{freq}{direc}{level}{sky}'
            if variable[:4] in ('rsdt', 'rldt') or variable == 'rluscs':
                continue
            files1 = data.get((key1, 'Amon', variable))
            files2 = data.get((key2, 'Amon', variable))
            missing = ', '.join(key for key, files in ((key1, files1), (key2, files2)) if not files)  # noqa: E501
            if missing:
                print(f'Warning: Experiment(s) {missing} missing for variable {variable!r}.')  # noqa: E501
                continue
            print(f'Variable: {variable}')
            dates1 = [_main_parts['dates'](file.name) for file in files1]
            dates2 = [_main_parts['dates'](file.name) for file in files2]
            print(*(file.name for file in (*files1, *files2)))
            print(*(date for date in (*dates1, *dates2)))
