#!/usr/bin/env python3
"""
Calculate feedback, forcing, and sensitivity estimates from ESGF data.
"""
import climopy as climo  # noqa: F401  # add accessor
from icecream import ic  # noqa: F401

from . import cdo, nco, Atted, Rename, CDOException
from .facets import (
    _glob_files,
    _item_dates,
    FacetPrinter,
    FacetDatabase,
)

__all__ = [
    'process_feedbacks',
]


# Feedback variables
# NOTE: Here albedo is taken from ratio of upwelling to downwelling all-sky surface
# shortwave radiation, and surface shortwave and longwave feedbacks are determined
# by regressing net (upwelling minus downwelling) radiation fluxes. Note since albedo
# is just the ratio of upwelling to downwelling surface shortwave, and kernels are
# the change in net shortwave per unit albedo, then surface albedo kernels are the
# simple function Ka = (rsus - rsds) / (rsus / rsds). So totally radiation dependent.
VARIABLES = [
    'rlut',  # all-sky upwelling LW TOA
    'rsut',  # all-sky upwelling SW TOA
    'rlutcs',  # clear-sky upwelling LW TOA
    'rsutcs',  # clear-sky upwelling SW TOA
    'rlds',  # all-sky downwelling LW surface
    'rsds',  # all-sky downwelling SW surface
    'rldscs',  # clear-sky downwelling LW surface
    'rsdscs',  # clear-sky downwelling SW surface
    'rlus',  # all-sky upwelling LW surface (for net LW surface budget)
    'rsus',  # all-sky upwelling SW surface (for net SW surface budget and albedo)
    'rsuscs',  # all-sky upwelling SW surface (for net SW surface budget and albedo)
]


def _control_feedback():
    """
    Calculate the feedbacks using covariance-variance ratios of deseasonalized
    month-to-month anomalies in a control climate time series.
    """


def _response_feedback():
    """
    Calculate the Gregory et al. regression feedbacks using time series from
    an abrupt forcing response experiment.
    """


def _simple_feedback():
    """
    Calculate the feedbacks using the ratio of of abrupt to control experiments. This
    requires an effective forcing estimated from a regression calculation.
    """


def _feedback_from_kernel():
    """
    Get the radiation due to a feedback kernel by multiplying and verticaly
    integrating the response of the quantity.
    """


def process_feedbacks(
    *paths, mode='control', kernels=True,
    series=None, control=None, response=None, printer=None, **constraints
):
    """
    Generate radiative feedback estimates using the input cmip files. Also
    retrieve the effective forcing and sensitivity estimates.

    Parameters
    ----------
    *paths : path-like, optional
        Location(s) of the climate and series data.
    mode : {'control', 'response', 'ratio'}
        The type of feedback to compute.
    kernels : bool, optional
        Whether to include kernel components.
    series : 2-tuple of int, default: (0, 150)
        The year range for time series data.
    control : 2-tuple of int, default: (0, 150)
        The year range for control climate data.
    response : 2-tuple of int, default: (120, 150)
        The year range for response climate data.
    printer : callable, default: `print`
        The print function.
    **constraints
        Passed to `_parse_constraints`.
    """
    # Find files and restrict to unique constraints
    # NOTE: This requires flagship translation or else models with different control
    # and abrupt runs are not grouped together. Not sure how to handle e.g. non-flagship
    # abrupt runs from flagship control runs but cross that bridge when we come to it.
    # NOTE: Paradigm is to use climate monthly mean surface pressure when interpolating
    # to model levels and keep surface pressure time series when getting feedback
    # kernel integrals. Helps improve accuracy since so much stuff depends on kernels.
    variables = VARIABLES.copy()
    variables = constraints.setdefault('variable', variables)
    print = printer or FacetPrinter('feedbacks', mode, **constraints)
    print('Generating database.')
    files, _ = _glob_files(*paths, project=constraints.get('project', None))
    facets = ('project', 'model', 'ensemble', 'grid')
    database = FacetDatabase(files, facets, flagship_translate=True, **constraints)

    # Calculate clear and full-sky feedbacks surface and TOA files
    # NOTE: This requires existence of feedback kernel files for surface albedo,
    # surface temperature, air temperature, specifid humidity. They will be created
    # at the same location if they do not already exist. Could also loop through
    # in separate function and put into separate database... but whatever dude.
    # NOTE: Unlike the method that uses a fixed SST experiment to deduce forcing and
    # takes the ratio of local radiative flux change to global temperature change as
    # the feedback, the average of a regression of radiative flux change against global
    # temperature change will not necessarily equal the regression of average radiative
    # flux change. Therefore compute both and compare to get idea of the residual. Also
    # consider local temperature change for point of comparison (Hedemman et al. 2022).
    series = series or (0, 150)
    control = control or (0, 150)
    response = response or (120, 150)
    kernels = ('ts', 'ta', 'hus', 'alb') if kernels else ()
    for group, data in database.items():
        group = dict(zip(database._group, group))
        string = ', '.join(f'{key}: {value}' for key, value in group.items())
        print(f'\nCalculating {mode} feedbacks:', string)
        key1 = 'piControl'
        key2 = 'abrupt4xCO2' if database._project == 'CMIP5' else 'abrupt-4xCO2'
        for variable in VARIABLES:
            # Get files and dates
            files1 = data.get((key1, 'Amon', variable))
            files2 = data.get((key2, 'Amon', variable))
            missing = ', '.join(key for key, files in ((key1, files1), (key2, files2)) if not files)  # noqa: E501
            if missing:
                print(f'Warning: Experiment(s) {missing} missing for variable {variable!r}.')  # noqa: E501
                continue
            print(f'Variable: {variable}')
            dates1 = [_item_dates(file.name) for file in files1]
            dates2 = [_item_dates(file.name) for file in files2]
            print(*(file.name for file in (*files1, *files2)))
            print(*(date for date in (*dates1, *dates2)))

            # Select files for derivations
            files = []
            for kernel in kernels:
                pass
