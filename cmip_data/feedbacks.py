#!/usr/bin/env python3
"""
Calculate feedback, forcing, and sensitivity estimates from ESGF data.
"""
from pathlib import Path

from .process import cdo, nco, standardize_horizontal, standardize_vertical

# Constants
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


def process_feedbacks(path='~/data'):
    """
    Generate radiative feedback estimates using the input cmip files.
    """
    # Obtain surface and TOA files
    path = Path(path).expanduser()
    for s in 'sl':
        for dir in 'ud':

    # Local Gregory regressions
    # Global Gregory regression
    # NOTE: Unlike the method that uses a fixed SST experiment to deduce forcing and
    # takes the ratio of local radiative flux change to global temperature change as
    # the feedback, the average of a regression of radiative flux change against global
    # temperature change will not necessarily equal the regression of average radiative
    # flux change. Therefore compute both and compare to get idea of the residual. Also
    # consider local temperature change for point of comparison (Hedemman et al. 2022).


def standardize_kernels():
    """
    Standardize the radiative kernel files for consistency with the cmip files.
    """
    pass
