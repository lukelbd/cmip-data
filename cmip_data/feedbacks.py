#!/usr/bin/env python3
"""
Calculate feedback, forcing, and sensitivity estimates from ESGF data.
"""
from .process import standardize_horizontal, standardize_vertical


def process_feedbacks(data):
    """
    Generate radiative feedback estimates using the input cmip files.
    """
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
