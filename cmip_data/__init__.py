#!/usr/bin/env python3
"""
Shared package for working with CMIP experiments.
"""
# Initialize cdo and nco bindings
# NOTE: Use 'conda install cdo' and 'conda install nco' for the command-line
# tools. Use 'conda install python-cdo' and 'conda install pynco' (or 'pip
# install cdo' and 'pip install pynco') for only the python bindings. Note for
# some reason cdo returns lists while nco returns strings with newlines, and for
# the signal catch fix must install from github with subdirectory using pip install
# 'git+https://github.com/lukelbd/cdo-bindings@fix-signals#subdirectory=python'.
# Also should use 'pip install git+https://github.com/nco/pynco.git' to fix Rename
# issue (although currently use prn_option() for debugging help anyway).
# NOTE: This requires cdo > 2.0.0 or cdo <= 1.9.1 (had trouble installing recent
# cdo in base environments so try uninstalling libgfortran4, libgfortran5, and
# libgfortran-ng then re-installing cdo and the other packages removed by that
# action). In cdo 1.9.9 ran into weird bug where the delname line before ap2pl caused
# infinite hang for ap2pl. Also again note cdo is faster and easier to use than nco.
# Compare 'time ncremap -a conserve -G latlon=36,72 tmp.nc tmp_nco.nc' to 'time cdo
# remapcon,r72x36 tmp.nc tmp_cdo.nc'. Also can use '-t 8' for nco and '-P 8' for cdo
# for parallelization but still makes no difference (note real time speedup is
# marginal and user time should increase significantly, consistent with ncparallel
# results). The only exception seems to be attribute and name management.
import os
from cdo import Cdo, CDOException  # noqa: F401
from nco import Nco, NCOException  # noqa: F401
from nco.custom import Atted, Rename  # noqa: F401
os.environ['CDO_TIMESTAT_DATE'] = 'first'
cdo = Cdo(options=['-s'])
nco = Nco()  # overwrite is default, and see https://github.com/nco/pynco/issues/56

# Import remaining tools
from .facets import *  # noqa: F401, F403
from .download import *  # noqa: F401, F403
from .process import *  # noqa: F401, F403
from .feedbacks import *  # noqa: F401, F403
from .kernels import *
from .load import *  # noqa: F401, F403
