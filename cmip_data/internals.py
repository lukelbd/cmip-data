#!/usr/bin/env python3
"""
Utilities related to facet handling and logging.
"""
import builtins
import re
from pathlib import Path

import numpy as np

__all__ = [
    'glob_files',
    'Database',
    'Logger',
]

# Corrupt files that have to be manually ignored after bulk downloads. Can test using
# e.g. for file in ta*nc; do echo $file && ncvardump ta $file 1>/dev/null; done
# since can get HDF5 errors during processing but header can be read without error.
# NOTE: At one point filtered out these files inside Database._init() to prevent even
# downloading them but this just triggers downloading the 150+ year files during the
# script filtering step. Now permit downloads and only filter out in glob_files().
# NOTE: As of 2022-07-20 cannot download CAS-ESM2-0 data (surface drag and integrated
# and model-level cloud data), skip downloading too-high-resolution CNRM-CM6-1-HR data,
# and auto-correct specific humidity GISS missing values in repair_files().
CORRUPT_FILES = [
    'cct_Amon_IITM-ESM_piControl_r1i1p1f1_gn_192601-193512.nc',  # identical values
    'cct_Amon_IITM-ESM_piControl_r1i1p1f1_gn_193601-194512.nc',  # not sure of issue
    'cct_Amon_IITM-ESM_piControl_r1i1p1f1_gn_194601-195012.nc',
    'cct_Amon_IITM-ESM_piControl_r1i1p1f1_gn_195101-196012.nc',
    'cct_Amon_IITM-ESM_piControl_r1i1p1f1_gn_196101-197012.nc',
    'cct_Amon_IITM-ESM_piControl_r1i1p1f1_gn_197101-198012.nc',
    'cct_Amon_IITM-ESM_piControl_r1i1p1f1_gn_198101-199012.nc',
    'cct_Amon_IITM-ESM_piControl_r1i1p1f1_gn_199101-200012.nc',
    'cct_Amon_IITM-ESM_piControl_r1i1p1f1_gn_200101-201012.nc',
    'cct_Amon_IITM-ESM_piControl_r1i1p1f1_gn_201101-202012.nc',
    'cct_Amon_IITM-ESM_piControl_r1i1p1f1_gn_202101-203012.nc',
    'cct_Amon_IITM-ESM_piControl_r1i1p1f1_gn_203101-204012.nc',
    'cct_Amon_IITM-ESM_piControl_r1i1p1f1_gn_204101-205012.nc',
    'cct_Amon_IITM-ESM_piControl_r1i1p1f1_gn_205101-206012.nc',
    'cct_Amon_IITM-ESM_piControl_r1i1p1f1_gn_206101-207012.nc',
    'cct_Amon_IITM-ESM_piControl_r1i1p1f1_gn_207101-208012.nc',
    'clivi_Amon_IITM-ESM_piControl_r1i1p1f1_gn_192601-193512.nc',  # out-of-range
    'clivi_Amon_IITM-ESM_piControl_r1i1p1f1_gn_193601-194512.nc',  # not missing values
    'clivi_Amon_IITM-ESM_piControl_r1i1p1f1_gn_194601-195012.nc',
    'clivi_Amon_IITM-ESM_piControl_r1i1p1f1_gn_195101-196012.nc',
    'clivi_Amon_IITM-ESM_piControl_r1i1p1f1_gn_196101-197012.nc',
    'clivi_Amon_IITM-ESM_piControl_r1i1p1f1_gn_197101-198012.nc',
    'clivi_Amon_IITM-ESM_piControl_r1i1p1f1_gn_198101-199012.nc',
    'clivi_Amon_IITM-ESM_piControl_r1i1p1f1_gn_199101-200012.nc',
    'clivi_Amon_IITM-ESM_piControl_r1i1p1f1_gn_200101-201012.nc',
    'clivi_Amon_IITM-ESM_piControl_r1i1p1f1_gn_201101-202012.nc',
    'clivi_Amon_IITM-ESM_piControl_r1i1p1f1_gn_202101-203012.nc',
    'clivi_Amon_IITM-ESM_piControl_r1i1p1f1_gn_203101-204012.nc',
    'clivi_Amon_IITM-ESM_piControl_r1i1p1f1_gn_204101-205012.nc',
    'clivi_Amon_IITM-ESM_piControl_r1i1p1f1_gn_205101-206012.nc',
    'clivi_Amon_IITM-ESM_piControl_r1i1p1f1_gn_206101-207012.nc',
    'clivi_Amon_IITM-ESM_piControl_r1i1p1f1_gn_207101-208012.nc',
    'clwvi_Amon_IITM-ESM_piControl_r1i1p1f1_gn_192601-193512.nc',  # out-of-range
    'clwvi_Amon_IITM-ESM_piControl_r1i1p1f1_gn_193601-194512.nc',  # not missing values
    'clwvi_Amon_IITM-ESM_piControl_r1i1p1f1_gn_194601-195012.nc',
    'clwvi_Amon_IITM-ESM_piControl_r1i1p1f1_gn_195101-196012.nc',
    'clwvi_Amon_IITM-ESM_piControl_r1i1p1f1_gn_196101-197012.nc',
    'clwvi_Amon_IITM-ESM_piControl_r1i1p1f1_gn_197101-198012.nc',
    'clwvi_Amon_IITM-ESM_piControl_r1i1p1f1_gn_198101-199012.nc',
    'clwvi_Amon_IITM-ESM_piControl_r1i1p1f1_gn_199101-200012.nc',
    'clwvi_Amon_IITM-ESM_piControl_r1i1p1f1_gn_200101-201012.nc',
    'clwvi_Amon_IITM-ESM_piControl_r1i1p1f1_gn_201101-202012.nc',
    'clwvi_Amon_IITM-ESM_piControl_r1i1p1f1_gn_202101-203012.nc',
    'clwvi_Amon_IITM-ESM_piControl_r1i1p1f1_gn_203101-204012.nc',
    'clwvi_Amon_IITM-ESM_piControl_r1i1p1f1_gn_204101-205012.nc',
    'clwvi_Amon_IITM-ESM_piControl_r1i1p1f1_gn_205101-206012.nc',
    'clwvi_Amon_IITM-ESM_piControl_r1i1p1f1_gn_206101-207012.nc',
    'clwvi_Amon_IITM-ESM_piControl_r1i1p1f1_gn_207101-208012.nc',
]

# Ensemble labels associated with flagship versions of the pre-industrial control
# and abrupt 4xCO2 experiments. Note the CNRM, MIROC, and UKESM models run both their
# control and abrupt experiments with 'f2' forcing, HadGEM runs the abrupt experiment
# with the 'f3' forcing and the (parent) control experiment with 'f1' forcing, and
# EC-Earth3 strangely only runs the abrupt experiment with the 'r8' realization
# and the (parent) control experiment with the 'r1' control realiztion.
# NOTE: See the download_process.py script for details on available models.
ENSEMBLES_FLAGSHIP = {
    ('CMIP5', None, None): 'r1i1p1',
    ('CMIP6', None, None): 'r1i1p1f1',
    ('CMIP6', 'piControl', 'CNRM-CM6-1'): 'r1i1p1f2',
    ('CMIP6', 'piControl', 'CNRM-CM6-1-HR'): 'r1i1p1f2',
    ('CMIP6', 'piControl', 'CNRM-ESM2-1'): 'r1i1p1f2',
    ('CMIP6', 'piControl', 'MIROC-ES2L'): 'r1i1p1f2',
    ('CMIP6', 'piControl', 'MIROC-ES2H'): 'r1i1p4f2',
    ('CMIP6', 'piControl', 'UKESM1-0-LL'): 'r1i1p1f2',
    ('CMIP6', 'piControl', 'UKESM1-1-LL'): 'r1i1p1f2',
    ('CMIP6', 'abrupt-4xCO2', 'CNRM-CM6-1'): 'r1i1p1f2',
    ('CMIP6', 'abrupt-4xCO2', 'CNRM-CM6-1-HR'): 'r1i1p1f2',
    ('CMIP6', 'abrupt-4xCO2', 'CNRM-ESM2-1'): 'r1i1p1f2',
    ('CMIP6', 'abrupt-4xCO2', 'MIROC-ES2L'): 'r1i1p1f2',
    ('CMIP6', 'abrupt-4xCO2', 'MIROC-ES2H'): 'r1i1p4f2',
    ('CMIP6', 'abrupt-4xCO2', 'UKESM1-0-LL'): 'r1i1p1f2',
    ('CMIP6', 'abrupt-4xCO2', 'UKESM1-1-LL'): 'r1i1p1f2',
    ('CMIP6', 'abrupt-4xCO2', 'HadGEM3-GC31-LL'): 'r1i1p1f3',
    ('CMIP6', 'abrupt-4xCO2', 'HadGEM3-GC31-MM'): 'r1i1p1f3',
    ('CMIP6', 'abrupt-4xCO2', 'EC-Earth3'): 'r8i1p1f1',
}

# ESGF facets obtained with get_facet_options() for SearchContext(project='CMIP5')
# and SearchContext(project='CMIP6'). See: https://github.com/WCRP-CMIP/CMIP6_CVs
# Includes mappings between shorthand 'aliases' (used with databases and other internal
# utilities) and standard names (used when searching with pyesgf), between cmip5 and
# cmip6 experiment names (so that the same string can be specified in function calls),
# and between cmip5 model identifiers (may differ between dataset ids and file names).
# The 'storage' and 'summarize' are for storage folder names and summarize reports.
# Experiments: https://wcrp-cmip.github.io/CMIP6_CVs/docs/CMIP6_experiment_id.html
# Institutions: https://wcrp-cmip.github.io/CMIP6_CVs/docs/CMIP6_institution_id.html
# Models/sources: https://wcrp-cmip.github.io/CMIP6_CVs/docs/CMIP6_source_id.html
FACETS_CMIP5 = [
    'access', 'cera_acronym', 'cf_standard_name', 'cmor_table', 'data_node',
    'ensemble', 'experiment', 'experiment_family', 'forcing', 'format',
    'index_node', 'institute', 'model', 'product', 'realm', 'time_frequency',
    'variable', 'variable_long_name', 'version'
]
FACETS_CMIP6 = [
    'access', 'activity_drs', 'activity_id', 'branch_method', 'creation_date',
    'cf_standard_name', 'data_node', 'data_specs_version', 'datetime_end',
    'experiment_id', 'experiment_title', 'frequency', 'grid', 'grid_label',
    'index_node', 'institution_id', 'member_id', 'nominal_resolution', 'realm',
    'short_description', 'source_id', 'source_type', 'sub_experiment_id', 'table_id',
    'variable', 'variable_id', 'variable_long_name', 'variant_label', 'version'
]
FACETS_DOTNAMES = {
    ('CMIP5', 'ACCESS1-0'): 'ACCESS1.0',
    ('CMIP5', 'ACCESS1-3'): 'ACCESS1.3',
    ('CMIP5', 'CSIRO-Mk3-6-0'): 'CSIRO-Mk3.6.0',
    ('CMIP5', 'bcc-csm1-1'): 'BCC-CSM1.1',
    ('CMIP5', 'inmcm4'): 'INM-CM4',
}
FACETS_LONGNAMES = {
    ('CMIP5', 'table'): 'cmor_table',
    ('CMIP5', 'institution'): 'institute',
    ('CMIP5', 'frequency'): 'time_frequency',
    ('CMIP6', 'variable'): 'variable_id',
    ('CMIP6', 'table'): 'table_id',
    ('CMIP6', 'model'): 'source_id',
    ('CMIP6', 'experiment'): 'experiment_id',
    ('CMIP6', 'ensemble'): 'variant_label',
    ('CMIP6', 'institution'): 'institution_id',
}
FACETS_RENAMES = {
    ('CMIP5', 'hist-nat'): 'historicalNat',
    ('CMIP5', 'hist-GHG'): 'historicalGHG',
    ('CMIP5', 'esm-hist'): 'esmHistorical',
    ('CMIP5', 'esm-piControl'): 'esmControl',
    ('CMIP5', 'abrupt-4xCO2'): 'abrupt4xCO2',
    ('CMIP6', 'historicalNat'): 'hist-nat',
    ('CMIP6', 'historicalGHG'): 'hist-GHG',
    ('CMIP6', 'esmHistorical'): 'esm-hist',
    ('CMIP6', 'esmControl'): 'esm-piControl',
    ('CMIP6', 'abrupt4xCO2'): 'abrupt-4xCO2',
}
FACETS_STORAGE = (  # naming of storage folders
    'project',
    'experiment',
    'table',
)
FACETS_SUMMARIZE = (  # grouping in summarize logs
    'project',
    'experiment',
    'table',
    'variable',
)

# Model institutes
# NOTE: This was copied from the INSTITUTES.txt file produced by manual_checks.py and
# include the project so we can filter models to both unique institutes and projects.
# Currently used in 'coupled' plotting functions (previously tried to just use names
# but this was unreliable -- for example ACCESS/CSIRO are both from CSIRO in Australia,
# MPI/ICON are both from MPI in Germany, and CESM/CCSM are both from NCAR in Colorado).
# NOTE: Some institutes changed their ids between cmip5 and cmip6 due to syntax updates
# (e.g. dash instead of space) or new collaborators (generally appended with dashes).
# Fixed this so institues can be uniquely identified, and put original ids in comments.
# NOTE: Here the preferred model variants from the same institution are arranged
# last (see _parse_projects in templates.py). Roughly similar to original alphabetical
# order; prefer more complex (ESMs over CMs), higher resolution, and later versions.
# NOTE: Include 2-character ISO 3166 country codes for quick reference.
# See: https://en.wikipedia.org/wiki/List_of_ISO_3166_country_codes
# There are several Chinese models from academies, ministries, and universities.
# See: http://jmr.cmsjournal.net/article/doi/10.1007/s13351-020-9164-0
# Here MCM is based on the 'Manabe' model and is very primitive compared to others.
# See: http://u.arizona.edu/~ronaldstouffer/MCM_Description_Summary.html
# Prefer ACCESS to CSIRO becase latter is more complex.
# See: https://www.researchgate.net/publication/258763480_The_ACCESS_coupled_model_Description_control_climate_and_evaluation  # noqa: E501
# Prefer ESM2M to ESM2G since it simulates surface climate better.
# See: https://journals.ametsoc.org/view/journals/clim/25/19/jcli-d-11-00560.1.xml
# Prefer more updated GISS-E2-H ocean model to GISS-E2-R ocean model.
# See: https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2013MS000265
# Prefer MPI-ESM-MR to MPI-ESM-LR since it has higher resolution.
# See: https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/jame.20038
# Prefer NorESM1-ME to NorESM1-M because latter includes biogeochecmial stuff
# See: https://gmd.copernicus.org/articles/6/687/2013/
# Prefer bcc-csm1-1-m to bcc-csm1-1 because this is 'moderate resolution' t106 over t63
# See: http://forecast.bcccsm.ncc-cma.net/web/channel-63.htm
# Prefer CESM2 without FV2 designations because these are 2 degree instead of 1 degree
# See: https://bb.cgd.ucar.edu/cesm/threads/difference-between-cesm2-and-cesm2-fv.5549/
# Prefer CNRM CMIP6 earth system model even though cmip5 had only climate model.
# See: https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2019MS001791
# Prefer EC-Earth3-Veg because air chemistry and carbon cycle physics not relevant.
# See: https://gmd.copernicus.org/articles/15/2973/2022/
# Prefer latest version MIROC-ES2H to older version ES2L (resolution seems to be same).
# See: https://progearthplanetsci.springeropen.com/articles/10.1186/s40645-020-00369-5
# Classify MPI-Ham variant under the same institution since it derives from there.
# See: https://ui.adsabs.harvard.edu/abs/2017EGUGA..19.7132F/abstract
# Prefer NorESM-MM to LM because has 1 degree resolution instead of 2
# See: https://noresm-docs.readthedocs.io/en/latest/start.html
# Prefer UKESM to HadGEM because former is full earth system model
# See: https://www.metoffice.gov.uk/research/approach/modelling-systems/new-flagship-climate-models  # noqa: E501
INSTITUTES_ABBREVS = {  # map institute ids to more familiar model abbreviations
    'CSIRO': 'CSIRO (AU)',  # 'ACCESS',  # Commonwealth Sci. + Ind. Research Org.
    'BNU': 'BNU (CN)',  # Beijing N. University, China
    'NCAR': 'NCAR (US)',  # 'CESM',  # National Center for Atmospheric Research, USA
    'THU': 'THU (CN)',  # 'CIESM',  # Beijing Tsinghua University, China
    'CAMS': 'CAMS (CN)',  # Chinese Academy of Meteorological Sciences, China
    'CAS': 'CAS (CN)',  # 'FGOALS',  # Chinese Academy of Sciences, China
    'CMCC': 'CMCC (IT)',  # 'CMCC-Esm',  # Cen. Euro-Med. Cambiamenti Climatici, Italy
    'CNRM-CERFACS': 'CNRM (FR)',  # Cen. National de Recherches Meteorologiques, France
    'CCCma': 'CCCma (CA)',  # 'CanESM',  # Can. Cen. Clim. Modelling + Analaysis, Canada
    'E3SM-Project': 'E3SM (US)',  # various USA institutes
    'EC-Earth-Consortium': 'EC-Earth (EU)',  # various European institutes
    'NOAA': 'GFDL (US)',  # 'NOAA',  # Geophysical Fluid Dynamics Laboratory, USA
    'NASA': 'GISS (US)',  # 'NASA',  # Goddard Institute for Space Studies, USA
    'MOHC': 'MOHC (UK)',  # 'HadGEM',  # Met Office Hadley Centre, UK
    'CCCR-IITM': 'IITM (IN)',  # Indian Institute of Tropical Meteorology, India
    'INM': 'INM (RU)',  # Institute for Numerical Mathematics, Moscow
    'IPSL': 'IPSL (FR)',  # Institut Pierre Simon Laplace, France
    'KIOST': 'KIOST (KR)',  # Korea Institute of Ocean Science & Technology, Korea
    'NIMS-KMA': 'KMA (KR)',  # 'KACE',  # Korea Meteorological Administration, Korea
    'UA': 'UA (US)',  # 'MCM-UA',  # University of Arizonta, USA
    'MIROC': 'MIROC (JP)',  # jaMstec/nIes/R-ccs/aOri Consortium, Japan
    'MPI-M': 'MPI (DE)',  # 'MPI-ESM',  # Max Planck Institut, Germany
    'MRI': 'MRI (JP)',  # Meteorological Research Institute, Japan
    'NUIST': 'NUIST (CN)',  # 'NESM',  # Nanjing U. of Information Sci. and Tech., China
    'NCC': 'NCC (NO)',  # 'NorESM',  # NorESM Climate modeling Consortium, Norway
    'BCC': 'BCC (CN)',  # Beijing Climate Center, China
    'AWI': 'AWI (DE)',  # Alfred Wegener Institute, Germany
    'SNU': 'SNU (KR)',  # 'SAM',  # Seoul National University, Korea
    'AS-RCEC': 'AS (TW)',  # 'AS',  # Taipei Academia Sinica, Taiwan
}
MODELS_INSTITUTES = {  # map model ids to institute ids with 'flagship' model last
    ('CMIP5', 'BNU-ESM'): 'BNU',
    ('CMIP5', 'CCSM4'): 'NCAR',
    ('CMIP5', 'CNRM-CM5'): 'CNRM-CERFACS',
    ('CMIP5', 'CNRM-CM5-2'): 'CNRM-CERFACS',  # latest version
    ('CMIP5', 'CanESM2'): 'CCCma',
    ('CMIP5', 'CSIRO-Mk3-6-0'): 'CSIRO',  # 'CSIRO-QCCCE',  # overwritten by ACCESS
    ('CMIP5', 'ACCESS1-0'): 'CSIRO',  # 'CSIRO-BOM',
    ('CMIP5', 'ACCESS1-3'): 'CSIRO',  # 'CSIRO-BOM',  # latest version
    ('CMIP5', 'FGOALS-s2'): 'CAS',  # 'LASG-IAP',
    ('CMIP5', 'FGOALS-g2'): 'CAS',  # 'LASG-CESS',  # highest resolution
    ('CMIP5', 'GFDL-CM3'): 'NOAA',  # 'NOAA GFDL',
    ('CMIP5', 'GFDL-ESM2G'): 'NOAA',  # 'NOAA GFDL',
    ('CMIP5', 'GFDL-ESM2M'): 'NOAA',  # 'NOAA GFDL',
    ('CMIP5', 'GISS-E2-R'): 'NASA',  # 'NASA-GISS',
    ('CMIP5', 'GISS-E2-H'): 'NASA',  # 'NASA-GISS',  # more complex
    ('CMIP5', 'HadGEM2-ES'): 'MOHC',
    ('CMIP5', 'IPSL-CM5A-LR'): 'IPSL',
    ('CMIP5', 'IPSL-CM5B-LR'): 'IPSL',
    ('CMIP5', 'IPSL-CM5A-MR'): 'IPSL',  # highest resolution
    ('CMIP5', 'MIROC5'): 'MIROC',
    ('CMIP5', 'MIROC-ESM'): 'MIROC',  # most complex
    ('CMIP5', 'MPI-ESM-P'): 'MPI-M',
    ('CMIP5', 'MPI-ESM-LR'): 'MPI-M',
    ('CMIP5', 'MPI-ESM-MR'): 'MPI-M',  # highest resolution
    ('CMIP5', 'MRI-CGCM3'): 'MRI',
    ('CMIP5', 'NorESM1-ME'): 'NCC',
    ('CMIP5', 'NorESM1-M'): 'NCC',  # highest resolution
    ('CMIP5', 'bcc-csm1-1'): 'BCC',
    ('CMIP5', 'bcc-csm1-1-m'): 'BCC',  # highest resoution
    ('CMIP5', 'inmcm4'): 'INM',
    ('CMIP6', 'ACCESS-CM2'): 'CSIRO',  # 'CSIRO-ARCCSS',
    ('CMIP6', 'ACCESS-ESM1-5'): 'CSIRO',  # most complex
    ('CMIP6', 'AWI-CM-1-1-MR'): 'AWI',
    ('CMIP6', 'BCC-CSM2-MR'): 'BCC',
    ('CMIP6', 'BCC-ESM1'): 'BCC',  # most complex
    ('CMIP6', 'CAMS-CSM1-0'): 'CAMS',
    ('CMIP6', 'CAS-ESM2-0'): 'CAS',
    ('CMIP6', 'CESM2-FV2'): 'NCAR',
    ('CMIP6', 'CESM2'): 'NCAR',
    ('CMIP6', 'CESM2-WACCM-FV2'): 'NCAR',
    ('CMIP6', 'CESM2-WACCM'): 'NCAR',  # highest resolution
    ('CMIP6', 'CIESM'): 'THU',
    ('CMIP6', 'CMCC-CM2-SR5'): 'CMCC',
    ('CMIP6', 'CMCC-ESM2'): 'CMCC',  # most complex
    ('CMIP6', 'CNRM-CM6-1'): 'CNRM-CERFACS',
    ('CMIP6', 'CNRM-CM6-1-HR'): 'CNRM-CERFACS',
    ('CMIP6', 'CNRM-ESM2-1'): 'CNRM-CERFACS',  # most complex
    ('CMIP6', 'CanESM5'): 'CCCma',
    ('CMIP6', 'E3SM-1-0'): 'E3SM-Project',
    ('CMIP6', 'EC-Earth3'): 'EC-Earth-Consortium',  # collaborative incl. many centers
    ('CMIP6', 'EC-Earth3-AerChem'): 'EC-Earth-Consortium',
    ('CMIP6', 'EC-Earth3-CC'): 'EC-Earth-Consortium',
    ('CMIP6', 'EC-Earth3-Veg-LR'): 'EC-Earth-Consortium',
    ('CMIP6', 'EC-Earth3-Veg'): 'EC-Earth-Consortium',  # most relevant
    ('CMIP6', 'FGOALS-f3-L'): 'CAS',
    ('CMIP6', 'FGOALS-g3'): 'CAS',  # highest resolution
    ('CMIP6', 'GFDL-CM4'): 'NOAA',  # 'NOAA-GFDL',
    ('CMIP6', 'GFDL-ESM4'): 'NOAA',  # 'NOAA-GFDL',  # most complex
    ('CMIP6', 'GISS-E2-1-G'): 'NASA',  # 'NASA-GISS',
    ('CMIP6', 'GISS-E2-1-H'): 'NASA',  # 'NASA-GISS',
    ('CMIP6', 'GISS-E2-2-G'): 'NASA',  # 'NASA-GISS',
    ('CMIP6', 'GISS-E2-2-H'): 'NASA',  # 'NASA-GISS',  # highest resolution
    ('CMIP6', 'IITM-ESM'): 'CCCR-IITM',
    ('CMIP6', 'INM-CM4-8'): 'INM',
    ('CMIP6', 'INM-CM5-0'): 'INM',  # latest version
    ('CMIP6', 'IPSL-CM6A-LR'): 'IPSL',
    ('CMIP6', 'IPSL-CM6A-LR-INCA'): 'IPSL',  # NOTE: only found in zelika table
    ('CMIP6', 'IPSL-CM5A2'): 'IPSL',  # NOTE: not yet found in anything
    ('CMIP6', 'IPSL-CM5A2-INCA'): 'IPSL',  # highest resolution
    ('CMIP6', 'KACE-1-0-G'): 'NIMS-KMA',
    ('CMIP6', 'KIOST-ESM'): 'KIOST',
    ('CMIP6', 'MCM-UA-1-0'): 'UA',
    ('CMIP6', 'MIROC6'): 'MIROC',
    ('CMIP6', 'MIROC-ES2L'): 'MIROC',
    ('CMIP6', 'MIROC-ES2H'): 'MIROC',  # latest version
    ('CMIP6', 'ICON-ESM-LR'): 'MPI-M',  # overwritten by MPI
    ('CMIP6', 'MPI-ESM-1-2-HAM'): 'MPI-M',  # 'HAMMOZ-Consortium',  # started from MPI
    ('CMIP6', 'MPI-ESM1-2-LR'): 'MPI-M',
    ('CMIP6', 'MPI-ESM1-2-HR'): 'MPI-M',  # highest resolution
    ('CMIP6', 'MRI-ESM2-0'): 'MRI',
    ('CMIP6', 'NESM3'): 'NUIST',
    ('CMIP6', 'NorCPM1'): 'NCC',
    ('CMIP6', 'NorESM2-LM'): 'NCC',
    ('CMIP6', 'NorESM2-MM'): 'NCC',  # highest resolution
    ('CMIP6', 'SAM0-UNICON'): 'SNU',
    ('CMIP6', 'TaiESM1'): 'AS-RCEC',
    ('CMIP6', 'HadGEM3-GC31-LL'): 'MOHC',  # overwritten by UKESM
    ('CMIP6', 'HadGEM3-GC31-MM'): 'MOHC',
    ('CMIP6', 'UKESM1-0-LL'): 'MOHC',  # ESM variant of HadGEM
    ('CMIP6', 'UKESM1-1-LL'): 'MOHC',  # latest version
}

# Sorting facets in file and folder names, in database groups and sub-groups,
# and in organizing wget script lines (see download.py for details).
# NOTE: Previously we removed bad or down nodes but this is unreliable and has to be
# updated regularly. Instead now prioritize local nodes over distant nodes and then
# wget script automatically skips duplicate files after successful download (see top).
SORT_FACETS = [
    'project',  # same as folder
    'model', 'source_id',
    'experiment', 'experiment_id',  # same as folder
    'ensemble', 'variant_label',
    'table', 'cmor_table', 'table_id',  # same as folder
    'variable', 'variable_id',
    'grid_label',
]
SORT_OPTIONS = {
    'experiment': [  # control runs come first
        'control',
        'historical',
        'hist',
    ],
    'node': [  # closer nodes come first
        'ucar.edu',  # colorado
        'anl.gov',  # illinois
        'llnl.gov',  # california
        'nccs.nasa.gov',  # maryland
        'gfdl.noaa.gov',  # new jersey
        'ec.gc.ca',  # canada
        'ceda.ac.uk',  # uk
        'ichec.ie',  # ireland
        'ipsl.upmc.fr',  # france
        'umr-cnrm.fr',  # france
        'dkrz.de',  # germany
        'dlr.de',  # germany
        'dwd.de',  # germany
        'pik-potsdam.de',  # germany
        'dmi.dk',  # denmark
        'nird.sigma2.no',  # norway
        'liu.se',  # sweden
        'csc.fi',  # finland
        'cineca.it',  # italy
        'cmcc.it',  # italy
        'bsc.es',  # spain
        'unican.es',  # spain
        'nci.org.au',  # australia
        'diasjp.net',  # japan
        'snu.ac.kr',  # south korea
        'pknu.ac.kr',  # south korea
        'apcc21.org',  # south korea
        'bcc.cma.cn',  # china
        'camscma.cn',  # china
        'fio.org.cn',  # china
        'lasg.ac.cn',  # china
        'tsinghua.edu.cn',  # china
        'sinica.edu.tw',  # taiwan
        'ru.ac.th',  # thailand
        'tropmet.res.in',  # india
    ],
}

# Node hosts for logon and urls for downloading
# Hosts are listed here: https://esgf.llnl.gov/nodes.html
# Nodes and statuses are listed here: https://esgf-node.llnl.gov/status/
# LLNL: 11900116 hits for CMIP6 (best!)
# DKRZ: 01009809 hits for CMIP6
# IPSL: 01452125 hits for CMIP6
URLS_HOSTS = {
    'llnl': 'esgf-node.llnl.gov',
    'ceda': 'esgf-index1.ceda.ac.uk',
    'dkrz': 'esgf-data.dkrz.de',
    'gfdl': 'esgdata.gfdl.noaa.gov',
    'ipsl': 'esgf-node.ipsl.upmc.fr',
    'jpl': 'esgf-node.jpl.nasa.gov',
    'liu': 'esg-dn1.nsc.liu.se',
    'nci': 'esgf.nci.org.au',
    'nccs': 'esgf.nccs.nasa.gov',
}
URLS_NODES = [
    'aims3.llnl.gov',
    'cmip.bcc.cma.cn',
    'cmip.dess.tsinghua.edu.cn',
    'cmip.fio.org.cn',
    'cordexesg.dmi.dk',
    'crd-esgf-drc.ec.gc.ca',
    'data.meteo.unican.es',
    'dataserver.nccs.nasa.gov',
    'dist.nmlab.snu.ac.kr',
    'dpesgf03.nccs.nasa.gov',
    'eridanus.eoc.dlr.de',
    'esg-cccr.tropmet.res.in',
    'esg-dn1.nsc.liu.se',
    'esg-dn1.ru.ac.th',
    'esg-dn2.nsc.liu.se',
    'esg.camscma.cn',
    'esg.lasg.ac.cn',
    'esg.pik-potsdam.de',
    'esg1.umr-cnrm.fr',
    'esg2.umr-cnrm.fr',
    'esgdata.gfdl.noaa.gov',
    'esgf-cnr.hpc.cineca.it',
    'esgf-data.csc.fi',
    'esgf-data.ucar.edu',
    'esgf-data1.ceda.ac.uk',
    'esgf-data1.diasjp.net',
    'esgf-data1.llnl.gov',
    'esgf-data2.ceda.ac.uk',
    'esgf-data2.diasjp.net',
    'esgf-data2.llnl.gov',
    'esgf-data3.ceda.ac.uk',
    'esgf-data3.diasjp.net',
    'esgf-ictp.hpc.cineca.it',
    'esgf-nimscmip6.apcc21.org',
    'esgf-node.cmcc.it',
    'esgf-node2.cmcc.it',
    'esgf.anl.gov',
    'esgf.apcc21.org',
    'esgf.bsc.es',
    'esgf.dwd.de',
    'esgf.ichec.ie',
    'esgf.nccs.nasa.gov',
    'esgf.nci.org.au',
    'esgf.rcec.sinica.edu.tw',
    'esgf1.dkrz.de',
    'esgf2.dkrz.de',
    'esgf3.dkrz.de',
    'noresg.nird.sigma2.no',
    'polaris.pknu.ac.kr',
    'vesg.ipsl.upmc.fr'
]

# Horizontal and vertical grid constants
# NOTE: Level values for various tables are listed in this PDF (where CMIP6 Amon
# uses the 19 level configuration and CMIP5 Amon uses the 17 level configuration):
# https://cmip6dr.github.io/Data_Request_Home/Documents/CMIP6_pressure_levels.pdf
# See process.py header for notes about vertical interpolation of model coordinates.
# NOTE: Lowest resolution in CMIP5 models is 64 latitudes 128 longitudes (also still
# used by BCC-ESM1 in CMIP6) and grids are mix of regular lon-lat and guassian (assume
# that former was interpolated). Select regular grid spacing with similar resolution but
# not coarser than necessary. See Horizontal grids->Grid description->Predefined grids
# for details. To summarize grid sizes try: unset models; for f in ts_Amon*; do
# model=$(echo $f | cut -d_ -f3); [[ " ${models[*]} " =~ " $model " ]] && continue
# models+=("$model"); echo "$model:" && ncdims $f | grep -E 'lat|lon' | tr -s ' ' | xargs; done  # noqa: E501
# 'r360x180'  # 1.0 deg, 'r180x90'  # 2.0 deg, 'r144x72'  # 2.5 deg
STANDARD_GRIDSPEC_CMIP = (
    'r72x36'  # 5.0 degree resolution
    # 'r144x72'  # 2.5 degree resolution
)
STANDARD_LEVELS_CMIP5 = 100 * np.array(
    [
        1000, 925, 850, 700, 600, 500, 400, 300, 250,  # tropospheric
        200, 150, 100, 70, 50, 30, 20, 10,  # stratospheric
    ]
)
STANDARD_LEVELS_CMIP6 = 100 * np.array(
    [
        1000, 925, 850, 700, 600, 500, 400, 300, 250,  # tropospheric
        200, 150, 100, 70, 50, 30, 20, 10, 5, 1,  # stratospheric (2 extra)
    ]
)


# Helper functions for parsing and naming based on facets
# NOTE: Join function is used for log files, script files, and folder names (which
# use simply <project_id>-<experiment_id>-<table_id>). Replace the dashes so that
# e.g. models and experiments with dashes are not mistaken for field delimiters.
# NOTE: Netcdf names are <variable_id>_<table_id>_<source_id>_<experiment_id>...
# ..._<member_id>[_<grid_label>[_<start_date>-<end_date>[-<climate_indicator>]]].nc
# where grid labels are only in cmip6 and climate indicator is used for e.g. pfull.
# See: https://github.com/WCRP-CMIP/CMIP6_CVs/blob/master/CMIP6_grid_label.json
_item_file = lambda line: (
    getattr(line, 'name', line).split()[0].strip("'")
)
_item_node = lambda line: (
    line.split()[1].strip("'").split('thredds')[0]
)
_item_dates = lambda file: (
    d if (d := _item_piece(file, -1))[0].isdecimal() else ''
)
_item_years = lambda file: tuple(
    int(s[:4]) for s in _item_dates(file).split('-')[:2]
)
_item_piece = lambda file, idx: (
    _item_file(file).split('.')[0].split('_')[idx]  # use part before .nc suffix
)
_item_parts = {
    'model': lambda file: _item_piece(file, 2),
    'experiment': lambda file: _item_piece(file, 3),
    'ensemble': lambda file: _item_piece(file, 4),
    'table': lambda file: _item_piece(file, 1),
    'variable': lambda file: _item_piece(file, 0),
    'grid': lambda file: g if (g := _item_piece(file, -2))[0] == 'g' else 'g',
}
_item_join = lambda *options, modify=True: '_'.join(
    item for opts in options if opts and (
        item := '-'.join(
            re.sub(r'\W+', '', opt.lower()) if modify else opt
            for opt in filter(
                None, opts.split(',') if isinstance(opts, str) else tuple(opts)
            )
        )
    )
)

# Helper function for sorting based on facets
# NOTE: Previously we sorted files alphabetically but this means when processing
# files for the first time we iterate over abrupt variables before the control
# variables required for drift correction are available. The generalized priority
# sorting ensures control data comes first and leaves room for additional restrictions.
# NOTE: Previously we detected and removed identical files available from multiple nodes
# but this was terrible idea. The scripts will auto-skip files that both exist and are
# recorded in the .wget cache file, so as they loop over files they ignore already
# downloaded entries. This lets us maximize probability and speed of retrieving files
# without having to explicitly filter bad nodes. Example indices for identical names:
# ('00_ACCESS', ..., '02_esgf-data1.llnl.gov'), ('00_ACCESS', ..., '22_esgf.nci.org.au')
_sort_part = lambda item, facet: (
    str(item) if '_' not in getattr(item, 'name', item)
    else _item_dates(item) if facet == 'dates'
    else _item_node(item) if facet == 'node'
    else _item_parts[facet](item)
)
_sort_index = lambda item, facet: (
    (part := _sort_part(item, facet))
    and (opts := (*SORT_OPTIONS.get(facet, ()), part.lower()))  # adds default entry
    and min(f'{i:02d}_{part}' for i, opt in enumerate(opts) if opt in part.lower())
)
_sort_facet = lambda items, facet: sorted(
    set(items), key=lambda item: _sort_index(item, facet)
)
_sort_facets = lambda items, facets: sorted(
    items, key=lambda item: tuple(_sort_index(item, facet) for facet in facets)
)


def _parse_constraints(reverse=False, restrict=True, **constraints):
    """
    Standardize the constraints, accounting for facet aliases and option renames.

    Parameters
    ----------
    reverse : bool, default: False
        Whether to reverse translate facet aliases.
    restrict : bool, default: True
        Whether to restrict to facets readable in script lines.
    **constraints
        The constraints.
    """
    # NOTE: This sets a default project when called by download_script, filter_script,
    # or process_files, and enforces a standard order for file and folder naming. Also
    # enforced that control runs come before abrupt runs for consistency with database.
    for key in ('flagship_filter', 'flagship_translate'):
        constraints.pop(key, None)
    constraints = {
        facet: list(set(opts.split(',') if isinstance(opts, str) else opts))
        for facet, opts in constraints.items()
    }
    renames = FACETS_RENAMES
    dotnames_encode = FACETS_DOTNAMES  # convert dash names to dot names
    dotnames_decode = {(_, dotname): name for (_, name), dotname in dotnames_encode.items()}  # noqa: E501
    longnames_encode = FACETS_LONGNAMES  # convert short names to long names
    longnames_decode = {(_, long): short for (_, short), long in longnames_encode.items()}  # noqa: E501
    projects = constraints.setdefault('project', ['CMIP6'])
    if len(projects) != 1:
        raise NotImplementedError('Non-scalar projects are not supported.')
    projects[:] = [project := projects[0].upper()]
    facets = (
        *(facet for facet in SORT_FACETS if facet in constraints),  # impose order
        *(facet for facet in constraints if facet not in SORT_FACETS),  # retain order
    )
    constraints = {
        (longnames_decode if reverse else longnames_encode)  # facet translations
        .get((project, facet), facet):
        _sort_facet(
            (
                (dotnames_decode if reverse else dotnames_encode)  # model translations
                .get(
                    (project, opt),
                    renames.get((project, opt), opt)  # experiment translations
                )
                for opt in constraints[facet]
            ),
            longnames_decode.get((project, facet), facet)  # need standard name here
        )
        for facet in facets
    }
    if not restrict:
        pass
    elif not constraints.keys() - _item_parts.keys() <= {'project'}:
        raise ValueError(f'Facets {constraints.keys()} must be subset of: {_item_parts.keys()}')  # noqa: E501
    return project, constraints


def _validate_ranges(variable, table='Amon'):
    """
    Return the quality control ranges from the source file.

    Parameters
    ----------
    variable : str
        The variable to select.
    table : str, optional
        The table to select.
    """
    # NOTE: This uses the official cmip quality control ranges produced for modeling
    # centers to use for validating simulations and published netcdf files. They
    # were copied into VALIDATE.txt from esgf website (see header for details).
    path = Path(__file__).parent.parent / 'VALIDATE.txt'
    data = open(path).read()
    keys = ('valid_min', 'valid_max', 'ok_min_mean_abs', 'ok_max_mean_abs')
    values = [None, None, None, None]
    for part in data.split('\n\n'):
        if (
            part[0] == '['
            and (m := re.match(r'\[(\w*)[-](\w*)\]', part))
            and m.groups() == (table, variable)
        ):
            for i, name in enumerate(keys):
                if m := re.search(rf'{name}(?:\[.\])?:\s*(\S*)\s', part):
                    values[i] = float(m.group(1))
            break
    return tuple(values)


def glob_files(*paths, pattern='*', project=None):
    """
    Return a list of files matching the input pattern and a second list of files
    containing corrupt or duplicate files (e.g. with an ``.nc4`` extension).

    Parameters
    ----------
    *path : path-like
        The source path(s).
    pattern : str, default: '*'
        The glob pattern preceding the file extension.
    """
    # NOTE: Could amend this to not look for subfolders if 'project' was not passed
    # but prefer consistency with Logger and Database of always using cmip6
    # as the default, and this is the only way to filter paths by 'project'.
    project = (project or 'cmip6').lower()
    paths = paths or ('~/data',)
    files = _sort_facets(  # have only seen .nc4 but this is just in case
        (
            file for ext in ('.nc', '.nc[0-9]') for path in paths
            for folder in Path(path).expanduser().glob(f'{project}*')
            for file in folder.glob(pattern + ext)
            if '_standard-' not in file.name  # skip intermediate files
            and file.name.count('_') >= 4  # skip temporary files
            and (
                file.stat().st_size > 0 or file.unlink()
                or print(f'Warning: Removed empty download file {file.name!r}')
            )
        ), facets=(*_item_parts, 'dates'),
    )
    files_corrupt = []
    files_duplicate = []
    files_filtered = [
        file for file in files if (
            file.suffix == '.nc' or file.parent / f'{file.stem}.nc' not in files
            or files_duplicate.append(file)  # always false
        ) and (
            file.name not in CORRUPT_FILES
            or files_corrupt.append(file)  # always false
        )
    ]
    if files_duplicate:
        message = ' '.join(file.name for file in files_duplicate)
        print(f'Ignoring duplicate files: {message}.')
    if files_corrupt:
        message = ' '.join(file.name for file in files_corrupt)
        print(f'Ignoring corrupt files: {message}.')
    return files_filtered, files_duplicate, files_corrupt


class Logger(object):
    """
    Simultaneously print the results and add them to an automatically named log file.
    """
    def __init__(self, prefix, *suffixes, backup=False, **constraints):
        """
        Parameters
        ----------
        prefix : str
            The log file prefix (joined with underscore).
        *suffixes : str, optional
            The optional log file suffixes (joined with dashes).
        **constraints
            The constraints (joined with dashes and underscores).
        backup : bool, optional
            Whether to backup existing files.
        """
        _, constraints = _parse_constraints(reverse=True, **constraints)
        name = _item_join(prefix, *constraints.values(), suffixes) + '.log'
        path = Path(__file__).parent.parent / 'logs' / name
        path.parent.mkdir(exist_ok=True)
        if path.is_file():
            if backup:
                print(f'Moving previous log to backup: {path}')
                path.rename(str(path) + '.bak')
            else:
                print(f'Removing previous log: {path}')
                path.unlink(missing_ok=True)
        self.path = path
        self.constraints = constraints

    def __call__(self, *args, sep=' ', end='\n'):
        """
        Parameters
        ----------
        *args
            The objects to print.
        sep : str, default: ' '
            Separation between objects
        end : str, default: '\\n'
            Ending after last printed object.
        """
        print(*args, sep=sep, end=end)
        with open(self.path, 'a') as f:
            f.write(sep.join(map(str, args)) + end)


class Database(object):
    """
    A nested dictionary that organizes file names and wget script lines based
    on two levels of facet options. This is used to group dictionaries of files
    for individual models, to create control-response pair groups, and to create
    groups of files destined for separate subfolders.
    """
    def __str__(self):
        return self.summarize(printer=lambda *a, **k: None)  # noqa: U100

    def __len__(self):  # length of nested lists
        return sum(len(data.values()) for data in self.database.values())

    def __bool__(self):
        return bool(len(self))

    def __iter__(self):  # iterate through nested lists
        for data in self.database.values(): yield from data.values()  # noqa: E701

    def __contains__(self, key):  # cloned dictionary operator
        return self.database.__contains__(key)

    def __getitem__(self, key):
        return self.database.__getitem__(key)

    def get(self, *args):
        return self.database.get(*args)

    def keys(self):
        return self.database.keys()

    def values(self):
        return self.database.values()

    def items(self):
        return self.database.items()

    def __init__(
        self, source, facets,
        flagship_translate=False,
        flagship_filter=False,
        **constraints
    ):
        """
        Parameters
        ----------
        source : sequence of str or path-like, optional
            The script lines or file paths.
        facets : sequence of str
            The facets to use as group keys.
        flagship_translate : bool, optional
            Whether to relable ensembles belonging to the flagship run.
        flagship_filter : bool, optional
            As with `flagship_translate` but also filter to only flagships.
        **constraints : optional
            The constraints.
        """
        # Initial stuff
        # NOTE: Since project is not included in script lines and file names we
        # propagate it here. Also must always be scalar since not included in
        # native file names or script lines and would be difficult to track.
        flagship_translate = flagship_translate or flagship_filter
        if flagship_filter:  # alias for flagship_translate=True and ensemble='flagship'
            constraints['ensemble'] = 'flagship'
        facets = facets.split(',') if isinstance(facets, str) else tuple(facets)
        project, constraints = _parse_constraints(reverse=True, **constraints)
        facets_group = ('project', *(facet for facet in _item_parts if facet in facets))
        facets_key = tuple(facet for facet in _item_parts if facet not in facets)

        # Add lists to sub-dictionaries
        # NOTE: Have special handling for feedback files -- only include when
        # users explicitly pass variable='feedbacks', exclude otherwise.
        self.database = {}
        for value in _sort_facets(source, facets=(*facets_group[1:], *facets_key)):
            parts = {'project': project}
            parts.update({facet: func(value) for facet, func in _item_parts.items()})
            key_flagship = (project, parts['experiment'], parts['model'])
            ens_default = ENSEMBLES_FLAGSHIP[project, None, None]
            ens_flagship = ENSEMBLES_FLAGSHIP.get(key_flagship, ens_default)
            if flagship_translate and parts['ensemble'] == ens_flagship:
                parts['ensemble'] = 'flagship'  # otherwise retain ensemble
            if any(opt not in constraints.get(facet, (opt,)) for facet, opt in parts.items()):  # noqa: E501
                continue  # item not in constraints
            if parts['variable'] == 'feedbacks' and 'feedbacks' not in constraints.get('variable', ()):  # noqa: E501
                continue  # require explicitly requesting bulk feedbacks file
            if parts['variable'] == 'derived' and 'derived' not in constraints.get('variable', ()):  # noqa: E501
                continue  # require explicitly requesting bulk derived file
            group = tuple(parts[facet] for facet in facets_group)
            data = self.database.setdefault(group, {})
            key = tuple(parts[facet] for facet in facets_key)
            values = data.setdefault(key, set())
            values.add(value)

        # Sort the results and store attributes
        # NOTE: This is important for e.g. databases of wget script lines
        # since otherwise duplicates will appear. Not sure of other uses.
        self.group = facets_group
        self.key = facets_key
        self.project = project
        self.constraints = constraints
        for group, data in self.database.items():
            for key, items in tuple(data.items()):
                data[key] = sorted(items)

    def filter(self, keys=None, always_include=None, always_exclude=None):
        """
        Filter the database keys. Compare to `.restrict`.

        Parameters
        ----------
        keys : sequence, optional
            Valid keys for the database.
        always_include : dict or sequence, optional
            The dictionari(es) describing valid constraints.
        always_exclude : dict or sequence, optional
            The dictionari(es) describing invalid constraints.
        """
        # Parse override definitions
        overrides = []
        for override in (always_include, always_exclude):
            override = override or ()
            if not isinstance(override, (tuple, list)):
                override = (override,)
            dicts = []
            for constraints in override:
                _, constraints = _parse_constraints(
                    reverse=True, project=self.project, **constraints
                )
                constraints.pop('project', None)
                dicts.append(constraints)
            overrides.append(dicts)

        # Apply filters to data
        always_include, always_exclude = overrides
        for group, data in self.database.items():
            for key in tuple(data):
                parts = dict(zip((*self.group, *self.key), (*group, *key)))
                if keys and key not in keys and not any(
                    all(parts[facet] in opts for facet, opts in constraints.items())
                    for constraints in always_include
                ):
                    del data[key]
                elif any(
                    all(parts[facet] in opts for facet, opts in constraints.items())
                    for constraints in always_exclude
                ):
                    del data[key]

    def summarize(self, missing=True, rawdata=False, printer=None, message=None):
        """
        Print information about the database and return the message.

        Parameters
        ----------
        missing : bool, optional
            Whether to include info about missing facets.
        rawdata : bool, optional
            Whether to show the raw data for each group.
        printer : callable, optional
            The print function.
        message : str
            An additional message
        """
        # Initial stuff
        lines = []
        print = lambda *args, **kwargs: (
            lines.append(' '.join(map(str, args)))
            or (printer or builtins.print)(*args, **kwargs)
        )
        flush = lambda dict_: tuple(
            print(f'  {facet}s ({len(opts)}):', ', '.join(map(str, _sort_facet(opts, facet))))  # noqa: E501
            for facet, opts in dict_.items() if opts
        )
        database = {
            group: (data, opts)
            for group, data in self.database.items()
            if (keys := tuple(key for key, values in data.items() if values))
            and (opts := {facet: set(opts) for facet, opts in zip(self.key, zip(*keys))})  # noqa: E501
        }

        # Print information
        if message:
            print(f'{message}:')
        for group, (data, opts) in database.items():
            print(', '.join(f'{facet}: {opt}' for facet, opt in zip(self.group, group)))
            if missing:
                print('missing:')
                mopts = {
                    facet: sorted(set(opt for (_, opts) in database.values() for opt in opts[facet]) - set(opts[facet]))  # noqa: E501
                    for facet in self.key
                }
                flush(mopts)
                print('present:')
            flush(opts)
            if rawdata:
                print('raw data:')
                rawdata = '\n'.join(
                    f"  {getattr(obj, 'name', obj)}"
                    for objs in data.values() for obj in objs
                )
                print(rawdata)
        print()
        return '\n'.join(lines)
