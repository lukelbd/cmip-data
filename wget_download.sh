#!/usr/bin/env bash
#------------------------------------------------------------------------------#
# Use RESTful syntax to download massive numbers of files
# The RESTful wget generator screws up often; get unexpected variable names
# and get truncation even when number of files is well within download limit.
# For safety, always use wget scripts with just 1 variable.
#------------------------------------------------------------------------------#
# Play around to see what's available with: https://esgf-node.llnl.gov/search/esgf-llnl/
# List of experiments: http://www.ipcc-data.org/sim/gcm_monthly/AR5/CMIP5-Experiments.html
# CF long and short names: http://cfconventions.org/Data/cf-standard-names/65/build/cf-standard-name-table.html
# Follows FAQ info here: https://www.earthsystemcog.org/projects/cog/doc/esgf/faq/wget#Wget%20Script%20File%20List%20Issues
# Also see guide here: https://www.earthsystemcog.org/projects/cog/doc/wget
# Nodes listed here: https://esgf.llnl.gov/nodes.html
# LLNL: https://esgf-node.llnl.gov/
# CEDA: https://esgf-index1.ceda.ac.uk/
# DKRZ: https://esgf-data.dkrz.de/
# GFDL: https://esgdata.gfdl.noaa.gov/
# IPSL: https://esgf-node.ipsl.upmc.fr/
# JPL:  https://esgf-node.jpl.nasa.gov/
# LIU:  https://esg-dn1.nsc.liu.se/
# NCI:  https://esgf.nci.org.au/
# NCCS: https://esgf.nccs.nasa.gov/
#------------------------------------------------------------------------------#
# The base URL
base='http://esgf-data.dkrz.de/esg-search/wget/?project=CMIP5'
base='http://esgf-node.llnl.gov/esg-search/wget/?project=CMIP5'

# Function that builds URL from input variable names
# WARNING: Options in different categories interpreted as logical AND, options
# in same category interpreted as logical OR.
# WARNING: Need to download variable-by-variable! Even below hard limit max
# of 10000 files, get silent truncation of models! Also get random unwanted
# variables if we use multiple vars in the URL.
build() {
  local exp vars url
  # Input data
  exp=$1
  table=$2 # the cmor table
  vars=("${@:3}") # can be single string or several strings
  # Build URL
  url="$base&experiment=$exp&ensemble=r1i1p1&cmor_table=$table" # will then filter files ourselves, e.g. only download 100 days!
  for var in "${vars[@]}"; do
    url="$url&variable=$var"
  done
  url="$url&limit=10000" # constants
  echo "$url"
}

#------------------------------------------------------------------------------#
# CMIP5 (climate model intercomparison project)
#------------------------------------------------------------------------------#
# Average data, for climate sensitivity and isentropic slope, mean circulation stuff
exps=(abrupt4xCO2 piControl)
vars=(ps ta ua va tas rlut rsut rsdt) # fluxes, radiation longwave upwelling surface, etc.
tables=(Amon) # also try 6hrPlev

# Daily data for correlation thing
# NOTE: Surface pressure not available, only sea-level pressure, which is useless
# for taking vertical mass-weighted averages.
# NOTE: Shortwave surface budget available, but TOA shortwave budget and
# clear-sky components not available. Need cfDay for that!
# Radiation longwave upwelling surface == rlus, et cetera
# vars=(ta hfls hfss rlds rlus rlut rsut) # fluxes, radiation longwave upwelling surface, etc.
exps=(piControl)
vars=(ta ua va hfls hfss rlds rlus rlut) # fluxes, radiation longwave upwelling surface, etc.
tables=(day)

#------------------------------------------------------------------------------#
# CFMIP (cloud feedback model intercomparison project)
# WARNING: Hoped had more data, but turns out only one model
#------------------------------------------------------------------------------#
# Average wind, temperature, radiation
# This is for climate sensitivity and isentropic slope, mean circulation stuff
# exps=(abrupt4xCO2 piControl)
# vars=(ta ua va rlut rsut rsdt) # fluxes, radiation longwave upwelling surface, etc.
# tables=(cfMon)  # also try 6hrPlev

# Average diabatic heating
# This is only available cfMip for 1 model
# vars=(ta ua va tntmp)
# tables=(cfMon)  # also try 6hrPlev

# Daily data
# Get latent heating from liquid water path change from centered finite difference
# plus the amount precipitated out (WARNING: only convective available,
# not stratiform, but probably much bigger and anyway cloud generation
# rate probably larger than the amount precipitated out)
# NOTE: 3D data is on model levels, so need surface pressure
# NOTE: There is no 'clear sky' rlus (just surface emission, no longwave reflection) or rsdt (just solar radiation)
# WARNING: Sea-level pressure only available for single model, so forget it.
exps=(piControl)
vars=(ps ta ua va hfls hfss rlds rlus rlut rsds rsus rsut rsdt rldscs rlutcs rsdscs rsuscs rsutcs) # fluxes, radiation longwave upwelling surface, etc.
tables=(cfDay) # WARNING: cf3hr data all unavailable


#------------------------------------------------------------------------------#
# Get wget script(s)
#------------------------------------------------------------------------------#
# Loop through time tables, experiments, and variables
for table in ${tables[@]}; do
  for exp in ${exps[@]}; do
    for var in ${vars[@]}; do
      url=$(build $exp $table $var)
      echo "Download url: $url"
      file=wgets/${exp}-${table}-${var}.sh
      wget -O - "$url" | grep -P "^(?!'(?!${var}_)|'.*\.nc4)" 1>$file
      nwords=$(cat "$file" | wc -l)
      [ $nwords -le 1 ] && echo "No files found." && rm "$file"
    done
  done
done
