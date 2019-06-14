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
  vars="${@:3}" # can be single string or several strings
  # Build URL
  url="$base&experiment=$exp&ensemble=r1i1p1&cmor_table=$table" # will then filter files ourselves, e.g. only download 100 days!
  for var in $vars; do
    url="$url&variable=$var"
  done
  url="$url&limit=10000" # constants
  echo "$url"
}

# Average data, for climate sensitivity and isentropic slope, mean circulation stuff
exps=(abrupt4xCO2 piControl)
vars=(ta ua va rlut rsut rsdt) # fluxes, radiation longwave upwelling surface, etc.
tables=(Amon) # also try 6hrPlev
# Daily data for correlation thing
# exps=(piControl)
# vars=(ta hfls hfss rlds rlus rlut) # fluxes, radiation longwave upwelling surface, etc.
# tables=(day)
# Diabatic heating itself only available below, and only for 1 model
# vars=(ta ua va tntmp)
# tables=(cfMon) # also try 6hrPlev

# Get wget script
for table in ${tables[@]}; do
  for exp in ${exps[@]}; do
    for var in ${vars[@]}; do
      url="$(build $exp $table $var)"
      echo "Download url: $url"
      file=wgets/wget-raw-${exp}-${table}-${var}.sh
      wget -O - "$url" | grep -P "^(?!'(?!${var}_)|'.*\.nc4)" 1>$file
      nwords=$(cat "$file" | wc -l)
      [ $nwords -eq 0 ] && echo "No files found." && rm "$file"
    done
  done
done
