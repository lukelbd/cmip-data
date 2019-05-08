#!/usr/bin/env bash
#------------------------------------------------------------------------------#
# Use RESTful syntax to download massive numbers of files
# Play around to see what's available with: https://esgf-node.llnl.gov/search/esgf-llnl/
# List of experiments: http://www.ipcc-data.org/sim/gcm_monthly/AR5/CMIP5-Experiments.html
# CF long and short names: http://cfconventions.org/Data/cf-standard-names/65/build/cf-standard-name-table.html
# Follows FAQ info here: https://www.earthsystemcog.org/projects/cog/doc/esgf/faq/wget#Wget%20Script%20File%20List%20Issues
# Also see guide here: https://www.earthsystemcog.org/projects/cog/doc/wget
#------------------------------------------------------------------------------#
# The base URL
# Nodes listed here: https://esgf.llnl.gov/nodes.html
# NOTE: Think we just use a data node for *searching*, but maybe download
# can happen through any node?
# LLNL: https://esgf-node.llnl.gov/
# CEDA: https://esgf-index1.ceda.ac.uk/
# DKRZ: https://esgf-data.dkrz.de/
# GFDL: https://esgdata.gfdl.noaa.gov/
# IPSL: https://esgf-node.ipsl.upmc.fr/
# JPL:  https://esgf-node.jpl.nasa.gov/
# LIU:  https://esg-dn1.nsc.liu.se/
# NCI:  https://esgf.nci.org.au/
# NCCS: https://esgf.nccs.nasa.gov/
base='http://esgf-data.dkrz.de/esg-search/wget/?project=CMIP5'

# Function that builds URL from input variable names
# NOTE: Options in different categories interpreted as logical AND, options
# in same category interpreted as logical OR.
# WARNING: Hard limit of max 10000 files
build() {
  local exp vars url
  exp=$1
  freqs="$2" # can be single string
  vars="${@:3}" # can be single string or several strings
  url="$base&experiment=$exp&ensemble=r1i1p1&limit=10000" # will then filter files ourselves, e.g. only download 100 days!
  for freq in $freqs; do
    url="$url&time_frequency=$freq"
  done
  for var in $vars; do
    url="$url&variable=$var"
  done
  echo "$url"
}

# Wget script will appear here
vars='ta ua va' # sometimes psl not available, so try to get it
# exps=(rcp85 2xCO2 amip4xCO2 abrupt4xCO2 amip historical piControl historicalNat)
exps=(abrupt4xCO2 piControl)
# freqs=('6hr 3hr' day mon)
freqs=(minClimo)
for freq in "${freqs[@]}"; do
  for exp in ${exps[@]}; do
    url="$(build $exp "$freq" "$vars")"
    echo "Download url: $url"
    file=wgets/wget-${exp}-${freq% *}.sh
    wget -O - "$url" 1>$file
  done
done
