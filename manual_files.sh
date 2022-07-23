#!/usr/bin/env bash
# This file contains scripts for manually downloading missing data not
# available on main server. Uses either wget or lftp to connect.
gfdl=false
mcmua=false

# Missing cmip5 downloads
# NOTE: Had constant issues trying to download with wget on macbook (triggered 500
# unknown command errors), with wget on remote server (command would hang), and with
# lftp on remote server (would connect but hang when trying to list content). Since
# it worked on macbook tried to use this script: https://serverfault.com/a/25286/427991
# but transfer to server with lrcp was too slow... another solution was to use curl
# with in-built globbing patterns using e.g. ts_..._[000000-015012]-[00000-015012].nc
# but this also failed... so elected to simply iterate over files using name convention.
# NOTE: Tried to devise similar script for logging into CEDA acrhive using
# ftp://ftp.ceda.ac.uk/badc/cmip5/data/cmip5/output1/ICHEC/EC-EARTH/ with username
# lukelbd and password generated from CEDA Webpage->My Account->Configure FTP Account
# but never worked... instead opted to manually browse and download from HTTP URLs
# with e.g. https://dap.ceda.ac.uk/badc/cmip5/data/cmip5/output1/ICHEC/. Also for
# some reason using 'curl -O <link>' with individual files failed everywhere, had
# to be done interactively, and failed to use curl -O -u 'username:password' or
# wget with username and password on command line (curl would seem to work but
# only return tiny files), so only download interactively... however in the end
# decided not to download this data at all since too much is missing.
if $gfdl; then
  # path=$HOME/Downloads  # macbook
  path=$HOME/scratch2/data-downloads  # server
  cd $path || { echo "Error: Cannot find destination $path"; exit 1; }
  for model in GFDL-CM3 GFDL-ESM2G GFDL-ESM2M; do
    for experiment in piControl abrupt4xCO2; do
      for variable in rsut rsutcs rsus rsuscs ta ts va vas zg ; do
        url=ftp://anonymous:anonymous@nomads.gfdl.noaa.gov
        url=$url/CMIP5/output1/NOAA-GFDL/$model/$experiment
        url=$url/mon/atmos/Amon/r1i1p1/v20110601/$variable
        head=${variable}_Amon_${model}_${experiment}_r1i1p1
        for year in $(seq 5 5 150); do
          date=$(printf %04d $((year - 4)))01-$(printf %04d $year)12
          file=${head}_${date}.nc
          echo "Getting file: $url/$file"
          curl -O "$url/$file"
        done
        # for i in 0 1; do  # globbing approach
        #   [[ $i -eq 0 ]] && glob='0-9' || glob='0-4'
        #   pattern="${variable}_*_0${i}[${glob}]*"
        #   echo "Getting pattern: $pattern"
        #   lftp -d -c "open $url; mget $pattern"
        # done
      done
    done
  done
fi

# Missing cmip6 radiative flux data
# NOTE: Here had to manually download and combine wget scripts... for some reason
# the online ESGF interface was giving weird results.
# NOTE: Use the fact that rtmt = rsdt - rsut - rlut --> rsut - rsdt = -1 * (rtmt + rlut)
# to create nominal 'rsut' files containing 'rsut - rsdt' values... then create a
# zero-valued dummy rsdt file and the places where we compute net 'rsdt - rsut' flux
# (e.g. for ocean + atmosphere energy transport) give the same result as true rsdt.
if $mcmua; then
  for experiment in picontrol abrupt4xco2; do
    path=$HOME/scratch5/cmip6-$experiment-amon
    cd $path || { echo "Error: Cannot find location $path"; exit 1; }
    for rtmt in rtmt_Amon_MCM-UA-1-0*; do
      rlut=${rtmt/rtmt/rlut}
      rsut=${rtmt/rtmt/rsut}
      rsdt=${rtmt/rtmt/rsdt}
      [ -r "$rlut" ] || { echo "Error: Cannot find file $rlut"; continue; }
      echo "Converting net downward flux to upward shortwave minus solar constant."
      echo "Command: cdo -mulc,-1 -add $rtmt $rlut $rsut."
      cdo -O \
        -setattribute,rsut@long_name="TOA Outgoing Minus Incident Shortwave Radiation" \
        -setattribute,rsut@standard_name="toa_outgoing_minus_incoming_shortwave_flux" \
        -chname,rtmt,rsut -mulc,-1 -add "$rtmt" "$rlut" "$rsut"
      echo "Converting net downward flux to dummy solar constant file."
      echo "Command: cdo -mulc,0 $rtmt $rsdt."
      cdo -O \
        -setattribute,rsdt@long_name="TOA Zero Times Incident Shortwave Radiation" \
        -setattribute,rsdt@standard_name="toa_zero_times_incoming_shortwave_flux" \
        -chname,rtmt,rsdt -mulc,0 "$rtmt" "$rsdt"
    done
  done
fi
