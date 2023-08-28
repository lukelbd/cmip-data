#!/usr/bin/env bash
#------------------------------------------------------------------------------#
# This script can do two things:
# 1. Prints available years as function of model for each variable (iterating
#    through files on disk or optionally the files inside a wget script).
# 2. Compute climatological averages from the 'nclimate' initial or final years
#    of the CMIP run. Default is 50 years.
# WARNING: Tempting to combine all variables into one file, but that adds an
# extra unnecessary step when you can just make your load script merge the
# DataArrays into a Dataset, and gets confusing when want to add new variables.
#------------------------------------------------------------------------------#
# Constants
shopt -s nullglob
overwrite=false
dryrun=false
data=$HOME/data
[[ "$OSTYPE" =~ darwin* ]] && root=$data || root=~/scratch

# Simple helper functions
# See: https://unix.stackexchange.com/a/13779/112647
# WARNING: Dates sent to awk scripts must be sorted!
year1() {
  local date=${1%-*}
  echo ${date::4}
}
year2() {
  local date=${1#*-}
  echo ${date::4}
}
minyear() {
  local min='$1 ~ /^[0-9]*(\.[0-9]*)?$/ { a[c++] = $1 }; END { print a[0] }'
  echo "$@" | tr ' ' $'\n' | cut -c-4 | awk "$min" | bc -l
}
maxyear() {
  local max='$1 ~ /^[0-9]*(\.[0-9]*)?$/ { a[c++] = $1 }; END { print a[c-1] }'
  echo "$@" | tr ' ' $'\n' | cut -c-4 | awk "$max" | bc -l
}

# Driver function
driver() {
  # Find available models and variables
  # TODO: Enable wget=false or wget=true not automatically
  local project experiment table vars  # avoid overwriting these, don't worry about others
  [ $# -ge 3 ] || { echo && echo "Error: At least 3 arguments required." && return 1; }
  project=$1
  experiment=$2
  table=$3
  vars=("${@:4}")  # TODO: also permit restricting models?
  [[ "$table" =~ Emon ]] && pattern=Amon || pattern=$table
  [[ "$pattern" =~ mon ]] && wget=false || wget=true  # show climate summaries from wget scripts?
  string=${project}-${experiment}-${pattern}
  if $wget; then  # search files in wget script
    eof=EOF--dataset.file.url.chksum_type.chksum
    input=$root/wgets/wget_${string}.sh
    [ -r "$input" ] || { echo && echo "Error: File $input not found." && return 1; }
    files_all=$(cat $input | sed -n "/$eof/,/$eof/p" | grep -v "$eof" | cut -d"'" -f2)
  else  # search files on disk
    input=${root}/${string}
    [ -d "$input" ] || { echo && echo "Error: Path $input not found." && return 1; }
    output=${data}/${string}
    [ -r "$output" ] || mkdir "$output" || { echo && echo "Error: Failed to make $output." && return 1; }
    files_all=$(find $input -name "*.nc" -printf "%f\n")
  fi

  # Iterate through models then variables
  # NOTE: The quotes around files are important! contains newlines!
  models=($(echo "$files_all" | cut -d'_' -f3 | sort | uniq))
  [ ${#vars[@]} -eq 0 ] && vars=($(echo "$files_all" | cut -d'_' -f1 | sort | uniq))
  echo
  echo "Table $table, experiment $experiment: ${#models[@]} models found"
  for model in ${models[@]}; do
    # [ "$model" != "CCSM4" ] && continue  # for testing
    echo
    echo "Input: ${input##*/}, Model: $model"
    for var in "${vars[@]}"; do
      # List files and get file info
      # NOTE: Files may indicate if they are 'spun up' with 'parent_experiment_id'
      # NOTE: Some files have extra field but date is always last so use rev.
      # TODO: Add year suffix? Then need to auto-remove files with different range.
      tmp=${output}/tmp.nc
      out=${output}/${var}_${table}_${model}_${experiment}_${project}.nc
      exists=false
      [ -r $out ] && [ "$(ncdump -h "$out" 2>/dev/null | grep 'UNLIMITED' | tr -dc 0-9)" -gt 0 ] && exists=true
      files=($(echo "$files_all" | grep "${var}_${table}_${model}_"))
      dates=($(echo "${files[@]}" | tr ' ' $'\n' | rev | cut -d'_' -f1 | rev | sed 's/\.nc//g' | sort))
      if [ ${#files[@]} -eq 0 ]; then
        echo "$(printf "%-20s" "$var (unfiltered):") not found"
        continue
      else
        ymin=$(minyear "${dates[@]%-*}")
        ymax=$(maxyear "${dates[@]#*-}")
        echo "$(printf "%-20s" "$var (unfiltered):") $ymin-$ymax (${#files[@]} files)"
      fi

      # Select files for averaging
      # NOTE: Use at least this many years from start of simulation (models are
      # already spun up, and this could help remove drift during control runs).
      unset files_climo dates_climo
      [[ $experiment =~ CO2 ]] && ny=$nresponse || ny=$nclimate
      for i in $(seq 1 "${#files[@]}"); do
        file=${files[i - 1]}
        date=${dates[i - 1]}
        # shellcheck disable=SC2046
        if [ $(stat -c "%s" "$input/$file") -eq 0 ]; then
          echo "Warning: Deleting empty file $file"
          rm "$input/$file"
        elif [ "$(year1 $date)" -le $((ymin + ny - 1)) ]; then
          files_climo+=("$file")
          dates_climo+=("$date")
        fi
      done
      ymin=$(minyear "${dates_climo[@]%-*}")
      ymax=$(maxyear "${dates_climo[@]#*-}")
      echo "$(printf "%-20s" "$var (filtered):") $ymin-$ymax (${#files_climo[@]} files)"
      $exists && ! $overwrite && echo 'Skipping (output exists)...' && continue
      $wget && echo 'Skipping (wget only)...' && continue
      $dryrun && echo 'Skipping (dry run)...' && continue
      [ ${#files_climo[@]} -eq 0 ] && echo 'Skipping (no files)...' && continue

      # Take zonal and time averages
      # NOTE: Optionally get chunks of N-yearly averages for calculating
      # feedbacks and Gregory plots or just total climatological average.
      # NOTE: Optionally use final years or intial years. Generally want initial
      # years of control run to avoid model drift and final years of abrupt run.
      # Note that runs published in CMIP are already spun up.
      # cmds=${files_climo[*]/#/-zonmean -selname,$var $input/}
      cmds=${files_climo[*]/#/-selname,$var $input/}
      [ ${#files_climo[@]} -gt 1 ] && cmds="-mergetime $cmds"
      cdo -s -O $cmds $tmp || { echo 'Warning: Merge failed.' && continue; }
      nt=$(cdo -s ntime $tmp)  # file time steps
      nty=$((ny * 12))  # desired total time steps
      ntc=$((nchunks * 12))  # desired chunk time steps
      if $chunks; then
        unset cmd
        for ti in $(seq 1 ntc $((nty - ntc))); do
          if [ $ti -gt $nt ]; then
            echo "Warning: Requested $nty time steps but file only has $nt time steps."
            break
          elif [ $((ti + ntc - 1)) -gt $nt ]; then
            echo "Warning: Averaging chunk size $ntc incompatible with $nt time steps."
            break
          elif $endyears; then
            t1=$((nt - ti - ntc))
            t2=$((nt - ti + 1))  # endpoint inclusive starts at one
          else
            t1=$((ti))
            t2=$((ti + ntc - 1))  # endpoint inclusive
          fi
          [ $t1 -lt 1 ] && t1=1
          [ $t2 -gt $nt ] && t2=$nt
          echo "$(printf "%-20s" "$var (timesteps):") $t1-$t2 ($(((t2 - t1 + 1) / 12)) years)"
          cmd="$cmd -mergetime -seltimestep,$t1/$t2 $tmp"
        done
        cdo -s -O -ensmean $cmd $out
      else
        unset cmd
        if [ $nty -gt $nt ]; then
          echo "Warning: Requested $nty time steps but file only has $nt time steps."
        elif $endyears; then  # ifnal years
          t1=$((nt - nty + 1))
          t2=$nt
        else  # initial years
          t1=1
          t2=$nty
        fi
        echo "$(printf "%-20s" "$var (timesteps):") $t1-$t2 ($(((t2 - t1 + 1) / 12)) years)"
        cmd="-seltimestep,$t1,$t2"
        cdo -s -O -ymonmean $cmd $tmp $out
      fi
      echo "Output: ${output##*/}/${out##*/}"
    done
  done
}

# Global variables
# nchunks=10  # for response time series using blockwise averages
# endyears=false  # whether to use end or start years

# Temperature data
# projects=(cmip6)
projects=(cmip5 cmip6)
experiments=(piControl)
tables=(Amon Emon)
# vars=(gs psl ua va)  # circulation variables
# vars=(ta hur hus cl clw cli clt clwvp clwvi clivi cct)  # thermodynamics variables
vars=(ta gs ua va hur hus cl clw cli)  # multi-level bariables
# vars=(psl clt cct clwvp clwvi clivi)  # single-level variables
chunks=false
nchunks=10
nclimate=50  # first 50 years
nresponse=50  # last 50 years
endyears=false  # whether to use end or start years

# Transport data
# projects=(cmip6)
# experiments=(piControl abrupt4xCO2)
# tables=(Emon)
# vars=(intuadse intvadse intuaw intvaw)
# chunks=false
# nchunks=10
# nclimate=100  # last 100 years
# nresponse=100  # last 100 years
# endyears=true  # whether to use end or start years

# Call main function
# TODO: Limit files to number of timesteps
for project in ${projects[@]}; do
  for experiment in ${experiments[@]}; do
    for table in ${tables[@]}; do
      string=$project-$experiment-$table
      $dryrun && log=logs/average_${string}.log || log=logs/average-dryrun_${string}.log
      [ -r $log ] && rm $log
      # driver $project $experiment $table ta | tee $log
      driver $project $experiment $table ${vars[@]} | tee $log
    done
  done
done
