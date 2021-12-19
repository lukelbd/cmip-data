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
dryrun=false
nclimate=50  # first 50 years
nresponse=100  # first 100 years
nchunks=10 # for response blockwise averages
data=$HOME/data
[[ "$OSTYPE" =~ darwin* ]] && root=$data || root=/mdata5/ldavis

# Helper functions
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
  [ $# -ge 3 ] || { echo && echo "Error: At least 3 arguments required." && return 1; }
  project=$1
  experiment=$2
  table=$3
  vars=("${@:4}")  # TODO: also permit restricting models?
  string=${project}-${experiment}-${table}
  [[ "$table" =~ "mon" ]] && wget=false || wget=true  # show climate summaries from wget scripts?
  if $wget; then  # search files in wget script
    eof=EOF--dataset.file.url.chksum_type.chksum
    input=$root/wgets/wget_${string}.sh
    [ -r "$input" ] || { echo && echo "Error: File $input not found." && return 1; }
    allfiles=$(cat $input | sed -n "/$eof/,/$eof/p" | grep -v "$eof" | cut -d"'" -f2)
  else  # search files on disk
    input=${root}/${string}
    [ -d "$input" ] || { echo && echo "Error: Path $input not found." && return 1; }
    output=${data}/${string}-avg
    [ -r "$output" ] || mkdir "$output" || { echo && echo "Error: Failed to make $output." && return 1; }
    allfiles=$(find $input -name "*.nc" -printf "%f\n")
  fi

  # Iterate through models then variables
  # NOTE: The quotes around files are important! contains newlines!
  models=($(echo "$allfiles" | cut -d'_' -f3 | sort | uniq))
  [ ${#vars[@]} -eq 0 ] && vars=($(echo "$allfiles" | cut -d'_' -f1 | sort | uniq))
  echo
  echo "Table $table, experiment $experiment: ${#models[@]} models found"
  for model in ${models[@]}; do
    # [ "$model" != "CCSM4" ] && continue  # for testing
    echo
    echo "Input: ${input##*/}, Model: $model"
    for var in "${vars[@]}"; do
      # List files and get file info
      # NOTE: Some files have extra field but date is always last so use rev.
      # TODO: Add year suffix? Then need to auto-remove files with different range.
      tmp=${output}/tmp.nc
      out=${output}/${var}_${table}_${model}_${experiment}_${project}.nc
      exists=false
      [ -r $out ] && [ "$(ncdump -h "$out" 2>/dev/null | grep 'UNLIMITED' | tr -dc 0-9)" -gt 0 ] && exists=true
      files=($(echo "$allfiles" | grep "${var}_${table}_${model}_"))
      dates=($(echo "${files[@]}" | tr ' ' $'\n' | rev | cut -d'_' -f1 | rev | sed 's/\.nc//g' | sort))
      if [ ${#files[@]} -eq 0 ]; then
        echo "$(printf "%-8s" "$var:") not found"
        continue
      else
        ymin=$(minyear "${dates[@]%-*}")
        ymax=$(maxyear "${dates[@]#*-}")
        parent=$(ncdump -h ${files[0]} 2>/dev/null | grep 'parent_experiment_id' | cut -d'=' -f2 | tr -dc a-zA-Z)
        echo "$(printf "%-8s" "$var:") $ymin-$ymax (${#files[@]} files, parent ${parent:-NA}"
      fi

      # Select files for averaging
      # NOTE: Use at least this many years from start of simulation (models are
      # already spun up, and this could help remove drift during control runs).
      unset files_climo dates_climo
      [[ $experiment =~ CO2 ]] && ny=$nresponse chunks=true || ny=$nclimate chunks=false
      for i in $(seq 1 "${#files[@]}"); do
        file=${files[i - 1]}
        date=${dates[i - 1]}
        if [ "$(year1 $date)" -le $((ymin + ny - 1)) ]; then
          files_climo+=("$file")
          dates_climo+=("$date")
        fi
      done
      ymin=$(minyear "${dates_climo[@]%-*}")
      ymax=$(maxyear "${dates_climo[@]#*-}")
      echo "$(printf "%-8s" "$var:") $ymin-$ymax (${#files_climo[@]} files, parent ${parent:-NA})"
      echo "Output: ${output##*/}/${out##*/}"
      $wget && echo 'Skipping (wget only)...' && continue
      $exists && echo 'Skipping (file exists)...' && continue
      $dryrun && echo 'Skipping (dry run)...' && continue

      # Take zonal and time averages
      # NOTE: Use the initial years from the files rather than trailing years.
      unset cmd
      cmds=${files_climo[*]/#/-zonmean -selname,$var $input/}
      [ ${#files_climo[@]} -gt 1 ] && cmds="-mergetime $cmds"
      cdo -s -O $cmds $tmp || { echo 'Warning: Merge failed.' && continue; }
      nt1=$(cdo -s ntime $tmp)  # file time steps
      nt2=$((ny * 12))  # final time steps
      if $chunks; then
        for iy in $(seq 0 nchunks $((ny - nchunks))); do
          ti=$((iy * 12 + 1))
          tf=$(((iy + nchunks) * 12))
          cmd="$cmd -ymonmean -seltimestep,$ti/$tf $tmp"
        done
        cdo -s -O -mergetime $cmd $out
      else
        [ $nt1 -ge $nt2 ] && cmd="-seltimestep,1/$nt2" # "$((nt1 - nt2 + 1))/$nt1"  # final years
        cdo -s -O -ymonmean $cmd $tmp $out
      fi
    done
  done
}

# Call main function
projects=(cmip5 cmip6)
# experiments=(piControl abrupt4xCO2)
experiments=(piControl)
# tables=(Amon cfDay day)  # for 'cfDay' and 'day' just show tables (see below)
tables=(Amon)
for project in ${projects[@]}; do
  for experiment in ${experiments[@]}; do
    for table in ${tables[@]}; do
      string=$project-$experiment-$table
      $dryrun && log=logs/average_${string}.log || log=logs/average-dryrun_${string}.log
      [ -r $log ] && rm $log
      driver $project $experiment $table ta | tee $log
    done
  done
done
