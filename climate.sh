#!/usr/bin/env bash
#------------------------------------------------------------------------------#
# This script does two things:
# 1. Prints available years as function of model for each variable; iterates
#    through files on disk or optionally the files inside a wget script
# 2. If the former, also computes climatological averages from 'nyears' final
#    years of the CMIP run
# WARNING: Tempting to combine all variables into one file, but that adds an
# extra unnecessary step when you can just make your load script merge the
# DataArrays into a Dataset, and gets confusing when want to add new variables.
#------------------------------------------------------------------------------#
# Settings
shopt -s nullglob
dryrun=true
nyears=30
# Folders
root=/mdata2/ldavis/cmip5
data=$HOME/data/cmip5
[ -d $data ] || mkdir $data
# Loop
exps=(piControl abrupt4xCO2)
tables=(Amon cfDay)
# Awk scripts
# See: https://unix.stackexchange.com/a/13779/112647
min='$1 ~ /^[0-9]*(\.[0-9]*)?$/ { a[c++] = $1 }; END { print a[0] }'
max='$1 ~ /^[0-9]*(\.[0-9]*)?$/ { a[c++] = $1 }; END { print a[c-1] }'

# Main function
driver() {
  for table in "${tables[@]}"; do
  for exp in "${exps[@]}"; do
    # Find available models and variables
    dir=$root/$exp-$table
    [ "$table" == "cfDay" ] && wget=true || wget=false # show climate summaries from wget scripts?
    if $wget; then
      # Search files listed in wget script!
      eof='EOF--dataset.file.url.chksum_type.chksum'
      wget="wgets/${exp}-${table}.sh"
      ! [ -r $wget ] && echo && echo "Warning: File $wget not found." && continue
      allfiles="$(cat $wget | sed -n "/$eof/,/$eof/p" | grep -v "$eof" | cut -d"'" -f2)"
    else
      # Search actual files on disk!
      ! [ -d $dir ] && echo && echo "Warning: Directory $dir not found." && continue
      allfiles="$(find $dir -name "*.nc" -printf "%f\n")"
    fi
    # Iterate through *models* first, then variables
    vars=($(echo "$allfiles" | cut -d'_' -f1 | sort | uniq)) # quotes around files important! contains newlines!
    models=($(echo "$allfiles" | cut -d'_' -f3 | sort | uniq))
    for model in ${models[@]}; do
      echo
      echo "Dir: ${dir##*/}, Model: $model"
      for var in "${vars[@]}"; do
        # List of files
        files=($(echo "$allfiles" | grep "${var}_${table}_${model}_"))
        [ ${#files[@]} -eq 0 ] && echo "$(printf "%-8s" "$var:") not found" && continue
        # Date range
        dates=($(echo "${files[@]}" | tr ' ' $'\n' | cut -d'_' -f6 | sed 's/\.nc//g'))
        imin=$(echo "${dates[@]%-*}" | tr ' ' $'\n' | cut -c-4 | awk "$min" | bc -l)
        imax=$(echo "${dates[@]#*-}" | tr ' ' $'\n' | cut -c-4 | awk "$max" | bc -l)
        # Parent experiment
        parent=$(ncdump -h ${files[0]} 2>/dev/null | grep 'parent_experiment_id' | cut -d'=' -f2 | tr -dc '[a-zA-Z]')
        [ -z "$parent" ] && parent="NA"
        # Print message
        echo "$(printf "%-8s" "$var:") $imin-$imax (${#files[@]} files, $((imax-imin+1)) years), parent $parent"
        $dryrun && continue
        # Test if climate already gotten
        tmp="$data/tmp.nc"
        out="$data/${var}_${exp}-${model}-${table}.nc"
        exists=false
        if [ -r $out ]; then
          header="$(ncdump -h $out 2>/dev/null)" # double quotes to keep newlines
          [ $? -eq 0 ] && [ "$(echo "$header" | grep 'UNLIMITED' | tr -dc '[0-9]')" -gt 0 ] && exists=true
        fi
        $exists && continue

        # Calculate climatological means
        # NOTE: Conventions for years, and spinup times, are all different!
        # We just take the final 30 years for each experiment.
        # Test if maximum year exceeds (simulation end year minus nyears)
        unset climo
        for i in $(seq 1 "${#files[@]}"); do
          file="${files[i-1]}"
          date="${dates[i-1]#*-}" # end date!
          date="$(echo ${date::4} | bc -l)"
          [ $date -ge $((imax - nyears + 1)) ] && climo+=("$file")
        done
        echo "Getting summary ${out##*/} with files ${climo[@]##*_}..."
        commands="${climo[@]/#/-zonmean -selname,$var }"
        [ ${#climo[@]} -gt 1 ] && commands="-mergetime $commands"
        cdo -s -O $commands $tmp
        [ $? -ne 0 ] && echo "Error: Merge failed." && exit 1

        # Make sure to only select final 30 years! These are
        # monthly means so will be 12*30 last timesteps.
        ntime=$(cdo -s ntime $tmp)
        if [ $ntime -lt $((nyears*12)) ]; then
          echo "Warning: Only $((ntime/12)) years available but $nyears requested."
          cdo -s -O -ymonmean $tmp $out
        else
          range="$((ntime-12*nyears+1))/$ntime"
          echo "Warning: Using timesteps ${range}."
          cdo -s -O -ymonmean -seltimestep,$range $tmp $out
        fi
      done
    done
  done
  done
}

# Call big function
if $dryrun; then
  [ -r climate.log ] && rm climate.log
  driver | tee climate.log
else
  driver
fi
