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
# Settings
shopt -s nullglob
dryrun=true
nclimate=50  # first 50 years
nresponse=100  # first 100 years
nchunks=10 # for response blockwise averages

# Loop
exps=(piControl abrupt4xCO2)
tables=(Amon cfDay day)  # for 'cfDay' and 'day' just show tables (see below)

# Folders
root=/mdata5/ldavis/cmip5
data=$HOME/data/cmip5
[ -d $data ] || mkdir $data

# Awk scripts
# See: https://unix.stackexchange.com/a/13779/112647
# WARNING: Dates must be sorted!
min='$1 ~ /^[0-9]*(\.[0-9]*)?$/ { a[c++] = $1 }; END { print a[0] }'
max='$1 ~ /^[0-9]*(\.[0-9]*)?$/ { a[c++] = $1 }; END { print a[c-1] }'

# Main function
date1() {
  date=${1%-*}
  echo ${date::4}
}
date2() {
  date=${1#*-}
  echo ${date::4}
}
driver() {
  for table in "${tables[@]}"; do
  for exp in "${exps[@]}"; do
    # Find available models and variables
    dir=$root/$exp-$table
    # wget=true
    [[ "$table" =~ "mon" ]] && wget=false || wget=true # show climate summaries from wget scripts?
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
    echo
    echo "Table $table, exp $exp: ${#models[@]} models found"
    for model in ${models[@]}; do
      # [ "$model" != "CCSM4" ] && continue # for testing
      echo
      echo "Dir: ${dir##*/}, Model: $model"
      for var in "${vars[@]}"; do
        # List of files
        files=($(echo "$allfiles" | grep "${var}_${table}_${model}_"))
        [ ${#files[@]} -eq 0 ] && echo "$(printf "%-8s" "$var:") not found" && continue
        # Date range
        dates=($(echo "${files[@]}" | tr ' ' $'\n' | cut -d'_' -f6 | sed 's/\.nc//g' | sort))
        ymin=$(echo "${dates[@]%-*}" | tr ' ' $'\n' | cut -c-4 | awk "$min" | bc -l)
        ymax=$(echo "${dates[@]#*-}" | tr ' ' $'\n' | cut -c-4 | awk "$max" | bc -l)
        # Parent experiment
        parent=$(ncdump -h ${files[0]} 2>/dev/null | grep 'parent_experiment_id' | cut -d'=' -f2 | tr -dc a-zA-Z)
        [ -z "$parent" ] && parent="NA"
        # Print message
        echo "$(printf "%-8s" "$var:") $ymin-$ymax (${#files[@]} files, $((ymax-ymin+1)) years), parent $parent"
        $dryrun && continue
        # Test if climate already gotten
        tmp=$data/tmp.nc
        out=$data/${var}_${exp}-${model}-${table}.nc
        exists=false
        if [ -r $out ]; then
          header=$(ncdump -h $out 2>/dev/null) # double quotes to keep newlines
          [ $? -eq 0 ] && [ "$(echo "$header" | grep 'UNLIMITED' | tr -dc 0-9)" -gt 0 ] && exists=true
        fi
        $exists && continue

        # Calculate climatological means
        # TODO: Allow multiple times in input files, then let plotting functions
        # select seasons or time periods!
        # NOTE: Conventions for years, and spinup times, are all different!
        # We just take the final 30 years for each experiment.
        # Test if maximum year exceeds (simulation end year minus nclimate)
        unset climo
        [[ $exp =~ "CO2" ]] && ny=$nresponse chunks=true || ny=$nclimate chunks=false
        for i in $(seq 1 "${#files[@]}"); do
          # Final n years
          # yr=$(date2 ${dates[i-1]}) # end date!
          # [ $yr -ge $((ymax - ny + 1)) ] && climo+=("${files[i-1]}")
          # First n years
          yr=$(date1 ${dates[i-1]}) # start date!
          [ $yr -le $((ymin + ny - 1)) ] && climo+=("${files[i-1]}")
        done
        # First just merge files
        echo "Getting summary ${out##*/} with files ${climo[*]##*_}..."
        cmds=${climo[*]/#/-zonmean -selname,$var }
        [ ${#climo[@]} -gt 1 ] && cmds="-mergetime $cmds"
        cdo -s -O $cmds $tmp || { echo "Error: Merge failed."; exit 1; }

        # Chunks
        # TODO: No averaging of response at all? Just lob off a standard
        # number of years of response and do all processing later on?
        unset cmd
        nty=$((ny * 12))
        nt=$(cdo -s ntime $tmp)
        if $chunks; then
          for iy in $(seq 0 nchunks $((nresponse - nchunks))); do
            ti=$((iy * 12 + 1))
            tf=$(((iy + nchunks) * 12))
            cmd="$cmd -ymonmean -seltimestep,$ti/$tf $tmp"
          done
          cdo -s -O -mergetime $cmd $out
        # Climatological averages
        else
          [ $nt -ge $nty ] && cmd="-seltimestep,1/$nty" # "$((nt - 12 * ny + 1))/$nt"  # final years
          [ $nt -ge $nty ] && cmd="-seltimestep,$((nt - 12 * ny + 1))/$nt"  # final years
          cdo -s -O -ymonmean $cmd $tmp $out
        fi
      done
    done
  done
  done
}

# Call big function
if $dryrun; then
  [ -r make_climatology.log ] && rm make_climatology.log
  driver | tee make_climatology.log
else
  driver
fi
