#!/usr/bin/env bash
#------------------------------------------------------------------------------#
# Prints the available times for different CMIP5 models
#------------------------------------------------------------------------------#
# Intro
shopt -s nullglob
nyears=30
root=/mdata2/ldavis/cmip5
exps=(abrupt4xCO2 piControl)
freqs=(mon)
vars=(ta ua va) # try ua and va too
# Awk script
# See: https://unix.stackexchange.com/a/13779/112647
min='
  BEGIN { c = 0; }
  $1 ~ /^[0-9]*(\.[0-9]*)?$/ { a[c++] = $1; }
  END { print a[0]; }
'
max='
  BEGIN { c = 0; }
  $1 ~ /^[0-9]*(\.[0-9]*)?$/ { a[c++] = $1; }
  END { print a[c-1]; }
'
# Loops
echo
for var in "${vars[@]}"; do
  for freq in "${freqs[@]}"; do
    # First get models for which have both control and global warming
    for exp in "${exps[@]}"; do
      # Test
      dir=$root/$exp-$freq
      echo "Dir: $dir, Param: $var"
      ! [ -d $dir ] && echo "Error: Directory $dir not found." && continue
      # Get models
      models=($(echo $dir/$var*.nc | tr ' ' $'\n' | cut -d'_' -f3 | sort | uniq))
      # Get dates
      unset message
      for model in "${models[@]}"; do
        # Message with information
        # WARNING: Can get 'cfMon' files along with 'Amon' files, and former
        # has different number of variables! Filter them out. Add to this for
        # other timesteps.
        # udates=($(echo "${dates[@]}" | tr ' ' $'\n' | sort | uniq))
        echo "Model: $model"
        [ "$freq" == "mon" ] && ffreq="Amon" || ffreq="$freq" # also downloads "cfMon"
        files=($(echo $dir/$var*$ffreq*$model*.nc))
        dates=($(echo "${files[@]}" | tr ' ' $'\n' | cut -d'_' -f6 | sed 's/\.nc//g'))
        imin=$(echo "${dates[@]%??-*}" | sort | uniq | tr ' ' $'\n' | awk "$min" | bc -l)
        imax=$(echo "${dates[@]%??}" | sort | uniq | tr ' ' $'\n' | cut -d'-' -f2 | awk "$max" | bc -l)
        parent=$(ncdump -h ${files[0]} | grep 'parent_experiment_id' | cut -d'=' -f2 | tr -dc '[a-zA-Z]')
        message+="$model: $imin-$imax (${#files[@]} files, $((imax-imin+1)) years), parent $parent"

        # Verify files are readable
        # for file in "${files[@]}"; do
        #   ncdump -h "$file" &>/dev/null
        #   if [ $? -ne 0 ]; then
        #     echo "Warning: File $file is corrupt."
        #     mv "$file" $root/corrupt
        #   fi
        # done

        # Test if climate already gotten
        tmp="$root/climate/tmp.nc"
        out="$root/climate/${var}_${exp}-${model}-${freq}.nc"
        ! [ -d $root/climate ] && mkdir $root/climate
        get=true
        if [ -r $out ]; then
          dump="$(ncdump -h $out)" # double quotes to keep newlines
          [ $? -eq 0 ] && [ "$(echo "$dump" | grep 'UNLIMITED' | tr -dc '[0-9]')" -gt 0 ] && get=false
        fi
        # Calculate climatological means
        # NOTE: Conventions for years, and spinup times, are all different!
        # We just take the final 30 years for each experiment.
        # if $get; then
        if true; then
          unset climo
          # Test if maximum year exceeds (simulation end year minus nyears)
          for i in $(seq 1 "${#files[@]}"); do
            file="${files[i-1]}"
            date="${dates[i-1]%??}"
            date="$(echo ${date#*-} | bc -l)"
            [ $date -ge $((imax - nyears + 1)) ] && climo+=("$file")
          done
          message+=", climo ${climo[@]##*_}"
          # Calculate climatology
          echo "Getting climate (${out##*/})..."
          commands="${climo[@]/#/-zonmean -selname,$var }"
          [ ${#climo[@]} -gt 1 ] && commands="-mergetime $commands"
          echo "Warning: Using date ranges ${climo[@]##*_}"
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
        fi
        message+="\n"
      done
      echo "Summary:"
      printf "$message" | column -t -s ':' -o ''
      echo
    done
  done
done
