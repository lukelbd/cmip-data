#!/usr/bin/env bash
#------------------------------------------------------------------------------#
# Compare wget files
#------------------------------------------------------------------------------#
# Function that gets NetCDF file names
# NOTE: Will filter so that only have like models
# NOTE: Filter to only files with at least air temperature!
# WARNING: Get random unwanted variables for some reason! Even when filter
# request by variable names!
eof='EOF--dataset.file.url.chksum_type.chksum'
ignore="^(?!'(?!ta_|ua_|va_)|'.*\.nc4)" # only way to do this is with Perl regex
models() {
  sed -n "/$eof/,/$eof/p" "$1" | grep -v "$eof" | grep -P "$ignore" | cut -d'_' -f3 | sort | uniq
}

# Different time frequencies
[ -r wget-filter.log ] && rm wget-filter.log
freqs=(6hr day mon)
for freq in ${freqs[@]}; do
  # Loop through experiments
  unset models_all models_grouped message_all message_forcing message_control
  # exps=(rcp85 amip4xCO2 abrupt4xCO2 amip historical piControl historicalNat)
  # exps=(rcp85 amip4xCO2 abrupt4xCO2 piControl historicalNat)
  exps=(abrupt4xCO2 piControl)
  for exp in ${exps[@]}; do
    file="wgets/wget-${exp}-${freq}.sh"
    models=$(models "$file")
    models_all+="$models"$'\n'
    models_grouped+=("$models")
  done

  # Message
  # printf "$models_all" | sort | uniq
  for model in $(printf "$models_all" | sort | uniq); do
    unset explist
    for i in $(seq 1 ${#exps[@]}); do
      exp="${exps[i-1]}"
      models="${models_grouped[i-1]}"
      [[ "$models" =~ "$model" ]] && explist+="$exp,"
    done
    # Test if forcing *and* natural here
    forcing=$(echo "$explist" | tr ',' $'\n' | grep -E 'CO2|rcp' |  xargs)
    control=$(echo "$explist" | tr ',' $'\n' | grep -v -E 'CO2|rcp' |  xargs)
    if [ -n "$forcing" ] && [ -n "$control" ]; then
      message_all+="$model: ${explist%,}"$'\n'
    elif [ -n "$forcing" ]; then
      message_forcing+="$model, "
    else
      message_control+="$model, "
    fi
  done
  # Generate wget script with models *filtered*
  # to those for which we have both control and forced run
  omit=$(echo "${message_forcing%, }, ${message_control%, }" | sed 's/, /|/g')
  unset message_extra
  for exp in ${exps[@]}; do
    file="wgets/wget-${exp}-${freq}.sh"
    filtered="wgets/wget-${exp}-${freq}-filtered.sh"
    cat "$file" | grep -v -E "$omit" | grep -P "$ignore" >"$filtered" # or sed '/regex/d'
    [[ "$exp" =~ CO2|rcp ]] && cat="Forcing" || cat="Control"
    message_extra+="$cat: $(cat "$file" | grep -v -E "$omit" | grep -P "^(?!'(?!ta_|ua_|va_)|'.*\.nc4)" | grep '\.nc' | cut -d'_' -f3 | sort | uniq | xargs)\n"
    # message_extra+="$exp: "$omit"\n$(cat "$file" | grep -v -E "$omit" | grep -P "^('(?!ta_|ua_|va_)|'.*\.nc4)" | grep '\.nc' | cut -d' ' -f1)\n"
  done
  # Print to logfile
  message_all=$(printf "${message_all}" | sort | column -t | sed 's/,/, /g')
  printf "\nTime frequency: ${freq}\n${message_all}
Just forcing: ${message_forcing%, }
Just control: ${message_control%, }
$message_extra
" | tee -a wget-filter.log
done

