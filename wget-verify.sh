#!/usr/bin/env bash
#------------------------------------------------------------------------------#
# Verifies that filtered wget scripts indeed have the same number of
# models as each other.
#------------------------------------------------------------------------------#
shopt -s nullglob
root=/mdata2/ldavis/cmip5
freqs=(6hr day mon)
echo
for freq in ${freqs[@]}; do
  echo "Frequency: $freq"
  exps=(abrupt4xCO2 piControl)
  for exp in ${exps[@]}; do
    # List of filtered model data we *should have* downloaded
    wget=wgets/wget-${exp}-${freq}-filtered.sh
    models=$(cat $wget | grep '.*\.nc' | cut -d'_' -f3 | sort | uniq | xargs)
    echo "$exp: $models"
    # Check for extra models downloaded due to bug in wget-filter.sh
    files=($root/${exp}-${freq}/*.nc)
    regex=$(echo $models | tr ' ' '|')
    for file in "${files[@]}"; do
      if ! [[ "$file" =~ $regex ]]; then
        echo "Extra file: $file"
        # rm "$file"
      fi
    done
  done
  echo
done
