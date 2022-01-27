#!/usr/bin/env bash
#------------------------------------------------------------------------------#
# Verify that files are valid
#------------------------------------------------------------------------------#
# Loop through every file in directory
# Takes a minute or two to run but surprisingly fast
# NOTE: So far have not encountered usual issues with corrupt files or
# zero length time dimensions; just have dummy zero-byte files.
shopt -s nullglob
root=/mdata2/ldavis/cmip5
for subdir in $root/*; do
  echo "Directory: ${subdir##*/}"
  for file in $subdir/*.nc; do
    # Detect zero-byte files; only issue observed so far
    # [ "$(echo "$(ncdump -h $file)" | grep 'UNLIMITED' | tr -dc '[0-9]')" -eq 0 ]; then
    # bytes=$(wc -c < $file) # slow because reads whole file
    bytes=$(ls -nl $file | awk '{print $5}')  # faster
    if [ $bytes -eq 0 ]; then
      echo "Removing empty file '${file##*/}'"
      rm $file
    fi
  done
done
exit 1