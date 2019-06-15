#!/usr/bin/env bash
#------------------------------------------------------------------------------#
# Verify that files are valid!
#------------------------------------------------------------------------------#
# Loop through every file in directory
# Takes a minute or two to run but surprisingly fast
shopt -s nullglob
root=/mdata2/ldavis/cmip5
for subdir in $root/*; do
  echo "Directory: ${subdir##*/}"
  for file in $subdir/*.nc; do
    header="$(ncdump -h $file 2>/dev/null)" # double quotes to keep newlines
    if [ $? -ne 0 ]; then
      echo "File ${file##*/} corrupt!"
    elif [ "$(echo "$header" | grep 'UNLIMITED' | tr -dc '[0-9]')" -eq 0 ]; then
      echo "File ${file##*/} empty!"
    fi
  done
done
exit 1
