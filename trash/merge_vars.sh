#!/usr/bin/env bash
#------------------------------------------------------------------------------#
# Merge variables from climate files into one file
# WARNING: No longer do this! Just do it with xarray! Really not much point!
#------------------------------------------------------------------------------#
# NOTE: For now we keep separate variable files and *combined* climate files
shopt -s nullglob
root=/mdata2/ldavis/cmip5
bases=($(echo $root/climate/*.nc | tr ' ' $'\n' | sed 's/^.\+_//g' | sort | uniq))
# Iterate
for base in "${bases[@]}"; do
  files=($root/climate/*_$base) # don't match the files named exactly "base"!
  if [ ${#files[@]} -gt 0 ]; then
    echo "Merging files: ${files[@]##*/}"
    cdo -s -O -merge "${files[@]}" "$root/climate/$base"
  fi
done
