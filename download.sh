#!/usr/bin/env bash
#------------------------------------------------------------------------------#
# Call various wget filtered scripts
#------------------------------------------------------------------------------#
# Vars
# exps=(abrupt4xCO2 piControl)
shopt -s nullglob
root=/mdata2/ldavis/cmip5
tables=(Amon)
exps=(piControl)
cwd=$(pwd)
# Loop
for table in ${tables[@]}; do
  for exp in ${exps[@]}; do
    # Make directory
    files=($cwd/wgets/wget-filtered-${exp}-${table}-*.sh)
    [ ${#files[@]} -eq 0 ] && echo "Warning: No files found." && continue
    dir="$root/${exp}-${table}"
    [ -d "$dir" ] || mkdir "$dir"
    for file in "${files[@]}"; do
      # Run script
      cd $cwd
      cp $file $dir
      cd $dir
      chmod 755 ${file##*/}
      ./${file##*/} -H -o https://esgf-node.llnl.gov/esgf-idp/openid/lukelbd
      # ./${file##*/} -H
    done
  done
done
