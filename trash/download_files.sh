#!/usr/bin/env bash
#------------------------------------------------------------------------------#
# Call various "summary" wget scripts
#------------------------------------------------------------------------------#
# exps=(abrupt4xCO2 piControl)
shopt -s nullglob
root=$HOME/ldavis/data/cmip
# root=/mdata2/ldavis/cmip5
exps=(piControl)
# exps=(abrupt4xCO2 piControl)
tables=(Amon)
projs=(cmip6 cmip5)
cwd=${0%/*}

# Loop over files
for proj in ${projs[@]}; do
  for table in ${tables[@]}; do
    for exp in ${exps[@]}; do
      file=$cwd/wgets/wget_${proj}-${exp}-${table}.sh
      dir=$root/${proj}-${exp}-${table}
      [ -d "$dir" ] || mkdir "$dir" || { echo "Failed to create directory."; exit 1; }
      cp "$file" "$dir"
      cd "$dir" || { echo "Failed to move to directory."; exit 1; }
      chmod 755 ${file##*/}
      ./${file##*/} -H -o https://esgf-node.llnl.gov/esgf-idp/openid/lukelbd
      # ./${file##*/} -H
    done
  done
done
