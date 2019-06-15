#!/usr/bin/env bash
#------------------------------------------------------------------------------#
# Call various "summary" wget scripts
#------------------------------------------------------------------------------#
# Vars
# exps=(abrupt4xCO2 piControl)
shopt -s nullglob
root=/mdata2/ldavis/cmip5
exps=(abrupt4xCO2 piControl)
tables=(Amon)
cwd=$(pwd)
# Loop
for table in ${tables[@]}; do
  for exp in ${exps[@]}; do
    # Make directory
    file=$cwd/wgets/${exp}-${table}.sh
    dir="$root/${exp}-${table}"
    [ -d "$dir" ] || mkdir "$dir"
    # Run script
    cd $cwd
    cp $file $dir
    cd $dir
    chmod 755 ${file##*/}
    ./${file##*/} -H -o https://esgf-node.llnl.gov/esgf-idp/openid/lukelbd
    # ./${file##*/} -H
  done
done
