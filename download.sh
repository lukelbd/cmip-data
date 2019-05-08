#!/usr/bin/env bash
#------------------------------------------------------------------------------#
# Call various wget filtered scripts
#------------------------------------------------------------------------------#
# Vars
root=/mdata2/ldavis/cmip5
freqs=(mon)
# exps=(abrupt4xCO2 piControl)
exps=(piControl)
cwd=$(pwd)
# Loop
for freq in ${freqs[@]}; do
  for exp in ${exps[@]}; do
    # Make directory
    file="$cwd/wgets/wget-${exp}-${freq}-filtered.sh"
    if ! [ -r "$file" ]; then
      echo "Warning: File $file not found."
      continue
    fi
    dir="$root/${exp}-${freq}"
    [ -d "$dir" ] || mkdir "$dir"
    # Run script
    cp $file $dir
    cd $dir
    chmod 755 ${file##*/}
    # ./${file##*/} -H -o https://esgf-node.llnl.gov/esgf-idp/openid/lukelbd
    ./${file##*/} -H
  done
done
