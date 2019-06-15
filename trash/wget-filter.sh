#!/usr/bin/env bash
#------------------------------------------------------------------------------#
# Generate "filtered" wget files, so we only have models with both control run
# and forcing run. Also filter to desired variables.
#------------------------------------------------------------------------------#
# Initial stuff
shopt -s nullglob
eof='EOF--dataset.file.url.chksum_type.chksum'
# For climatology
# exps=(abrupt4xCO2 piControl)
# tables=(Amon)
# For daily data correlation thing
exps=(piControl)
tables=(day)

# Function that gets NetCDF file names
models() {
  sed -n "/$eof/,/$eof/p" "$1" | grep -v "$eof" | cut -d'_' -f3 | sort | uniq
}
# Different CMOR tables
[ -r wget-filter.log ] && rm wget-filter.log
{
for table in ${tables[@]}; do
  # Filter wget scripts to select models
  # 1) Filter to models for which *all* variables are available
  # 2) Add these filtered models to bash array of models for each experiment
  unset models_all models_grouped models_missing_var models_missing_exp models_intersection
  for exp in ${exps[@]}; do
    # Get list of all models, and list of grouped models
    unset ivars imodels_all imodels_grouped imodels_missing imodels_intersection
    wgets=(wgets/wget-${exp}-${table}-*.sh)
    [ ${#wgets[@]} -eq 0 ] && echo "Warning: No $exp $table wget scripts found. Run wget.sh." && exit 1
    for file in ${wgets[@]}; do
      ivar=${file%.sh}
      ivar=${ivar##*-}
      ivars+=($ivar)
      imodels_grouped+=("$(models "$file" | xargs)")
    done
    imodels_all=($(echo ${imodels_grouped[@]} | tr ' ' $'\n' | sort | uniq))
    # Iterate through each model, make sure it appears in every group
    for model in ${imodels_all[@]}; do
      inall=true
      for group in "${imodels_grouped[@]}"; do
        [[ " $group " =~ " $model " ]] || inall=false
      done
      $inall && imodels_intersection+=($model) || imodels_missing+=($model) models_missing_var+=($model)
    done
    # Save *these* models for each experiment
    models_grouped+=("$(echo "${imodels_intersection[@]}")") # save as single string
    echo
    echo "Experiment: $exp, Table: $table"
    echo "All variables: ${ivars[@]}"
    echo "Good models: ${imodels_intersection[@]}"
    for model in ${imodels_missing[@]}; do
      unset missing
      for file in ${wgets[@]}; do
        ivar=${file%.sh}
        ivar=${ivar##*-}
        cat $file | grep "_${model}_" &>/dev/null
        [ $? -ne 0 ] && missing+="$ivar "
      done
      echo "$model missing: $missing"
    done
  done
  models_all=($(echo ${models_grouped[@]} | tr ' ' $'\n' | sort | uniq))

  # 3) Filter to models available in control *and* forcing run
  for model in ${models_all[@]}; do
    # List of experiments for which the model is found
    inall=true
    for group in "${models_grouped[@]}"; do
      [[ " $group " =~ " $model " ]] || inall=false
    done
    $inall && models_intersection+=($model) || models_missing_exp+=($model)
  done
  # Message
  echo
  echo "Table: $table"
  echo "All variables and experiments: ${models_intersection[@]}"
  for model in "${models_missing_exp[@]}"; do
    unset missing
    for exp in ${exps[@]}; do
      wgets=(wgets/wget-${exp}-${table}-*.sh)
      for file in "${wgets[@]}"; do
        cat $file | grep "_${model}_" &>/dev/null
        [ $? -ne 0 ] && missing+="$exp "
      done
    done
    echo "$model missing: $(echo $missing | tr ' ' $'\n' | sort | uniq | xargs)"
  done

  # Finally apply filtered wget script
  unset message_extra
  omit=$(echo " ${models_missing_var[@]} ${models_missing_exp[@]} " | \
         sed 's/ /_ _/g;s/^_ /(/;s/ _$/)/;s/ \?__ \?//g' | tr ' ' '|')
  echo "Omit: $omit"
  for exp in ${exps[@]}; do
    wgets=(wgets/wget-${exp}-${table}-*.sh)
    for file in "${wgets[@]}"; do
      # Make filtered script
      filtered=${file/raw/filtered}
      cat "$file" | grep -v -E "$omit" >"$filtered" # or sed '/regex/d'
      # echo "File ${filtered##*/}: $(models "$filtered" | xargs)" # verify!
    done
  done
done
} | tee -a wget-filter.log
