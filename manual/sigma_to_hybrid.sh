sigma_to_hybrid() {
  for file in "$@"; do
    ncap2 -O -s '
      b[$lev] = lev;
      ap[$lev] = (1 - lev) * ptop;
      b_bnds[$lev, $bnds] = lev_bnds;
      ap_bnds[$lev, $bnds] = (1 - lev_bnds) * ptop;
    ' "$file" "$file" \
    && ncatted -O \
      -a long_name,lev,o,c,"atmospheric model level" \
      -a standard_name,lev,o,c,"atmosphere_hybrid_sigma_pressure_coordinate" \
      -a formula,lev,o,c,"p = ap + b*ps" \
      -a formula_terms,lev,o,c,"ap: ap b: b ps: ps" \
      -a long_name,lev_bnds,o,c,"atmospheric model level bounds" \
      -a standard_name,lev_bnds,o,c,"atmosphere_hybrid_sigma_pressure_coordinate" \
      -a formula,lev_bnds,o,c,"p = ap + b*ps" \
      -a formula_terms,lev_bnds,o,c,"ap: ap_bnds b: b_bnds ps: ps" \
      -a ,ap,d,, -a ,b,d,, -a ,ap_bnds,d,, -a ,b_bnds,d,, \
      -a long_name,ap,o,c,"vertical coordinate formula term: ap(k)" \
      -a long_name,b,o,c,"vertical coordinate formula term: b(k)" \
      -a long_name,ap_bnds,o,c,"vertical coordinate formula term: ap(k+1/2)" \
      -a long_name,b_bnds,o,c,"vertical coordinate formula term: b(k+1/2)" \
    "$file" "$file" \
    && ncks -C -O -x -v ptop "$file" "$file" \
    || echo "Warning: Failed to translate file '$file'."
  done
}
