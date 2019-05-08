# Summary
The `wget.sh` script is used to download ESGF wget scripts that can be used
to download CMIP5 data. The `wget-filter.sh`
is used to generated filtered scripts that only download model data
for which both "forced" runs (any experiment name with the substring `rcp` or `CO2` inside) and "control" runs (all other experiment names) are available. The `download.sh` script downloads data using these filtered wget scripts and organizes the
NetCDF files into subfolders.

The `climate.sh` script is used to generate 30-year climatologies for the trailing 30 years of the downloaded simulations, and consolidate the resulting files containing individual variables into a single climatology file. The `clean.sh` script is used to delete accidentally downloaded NetCDF files.

