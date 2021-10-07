Summary
-------

The `wget-download.sh` script is used to download ESGF wget scripts that can be used to
download CMIP5 data. The `wget-filter.py` module is used to generated filtered scripts
that only download model data for which both "forced" runs and "control" runs are
available. It also filters given model experiments to download all variables within the
same time period -- for example, if we have 500 years of 2-D radiative fluxes, but only
100 years of air temperature, we just download the first 100 years of fluxes. The
`wget-call.sh` script downloads data using these filtered wget scripts, and organizes
the NetCDF files into appropriate subfolders. The `climate.sh` script is used to
generate climatologies from the downloaded files. It can also be used to just generate
tables of the available time periods for each variable and each model run, using files
downloaded to the scratch folder or just be inspecting a wget script directly.
<!-- The `clean.sh` script is used to delete accidentally downloaded NetCDF files. -->
