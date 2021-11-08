Emergent constraints
--------------------

`wget_download.sh` downloads ESGF wget scripts for retrieving CMIP5 data.
`wget_filter.py` filters the ESGF wget scripts to model data for which both "forced"
runs and "control" runs are available. It also filters given model experiments to
download all variables within the same time period -- e.g. if we have 500 years of 2D
radiative fluxes but only 100 years of air temperature we just download the first 100
years of fluxes. `wget-call.sh` downloads data using the filtered wget scripts and
organizes the NetCDF files into appropriate subfolders. `climate.sh` generates
climatologies from the downloaded files. It can also print tables of the available time
periods for each variable and each model run in dry-run mode, by inspecting downloaded
files or wget scripts.
