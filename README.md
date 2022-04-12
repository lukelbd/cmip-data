What's here?
------------

The `cmip_data.load` module contains functions for loading cmip data, radiative
kernels, and climate feedback and sensitivity tables. The `cmip_data.download`
module contains functions for downloading and averaging cmip data. The
`download_script` function creates ESGF wget scripts for retrieving CMIP5 data,
and the `filter_script` function consolidates these into a single scripts and filters to
within the desired time range -- e.g. if there are 500 years of data but we only need a
100 year climatology. The `cmip_data.process` module can then be used to create
monthly climatologies or annual time series from files downloaded via the wget
scripts, optionally with standardized horizontal and vertical coordinates.
