#!/usr/bin/env python3
"""
Convenient wrappers for python APIs to download CMIP data.
"""
__all__ = [
    'cmip',
]


# CMIP constants. Results of get_facet_options() for SearchContext(project='CMIP5')
# and SearchContext(project='CMIP6') using https://esgf-node.llnl.gov/esg-search
# for the SearchConnection URL. Conventions changed between projects so e.g.
# 'experiment', 'ensemble', 'cmor_table', and 'time_frequency' in CMIP5 must be
# changed to 'experiment_id', 'variant_label', 'table_id', and 'frequency' in CMIP6.
# Note 'member_id' is equivalent to 'variant_label' if 'sub_experiment_id' is unset
# and for some reason 'variable' and 'variable_id' are kepts synonyms in CMIP5.
# URL https://esgf-node.llnl.gov/esg-search:     11900116 hits for CMIP6 (use this one!)
# URL https://esgf-data.dkrz.de/esg-search:      01009809 hits for CMIP6
# URL https://esgf-node.ipsl.upmc.fr/esg-search: 01452125 hits for CMIP6
CMIP5_FACETS = [
    'access', 'cera_acronym', 'cf_standard_name', 'cmor_table', 'data_node',
    'ensemble', 'experiment', 'experiment_family', 'forcing', 'format',
    'index_node', 'institute', 'model', 'product', 'realm', 'time_frequency',
    'variable', 'variable_long_name', 'version'
]
CMIP6_FACETS = [
    'access', 'activity_drs', 'activity_id', 'branch_method', 'creation_date',
    'cf_standard_name', 'data_node', 'data_specs_version', 'datetime_end',
    'experiment_id', 'experiment_title', 'frequency', 'grid', 'grid_label',
    'index_node', 'institution_id', 'member_id', 'nominal_resolution', 'realm',
    'short_description', 'source_id', 'source_type', 'sub_experiment_id', 'table_id',
    'variable', 'variable_id', 'variable_long_name', 'variant_label', 'version'
]


def cmip(url=None):
    """
    Download CMIP5 model data.

    Parameters
    ----------
    url : str, default: 'https://esgf-node.llnl.gov/esg-search'
        The search URL.
    """
    # Data requests
    from pyesgf.search import SearchConnection
    if url is None:
        url = 'https://esgf-node.llnl.gov/esg-search'
    conn = SearchConnection(url, distrib=True)
    ctx = conn.new_context()
    return ctx


def cesm():
    """
    Download large ensemble data.
    """
    raise NotImplementedError
