#!/usr/bin/env python
"""
Download wget files using pyesgf. Will eventually repace download_wgets.sh.

Todo
----
Place wget files into file names that denote their contents. Use the parsing
utilities from filter_wgets.py. Also incorporate into climopy... or consider
replacing climopy download utilities with companion project.
"""
# Log on with OpenID
# LLNL: https://esgf-node.llnl.gov/
# CEDA: https://esgf-index1.ceda.ac.uk/
# DKRZ: https://esgf-data.dkrz.de/
# GFDL: https://esgdata.gfdl.noaa.gov/
# IPSL: https://esgf-node.ipsl.upmc.fr/
# JPL:  https://esgf-node.jpl.nasa.gov/
# LIU:  https://esg-dn1.nsc.liu.se/
# NCI:  https://esgf.nci.org.au/
# NCCS: https://esgf.nccs.nasa.gov/
# Nodes listed here: https://esgf.llnl.gov/nodes.html
import sys
from pathlib import Path
from pyesgf.logon import LogonManager
from pyesgf.search import SearchConnection

# Constants
DATA = Path.home() / 'data'
if sys.platform == 'darwin':
    ROOT = Path.home() / 'data'
else:  # TODO: add conditionals?
    ROOT = Path('/mdata5') / 'ldavis'

if __name__ == '__main__':
    # Log on
    lm = LogonManager()
    host = 'esgf-node.llnl.gov'
    if not lm.is_logged_on():
        lm.logon(username='lukelbd', password=None, hostname=host)

    # Create CMIP5 and CMIP6 contexts
    url = 'https://esgf-node.llnl.gov/esg-search'
    conn = SearchConnection(url, distrib=True)
    cmip6 = conn.new_context(project='CMIP6')
    cmip5 = conn.new_context(project='CMIP5')
    cmip5_control = cmip5_response = None
    cmipy_control = cmip6_response = None

    # Integrated transport
    # cmip6_control = cmip6.constrain(
    #     experiment_id=['piControl'],
    #     variable_id=['intuadse', 'intvadse', 'intuaw', 'intvaw'],
    #     variant_label=['r1i1p1f1'],
    # )
    # cmip6_response = cmip6.constrain(
    #     experiment_id=['abrupt-4xCO2'],
    #     variable_id=['intuadse', 'intvadse', 'intuaw', 'intvaw'],
    #     variant_label=['r1i1p1f1'],
    # )

    # Climate everything
    cmip6_control = cmip6.constrain(
        experiment_id=['piControl'],
        variable_id=[
            'ta', 'hur', 'hus', 'cl', 'clw', 'cli', 'clwvp', 'clwvi', 'clivi', 'cct',
            # 'cldwatmxrat27', 'cldicemxrat27'
        ],
        variant_label=['r1i1p1f1'],
        table_id=['Amon'],
    )
    cmip5_control = cmip5.constrain(
        experiment=['piControl'],
        variable=['ta'],
        ensemble=['r1i1p1'],
        cmor_table=['Amon'],
    )

    # Climate temperature
    # cmip6_response = cmip6.constrain(
    #     experiment_id=['abrupt-4xCO2'],
    #     variable_id=['tas', 'rlut', 'rsut', 'rlutcs', 'rsutcs'],
    #     variant_label=['r1i1p1f1'],
    #     table_id=['Amon'],
    # )
    # cmip5_response = cmip5.constrain(
    #     experiment=['abrupt4xCO2'],  # no dash
    #     variable=['tas', 'rlut', 'rsut', 'rlutcs', 'rsutcs'],
    #     ensemble=['r1i1p1'],
    #     cmor_table=['Amon'],
    # )

    # Iterate over contexts
    # NOTE: Idea is that wget file names should be standardized like climatology
    # directory structures.
    # How to standardize directory structure?
    ctxs = (cmip6_control, cmip5_control, cmip6_response, cmip5_response)
    for i, ctx in enumerate(ctxs):
        if ctx is None:
            continue
        # Create wget name
        print(f'Context {i}:', ctx, ctx.facet_constraints)
        print(f'Hit count {i}:', ctx.hit_count)
        keys = (
            'project',
            ('experiment', 'experiment_id'),
            ('cmor_table', 'table_id'),
            ('variable_id', 'variable'),
        )
        parts = []
        for j, key in enumerate(keys):  # constraint components to use in file name
            key = (key,) if isinstance(key, str) else key
            opts = sum((ctx.facet_constraints.getall(k) for k in key), start=[])
            part = '-'.join(opt.replace('-', '') for opt in opts)
            if j == 2:
                part = part or 'Amon'  # TODO: remove
            if 'project' in key:
                part = part.lower()
            parts.append(part)
        # Write wget file
        for j, ds in enumerate(ctx.search()):  # iterate over models and dates
            print(f'Dataset {j}:', ds)
            fc = ds.file_context()
            name = 'wget_' + '_'.join((*parts, str(j))) + '.sh'
            path = Path(ROOT, 'wgets', name)
            if path.exists():
                print('Skipping script:', name)
            else:
                try:
                    wget = fc.get_download_script()
                except Exception:
                    print('Download failed:', name)
                else:
                    print('Creating script:', name)
                    with open(path, 'w') as f:
                        f.write(wget)
