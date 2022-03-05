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
    if not lm.is_logged_on():  # surface orography
        lm.logon(username='lukelbd', password=None, hostname=host)

    # Create CMIP5 and CMIP6 contexts
    facets6 = 'project,experiment_id,variant_label,table_id,variable_id'
    facets5 = 'project,experiment,ensemble,cmor_table,variable'
    url = 'https://esgf-node.llnl.gov/esg-search'
    conn = SearchConnection(url, distrib=True)
    cmip6 = conn.new_context(
        project='CMIP6',
        variant_label=['r1i1p1f1'],
        table_id=['Amon', 'Emon', 'fx'],
        facets=facets6,
    )
    cmip5 = conn.new_context(
        project='CMIP5',
        ensemble=['r1i1p1'],
        cmor_table=['Amon', 'Emon', 'fx'],
        facets=facets5,
    )
    cmip5_control = cmip5_response = None
    cmip6_control = cmip6_response = None

    # Integrated transport
    # vars = ['intuadse', 'intvadse', 'intuaw', 'intvaw']
    # cmip6_control = cmip6.constrain(
    #     experiment_id=['piControl'],
    #     variable_id=vars,
    # )
    # cmip6_response = cmip6.constrain(
    #     experiment_id=['abrupt-4xCO2'],
    #     variable_id=vars,
    # )

    # Residual transport and net feedbacks
    # NOTE: Need abrupt 4xCO2 radiation data for 'local feedback parameter' estimates
    # and pre-industrial control radition data for 'effective relaxation timescale'
    # estimates. Then atmospheric parameters used for regressions and correlations
    # (after multiplying by radiative kernel to get 'radiation due to parameter'
    # then regressing or correlating with surface temperature). Also note albedo
    # is taken from ratio of upwelling to downwelling surface solar.
    # vars = [
    #     'rtmt',  # net TOA LW and SW
    #     'rls',  # net surface LW
    #     'rss',  # net surface SW
    #     'hsls',  # surface upward LH flux
    #     'hsfs',  # surface upward SH flux
    # ]
    # vars = [
    #     'ta',  # air temperature
    #     'hus',  # specific humidity
    #     'tas',  # surface temperature
    #     'rsdt',  # downwelling SW TOA (identical to solar constant)
    #     'rlut',  # upwelling LW TOA
    #     'rsut',  # upwelling SW TOA
    #     'rlds',  # downwelling LW surface
    #     'rsds',  # downwelling SW surface
    #     'rlus',  # upwelling LW surface
    #     'rsus',  # upwelling SW surface
    #     'rlutcs',  # upwelling LW TOA (clear-sky)
    #     'rsutcs',  # upwelling SW TOA (clear-sky)
    #     'rsuscs',  # upwelling SW surface (clear-sky) (in response to downwelling)
    #     'rldscs',  # downwelling LW surface (clear-sky)
    #     'rsdscs',  # downwelling SW surface (clear-sky)
    #     'hsls',  # surface upward LH flux
    #     'hsfs',  # surface upward SH flux
    # ]
    # cmip5_control = cmip5.constrain(
    #     experiment=['piControl'],
    #     variable=vars,
    # )
    # cmip5_response = cmip5.constrain(
    #     experiment=['abrupt4xCO2'],  # no dash
    #     variable=vars,
    # )
    # cmip6_control = cmip6.constrain(
    #     experiment_id=['piControl'],
    #     variable_id=vars,
    # )
    # cmip6_response = cmip6.constrain(
    #     experiment_id=['abrupt-4xCO2'],  # include dash
    #     variable_id=vars,
    # )

    # Climate everything
    vars = ['cldwatmxrat27', 'cldicemxrat27']  # special cloud variables
    vars = [
        # Phase 3
        'ts',  # surface temperature
        'hurs',  # near-surace relative humidity
        'huss',  # near-surace specific humidity
        'prw',  # water vapor path
        'pr',  # precipitation
        # Phase 2
        # 'psl',  # sea-level pressure
        # 'gs',  # geopotential height
        # 'ua',  # zonal wind
        # 'va',  # meridional wind
        # Phase 1
        # 'ta',  # air temperature
        # 'hur',  # relative humidity
        # 'hus',  # specific humidity
        # 'cl',  # percent cloud cover
        # 'clt',  # total percent cloud cover
        # 'clw',  # mass fraction cloud water
        # 'cli',  # mass fraction cloud ice
        # 'clwvi',  # condensed water path
        # 'clivi',  # condensed ice path
        # 'cct',  # convective cloud top pressure
    ]
    cmip6_control = cmip6.constrain(
        experiment_id=['piControl'],
        variable_id=vars,
    )
    cmip5_control = cmip5.constrain(
        experiment=['piControl'],
        variable=vars,
    )

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
        parts = []
        for j, keys in enumerate(zip(facets5.split(','), facets6.split(','))):
            opts = sum((ctx.facet_constraints.getall(k) for k in keys), start=[])
            part = '-'.join(opt.replace('-', '') for opt in sorted(set(opts)))
            if j == 2:
                part = part or 'Amon'  # TODO: remove kludge?
            if 'project' in keys:
                part = part.lower()
            parts.append(part)
        # Write wget file
        for j, ds in enumerate(ctx.search()):
            print(f'Dataset {j}:', ds)
            fc = ds.file_context()
            fc.facets = ctx.facets  # TODO: report bug and remove?
            name = 'wget_' + '_'.join((*parts, format(j, '05d'))) + '.sh'
            path = Path(ROOT, 'wgets', name)
            if path.exists() and False:
                print('Skipping script:', name)
                continue
            try:
                wget = fc.get_download_script()
            except Exception:
                print('Download failed:', name)
            else:
                print('Creating script:', name)
                with open(path, 'w') as f:
                    f.write(wget)