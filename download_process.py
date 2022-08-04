#!/usr/bin/env python3
"""
File for downloading and merging relevant data.
"""
import cmip_data

# Global arguments
# NOTE: Mark Zelinka CMIP5 is missing r1i1p1 CNRM-CM5-2 (not sure why) and EC-Earth
# (also skip this one because it provides only partial data and recently disappeared
# from esgf... even tried downloading from CEDA but missing data), increasing ensemble
# members from 29 to 30. Also for some reason CSIRO-Mk3L-1-2 has only published abrupt
# data but not control. Verified that unlike CMIP6 there are no non-r1i1p1 flagships.
# NOTE: Mark Zelinka CMIP6 is missing r1i1p1f1 CAS-ESM2-0, FIO-ESM-2-0, ICON-ESM-LR,
# KIOST-ESM, and MCM-UA-1-0 (older versions also missed EC-Earth3-CC, EC-Earth3-Veg-LR,
# GISS-E2-2-H but they were recently added). However in initial search, failed to find
# non-r1i1p1f1 flagship abrupt runs of EC-Earth3 (uses realization 'r8', verified that
# parent_variant_label is r1i1p1f1, and note 'r3' appears available but web interface
# downloads zero-byte files), abrupt runs of HadGEM3-GC31-LL and HadGEM3-GC31-MM
# (use forcing 'f3', also verified that parent_variant_label is r1i1p1), both control
# and abrupt runs of CNRM-CM6-1, CNRM-CM6-1-HR, CNRM-ESM2-1, MIROC-ES2L, UKESM1-0-LL
# (use forcing 'f2', and unlike HadGEM, control runs also use forcing 'f2'), and
# control runs of IPSL-CM6A-LR-INCA (comment attribute indicates this was started from
# an IPSL-CM6A-LR spinup and a full INCA version of the control run was never published
# -- so we skip processing this one, and it is useless anyway since we are trying to
# constrain the response from the control climate). There is also a MIROC-ES2H model
# that uses both 'f2' and 'p4' that Mark missed (the equivalent 'f2' and 'p1' through
# 'p3' runs only include 1 year of data... weird), and a similar UKESM1-1 model with the
# same r1i1p1f2 variant (no pfull yet but also no model level data available). Verified
# there are no other missing control-abrupt pairs (candidates are AWI-ESM-1-1-LR,
# CMCC-CM2-HR4, CanESM5-CanOE, E3SM-1-1, E3SM-1-1-ECA, EC-Earth3-LR, and NorESM1-F,
# but all are missing abrupt simulations). In sum we added 5 models but missed 9 models
# for a total of 4 models fewer (49 instead of 53). We can rectify everything but the
# IPSL model, plus add the extra MIROC and UKESM models, but *omit* the FIO-ESM2 model
# because it is missing too much data from branch time 300 (see bottom), providing a
# total of 5 models more than Mark (58 instead of 53). Note that for MCM-UA-1-0, we
# download 'rtmt' and compute 'rsut - rsdt' from the difference (see 'manual_data.sh').
climate = True
series = True
analysis = True  # control and response data for feedback analysis?
circulation = True  # control and response data for implicit circulation stuff?
constraints = True  # control data for emergent constraints?
dependencies = True  # dependency data for emergent constraintss?
explicit = True  # control and response data for explicit circulation stuff?
download = False
filter = False
process = False
feedbacks = True
summarize = False

# Pre-industrial control and response data for constraints, transport, and feedbacks
# NOTE: The abrupt and control radiation data is needed for 'local feedback parameter'
# esimates, and control radiation data is needed for 'relaxation feedback parameter'
# estimates. Then 'ts', 'ta', and 'hus' are needed for component feedaback breakdowns
# (after multiplying by radiative kernels to get 'radiation due to parameter' then
# either regressing against surface temperature or correlating with surface temp). Note
# that because Clausius-Clapeyron is a non-linear function of 't' and 'q', average
# relative humidity is not same as relative humidity of average, however neglect this
# in e.g. relative humidity feedback breakdowns so will also ignore in climate averages.
# Also note albedo is calculated from ratio of upwelling to downwelling surface SW
# radiation rather than some sort of direct surface snow/ice measurement.
# NOTE: The abrupt and control energetic and water data is needed for 'moist/dry
# static energy transport' esimates. To derive dry transport use radiation across
# top-of-atmosphere and surface boundaries plus sensible heat flux then subtract latent
# heat transport. To derive latent transport use both turbulent surface evaporation and
# precipitation, where assumption is 1) water/snow/ice transport is neglible compared to
# humidity transport (valid since vertically integrated humidity is always larger
# than vertically integrated water/snow/ice, should make plots to demonstrate), and
# therefore 2) precipitation falls where it was formed (implying it equals the component
# of vertically integrated latent heat released by forming hydrometeors not balanced
# by latent heat absorbed by evaporating hydrometeors -- assumptions are mentioned in
# detailed breakdown papers, e.g. Mayer et al. (2019) and Donohue et al. (2020).
# NOTE: The transport data permits sanity checks against energetic and water residuals.
# Should produce comparison plots of dry transport derived from (1) energy balance
# and (2) explicit 'intvadse' or 'vt' minus 'v' * 'z' (which has only marginal transient
# component due to geostrophic balance holding well where 'z' flux matters, see the
# dynamical core simulations for details), plus comparison plots of moist transport
# derived from (1) energy balance and (2) explicit 'intvaw' or 'vqint' (should be
# identical, have different names just because they were designed for different
# intercomparisons and there is no 'controlled vocabulary' for variable_id). Note even
# though 'intvaw' says 'water' have verified it includes vapor by plotting and comparing
# its magnitude to dry transport -- also perhaps 'vqint' omits water/snow/ice flux but
# assumption is this component is small (see above discussion).
kw_constant = {
    'table': 'fx',
    'variable': [
        'orog',  # orography (currently unused)
        'sftlf',  # land fraction (possibly useful for averages e.g. constraints)
    ],
}
kw_analysis = {
    'table': 'Amon',
    'experiment': ['piControl', 'abrupt-4xCO2'],
    'variable': [
        'ps',  # surface pressure for kernels
        'ta',  # air temperature for kernels
        'ts',  # surface temperature for kernels
        'hus',  # specific humidity for kernels
        'rsdt',  # downwelling SW TOA (identical to solar constant)
        'rlut',  # upwelling LW TOA
        'rsut',  # upwelling SW TOA
        'rlds',  # downwelling LW surface
        'rsds',  # downwelling SW surface
        'rlus',  # upwelling LW surface
        'rsus',  # upwelling SW surface
        'rlutcs',  # upwelling LW TOA (clear-sky)
        'rsutcs',  # upwelling SW TOA (clear-sky)
        'rsuscs',  # upwelling SW surface (clear-sky) (in response to downwelling)
        'rldscs',  # downwelling LW surface (clear-sky)
        'rsdscs',  # downwelling SW surface (clear-sky)
        # 'rtmt',  # net TOA LW and SW (Amon table, use individual components instead)
        # 'rls',  # net surface LW (Emon table, use individual components instead)
        # 'rss',  # net surface SW (Emon table, use individual components instead)
    ],
}
kw_circulation = {
    'table': 'Amon',
    'experiment': ['piControl', 'abrupt-4xCO2'],
    'variable': [  # NOTE: tauu is balanced by cemf, tauv by Coriolis torque
        'pr',  # water/ice precipitation for transport (Lv+Ls gained)
        'prsn',  # ice precipitation for transport (Ls gained, use to isolate Lv)
        'evspsbl',  # water/ice evaporation for transport (Lv+Ls lost)
        'hfls',  # surface upward LH flux for transport (Lv+Ls gained)
        'hfss',  # surface upward SH flux for transport (heat gained)
        'sbl',  # ice evaporation for transport (Ls lost, use to isolate Lv)
        'ua',  # zonal wind for circulation
        'va',  # meridional wind for circulation
        'uas',  # surface wind for circulation
        'vas',  # surface wind for circlation
        'psl',  # sea-level pressure for circulation
        'tauu',  # surface friction for circulation (indicates eddy jet strength)
        'tauv',  # surface friction for circulation (indicates zonal jet strength)
        'zg',  # geopotential height for circulation
    ]
}
kw_constraints = {
    'table': 'Amon',
    'experiment': ['piControl', 'abrupt-4xCO2'],
    'variable': [
        'cl',  # percent cloud cover
        'clt',  # total percent cloud cover
        'cct',  # convective cloud top pressure
        'clw',  # mass fraction cloud water/snow/ice
        'cli',  # mass fraction cloud snow/ice
        'clwvi',  # vertically integrated condensed cloud water/snow/ice
        'clivi',  # vertically integrated condensed cloud snow/ice
        'huss',  # near-surface specific humidity
        'prw',  # vertically integrated water vapor path
    ],
}
kw_dependencies = {
    'table': ('Amon', 'AERmon'),
    'experiment': ('piControl', 'control-1950'),
    'variable': 'pfull',
    'model': [  # NOTE: _parse_constraints handles cmip5 conversion between dots/dashes
        'ACCESS1-0',  # available in Amon piControl
        'ACCESS1-3',  # available in Amon piControl
        'HadGEM2-ES',  # available in Amon piControl
        'ACCESS-CM2',  # available in Amon piControl
        'ACCESS-ESM1-5',  # available in Amon piControl
        'HadGEM3-GC31-LL',  # only available in Amon control-1950 (skip AERmon)
        'HadGEM3-GC31-MM',  # only available in Amon control-1950 (skip AERmon)
        'KACE-1-0-G',  # available in Amon piControl
        'UKESM1-0-LL',  # only available in AERmon (see process.py)
        'UKESM1-1-LL',  # not available yet but will likely match UKESM1-0-LL
    ],
}
kw_explicit = {
    'table': 'Emon',
    'experiment': ['piControl', 'abrupt-4xCO2'],
    'variable': [
        'uv',  # HighResMIP (vertically integrate to get EMF)
        'u2',  # HighResMIP (vertically integrate to get EKE)
        'v2',  # HighResMIP (vertically integrate to get EKE)
        'ut',  # HighResMIP (vertically integrate and combine with 'v' * 'z' to get DSE)
        'vt',  # HighResMIP (vertically integrate and combine with 'v' * 'z' to get DSE)
        'uqint',  # HighResMIP (already vertically integrated for MSE)
        'vqint',  # HighResMIP (already vertically integrated for MSE)
        'intuadse',  # PMIP limited availability
        'intvadse',  # PMIP limited availability
        'intuaw',  # PMIP limited availability
        'intvaw',  # PMIP limited availability
    ]
}

# Download the wget files
# NOTE: Facets and options supplied to constrain filter are standardized between
# cmip5 and cmip6 synonyms. So can loop through both projects with same keywords.
dicts = []
if analysis:
    dicts.append(kw_analysis)
if circulation:
    dicts.append(kw_circulation)
if constraints:
    dicts.append(kw_constraints)
if dependencies:
    dicts.append(kw_dependencies)
if explicit:
    dicts.append(kw_explicit)
if download:
    unfiltered = []
    for kwargs in dicts:
        if kwargs is kw_explicit:
            projects = ('CMIP6',)
        else:
            projects = ('CMIP6', 'CMIP5')
        for project in projects:
            if kwargs is kw_dependencies:
                folder = '~/scratch2/data-dependencies'
            elif project == 'CMIP5':
                folder = '~/scratch2'
            elif project == 'CMIP6':
                folder = '~/scratch5'
            script = cmip_data.download_script(
                folder,
                node='llnl',
                openid='https://esgf-node.llnl.gov/esgf-idp/openid/lukelbd',
                username='lukelbd',
                flagship_filter=True,
                project=project,
                logging=True,
                **kwargs,
            )
            unfiltered.append(script)

# Filter the resulting wget files
# NOTE: Skip EC-EARTH data since it has only partial availability... text file
# on CEDA server seems to indicate a bunch of data was deleted and improperly
# backed up and apparently no one else on the planet had copies.
# NOTE: Skip model-level cloud data for CNRM-CM6-1-HR since it is much larger than any
# other source (4GB for 10 years of data compared to next highest of 2GB for 10 years
# of HadGEM-31-MM data or 300MB to 1GB for 10 years of most other data -- files have
# 91 levels, 360 latitudes, 720 longitudes, way more than other models. For summary
# try the following: cd ~/scratch5 && ll cmip*/cl{,i,w}_* | cut -d' ' -f5- | sort -h
# NOTE: Only want to bother downloading constraint data with an equivalent abrupt
# simulation (indicating that feedbacks are available) but don't actually want to
# download this data so use the exclude fitler to avoid it. Also don't want abrupt
# pfull data since not always available. And need to include control pfull data from
# other tables/experiments (see process.py comments) so use always_include.
if filter:
    filtered = []
    for kwargs in dicts:
        if kwargs is kw_explicit:
            projects = ('CMIP6',)
        else:
            projects = ('CMIP6', 'CMIP5')
        for project in projects:
            exclude = []
            if kwargs is kw_constraints:
                exclude.append({'experiment': 'abrupt4xCO2'})
            if project == 'CMIP5':
                exclude.append({'model': 'EC-EARTH'})  # incomplete availability
            else:
                exclude.append({'model': 'FIO-ESM-2-0'})  # incomplete availability
            if kwargs is kw_dependencies:
                exclude.append({'table': 'AERmon', 'model': ['HadGEM3-GC31-LL', 'HadGEM3-GC31-MM']})  # noqa: E501
            elif kwargs is kw_constraints:
                exclude.append({'model': 'CNRM-CM6-1-HR', 'variable': ['cl', 'clw', 'cli']})  # noqa: E501
            if kwargs is kw_dependencies:
                folder = '~/scratch2/data-dependencies'
            elif project == 'CMIP5':
                folder = '~/scratch2'
            elif project == 'CMIP6':
                folder = '~/scratch5'
            scripts = cmip_data.filter_script(
                folder,
                maxyears=150,
                endyears=False,
                always_include=None,
                always_exclude=exclude,
                flagship_filter=True,
                project=project,
                logging=True,
                **kwargs
            )
            filtered.extend(scripts)

# Average and standardize the resulting files
# NOTE: Here follow Armour et al. 2019 methodology of taking difference between final
# 30 years of the 150 years required by the DECK abrupt-4xco2 experiment protocol. Also
# again exclude constraint data from processing (pfull is excluded in process_files).
# See: https://pcmdi.llnl.gov/CMIP6/Guide/dataUsers.html
if process:
    nodrifts = (False,)
    # nodrifts = (True,)
    # nodrifts = (False, True)
    for nodrift in nodrifts:
        for kwargs in dicts:
            if kwargs is kw_dependencies:
                projects = ()
            elif kwargs is kw_explicit:
                projects = ('CMIP6',)
            else:
                projects = ('CMIP6', 'CMIP5')
            for project in projects:
                experiments = {'piControl': 150, 'abrupt-4xCO2': (120, 150)}
                if kwargs is kw_constraints:
                    del experiments['abrupt-4xCO2']
                if not climate:
                    experiments.clear()
                if project == 'CMIP5':
                    folder = '~/scratch2'
                elif project == 'CMIP6':
                    folder = '~/scratch5'
                for experiment, years in experiments.items():
                    kw = {**kwargs, 'experiment': experiment}
                    cmip_data.process_files(
                        folder,
                        output='~/data',
                        search='~/scratch2/data-dependencies',
                        method='gencon',
                        years=years,
                        climate=True,
                        nodrift=nodrift,
                        overwrite=False,  # TODO: change back
                        dryrun=False,
                        flagship_filter=True,
                        project=project,
                        logging=True,
                        **kw,
                    )
                experiments = {'piControl': 150, 'abrupt-4xCO2': 150}
                if kwargs is not kw_analysis:
                    experiments.clear()
                if not series:
                    experiments.clear()
                for experiment, years in experiments.items():
                    kw = {**kwargs, 'experiment': experiment}
                    cmip_data.process_files(
                        folder,
                        output='~/scratch2/data-processed',
                        search='~/scratch2/data-dependencies',
                        constants='~/data',
                        method='gencon',
                        years=years,
                        climate=False,
                        nodrift=nodrift,
                        flagship_filter=True,
                        overwrite=False,  # TODO: change back
                        project=project,
                        logging=True,  # ignored if dryrun true
                        dryrun=False,
                        **kw,
                    )

# Calculate the response, control, and anomaly feedbacks (note control anomaly
# feedbacks are impossible because cannot select a period to use for anomalies).
# NOTE: The residual feedback will only be calculated if all kernels
# for the associated flux are requested. Otherwise this is bypassed.
if feedbacks:
    nodrifts = (False,)
    # nodrifts = (True,)
    # nodrifts = (False, True)
    for nodrift in nodrifts:
        experiments = (
            ('abrupt4xCO2', False),  # regression of series
            ('piControl', False),  # regression of series
            ('abrupt4xCO2', True),  # ratio of anomalies
        )
        for experiment, ratio in experiments:
            projects = ('CMIP6', 'CMIP5')
            for project in projects:
                cmip_data.process_feedbacks(
                    '~/data',  # source climate location
                    '~/scratch2/data-processed',  # source series location
                    feedbacks='~/data',  # output feedback location
                    kernels='~/data',  # input kernels location
                    fluxes='~/scratch2/data-processed',  # output flux location
                    ratio=ratio,
                    source='eraint',
                    project=project,
                    experiment=experiment,
                    flagship_filter=True,
                    nodrift=nodrift,
                    overwrite=False,
                    logging=True,  # ignored if dryrun true
                    dryrun=False,
                )

# Update the summary logs once finished
# NOTE: Have 'intvadse' and 'intvaw' for IPSL-CM6A-LR, CNRM-CM6-1, CNRM-ESM2-1, and
# CNRM-CM6-1-HR (last one was briefly unavailable). Have only 'intvaw' for ACCESS-CM2,
# ACCESS-CM1-5, MIROC-ES2L, and MIROC-ES2H, and only 'intvadse' for IPSL-CM5A2-INCA.
# Have control 'vt' and 'vqint' for HadGEM3-GC31-LL, HadGEM3-GC31-MM, and UKESM1-0-LL.
# NOTE: Obtained missing GFDL data from the GFDL portal manually. Ignored missing
# EC-EARTH data because even CEDA data portal is missing critical data. Derived
# missing MCM-UA-1-0 shortwave top-of-atmosphere data by combining rtmt data with
# rlut data to get 'rsut - rsdt' as a residual. See manual_files.sh for details.
# NOTE: Still need to download missing FGOALS-g2, CAMS-CSM1-0, GFDL-ESM4, NorESM2-LM,
# NorESM2-MM, and TaiESM1 flux for surface and atmosphere feedbacks, plus FIO-ESM-2-0
# for missing top-of-atmosphere cloud feedbacks. Notably, among the models Mark
# excluded (CAS-ESM2-0, FIO-ESM-2-0, ICON-ESM-LR, KIOST-ESM, and MCM-UA-1-0), there
# is either missing data required for TOA cloud feedbacks (in the case of FIO and MCM),
# control data years out-of-sync with abrupt years (in the case of FIO, ICON, KIOST),
# or... not sure what else (possibly quality control/acquisition issues for CAS-ESM2).
# * The ICON model switches time units -- control data starts at year 4000, abrupt
#   data at year 1850, and abrupt attributes state that branch_time_in_parent = 0 and
#   branch_time_in_parent = 0 -- but abrupt attributes also state parent_time_units =
#   "days since 4000" (and control data time units are indeed "days since 4000") while
#   the time units of abrupt data are "days since 1850".
# * The KIOST model switches abrupt times -- control data starts at year 3189, abrupt
#   data at year 1850 (both with time units "days since 1850"), and abrupt attributes
#   state that branch_time_in_child = 0 days, branch_time_in_parent = 488370 days
#   (i.e. 1339 365-day calendar years --> 1850 + 1339 = the start year 3189). This
#   center published *pre-branching-time-only* control data from year 2689 in earlier
#   versions... possible that older versions of abrupt runs started from here.
# * The FIO model switches abrupt times -- control data starts at year 0300, abrupt
#   data at year 0001 (both with time units "days since 0001"), and abrupt attributes
#   state that branch_time_in_child = 0 days, branch_time_in_parent = 109500 days
#   (i.e. 300 365-day calendar years --> 0001 + 0300 = the start year 0300). This
#   center published *post-branching-time only* control data from year 0400 in later
#   versions... but some data (e.g. ta, hus) is only available from year 0400.... could
#   use it anyway, but notably surface pressure ps is only available from year 0300,
#   so would be weird to pretend year 0400 is the branching time. Forget this model.
# * The UKESM-1-1 model switches control and abrupt times -- control data starts at
#   year 2743, abrupt data starts at year 1850 (both with time units "days since 1850"),
#   and abrupt attributes state branch_time_in_child = branch_time_in_parent = 0 days,
#   but control attributes state branch_time_in_child = 321480 (i.e. 893 360-day years
#   --> 1850 + 893 = 2743)... so control dates jump relative to *spinup*. However other
#   models Mark *does* use also have date differences (e.g. UKESM-1-0 abrupt data
#   starts at 1850 and control data starts at 1960) so not sure why they are excluded.
# * Missing data (after adding rsut and rsdt MCM data with residual):
#   rlds: MCM-UA-1-0
#   rldscs: CAMS-CSM1-0, GFDL-ESM4, MCM-UA-1-0, NorESM2-LM, NorESM2-MM, TaiESM1
#   rlus: MCM-UA-1-0
#   rlutcs: MCM-UA-1-0
#   rsds: MCM-UA-1-0
#   rsdscs: CAMS-CSM1-0, GFDL-ESM4, MCM-UA-1-0
#   rsus: MCM-UA-1-0
#   rsuscs: CAMS-CSM1-0, GFDL-ESM4, MCM-UA-1-0
#   rsutcs: MCM-UA-1-0
if summarize:
    for project in ('cmip6', 'cmip5'):
        folders_downloads = ('~/scratch2', '~/scratch5')
        folders_processed = ('~/scratch2/data-processed', '~/data')
        cmip_data.summarize_downloads(
            *folders_downloads,
            project=project,
            flagship_translate=True,
        )
        cmip_data.summarize_processed(
            *folders_downloads,
            *folders_processed,
            project=project,
            flagship_translate=True,
        )
        cmip_data.summarize_ranges(
            *folders_processed,
            project=project,
            flagship_translate=True,
        )
