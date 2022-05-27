#!/usr/bin/env python3
"""
File for downloading and merging relevant data.
"""
# WARNING: List corrupt files that have to be manually deleted here. Can test using
# e.g. for file in ta*nc; do echo $file && ncvardump ta $file >/dev/null; done
# since often get HDF5 errors during processing but header can be read without error.
# ./cmip6-picontrol-amon/clwvi_Amon_NorESM2-MM_piControl_r1i1p1f1_gn_132001-132912.nc
import cmip_data

# Global arguments
# NOTE: Here have 'intvadse' and 'intvaw' for IPSL-CM6A-LR, CNRM-CM6-1, CNRM-ESM2-1, and
# CNRM-CM6-1-HR (last one was briefly unavailable). Have only 'intvaw' for ACCESS-CM2,
# ACCESS-CM1-5, MIROC-ES2L, and MIROC-ES2H, and only 'intvadse' for IPSL-CM5A2-INCA.
# Have control 'vt' and 'vqint' for HadGEM3-GC31-LL, HadGEM3-GC31-MM, and UKESM1-0-LL.
# NOTE: Here 'tauu' and 'tauv' are for jet calculations, since time-vertical-zonal
# mean tauu is balanced by eddy-momentum convergence (integrated meridional wind is zero
# due to mass conservation) and tauv is balanced by Coriolis torque from zonal wind.
# Pair with residual eddy-energy transport for 'storm track' and 'eddy jet' metrics.
# NOTE: Mark Zelinka CMIP5 is missing CNRM-CM5-2 (not sure why) and EC-Earth (because it
# provides only partial data and recently disappeared from esgf), increasing ensemble
# members from 29 to 31. Also for some reason CSIRO-Mk3L-1-2 has only published abrupt
# data but not control data. Unlike CMIP6 verified there are no non-r1i1p1 flagships.
# NOTE: Mark Zelinka CMIP6 is missing CAS-ESM2-0, FIO-ESM-2-0, ICON-ESM-LR, KIOST-ESM,
# and MCM-UA-1-0 (older version also missed EC-Earth3-CC and GISS-E2-2-H but they were
# recently added). However in search, failed to find control run of IPSL-CM6A-LR-INCA
# (tried ESGF website and seems it was removed, also checked that abrupt parent is
# indeed the same model and searched the IPSL ESGF node -- note CMCC-CM2-HR4 also only
# published abrupt data so was ignored by Mark and ignored by us), abrupt simulations of
# EC-Earth3 (uses realization 'r8', verified that 'parent_variable_label' is 'r1i1p1f1',
# and note 'r3' is also available but the online interface downloads zero-byte files),
# abrupt simulations of HadGEM3-GC31-LL and HadGEM3-GC31-MM (uses forcing 'f3', also
# verified that 'parent_variable_label' is 'r1i1p1'), and both control and abrupt
# simulations of CNRM-CM6-1, CNRM-CM6-1-HR, CNRM-ESM2-1, MIROC-ES2L, and UKESM1-0-LL
# (uses forcing 'f2', and in contrast with HadGEM3 models the control forcing is also
# 'f2'). There is also a MIROC-ES2H model that uses both 'f2' and 'p4' that Mark missed
# (the equivalent 'f2' and 'p1' through 'p3' runs only include 1 year of data... weird).
# Searched around and seems there are no other missing non-'r1i1p1f1' simulation pairs
# (found only 'p2' ensemble of CanESM5-CANOE but only the control run is available).
# In sum we added 5 models but missed 9 models for a total of 4 models fewer (47 instead
# of 52). We can rectify everything but the IPSL model, plus add the extra MIROC model,
# to provide a total of 5 models more than Mark Zelinka (57 instead of 52).
series = True  # get feedback time series?
climate = True  # get all climate averages?
feedbacks = True  # control and response data for feedbacks?
constrain = True  # control data for constraints?
circulation = True  # control and response data for implicit circulation stuff?
explicit = True  # control and response data for explicit circulation stuff?
download = False
filter = False
process = True

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
# NOTE: The abrupt and control energetic and water data is also needed for 'moist/dry
# static energy transport' esimates. To derive dry transport use radiation across
# top-of-atmosphere and surface boundaries plus sensible heat flux then subtract latent
# heat transport. To derive latent transport use both turbulent surface evaporation and
# precipitation, where assumption is 1) water/snow/ice transport is neglible compared to
# humidity transport (valid since vertically integrated humidity is always larger
# than vertically integrated water/snow/ice, should make plots to demonstrate), and
# similarly 2) precipitation falls where it was formed (implying it equals the component
# of vertically integrated latent heat released by forming hydrometeors not balanced
# by latent heat absorbed by evaporating hydrometeors -- should research literature
# on these assumptions). Armour et al. (2019) only mention falling snow but mistaken.
# The LH flux convergence is then written dLH/dt = Lv * (pr - prsn) + Ls * prsn - hsls
# where hsls is explicit or can be found with Lv * (evspsbl - sbl) + Ls * sbl (include
# the implicit calculation only for sanity checks since sbl is often not provided).
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
kw_feedbacks = {
    'table': 'Amon',
    'experiment': ['piControl', 'abrupt-4xCO2'],
    'variable': [
        'ta',  # air temperature for kernels
        'ts',  # surface temperature for kernels
        'hus',  # specific humidity for kernels
        'huss',  # near-surface specific humidity for consistency
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
        # 'rtmt',  # net TOA LW and SW (use individual components instead)
        # 'rls',  # net surface LW (use individual components instead)
        # 'rss',  # net surface SW (use individual components instead)
    ],
}
kw_constrain = {
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
        'prw',  # vertically integrated water vapor path
        'pfull',  # model level pressure (provided only for hybrid height models)
        'ps',  # monthly surface pressure (for hybrid pressure sigma interpolation)
        # 'pfull',  # pressure at model full levels (use hybrid coords instead)
        # 'phalf',  # pressure at model half levels (use hybrid coords instead)
    ],
}
kw_circulation = {
    'table': 'Amon',
    'experiment': ['piControl', 'abrupt-4xCO2'],
    'variable': [
        'hfls',  # surface upward LH flux for transport (Lv+Ls gained)
        'hfss',  # surface upward SH flux for transport (heat gained)
        'pr',  # water/snow/ice precipitation for transport (Lv+Ls gained)
        'prsn',  # snow/ice precipitation for transport (Ls gained, use to isolate Lv)
        'evspsbl',  # evaporation/transpiration/sublimation for transport (Lv+Ls lost)
        'sbl',  # sublimation for transport (Ls lost, use to isolate Lv)
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
kw_explicit = {
    'table': 'Emon',
    'experiment': ['piControl', 'abrupt-4xCO2'],
    'variable': [
        'uv',  # HighResMIP (vertically integrate to get EMF)
        'u2',  # HighResMIP (vertically integrate to get EKE)
        'v2',  # HighResMIP (vertically integrate to get EKE)
        'ut',  # HighResMIP (vertically integrate and combine with 'v' * 'z' to get DSE)
        'vt',  # HighResMIP (vertically integrate and combine with 'v' * 'z' to get DSE)
        'uqint',  # HighResMip (already vertically integrated for MSE)
        'vqint',  # HighResMIP (already vertically integrated for MSE)
        'intuadse',  # PMIP limited availability
        'intvadse',  # PMIP limited availability
        'intuaw',  # PMIP limited availability
        'intvaw',  # PMIP limited availability
    ]
}

# Download the wget files
# NOTE: Facets and options supplied to constrain filter are standardized between
# cmip5 and cmip6 synonyms. So can loop through both projects with about keywords.
dicts = []
if feedbacks:
    dicts.append(kw_feedbacks)
if constrain:
    dicts.append(kw_constrain)
if circulation:
    dicts.append(kw_circulation)
if explicit:
    dicts.append(kw_explicit)
if download:
    unfiltered = []
    for kwargs in dicts:
        projects = ('CMIP6',) if kwargs is kw_explicit else ('CMIP6', 'CMIP5')
        for project in projects:
            folder = '~/scratch2' if project == 'CMIP5' else '~/scratch5'
            script = cmip_data.download_script(
                folder,
                project=project,
                node='llnl',
                openid='https://esgf-node.llnl.gov/esgf-idp/openid/lukelbd',
                username='lukelbd',
                flagship_translate=True,
                **kwargs,
            )
            unfiltered.append(script)

# Filter the resulting wget files
# NOTE: Here only want to bother downloading constraint data with an equivalent
# abrupt simulation (indicating feedbacks are available) but don't actually want to
# download this data so use the exclude fitler to avoid it. Also don't want abrupt
# pfull data since not always available. And need to include control pfull data from
# other tables/experiments (see process.py comments) so use always_include.
if filter:
    filtered = []
    for kwargs in dicts:
        projects = ('CMIP6',) if kwargs is kw_explicit else ('CMIP6', 'CMIP5')
        for project in projects:
            folder = '~/scratch2' if project == 'CMIP5' else '~/scratch5'
            models_skip = (
                'FGOALS-g2',  # sigma coordinates treated as hybrid
                'FGOALS-g3',  # sigma coordinates treated as hybrid
                'E3SM-1-1',  # does not have abrupt simulation
                'E3SM-1-1-ECA',  # does not have abrupt simulation
                'IPSL-CM5A2',  # special hybrid coordinates interpolated
                'IPSL-CM5A2-INCA',  # special hybrid coordinates interpolated
                'IPSL-CM6A-LR',  # special hybrid coordinates interpolated
                'IPSL-CM6A-LR-INCA',  # special hybrid coordinates interpolated
            )
            scripts = cmip_data.filter_script(
                folder,
                project=project,
                maxyears=150,
                endyears=False,
                always_include={'variable': 'pfull'},  # i.e. bypass intersection
                always_exclude=(
                    {'variable': 'pfull', 'model': models_skip},
                    {'variable': kw_constrain['variable'], 'experiment': 'abrupt-4xCO2'}
                ),
                flagship_translate=True,
                **kwargs
            )
            filtered.extend(scripts)

# Average and standardize the resulting files
# NOTE: Here follow Armour et al. 2019 methodology of taking difference between final
# 30 years of the 150 years required by the DECK abrupt-4xco2 experiment protocol. Also
# again exclude constraint data from processing (pfull is excluded in process_files).
# See: https://pcmdi.llnl.gov/CMIP6/Guide/dataUsers.html
if process:
    for nodrift in (True,):
        for kwargs in dicts:
            projects = ('CMIP6',) if kwargs is kw_explicit else ('CMIP6', 'CMIP5')
            for project in projects:
                folder = '~/scratch2' if project == 'CMIP5' else '~/scratch5'
                experiments = {'piControl': 150, 'abrupt-4xCO2': (120, 150)}
                if kwargs is kw_constrain:
                    del experiments['abrupt-4xCO2']
                if not climate:
                    experiments.clear()
                for experiment, years in experiments.items():
                    kw = {**kwargs, 'experiment': experiment}
                    cmip_data.process_files(
                        folder,
                        output='~/data',
                        project=project,
                        nodrift=nodrift,
                        climate=True,
                        overwrite=False,
                        dryrun=False,
                        years=years,
                        **kw,
                    )
                experiments = {'abrupt-4xCO2': 150, 'piControl': 150}
                if kwargs is not kw_feedbacks:
                    experiments.clear()
                if not series:
                    experiments.clear()
                for experiment, years in experiments.items():
                    kw = {**kwargs, 'experiment': experiment}
                    cmip_data.process_files(
                        folder,
                        output='~/scratch2/data-series',
                        project=project,
                        nodrift=nodrift,
                        climate=False,
                        overwrite=False,
                        dryrun=False,
                        years=years,
                        **kw,
                    )

# Calculate the correlation, regression, and ratio feedbacks
if process and feedbacks:
    pass

# Update the summary logs once finished
# NOTE: The missing files for different feedback variables are tabulated below.
# Checked the filter logs and same files are missing. Then checked the download logs
# and for CMIP5 there were no failed dataset downloads but for CMIP6 there were several
# that sometimes corresponded to missing files. Consider manually searching the
# web interface or individually going onto model center websites. Currently should
# consider fixing CMIP5 GFDL and FGOALS and CMIP6 FIO to get tropopause feedbacks, and
# then NorESM2 and TaiESM1 (and possibly CAMS and GFDL-ESM4) to get surface feedbacks.
# CMIP5 missing feedback models (copied 2022-05-08):
# rsut: GFDL-CM3, GFDL-ESM2G, GFDL-ESM2M
# rsutcs: GFDL-CM3, GFDL-ESM2G, GFDL-ESM2M
# rsus: GFDL-CM3, GFDL-ESM2G, GFDL-ESM2M
# rsuscs: FGOALS-g2, GFDL-CM3, GFDL-ESM2G, GFDL-ESM2M
# CMIP6 missing feedback models (copied 2022-05-08):
# rsdt: MCM-UA-1-0
# rlutcs: MCM-UA-1-0
# rsut: MCM-UA-1-0
# rsutcs: FIO-ESM-2-0, MCM-UA-1-0
# rlus: MCM-UA-1-0
# rsus: MCM-UA-1-0
# rsuscs: CAMS-CSM1-0, GFDL-ESM4, MCM-UA-1-0
# rlds: MCM-UA-1-0
# rldscs: CAMS-CSM1-0, GFDL-ESM4, MCM-UA-1-0, NorESM2-LM, NorESM2-MM, TaiESM1
# rsds: MCM-UA-1-0
# rsdscs: CAMS-CSM1-0, GFDL-ESM4, MCM-UA-1-0
if False:
    for project in ('cmip6', 'cmip5'):
        kw = {'project': project, 'flagship_translate': True}
        folders_downloads = ('~/scratch2', '~/scratch5')
        folders_processed = ('~/data', '~/scratch2/data-series')
        cmip_data.summarize_downloads(*folders_downloads, **kwargs)
        cmip_data.summarize_processed(*folders_processed, **kwargs)
