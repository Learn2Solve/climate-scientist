# Domain Reference: Compound Flooding in UTZ Systems

This file is reference material for the agent. Read on-demand, not required upfront.

---

## Hydrologic / hydraulic concepts
- Rainfall-runoff transformation and hydrograph generation
- Downstream boundary control / backwater effect
- Tidal propagation and storm tide vs. non-tidal residual
- Floodplain storage, retention, and overflow/overtopping pathways
- Manning roughness and terrain control on flow routing
- Wetting-drying dynamics in 2D shallow water models
- Phasing / lag between flood drivers

## Compound-event concepts
- Dependence vs. independence of flood drivers
- Joint exceedance and joint return period
- Nonstationarity under climate change
- Driver dominance and regime shift
- Threshold behavior and pathway activation
- Additive vs. non-additive climate-change effects (synergy/antagonism)

## Climate-change concepts
- Sea-level rise (local vs. global, rate uncertainty)
- Changes in extreme precipitation (intensity-duration-frequency shifts)
- Storm climatology change (frequency, intensity, track, size)
- Scenario stress testing vs. probabilistic ensemble analysis
- Uncertainty communication and decision-relevant framing

## Geomorphic / estuarine concepts
- Estuary / tidal-river transition: where coastal influence fades inland
- Longitudinal gradient in driver dominance
- Slope effects on tidal damping and backwater reach
- Floodplain connectivity: lateral vs. longitudinal
- Geomorphic and anthropogenic controls on flow routing
- Topographic choke points, storage cells, gates (berms, rail embankments, culverts)

## UTZ-specific considerations
The "upper transition zone" is the inland reach of an estuary where:
- Coastal influence is present but not always dominant
- River slope, floodplain shape, and connectivity matter strongly
- SLR may alter the flood regime in threshold-like ways
- Compounding may occur in floodplains or storage areas rather than the main channel
- The boundary between "fluvial problem" and "coastal problem" is moving inland

## Seed paper's modeling approach
- PCSWMM: proprietary 1D hydrology (rainfall -> runoff -> hydrograph)
- HEC-RAS 2D: free 2D hydraulic model (flood routing over terrain)
- One-way coupling: hydrology feeds hydraulics but hydraulics does not feed back
- This omits: drainage-system backing up, two-way tide-river interaction in channel
- Open alternatives: EPA SWMM (free 1D hydrology), HEC-RAS (free), LISFLOOD-FP (open 2D)

## Data landscape for Eastwick / Delaware estuary
- NOAA CO-OPS: tide and water level records (Philadelphia, Marcus Hook, etc.)
- USGS NWIS: streamgage data (Darby Creek, Cobbs Creek, Schuylkill)
- NOAA Atlas 14/PFDS: precipitation frequency data (Atlas 15 emerging)
- PASDA: Pennsylvania geospatial layers including elevation
- NLCD/Annual NLCD: land cover and imperviousness
- NEXRAD: radar rainfall archive
- Local survey data (bridge-culvert geometry, bathymetry): likely request-only or unavailable
