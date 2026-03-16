# Compound Flooding Research Agent Brief

## Purpose
Use the seed paper below as a starting point to generate a **Stage-1 research proposal** on the **mechanisms of compound flooding**, with an emphasis on **upper transition zone (UTZ) / estuarine-riverine systems** where flood hazard may shift from fluvial-dominated to coastal-compound dominated under climate change.

This is **not** an exact replication task.

The immediate goal is to help a climate-science autonomous agent:
1. understand the seed paper deeply,
2. map the related literature,
3. audit reproducibility and data availability,
4. identify research gaps,
5. formulate new, testable hypotheses, and
6. produce a strong **research proposal** that could later evolve into a Nature-style manuscript.

---

## Seed paper
**Primary seed paper**
- Mita, K. S., Orton, P., Montalto, F., & Anbessie, T. (2025). *Accumulating climate change influences on extreme coastal, fluvial, and compound flooding in the upper transition zone*. Journal of Hydrology, 663, 134247. https://doi.org/10.1016/j.jhydrol.2025.134247

### What the seed paper does
The paper studies Eastwick (southwest Philadelphia), a neighborhood near the inland limit of a tidal estuarine-riverine system, and asks how future extreme flood hazard changes when multiple climate change influences act on multiple flood drivers.

Its framework is intentionally simplified and conservative:
- three event types: **fluvial flood (FF)**, **storm surge / coastal flood (SSF)**, and **compound flood (CF)**,
- three time horizons: **baseline**, **mid-century**, **late-century**,
- climate change influences (CCIs): **rainfall increase**, **sea-level rise (SLR)**, and **storm climatology / storm-surge increase**,
- model chain: **1D watershed hydrology (PCSWMM)** feeding **2D flood hydraulics (HEC-RAS 2D)** through one-way coupling.

### Core findings to treat as the seed paper's main claims
1. In the UTZ, future hazard can shift from mainly fluvial to increasingly coastal-compound.
2. Considering only one climate driver can materially underestimate future flood extent and depth.
3. Compounding in this case is not only a river-channel story; it can accumulate in nearby floodplain storage areas and then enter the built neighborhood through vulnerable pathways.
4. Local topography and connectivity strongly mediate whether floodplains buffer or worsen flood hazard.

### Why this paper is a good starting point
It is a good **mechanism-seeking seed paper**, not because it solves everything, but because it gives a clean structure for asking new questions:
- What is the physics of compounding in UTZ systems?
- When does a floodplain act as protection vs. a conduit?
- How much of the result depends on event construction choices (e.g., peak alignment)?
- Which insights are site-specific and which are transferable?

---

## Recommended stance for this project
Treat the seed paper as:
- a **conceptual anchor**,
- a **starting taxonomy of flood drivers and event types**,
- a **candidate mechanism paper**,
- and a **launchpad for a new proposal**,

but **not** as the single truth or as a result to imitate mechanically.

The agent should explicitly distinguish among:
- **paper-supported facts**,
- **reasonable inferences**,
- **open research questions**, and
- **new hypotheses proposed in this project**.

---

## Stage-1 objective
Produce a **research proposal** for a new study on compound flooding mechanisms that is:
- scientifically credible,
- literature-grounded,
- explicit about reproducibility constraints,
- realistic about data access,
- and oriented toward a later experimental phase.

### Stage-1 outputs must answer
1. What exactly is the research problem?
2. Why is it important scientifically and practically?
3. What does the seed paper already establish?
4. What remains unresolved or weakly tested?
5. What new hypotheses are worth pursuing?
6. Which future directions are feasible with open or request-access data?
7. What would a rigorous Phase-2 study look like?

---

## Hard constraints for Stage-1
### Do not do yet
- Do **not** claim exact reproduction of the seed paper.
- Do **not** assume the calibrated model files are publicly available.
- Do **not** depend on paid software or paid data unless clearly labeled as optional.
- Do **not** run expensive end-to-end hydrodynamic experiments in this stage.
- Do **not** present speculative mechanism claims as established facts.

### Required behavior
- Read the seed paper carefully.
- Build a focused literature review around it.
- Audit what is open, what is on-request, and what is likely proprietary or locally assembled.
- Generate new hypotheses and rank them by novelty, tractability, and likely scientific value.
- Prefer transferable ideas over narrow case-specific observations.

---

## What the agent should understand before proposing anything

### A. Compound flooding is not just “two bad things at once”
The agent should understand that compound flooding involves:
- multiple flood drivers,
- possible dependence between those drivers,
- timing / phasing effects,
- nonlinear hydraulic interactions,
- and pathway activation that can differ from single-driver floods.

### B. UTZ systems are special
The agent should treat **upper transition zones** as places where:
- coastal influence is present but not always dominant,
- river slope, floodplain shape, and connectivity matter strongly,
- sea-level rise may alter the flood regime in a threshold-like way,
- the location of compounding may occur in floodplains or storage areas rather than the main channel itself.

### C. Mechanism questions matter as much as hazard maps
The project is not only about “how much more area floods?”
It is also about:
- **where the water stores**,
- **how pathways open**,
- **when downstream boundary conditions trap upstream runoff**,
- **which barriers / berms / ditches / rail embankments act as gates**,
- and **how future climate drivers change those processes**.

---

## Quick reproducibility and data-access audit
This section should be treated as a starting point and independently verified by the agent.

| Item | Likely status | Notes |
|---|---|---|
| Seed paper PDF | Open-access paper | The article is open access, but not all supporting materials are bundled in a public repo. |
| Seed paper data package | **On request** | The paper states: “Data will be made available on request.” |
| PCSWMM model | Proprietary / commercial software | Useful but not ideal as a hard dependency for reproducible autonomous research. |
| EPA SWMM engine | Open source / free | A possible substitute or benchmark path if a similar hydrologic workflow is needed later. |
| HEC-RAS | Publicly available / free to download | Useful for Phase-2 if hydraulic simulation is pursued. |
| NOAA Atlas 14 / PFDS precipitation frequency data | Public | Atlas 15 is emerging, so the agent should note versioning when designing future work. |
| NOAA CO-OPS water level / tide data | Public | Appropriate for tide / water-level / non-tidal residual analysis. |
| USGS streamgage data (NWIS) | Public | Appropriate for discharge / stage / historical hydrograph analysis. |
| PASDA / Pennsylvania geospatial layers | Public | Includes open geospatial data, including elevation resources. |
| NLCD / Annual NLCD | Public | Land cover / imperviousness for roughness and exposure context. |
| NEXRAD radar archive | Public | Candidate rainfall input source if spatial rainfall is needed later. |
| Local field surveys / bridge-culvert geometry / custom bathymetry | Unclear or request-based | Often the real bottleneck for faithful reproduction. |
| Calibrated Eastwick model files used in the paper | Not obviously public | Assume unavailable unless located explicitly. |

### Practical implication
For Stage-1, **exact Eastwick reproduction is not the optimal objective**.
A better objective is to produce a research proposal that is:
- informed by Eastwick,
- aware of the data bottlenecks,
- and designed so that a Phase-2 study can either:
  1. use Eastwick with request-access data, or
  2. pivot to a more open benchmark site.

---

## Main research question for the proposal
**How do multiple climate change influences interact with local topography, timing, and floodplain connectivity to generate compound flooding in upper transition zone systems, and when do these interactions produce flood regimes that cannot be inferred from single-driver analysis?**

### Supporting questions
1. **Mechanism:** What specific physical mechanisms create compound amplification in UTZ settings?
2. **Thresholds:** Is there a threshold-like transition from fluvial-dominated hazard to coastal-compound hazard under SLR?
3. **Timing:** How sensitive are results to the relative timing of rainfall-runoff peaks and coastal water-level peaks?
4. **Floodplain role:** Under what conditions does floodplain storage mitigate flooding vs. reroute it toward developed areas?
5. **Transferability:** Which findings from Eastwick are likely general across UTZ systems?
6. **Methodology:** What is the right balance between deterministic stress tests and probabilistic / nonstationary compound analysis?
7. **Reproducibility:** Can similar qualitative insights be recovered with a more open toolchain and more openly accessible data?

---

## What the agent should extract from the seed paper
The agent should produce a structured decomposition of the paper with the following fields:

### 1. Problem framing
- What gap does the paper claim to fill?
- What is new relative to prior coastal-only or fluvial-only flood studies?
- How does the paper define the UTZ concept?

### 2. Event design choices
- How are FF, SSF, and CF defined?
- Which are historical events and which are synthetic / designed?
- How is the 100-year label used, and where could that label be misleading?
- What assumptions are introduced by peak alignment in the compound event?

### 3. Climate change assumptions
- How are rainfall increase, SLR, and storm climatology change represented?
- Are these deterministic stress tests or probabilistic projections?
- What is gained and lost by this setup?

### 4. Modeling architecture
- Why use one-way coupling?
- What does 1D hydrology contribute and what does 2D hydraulics contribute?
- Which physical processes are omitted or simplified?

### 5. Mechanism claims
- Where exactly does the paper say compounding occurs?
- What evidence supports “floodplain accumulation rather than channel-only compounding”?
- What role is assigned to JHR / storage / overflow pathways?

### 6. Limitations
At minimum, the agent should assess:
- event construction choices,
- deterministic scenario design,
- fixed land use assumption,
- omitted drainage-system details,
- boundary placement assumptions,
- calibration transparency,
- availability of local survey data,
- and external validity beyond Eastwick.

---

## Literature review instructions
The literature review should be **tight, mechanism-oriented, and selective**, not broad and generic.

### Priority literature buckets
#### Bucket 1: Direct lineage / local context
Start with the seed paper's direct methodological and conceptual lineage.
Suggested starting items:
- Mita et al. (2025) - seed paper
- Mita et al. (2023) - Eastwick sea-level-rise / transition-to-compounding study
- Nasrollahi (2024) - dissertation with additional modeling details

#### Bucket 2: Compound flooding reviews and syntheses
Build a compact but high-quality picture of the field.
Suggested starting items:
- Green et al. (2025) - comprehensive review of compound flooding in coastal and estuarine regions
- Relevant review / synthesis on compound pluvial-fluvial flooding if useful for taxonomy and methodology

#### Bucket 3: Estuary / tidal-river / transition-zone sensitivity studies
Look for work on:
- river-surge interaction,
- estuarine sensitivity to SLR,
- flood zone transitions,
- spatial heterogeneity inside estuaries,
- and changing driver dominance along estuarine gradients.
Suggested starting items:
- Orton et al. (2020)
- Ghanbari et al. (2021)
- Sensitivity-of-estuaries and threshold / timing papers

#### Bucket 4: Timing / lag / phasing papers
This is especially important because the seed paper uses peak alignment to construct a maximum compound event.
The agent should ask:
- Is worst flooding always produced by exact peak coincidence?
- How does lag sensitivity vary by basin scale, estuary shape, and floodplain connectivity?

#### Bucket 5: Modeling framework papers
Focus on:
- one-way vs. two-way coupling,
- statistical + hydraulic hybrid methods,
- event-based vs. continuous simulation,
- deterministic stress testing vs. probabilistic frameworks,
- open vs. proprietary toolchains.

#### Bucket 6: Human-modified floodplain / pathway activation studies
Look for papers on:
- embankments,
- rail corridors,
- berms,
- levees,
- culverts,
- floodplain restoration,
- and topographic controls on rerouting floodwater.

### Required literature review product
Create a table with columns:
- citation,
- study region,
- flood-driver types,
- event definition,
- model / method,
- climate treatment,
- main physical mechanism,
- main limitation,
- relevance to UTZ mechanism research,
- relevance to an open-data follow-up study.

---

## Domain knowledge the agent should actively use
The agent should reason with the following concepts explicitly rather than implicitly.

### Hydrologic / hydraulic concepts
- rainfall-runoff transformation
- hydrograph generation
- downstream boundary control / backwater effect
- tidal propagation
- storm tide vs. non-tidal residual
- floodplain storage and retention
- overflow / overtopping pathways
- Manning roughness and terrain control
- wetting-drying dynamics
- phasing / lag between drivers

### Compound-event concepts
- dependence vs. independence
- joint exceedance and joint return period
- nonstationarity under climate change
- driver dominance and regime shift
- threshold behavior and pathway activation
- additive vs. non-additive climate-change effects

### Climate-change concepts
- sea-level rise
- changes in extreme precipitation
- storm climatology change
- scenario stress testing vs. probabilistic ensemble analysis
- local vs. regional projections
- uncertainty communication

### Geomorphic / estuarine concepts
- estuary / tidal-river transition
- longitudinal gradient in driver dominance
- slope effects
- floodplain connectivity
- geomorphic and anthropogenic controls on routing
- topographic choke points / storage cells / gates

---

## Hypothesis generation task
The agent should generate **at least 5 candidate hypotheses**, then rank the top 3.

### Strong candidate directions
These are not mandatory final answers; they are starting points.

#### Hypothesis family 1: Floodplain-mediated compounding
**H1.** In UTZ systems, the strongest compounding signal may emerge in connected floodplain storage areas rather than in the main channel, especially where local topography creates storage-and-release behavior.

#### Hypothesis family 2: Threshold-like hazard transition
**H2.** There exists a site-specific SLR threshold beyond which fluvial-dominated extreme flooding transitions rapidly toward coastal-compound dominance, and the threshold is controlled by channel slope, floodplain storage capacity, and anthropogenic barriers.

#### Hypothesis family 3: Timing matters more than peak coincidence alone
**H3.** Maximum inundation in UTZ systems is not always produced by exact coincidence of peak discharge and peak coastal water level; lag structure can produce larger flooding depending on rising-tide trapping, travel times, and floodplain filling dynamics.

#### Hypothesis family 4: Moderate-moderate compounding can outperform extreme-single-driver events
**H4.** In some UTZ settings, moderate but dependent sea-level and river-flow conditions may produce broader flooding than a more extreme single driver, especially in middle-estuary / transition reaches.

#### Hypothesis family 5: Open workflow can recover transferable mechanisms
**H5.** A more open and reproducible modeling pipeline can recover the main qualitative mechanism claims of proprietary Eastwick workflows even if exact flood extents differ.

#### Hypothesis family 6: Human-built connectivity controls future pathway activation
**H6.** Rail corridors, berms, ditches, culverts, and partially enclosed floodplains act as hidden control points that determine whether future CCIs are buffered locally or redirected into urban neighborhoods.

### Hypothesis ranking criteria
Rank each hypothesis by:
- novelty,
- scientific importance,
- feasibility in Phase-2,
- dependence on proprietary or inaccessible data,
- transferability beyond Eastwick,
- and suitability for a strong paper.

---

## Expected critique of the seed paper
The proposal should not be timid. It should identify what is strong and what is fragile.

### Strengths to acknowledge
- clear event taxonomy,
- practical engineering relevance,
- useful decomposition into separate vs. combined CCIs,
- strong emphasis on spatial mechanism,
- transferable screening logic.

### Fragilities to interrogate
- the “100-year” labels are not identical across event types,
- the fluvial event is based on a design rainfall rather than a directly estimated 100-year discharge,
- the compound event uses a worst-case peak-alignment construction,
- the climate scenarios are deterministic and conservative rather than probabilistic,
- model reproducibility depends on local calibration and possibly local survey data,
- land use and local drainage evolution are held fixed,
- and the framework is more of a screening analysis than a full multivariate risk assessment.

The point is not to dismiss the paper. The point is to identify the best next questions.

---

## Recommended Phase-1 workflow for the agent

### Task 1. Paper dissection
Produce a structured memo that identifies:
- claims,
- assumptions,
- event definitions,
- data dependencies,
- model dependencies,
- mechanism claims,
- and unresolved questions.

### Task 2. Reproducibility audit
For every major input or method component, classify it as:
- open and easily accessible,
- open but labor-intensive,
- available on request,
- proprietary software dependent,
- or unclear.

### Task 3. Focused literature review
Build the literature matrix described above.
Do not optimize for paper count; optimize for relevance.

### Task 4. Mechanism map
Produce a conceptual diagram or structured description of the hypothesized mechanism pathways:
- channel backwater,
- floodplain storage,
- overflow pathway activation,
- timing / lag sensitivity,
- and future driver intensification.

### Task 5. Research gap synthesis
Identify 3-5 sharply stated gaps, for example:
- timing sensitivity remains weakly explored,
- floodplain pathway activation is insufficiently generalized,
- deterministic scenario design misses uncertainty in dependence changes,
- and reproducible open-data UTZ benchmarks are scarce.

### Task 6. Hypothesis generation and ranking
Generate, score, and select the top hypotheses.

### Task 7. Proposal drafting
Draft a research proposal that could plausibly become a high-impact paper if Phase-2 succeeds.

---

## What Phase-2 could look like (proposal only, do not execute yet)
The Stage-1 proposal should outline one or more concrete Phase-2 study designs.

### Candidate Phase-2 pathway A: Eastwick-centered follow-up
Use Eastwick as the primary system if enough request-access or open supporting data can be obtained.

Potential components:
- reconstruct open substitutes for paper inputs where possible,
- obtain missing local geometry / bathymetry if feasible,
- test lag sensitivity instead of only exact peak alignment,
- compute a synergy metric for separate vs. combined CCIs,
- examine threshold SLR levels for pathway activation,
- compare deterministic stress tests with probabilistic event sampling.

### Candidate Phase-2 pathway B: Open benchmark UTZ site
If Eastwick proves too dependent on inaccessible local data, identify a second site with:
- open DEM / bathymetry / boundary data,
- long tide and streamgage records,
- documented floodplain connectivity,
- and enough event history to support both statistics and dynamics.

### Candidate Phase-2 pathway C: Mechanism-first idealized modeling
Build an idealized UTZ model to isolate mechanism.
This may be especially powerful if the primary goal is to understand:
- floodplain storage thresholds,
- slope/connectivity controls,
- and timing/lag effects.

### Candidate methodological additions for Phase-2
- nonstationary copula or joint-probability analysis,
- lag-conditioned event sets,
- global sensitivity analysis,
- open-source hydrologic-hydraulic workflow benchmarking,
- pathway activation mapping,
- and transferability analysis across multiple UTZ systems.

---

## Deliverables required from the agent
The agent should produce the following outputs in Stage-1.

### Deliverable 1. Executive summary
A short, decision-oriented note answering:
- What should we study next?
- Why this direction?
- What is feasible now?
- What is blocked by data or software?

### Deliverable 2. Seed paper audit memo
A compact technical decomposition of the seed paper.

### Deliverable 3. Literature matrix
A structured table of the most relevant prior work.

### Deliverable 4. Reproducibility and data-feasibility memo
Explicitly list:
- open data,
- request-only data,
- proprietary software dependencies,
- and likely substitutes.

### Deliverable 5. Ranked hypotheses
At least 5 candidate hypotheses, top 3 prioritized.

### Deliverable 6. Research proposal
The proposal should include:
- title,
- abstract,
- background and significance,
- gap statement,
- research questions,
- hypotheses,
- proposed datasets,
- proposed methods,
- expected results / contributions,
- risks and contingencies,
- and a milestone plan.

### Deliverable 7. Nature-style manuscript blueprint
Produce a draft outline with sections such as:
- Title
- Abstract
- Introduction
- Results (planned result logic, not fabricated findings)
- Discussion
- Methods
- Data and code availability plan
- References

Important: the agent must **not invent results**. The Nature-style outline should describe the intended study architecture, not pretend the work is already complete.

---

## Suggested structure for the final proposal

### Title
A mechanism-focused title, not just a site title.
Example style:
- *Floodplain-mediated compound flooding in upper transition zones under accumulating climate stressors*
- *Thresholds, timing, and connectivity in compound flooding of estuarine-riverine transition systems*

### Abstract
Keep it hypothesis-driven and mechanism-oriented.

### Introduction
Should cover:
- why compound flooding matters,
- why UTZ systems are under-studied,
- why separate-driver analysis is insufficient,
- and what unresolved mechanism question the project targets.

### Research gap
State the gap sharply.
For example:
- “Existing UTZ flood studies show increasing compounding under climate change, but the mechanisms governing when floodplains buffer versus amplify compound flooding remain weakly generalized and poorly tested against timing sensitivity.”

### Research questions and hypotheses
List them clearly and testably.

### Data strategy
Separate data into:
- mandatory,
- optional,
- request-access,
- and substitute datasets.

### Methods
The proposal should state both:
- a **minimum viable Phase-2 workflow**, and
- a **best-case enhanced workflow**.

### Expected contributions
Possible contribution categories:
- new mechanism insight,
- new UTZ conceptual model,
- improved compound-event design,
- open/reproducible workflow,
- transferable screening framework,
- policy relevance for floodplain management.

### Risk and contingency plan
This section is mandatory.
If Eastwick data access fails, what is the backup plan?

---

## Evaluation rubric for the agent's own work
The agent should score its own proposal on:
1. scientific significance,
2. clarity of mechanism,
3. literature grounding,
4. novelty,
5. feasibility with accessible data,
6. dependence on proprietary tools,
7. transferability beyond one site,
8. and suitability for a strong paper.

The agent should explain trade-offs explicitly.

---

## Recommended starter reading list
### Core local / seed lineage
1. Mita, K. S., Orton, P., Montalto, F., & Anbessie, T. (2025). *Accumulating climate change influences on extreme coastal, fluvial, and compound flooding in the upper transition zone*. Journal of Hydrology, 663, 134247. https://doi.org/10.1016/j.jhydrol.2025.134247
2. Mita, K. S., Orton, P., Montalto, F., & Anbessie, T. (2023). *Sea Level Rise-Induced Transition from Rare Fluvial Extremes to Chronic and Compound Floods*. Water, 15(14), 2671. https://doi.org/10.3390/w15142671
3. Nasrollahi, F. (2024). *Modeling the effectiveness of flood adaptation strategies under climate change* (PhD dissertation, Drexel University). https://doi.org/10.17918/00010472

### Compound flooding review / synthesis
4. Green, J., Haigh, I. D., Quinn, N., Neal, J., Wahl, T., Wood, M., Eilander, D., de Ruiter, M., Ward, P., & Camus, P. (2025). *A comprehensive review of compound flooding literature with a focus on coastal and estuarine regions*. Natural Hazards and Earth System Sciences, 25, 747-816. https://doi.org/10.5194/nhess-25-747-2025

### Estuary / transition / flood hazard change
5. Orton, P. M., Conticello, F. R., Cioffi, F., Hall, T. M., Georgas, N., Lall, U., Blumberg, A. F., & MacManus, K. (2020). *Flood hazard assessment from storm tides, rain and sea level rise for a tidal river-estuary*. Natural Hazards, 102, 729-757. https://doi.org/10.1007/s11069-018-3251-x
6. Ghanbari, M., Arabi, M., Kao, S.-C., Obeysekera, J., & Sweet, W. (2021). *Climate Change and Changes in Compound Coastal-Riverine Flooding Hazard Along the U.S. Coasts*. Earth's Future, 9(5). https://doi.org/10.1029/2021EF002055
7. Harrison, T., et al. (2024). *Thresholds for estuarine compound flooding using a combined hydrodynamic-statistical modelling approach*. Natural Hazards and Earth System Sciences, 24, 973-1002. https://doi.org/10.5194/nhess-24-973-2024
8. Dykstra, S. L., & Dzwonkowski, B. (2021/2022 lineage if relevant to lag sensitivity; verify exact citation before final use in proposal)

### Optional supporting directions
9. Papers on storm-type-conditioned dependence, lag sensitivity, and moderate-moderate compounding
10. Papers on open-source hydrologic-hydraulic coupling and floodplain connectivity under climate change

---

## Suggested official data and tool sources to verify
- NOAA precipitation frequency data / Atlas 14 / PFDS
- NOAA CO-OPS water levels and tides
- USGS NWIS streamgage services
- PASDA / Pennsylvania geospatial portal
- USGS / MRLC Annual NLCD
- NOAA NCEI NEXRAD archive
- USACE HEC-RAS download and documentation
- EPA SWMM
- PCSWMM official product documentation

---

## Final instruction to the agent
Use the seed paper to identify a **new, mechanism-driven research direction** on compound flooding in upper transition zones.

Your job in Stage-1 is to determine:
- **what is scientifically worth doing next**,
- **what is actually feasible**,
- **which hypotheses deserve Phase-2 testing**,
- and **how to structure that work into a strong research proposal**.

Do not optimize for volume of literature.
Optimize for:
- precision,
- mechanism,
- transferability,
- reproducibility,
- and decision usefulness.
