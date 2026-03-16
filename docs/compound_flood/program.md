# Compound Flood Research - autoresearch

This is an autonomous research task. The agent reads reference materials, conducts
literature research, and produces structured deliverables for a research proposal on
compound flooding mechanisms in upper transition zone (UTZ) systems.

The seed paper is Mita et al. (2025), "Accumulating climate change influences on
extreme coastal, fluvial, and compound flooding in the upper transition zone."

## Setup

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar15`).
   The branch `autoresearch/compound-flood` should already exist.
2. **Read the reference materials** (all immutable, do not modify):
   - `docs/compound_flood/ref_brief.md` -- the full research brief with all requirements
   - `docs/compound_flood/ref_domain.md` -- domain knowledge: hydrology, compound events, UTZ
   - `docs/compound_flood/ref_literature.md` -- seed literature and search directions
3. **Initialize results.tsv**: Create `docs/compound_flood/results.tsv` with the header row.
4. **Confirm and go**.

Once you get confirmation, begin producing deliverables.

## Reference material summary

You must read the full `ref_brief.md` for detailed requirements. The short version:

**Seed paper (Mita et al. 2025)**:
- Site: Eastwick, Philadelphia - near the inland limit of a tidal estuarine-riverine system
- Framework: 3 event types (FF, SSF, CF) x 3 time horizons x 3 climate drivers
- Model chain: 1D hydrology (PCSWMM) -> 2D hydraulics (HEC-RAS 2D), one-way coupling
- Core claim: in UTZ systems, future hazard shifts from fluvial to coastal-compound
- Key limitation: deterministic stress tests, peak-aligned compound event, local calibration

**What makes this paper a good seed**:
- Clean structure for asking mechanism questions
- Identifies floodplain-mediated compounding (not just channel interaction)
- Highlights topographic and connectivity controls
- Raises questions about transferability beyond one site

## The deliverables

Produce 7 deliverables, in order. Each builds on previous ones.
All outputs go in `docs/compound_flood/output/`.

### D1: Seed paper audit
File: `01_paper_audit.md`

Structured decomposition:
1. Problem framing: what gap does the paper claim to fill?
2. Event design: how are FF/SSF/CF defined? What does "100-year" mean for each?
3. Climate assumptions: deterministic stress tests - what is gained, what is lost?
4. Modeling architecture: why one-way coupling? What is omitted?
5. Mechanism claims: where does compounding occur? What evidence supports it?
6. Strengths: at least 3 methodological strengths
7. Fragilities: at least 5 things that are fragile or weakly tested

Label everything: [paper-stated] / [inferred] / [open question].

**Pass criteria**: at least 5 fragilities identified; mechanism claims decomposed with
evidence assessment; clear distinction between paper-stated facts and inferences.

### D2: Reproducibility and data-access audit
File: `02_reproducibility_memo.md`

For every major input/method component, classify as:
- Open and easily accessible (with URL/source)
- Open but labor-intensive
- Available on request
- Proprietary software dependent
- Unclear or likely unavailable

Cover: precipitation data, water level/tide data, streamgage data, DEM/elevation,
land cover, radar rainfall, model software (PCSWMM, HEC-RAS, EPA SWMM), local
survey data, calibrated model files.

State the practical implication: what can a follow-up study actually do?

**Pass criteria**: every item in the brief's data table (ref_brief.md, "Quick
reproducibility" section) is addressed; at least one open substitute is proposed
for each proprietary dependency.

### D3: Literature matrix
File: `03_literature_matrix.md`

Search for and organize papers across 6 buckets:
1. Direct lineage / local context
2. Compound flooding reviews and syntheses
3. Estuary / tidal-river / transition-zone sensitivity
4. Timing / lag / phasing
5. Modeling frameworks (1-way vs 2-way coupling, open vs proprietary)
6. Human-modified floodplain / pathway activation

Table columns: citation | study region | flood-driver types | event definition |
model/method | climate treatment | main physical mechanism | main limitation |
relevance to UTZ research | relevance to open-data follow-up

Target: 15-25 papers. Optimize for relevance, not count.
Mark citation confidence: [verified] / [to verify] / [uncertain].

Use web search tools (Google Scholar, Semantic Scholar, OpenAlex) when available.
If web search is unavailable, work from ref_literature.md and cite additional papers
you are confident exist from your training data, marking them [to verify].

**Pass criteria**: at least 12 papers in the matrix; all 6 buckets represented;
at least 3 papers not already in ref_literature.md.

### D4: Mechanism map
File: `04_mechanism_map.md`

Structured description of compound flooding mechanism pathways in UTZ systems.
This is not a diagram - it is a text-based conceptual model.

For each mechanism pathway:
- Description of the physical process
- Under what conditions it activates
- Which drivers and climate change influences control it
- Where in the UTZ it is most relevant
- What determines whether it amplifies or buffers flooding
- Connection to the seed paper's findings (if any)

Minimum pathways to cover:
1. Channel backwater from coastal water levels
2. Floodplain storage filling and overflow
3. Pathway activation through anthropogenic features (berms, rail, culverts)
4. Timing/lag interaction between fluvial peak and coastal peak
5. Threshold-like regime shift under SLR
6. Drainage system capacity exceedance

**Pass criteria**: all 6 pathways addressed; each pathway linked to at least one
specific hypothesis or research question; conditions for amplification vs. buffering
stated for at least 3 pathways.

### D5: Research gap synthesis
File: `05_gap_synthesis.md`

3-5 sharply stated gaps. For each gap:
- What is known (with citations from D3)
- What is NOT known or weakly tested
- Why it matters (scientific and practical significance)
- How it connects to the proposed research

Gaps should be specific enough to generate hypotheses, not generic calls for
"more research."

**Pass criteria**: at least 3 gaps stated; each gap supported by citations; none
are simple restatements of the seed paper's own "future work" section.

### D6: Hypotheses (generate + rank)
File: `06_hypotheses.md`

Generate at least 5 candidate hypotheses. For each:
- Falsifiable statement (a specific test could reject it)
- Physical mechanism targeted
- Data/method needed to test it
- Why the seed paper does not already answer it
- Scores (1-5 each): novelty, scientific importance, feasibility with open data,
  transferability beyond Eastwick, suitability for a strong paper

Rank the top 3 with justification.

Candidate hypothesis families to consider (starting points, not mandatory):
- H1: Floodplain-mediated compounding dominates over channel compounding in UTZ
- H2: SLR threshold exists for regime transition from fluvial to compound dominance
- H3: Maximum inundation is not always at exact peak coincidence (lag matters)
- H4: Moderate-moderate compounding can exceed extreme single-driver events
- H5: Open workflow can recover main mechanism insights from proprietary pipelines
- H6: Anthropogenic features (rail, berms) are hidden controls on pathway activation

**Pass criteria**: at least 5 hypotheses; at least 3 are falsifiable; at least 2
are not restatements of the seed paper's limitations; each names a specific mechanism.

### D7: Research proposal + manuscript blueprint
File: `07_proposal.md`

Full proposal:
- Title (mechanism-focused, not site-focused)
- Abstract (hypothesis-driven, under 250 words)
- Background and significance (why compound flooding, why UTZ, why now)
- Gap statement (from D5, sharpened)
- Research questions and hypotheses (from D6, top 3)
- Data strategy: mandatory / optional / request-access / open substitutes
- Methods:
  - Minimum viable Phase-2 workflow (what can be done with open data)
  - Best-case enhanced workflow (if request-access data is available)
- Phase-2 pathway options:
  A. Eastwick-centered follow-up
  B. Open benchmark UTZ site
  C. Mechanism-first idealized modeling
- Expected contributions
- Risks and contingencies (what if Eastwick data access fails?)
- Milestone plan

Append: Nature-style manuscript blueprint
- Title, Abstract, Introduction, Results (planned logic), Discussion, Methods,
  Data/code availability plan, References
- Do NOT invent results. Describe the intended study architecture.

Final section: self-evaluation against the rubric (see below).

**Pass criteria**: all sections present; at least one Phase-2 pathway fully specified;
risk/contingency section addresses data access failure; hypotheses are traceable
back to D6.

## The research loop

This is a sequential pipeline, not an infinite loop. But for each deliverable,
iterate until it passes.

```
FOR EACH DELIVERABLE (D1 through D7):
  1. Read all reference materials + all previous deliverables produced so far
  2. If the deliverable needs literature (especially D3), use web search tools
  3. Write the deliverable to its output file
  4. Check against the pass criteria listed above
  5. If it passes: git commit with message "D{N}: {short description}"
  6. If it does not pass: revise (max 2 revision attempts), then commit best version
  7. Log to results.tsv
  8. Move to next deliverable
```

After all 7 deliverables are complete, write an executive summary:
File: `output/00_executive_summary.md`
- What should we study next?
- Why this direction?
- What is feasible now?
- What is blocked by data or software?

Commit: "D0: executive summary"

## Logging results

Log each deliverable to `docs/compound_flood/results.tsv` (tab-separated).

Header and columns:

```
deliverable	status	revisions	notes
```

1. deliverable: filename (e.g. 01_paper_audit)
2. status: `complete` or `partial` (if pass criteria not fully met)
3. revisions: number of revisions (0 = first draft passed)
4. notes: short description of what was produced

Example:

```
deliverable	status	revisions	notes
01_paper_audit	complete	0	7 fragilities; 4 mechanism claims decomposed
02_reproducibility_memo	complete	1	all 14 items classified; 3 open substitutes proposed
03_literature_matrix	complete	0	19 papers; 6 buckets covered; 5 new beyond seed list
```

Do NOT commit results.tsv to git (keep it untracked for the human to review).

## Self-evaluation rubric

After completing D7, score the entire body of work on these 8 criteria (1-5 each):

1. Scientific significance - does the proposal address a real, important gap?
2. Clarity of mechanism - is the physical reasoning specific and testable?
3. Literature grounding - are claims supported by evidence from D3?
4. Novelty - does it go beyond restating the seed paper?
5. Feasibility with accessible data - can Phase-2 actually be done?
6. Tool independence - does it avoid hard dependency on proprietary tools?
7. Transferability beyond one site - do insights generalize?
8. Paper suitability - could this contribute to a strong publication?

Include the rubric scores and trade-off discussion in D7's final section.

## What you CAN do

- Read and analyze reference materials
- Search for literature using web tools (if available)
- Write structured documents to the output directory
- Use your training knowledge for domain reasoning
- Mark uncertainty: [verified] / [to verify] / [uncertain] / [inferred]
- Git commit each completed deliverable

## What you CANNOT do

- Modify reference files (ref_brief.md, ref_domain.md, ref_literature.md)
- Fabricate citations - if you are not sure a paper exists, say so
- Claim reproduction of the seed paper
- Present speculative claims as established facts
- Run hydrodynamic simulations (this is Stage-1, not Phase-2)
- Add new dependencies or install software

## Autonomy

Once the deliverable pipeline begins, do NOT pause to ask the human for permission
to continue. Complete all 7 deliverables + executive summary before stopping.

If web search fails or is unavailable for a particular query, work from reference
materials and your training knowledge. Mark anything that needs verification.

If you get stuck on a deliverable, write what you can, log it as `partial` in
results.tsv, and move to the next one. Do not spin on a single deliverable.

The human may be away. Finish everything, then stop.
