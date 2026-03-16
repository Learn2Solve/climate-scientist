# Compound Flood Research - autoresearch

This is an autonomous research task. You (Claude Code) are the researcher.
Read reference materials, search literature, download and read papers, verify
data sources, and produce structured deliverables for a research proposal on
compound flooding mechanisms in upper transition zone (UTZ) systems.

The seed paper is Mita et al. (2025), "Accumulating climate change influences on
extreme coastal, fluvial, and compound flooding in the upper transition zone."

---

## Setup

1. Read the reference materials (all immutable, do not modify):
   - `docs/compound_flood/ref_brief.md` - full research brief with all requirements
   - `docs/compound_flood/ref_domain.md` - domain knowledge: hydrology, compound events, UTZ
   - `docs/compound_flood/ref_literature.md` - seed literature and search directions
2. Check if the seed paper PDF exists at `docs/papers/mita_2025.pdf`. If yes, read it.
   If not, work from the summary in ref_brief.md.
3. Create `docs/compound_flood/results.tsv` with the header row.
4. Begin producing deliverables. Do not wait for confirmation.

---

## Tools and how to use them

You have everything you need as built-in tools. Here is how to use them for
research tasks.

### Searching for papers

Use these approaches in order of preference:

1. **WebSearch** - search Google Scholar, Semantic Scholar, or general web
   for papers by topic, author, or keyword.
   Example queries:
   - "compound flooding upper transition zone mechanism site:scholar.google.com"
   - "estuarine compound flooding timing lag sensitivity"
   - "Mita Orton Montalto compound flooding 2025"

2. **Grok DeepSearch** (mcp__grok-deepsearch__deepsearch) - for deeper
   searches that need synthesis across multiple sources.

3. **Hugging Face paper search** (mcp__hugging-face__paper_search) - for
   ML/AI-adjacent papers if relevant.

### Downloading papers

1. **WebFetch** a paper's DOI page or direct PDF link.
   - DOI URLs: `https://doi.org/10.1016/j.jhydrol.2025.134247`
   - Publisher pages often have PDF links or full text HTML.
   - Open access papers can be fetched directly.

2. Save PDFs to `docs/compound_flood/papers/` using Bash:
   ```
   curl -L -o docs/compound_flood/papers/author_year.pdf "URL"
   ```

3. For paywalled papers: fetch the abstract/metadata page instead.
   Record what you could and could not access.

### Reading papers

1. **Read tool** - reads PDFs directly. Use this on any PDF in
   `docs/compound_flood/papers/`. Claude Code is multimodal and can
   read PDF content.

2. **WebFetch** - for HTML full-text versions on publisher sites.

3. **Firecrawl** (mcp__firecrawl__firecrawl_scrape) - for scraping
   publisher pages that WebFetch doesn't handle well.

### Verifying data sources

To check if datasets and APIs are actually accessible, use WebFetch on:
- NOAA CO-OPS: `https://tidesandcurrents.noaa.gov/api/datagetter?...`
- USGS NWIS: `https://waterservices.usgs.gov/nwis/iv/?...`
- NOAA Atlas 14: `https://hdsc.nws.noaa.gov/pfds/`
- PASDA: `https://www.pasda.psu.edu/`

You don't need to download full datasets. Just verify the endpoint responds
and note what data is available (date range, variables, formats).

### Writing deliverables

Use **Write** to create files in `docs/compound_flood/output/`.
Use **Edit** for revisions.
Use **Bash** for git commits.

---

## Reference material summary

Read the full `ref_brief.md` for detailed requirements. The short version:

**Seed paper (Mita et al. 2025)**:
- Site: Eastwick, Philadelphia - inland limit of tidal estuarine-riverine system
- Framework: 3 event types (FF, SSF, CF) x 3 time horizons x 3 climate drivers
- Model chain: 1D hydrology (PCSWMM) -> 2D hydraulics (HEC-RAS 2D), one-way coupling
- Core claim: in UTZ systems, future hazard shifts from fluvial to coastal-compound
- Key limitation: deterministic stress tests, peak-aligned compound event, local calibration

**What makes this paper a good seed**:
- Clean structure for asking mechanism questions
- Identifies floodplain-mediated compounding (not just channel interaction)
- Highlights topographic and connectivity controls
- Raises questions about transferability beyond one site

---

## The deliverables

Produce 7 deliverables, in order. Each builds on previous ones.
All outputs go in `docs/compound_flood/output/`.

### D1: Seed paper audit
File: `01_paper_audit.md`

If the seed paper PDF is available, read it thoroughly before writing this.
If not, work from ref_brief.md's detailed summary.

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

**Verify at least 3 data sources** by actually hitting their APIs or web pages
with WebFetch. Don't just list URLs - confirm they work and note what's available.

State the practical implication: what can a follow-up study actually do?

**Pass criteria**: every item in ref_brief.md's data table is addressed; at least
one open substitute proposed for each proprietary dependency; at least 3 sources
verified via actual web access.

### D3: Literature matrix
File: `03_literature_matrix.md`

**Actively search for papers.** Do not rely only on ref_literature.md.
Use WebSearch and/or Grok DeepSearch to find papers across 6 buckets:
1. Direct lineage / local context
2. Compound flooding reviews and syntheses
3. Estuary / tidal-river / transition-zone sensitivity
4. Timing / lag / phasing
5. Modeling frameworks (1-way vs 2-way coupling, open vs proprietary)
6. Human-modified floodplain / pathway activation

For each paper found:
- Try to fetch its abstract or full text (WebFetch on DOI page)
- If it's open access, download the PDF to docs/compound_flood/papers/
- Read at least the abstract and conclusions of each paper you cite

Table columns: citation | study region | flood-driver types | event definition |
model/method | climate treatment | main physical mechanism | main limitation |
relevance to UTZ research | relevance to open-data follow-up

Target: 15-25 papers. Optimize for relevance, not count.
Mark citation confidence: [verified] / [to verify] / [uncertain].

**Pass criteria**: at least 12 papers in the matrix; all 6 buckets represented;
at least 3 papers not already in ref_literature.md; at least 5 papers whose
abstracts you actually read (not just cited from memory).

### D4: Mechanism map
File: `04_mechanism_map.md`

Structured description of compound flooding mechanism pathways in UTZ systems.
Text-based conceptual model (not a diagram).

For each mechanism pathway:
- Description of the physical process
- Under what conditions it activates
- Which drivers and climate change influences control it
- Where in the UTZ it is most relevant
- What determines whether it amplifies or buffers flooding
- Connection to the seed paper's findings (if any)
- Supporting evidence from literature (cite papers from D3)

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

Gaps must be specific enough to generate hypotheses. Not "more research is needed"
but "the sensitivity of compound flood extent to lag between fluvial and coastal
peaks has been tested in only 2 studies, both in lower-estuary settings."

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
risk/contingency section addresses data access failure; hypotheses traceable to D6.

---

## The research loop

Sequential pipeline. For each deliverable, iterate until it passes.

```
FOR EACH DELIVERABLE (D1 through D7):
  1. Read all reference materials + all previous deliverables produced so far
  2. If the deliverable needs literature:
     a. Search for papers (WebSearch, Grok DeepSearch)
     b. Fetch and read promising papers (WebFetch, Read)
     c. Save useful PDFs to docs/compound_flood/papers/
  3. Write the deliverable to its output file
  4. Re-read your output and check against the pass criteria
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

---

## Logging results

Log each deliverable to `docs/compound_flood/results.tsv` (tab-separated).

```
deliverable	status	revisions	notes
01_paper_audit	complete	0	7 fragilities; 4 mechanism claims decomposed
02_reproducibility_memo	complete	1	all 14 items classified; 3 open substitutes proposed
03_literature_matrix	complete	0	19 papers; 6 buckets covered; 5 new beyond seed list
```

Status: `complete` or `partial`.
Do NOT commit results.tsv to git (keep it untracked).

---

## Self-evaluation rubric

After completing D7, score the entire body of work on 8 criteria (1-5 each):

1. Scientific significance - does the proposal address a real, important gap?
2. Clarity of mechanism - is the physical reasoning specific and testable?
3. Literature grounding - are claims supported by evidence from D3?
4. Novelty - does it go beyond restating the seed paper?
5. Feasibility with accessible data - can Phase-2 actually be done?
6. Tool independence - does it avoid hard dependency on proprietary tools?
7. Transferability beyond one site - do insights generalize?
8. Paper suitability - could this contribute to a strong publication?

Include rubric scores and trade-off discussion in D7's final section.

---

## Hard constraints

- Do NOT modify reference files (ref_brief.md, ref_domain.md, ref_literature.md)
- Do NOT fabricate citations - if unsure whether a paper exists, say so
- Do NOT claim reproduction of the seed paper
- Do NOT present speculative claims as established facts
- Do NOT invent results for the manuscript blueprint
- Do NOT optimize for volume - optimize for precision and testability
- Label everything: [paper-stated] / [inferred] / [open question] / [new hypothesis]
- Keep each deliverable under 2500 words

---

## Autonomy

Once you begin, do NOT pause to ask the human for permission to continue.
Complete all 7 deliverables + executive summary before stopping.

If a web search fails, work from reference materials and your training knowledge.
Mark anything unverified as [to verify].

If you get stuck on a deliverable, write what you can, log it as `partial`,
and move to the next one. Do not spin.

The human may be away. Finish everything, then stop.
