# Compound Flood Research Agent - program.md

## Goal
Given a seed paper on compound flooding in upper transition zone (UTZ) systems,
generate novel, testable research hypotheses that go beyond the seed paper's findings.

This is a hypothesis-generation and research-direction task, not a paper-writing task.

---

## Seed paper summary
**Mita et al. (2025)** - "Accumulating climate change influences on extreme coastal,
fluvial, and compound flooding in the upper transition zone."
Journal of Hydrology, 663, 134247.

**Site**: Eastwick, southwest Philadelphia - near the inland limit of a tidal
estuarine-riverine system (Darby Creek / tidal Schuylkill / Delaware estuary).

**Framework**:
- Three event types: fluvial flood (FF), storm surge flood (SSF), compound flood (CF)
- Three time horizons: baseline, mid-century, late-century
- Three climate change influences (CCIs): rainfall increase, SLR, storm climatology change
- Model chain: 1D hydrology (PCSWMM) -> 2D hydraulics (HEC-RAS 2D), one-way coupling

**Core claims**:
1. In UTZ systems, future hazard shifts from fluvial-dominated to coastal-compound
2. Single-driver analysis materially underestimates future flood extent and depth
3. Compounding accumulates in floodplain storage areas, not just the main channel
4. Local topography and connectivity mediate whether floodplains buffer or worsen flooding

**Key limitations** (for the agent to interrogate, not dismiss):
- "100-year" labels are not identical across event types
- Compound event uses worst-case peak-alignment construction
- Climate scenarios are deterministic stress tests, not probabilistic
- Reproducibility depends on local calibration data (likely unavailable)
- Land use held fixed; drainage evolution ignored

The full paper PDF is at `docs/papers/mita_2025.pdf` if available.
For reference material, see `docs/compound_flood/ref_domain.md` and `ref_literature.md`.

---

## What you produce

### Round 1: Paper audit + raw hypotheses
Write to `docs/compound_flood/round1_output.md`:

**Part A - Seed paper audit** (structured decomposition):
- What gap does the paper claim to fill?
- Event design choices and their assumptions
- Mechanism claims and supporting evidence
- What is strong vs. fragile in the methodology

**Part B - Hypotheses** (at least 5):
Each hypothesis must include:
- A falsifiable statement (not a vague direction)
- The physical mechanism it targets
- What data/method would test it
- Why the seed paper does not already answer it

### Round 2: Ranking + feasibility (after human review of Round 1)
Write to `docs/compound_flood/round2_output.md`:
- Rank top 3 hypotheses by: novelty, scientific importance, feasibility with open data, transferability beyond Eastwick
- For each top hypothesis: data requirements, method sketch, key risks
- Identify 5-10 papers that would need to be read (not a full review - just pointers with why each matters)

### Round 3: Mini-proposal (after human review of Round 2)
Write to `docs/compound_flood/round3_output.md`:
- Title, abstract, research questions, proposed methods
- Data strategy: mandatory vs. optional vs. request-access
- Minimum viable Phase-2 workflow
- Risk and contingency plan

---

## Hard constraints
- Do NOT claim reproduction of the seed paper
- Do NOT assume calibrated model files are publicly available
- Do NOT present speculative mechanism claims as established facts
- Do NOT fabricate literature - if you are unsure whether a paper exists, say so
- Do NOT optimize for volume. Optimize for precision and testability.
- Clearly label: paper-supported fact / reasonable inference / open question / new hypothesis

---

## Pass/fail criteria for Round 1
Your output PASSES if:
1. At least 3 of 5 hypotheses are falsifiable (a specific test could reject them)
2. At least 2 hypotheses are NOT restatements of the seed paper's own limitations
3. Each hypothesis names a specific physical mechanism (not just "more research needed")
4. The paper audit correctly identifies at least 3 methodological fragilities
5. You distinguish your inferences from paper-stated facts

Your output FAILS if:
- Hypotheses are generic directions ("study more sites") rather than testable claims
- All hypotheses require proprietary data with no open-data alternative
- The audit is a summary rather than a critical decomposition
- You fabricate citations or findings

---

## What you should understand about the domain
Reference: `docs/compound_flood/ref_domain.md`

The short version:
- Compound flooding is not "two bad things at once" - it involves nonlinear hydraulic
  interactions, timing/phasing, pathway activation, and driver dependence
- UTZ systems sit where coastal and fluvial influences compete; SLR can shift the
  balance in threshold-like ways
- The mechanism question ("where does water store, how do pathways open, when does
  backwater trap runoff") matters as much as the hazard mapping question
- Floodplain connectivity and anthropogenic features (berms, rail corridors, culverts)
  can act as hidden control points

---

## Execution notes
- Start with Round 1 only. Wait for human feedback before Round 2.
- If the seed paper PDF is available, read it. If not, work from the summary above.
- When citing literature, mark confidence: [verified] vs. [to verify] vs. [uncertain].
- Keep each round's output under 2000 words. Density over length.
