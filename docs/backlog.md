# Backlog — potential additions / optimizations

Created 2026-05-30. Prioritised by grade-impact vs. effort, grounded in the current state
(post max-grade remediation, commit 43c2ee0). Not commitments — a menu to pick from.

## Tier 1 — highest value, do first

### 1. Correctness review of this session's new code
- **What:** Run `/code-review` (or an adversarial pass) over the remediation diff — the co-customer
  projection, convergence rewrite, winsorise, `fit_and_score` caching. `checks.py` confirms patterns
  *exist*, not that they're *correct*; ~250 new lines have not had a correctness review.
- **Why:** Biggest open unknown. Cheapest way to move confidence from "works" to "verified-correct".
- **Effort:** low. **Grade impact:** protects everything already built.

### 2. presentation.pdf (graded deliverable, currently unverified)
- **What:** The ≤10-slide NOVA-IMS management deck is a *separate graded artefact* produced by the
  human from the notebook's charts/tables. I haven't seen it; can't vouch for it.
- **Why:** It's graded alongside the notebook. A strong notebook with a weak/missing deck loses easy points.
- **Effort:** medium (human-led; I can draft structure + map each slide to a notebook figure).
- **Grade impact:** high — it's half the visible deliverable at the oral.

## Tier 1b — class-coverage gaps (from docs/lab_coverage.md, 2026-05-30)
Techniques the weekly labs taught that our project does not use. Relevant to the
"everything done in class must be present" bar. Full analysis in `docs/lab_coverage.md`.

### 11. UDF family — `pandas_udf` + `applyInPandas`  [Week 6 marquee]
- **What:** We use one plain `F.udf`. The labs spent a week on the UDF trio. `applyInPandas`
  (grouped split-apply-combine) runs on executors — scalable, consistent with our safety stance.
- **Where it'd fit:** e.g. a `pandas_udf` for a vectorised per-seller calc, or `applyInPandas`
  for a per-group transform we currently do with Window/groupBy.
- **Effort:** low-medium. **Value:** medium — closes the most-emphasised Week-6 gap.

### 12. Spark–DL integration — `TorchDistributor` / `predict_batch_udf`  [Week 9 emphasis]
- **What:** Week 9's whole point was running PyTorch *inside Spark*, not building a net. We train
  the LSTM driver-side (`toPandas`→PyTorch). Wrap training in `TorchDistributor(local_mode=True)`
  and/or score via `predict_batch_udf`.
- **Effort:** medium. **Value:** medium-high — directly demonstrates the DL lab's actual subject.

### 13. `StringIndexer` / `OneHotEncoder`  [Weeks 7 & 8, taught twice]
- **What:** No categorical encoding anywhere (features are numeric). One-hot `seller_state` (or
  product category) into the demand feature Pipeline.
- **Effort:** low. **Value:** medium — common technique, taught twice, currently absent.

### 14. `ClusteringEvaluator` / silhouette  [Week 8]  (= old #6)
- **What:** Justify k=4 with a silhouette score, not just the WSSSE elbow. Merges with the earlier
  KMeans-validation item.
- **Effort:** low. **Value:** low-medium.

## Tier 2 — bonus / real upside

### 3. Streaming (Structured Streaming) — the brief's bonus  ⭐ (requested)
- **What:** Simulated Structured Streaming version of the weekly order-volume aggregation: drip the
  existing order rows into a watched directory, `spark.readStream` → `window("...", "10 minutes")`
  count → memory/console sink → run a few triggers → `.stop()`. New `pipeline/streaming.py` + one
  notebook subsection (e.g. §3.9 or appendix) + safety-log note.
- **Why:** The brief lists streaming as a **bonus** (CLAUDE.md §8, "tolerated absent"). It's the
  natural production form of our early-warning framing (§2 Velocity, §7 rec #6 already promise it).
  Adds points on top; absence costs nothing against core bars.
- **Caveat:** Dataset is a static historical dump, so the demo is necessarily *simulated*. Must execute
  with visible output in nbconvert without hanging (careful trigger/await/stop, or `trigger(once=True)`).
- **Effort:** medium (~30-40 LOC + cell hygiene). **Grade impact:** bonus only, but a clean velocity demo.

### 4. Value-weighted graph edges (addresses our own stated limitation)
- **What:** §5 Key Takeaways honestly flags that edges are item-count weighted, not revenue weighted.
  Re-weight the co-customer projection (and/or PageRank) by revenue/recency and report whether the
  structural-criticality ranking shifts.
- **Why:** Turns a stated limitation into an analysis; could further decorrelate centrality from degree
  and strengthen the graph-unique story (already our strongest, most-weighted area).
- **Effort:** medium. **Grade impact:** medium — deepens the heaviest-weighted rubric area.

## Tier 3 — calibration / analytical polish

### 5. Risk-band calibration — why is CRITICAL empty?
- **What:** Current bands are 2900 SAFE / 67 WARNING / **0 CRITICAL** (threshold >0.75). Examine the
  `risk_score` distribution — is CRITICAL structurally unreachable given the normalisation? Either
  justify it explicitly in §6 or recalibrate thresholds to the observed distribution (e.g. quantile bands).
- **Why:** A grader/oral examiner will likely ask "why does no seller hit CRITICAL?" — have a crisp answer
  or a calibrated banding.
- **Effort:** low. **Grade impact:** low-medium (defends a visible result).

### 6. KMeans cluster validation — add silhouette
- **What:** Archetypes use the WSSSE elbow for k=4. Add a silhouette score (`ClusteringEvaluator`) as a
  second validation signal.
- **Effort:** low. **Grade impact:** low (incremental rigor on an already-justified choice).

### 7. Confirm geo_centroids earns its place
- **What:** `geo_centroids.parquet` (19k zip centroids) is built + broadcast. Verify it drives a real
  feature/insight vs. just supplying seller-state context; if underused, either add a distance feature
  or trim the framing so it's not dead weight.
- **Effort:** low (investigate first). **Grade impact:** low.

## Tier 4 — engineering quality (not directly graded)

### 8. Cold-cache / submission timeout robustness
- **What:** A truly cold cache exceeds `build_zip.sh`'s 3600s per-cell timeout on the demand CV (~45min).
  Committed parquets make the graded build cache-skip, but if `outputs/` is wiped it'll fail. Options:
  bump the timeout to 7200s in `run.sh` + `build_zip.sh` (+ update CLAUDE.md §10/§11), or document a
  "warm the cache first" step.
- **Why:** Operational safety net for submission day, not a grade dimension.
- **Effort:** low. **Grade impact:** none directly; prevents a submission-day failure.

### 9. Lightweight pipeline tests
- **What:** A small pytest suite over the pure pipeline functions (schemas, deficit formula, normalisation
  bounds) alongside the existing `diff_parquets.py` guard.
- **Effort:** medium. **Grade impact:** none directly; reviewer-confidence / robustness.

### 10. Oral-exam Q&A prep doc
- **What:** A short `docs/oral_prep.md` anticipating likely questions (why GBT, why the deficit isn't a
  degree proxy, why CRITICAL is empty, why uniform docs, streaming-at-scale) with crisp answers.
- **Effort:** low. **Grade impact:** indirect — the oral is 50% of the exam weight.

## Accepted trade-offs (decided — NOT action items)
- **Documentation house-style** (markdown above every cell, "What this means" callouts, 🎯 emoji boxes):
  deliberately kept for the non-technical management audience (decision 2026-05-29). Known AI-tell
  perception risk; defended at the oral as a house style. Do not re-litigate unless the user reopens it.
