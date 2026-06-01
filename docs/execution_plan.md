# Execution Plan — Top-Grade Remediation (2026-06-01)

Supersedes the completed 2026-05-29 pass (captured in `decisions_log.md`).
Goal: convert a polished-but-overclaiming project into a defensible top-grade
submission. Driven by the 2026-06-01 sanity check + the official brief
(`docs/references/project_brief.pdf`). All heavy compute (CV, notebook execute)
is deferred to a single `./run.sh` tonight; everything below is code prepared in
advance so that run does the work. **Approval-gated per CLAUDE.md §9.3.**

Controlling brief facts:
- "**The .ipynb is the only code we will consider.**" → Spark code must be
  *visible* in the notebook, not only callable from `src/`.
- Grade depends primarily on correct/idiomatic Spark use; graph analytics is the
  heaviest-weighted area.
- Oral Q&A probes modelling choices → every claim must survive a one-line check.

---

## Workstream A — Notebook code visibility (structural fix) — HIGHEST ROI

Problem: ~62/71 code cells are thin `olist.*` calls; only 2 use
`inspect.getsource`. A grader reading only the notebook sees the implementing
Spark code for ~2.5 of 9 required demonstrations. **None of the heavy ones
(Pipelines, CV/MLlib, LSTM, GraphFrames, Streaming, applyInPandas) are visible.**

Fix: extend the existing `print(inspect.getsource(fn))` pattern in
`scripts/build_main.py` to one cell per hidden primitive, placed in its section.
Functions to surface (verify exact names on implement):

| Section | Function(s) to `getsource` |
|---|---|
| §3.5 feature Pipeline | `demand.build_feature_pipeline` (VectorAssembler, lag/rolling/calendar) |
| §3.6 MLlib + CV | `demand.fit_and_score` (GBT+RF + ParamGridBuilder + CrossValidator + RegressionEvaluator) |
| §4.4 NLP Pipeline | `sentiment.build_nlp_pipeline` (Tokenizer→StopWordsRemover[pt]→HashingTF→IDF→LR) |
| §4.6.1 LR CV | `sentiment.fit_nlp_pipeline` (CV grid) |
| §4.6.2 Deep Learning | LSTM model class + `sentiment.train_lstm` (TorchDistributor + predict_batch_udf) |
| §4.7.3 UDF | `sentiment.seller_sentiment_slopes` (applyInPandas) |
| §5.4–5.6 GraphFrames | `network.build_graph_frame`, `compute_pagerank`, `compute_connected_components`, `compute_shared_customer_motifs`, `compute_bfs_backups`, `compute_delayed_subgraph_pagerank`, `compute_cocustomer_centrality`, `compute_substitution_communities` |
| §8 Streaming | `streaming.run_weekly_volume_stream` (readStream/writeStream/trigger) |

Each is a one-line print cell → instant output, survives
`assert_notebook_outputs.py`. No new compute. Add a short markdown lead before
each ("The Spark code behind this step:") so it reads as method disclosure, not a
dump. Moves the heavy/graph rubric lines from near-invisible to fully visible.

---

## Workstream B — Demand forecasting: a forecast that actually beats a baseline

Problem (verified): `weekly_order_count` mean 3.11 / median 2 / 68% ≤ 2. The
tuned seller-level GBT (RMSE ~4.92) **loses to "predict last week" (lag-1 RMSE
~3.9)**, RMSE > target mean, and `forecast_uplift_pct` is **not even used** in the
index (the demand axis is `avg_delay_days`). Forecaster is both weak and decorative.

Fix — turn the weakness into a maturity story (3 parts):

1. **Keep the seller-level model as an honest "wrong granularity" demonstration.**
   In `demand.fit_and_score`, add naive baselines on the *same test rows*: `lag_1`
   (persistence), rolling-4-mean, global-mean → report RMSE next to GBT/RF. State
   plainly that per-seller weekly counts are too sparse to beat persistence. Keeps
   the GBT/RF+CV mechanics (rubric) and adds DS maturity (measured vs baseline).
   - Eval rigor: temporal split by **calendar week**, not per-seller row-number
     (current `approxQuantile(week_num,0.8)` makes the test set long-tenure-biased);
     fit the `Imputer` on **train only** (current fit-before-split leaks).

2. **Add a regional forecast that genuinely beats baseline — the real tool.**
   New in `demand.py`: `build_regional_weekly_volume` → `(seller_state, year_week)`
   (dense ~27×~100 series) and `fit_regional_forecast` (GBT+RF under CV; features
   lag_1/lag_2/lag_4, rolling-4 mean, month, week_of_year, state index; target
   next-week volume; baselines lag_1 + rolling). Write
   `outputs/nb1_regional_demand_forecast.parquet`
   `(seller_state, year_week, actual, predicted, model_rmse, baseline_rmse)`.
   Dense → the model should beat persistence → a *genuine* "where/when will demand
   spike" tool (matches the brief storyline). CV here is fast.

3. **Stop implying the forecast feeds the index.** Per-seller demand-risk axis stays
   `avg_delay_days` (honest); say so. Drop / clearly relabel `forecast_uplift_pct`
   (it manufactures a fake +30% growth for ~all sellers). Regional forecast stands
   alone as the insight/tool.

Notebook §3 narrative: seller-level (honest sparse finding) → regional forecast
(beats baseline) → per-seller delay risk (feeds §6).

**Tonight's heavy run:** the existing seller-level CV is the long part; regional CV
is light. Both run under one `./run.sh`.

---

## Workstream C — Risk index: make all three signals actually contribute (Findings 1+6)

Problem (verified): `corr(risk_score, sentiment_norm)=0.963`; demand_norm
(std 0.034) and network_norm (std 0.056) barely move anyone. Composite is ~96% the
inverted star rating. CRITICAL band is mathematically dead (max 0.705 < 0.75 → 0).

Fix in `convergence.py`:
1. Replace min-max(p1/p99) with **percentile-rank normalization** per component
   (`percent_rank()` window → each uniform on [0,1]). Equal weights → equal
   *influence*; all three signals genuinely move the score.
2. **Percentile-based bands** on the composite (CRITICAL = top ~5%, WARNING = next
   ~15%, SAFE = rest) so the top tier always identifies the worst sellers. Keep band
   names. (Alt: keep fixed thresholds + explicitly justify empty CRITICAL — weaker.)
3. Rewrite §6.2: drop "orthogonality validates the composite"; new framing =
   orthogonal *and* now equally weighted; print `corr(risk_score, each)` to prove
   balance.

Impact: `seller_risk_index.parquet` recomputed; band counts, "67 WARNING / 0
CRITICAL", "57 escalate_no_backup", top-20, slides 7/8/9 numbers WILL change →
reconcile afterwards.

---

## Workstream D — Graph "substitutability" honesty (Finding 5)

Problem (verified): defended with Pearson `corr(in_degree, deficit)=0.38`, but
**Spearman = 0.89** — by the rank metric that drives triage it is largely a degree
proxy.

Fix:
1. §5 degree-proxy check reports **both Pearson and Spearman**; stop claiming "not a
   degree proxy".
2. Reframe `substitutability_deficit` as a **degree-adjusted refinement** ("among
   equally-large sellers, ranks by how few substitutes exist") — true and defensible.
3. (Stretch) add a degree-orthogonal variant (residual of deficit on in_degree) as a
   secondary signal. Only if time permits; the honest reframe is the required fix.

---

## Workstream E — Sentiment honesty + a real use (Finding 3)

Problem: LSTM/LogReg labels derive from the star rating Olist already has; classifier
output is never used downstream → circular.

Fix:
1. Reframe §4: NLP + LSTM are the brief-mandated classifier/DL demonstrations,
   validated at AUC ~0.96. Do **not** claim operational value beyond the star rating.
2. (Stretch) add a **star–text mismatch flag** (model sentiment disagrees with stars)
   — genuine info stars lack; aggregate per seller. Only if time permits.
3. Keep the honest lead-indicator null; tighten "peak |ρ| 0.015" →
   "indistinguishable from zero at all lags 0–8w".

---

## Workstream F — Presentation reconciliation (after reruns)

- Reference slide added (slide 11). **Before submission fold back to ≤10 slides**
  (merge references into the closing slide) per the brief.
- Reduce remaining jargon for the non-technical audience.
- Re-export from the **official NOVA IMS PowerPoint template → PDF** (HTML deck is a
  draft, not the gradable format). *(Owner: human.)*
- After B/C rerun, update changed numbers on slides 7/8/9 + CLAUDE.md §7 +
  `decisions_log.md`.

---

## Sequencing

1. **Now (no compute):** Workstream A cells in `build_main.py`; code for B/C/D/E in
   `pipeline/*.py`; adjust `checks.py`; regenerate notebook source (`build_main.py`).
   Commit per workstream.
2. **Tonight (heavy):** `OLIST_FORCE_ALL=1 ./run.sh` — recomputes all parquets with
   new logic + heavy CV, executes `main.ipynb` with outputs (incl. new getsource cells).
3. **After run:** `checks.run_all()`; `assert_notebook_outputs.py`; reconcile headline
   numbers into CLAUDE.md §7, `decisions_log.md`, slides 7–9.
4. **Validate fixes:** `corr(risk_score, each)` balanced; CRITICAL band fires; regional
   forecast RMSE < baseline RMSE; Spearman reported.

## Expected metric drift (authorized — exceeds CLAUDE.md §7 ±1% gate)
Risk bands, escalate count, top-20, demand RMSE/baseline all change. Intended;
reconcile after the run.

## One open design choice
Workstream B keeps the seller-level model (honest demonstration) **and** adds a
regional forecast (the genuine tool). Alternative: replace seller-level entirely.
Recommendation: **keep + add** — more Spark demonstrated, and the stronger "we
measured, found the wrong granularity, fixed it" story.
