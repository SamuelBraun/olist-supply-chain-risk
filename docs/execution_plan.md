# Execution Plan — Max-Grade Remediation Pass

Created 2026-05-29. Supersedes the 2026-04-20 four-notebook build plan (that work is long done).
Drives the project from "strong, a few weak spots" to max-grade. Scope = every item the
2026-05-29 audit flagged weak/partial. **Approval-gated per CLAUDE.md §9.3 — do not implement until signed off.**

## Audit findings being addressed

| # | Finding | Severity | Phase |
| --- | --- | --- | --- |
| A | Graph analytics broad but not graph-*unique*: PageRank r=0.9998 vs in-degree; `network_risk_score` r=0.87 vs degree, zero for 68% of sellers; motif + BFS computed then **dropped** before the risk index | 🔴 highest (heaviest-weighted rubric area) | 3, 4 |
| B | Stale "RandomForest selected" prose contradicts code/parquet (GBT selected) | 🟠 reviewer-visible | 0 |
| C | RDD section decorative (output unused, fragile `split(",")`); reconciliation chain inlined in notebook (`build_main.py:371-388`) violates "no logic in cells" | 🟠 | 1 |
| D | No outlier treatment; raw delay feeds GBT label and distorts min-max normalisation range | 🟡 | 2 |
| E | Lazy-eval / transformation-vs-action never explicitly named (only class-material concept not surfaced) | 🟡 | 1 |
| F | LSTM AUC drift: safety log 0.9630 vs spec 0.9619 | 🟢 cosmetic | 6 |
| G | Stale "§1 + §2 only" scaffolding comment in `build_main.py:1-17` | 🟢 | 0 |

**Expected side effect (authorised by this plan):** headline metrics will move materially (>1%) —
risk-band counts especially. CLAUDE.md §7 normally requires stop-and-ask on >1% drift; this plan is
that authorisation. New values get written back to §7 in Phase 6.

---

## Phase 0 — Quick correctness fixes (no recompute)
Low-risk; only a notebook rebuild, no parquet recompute.

1. **B — GBT narrative.** `scripts/build_main.py` ~L477 and ~L543-544: replace hardcoded "RF wins / RF is selected" with text interpolated from `best_name` / `metrics_pd` so prose tracks the dynamically-rendered table and can't drift again.
2. **G — stale header.** Delete "Sections 3–7 are added in subsequent commits; this commit lands §1 + §2 only." from `build_main.py:1-17`.
3. Rebuild notebook (`scripts/build_main.py`), re-execute affected cells, eyeball.
4. Commit: `fix(nb): GBT-correct model-selection narrative + drop stale header`.

## Phase 1 — RDD: make it load-bearing + name lazy evaluation
Addresses C and E. Recomputes demand chain.

1. **Move the inlined reconciliation into `src`.** New `demand.rdd_vs_typed_reconciliation(spark)` holding the `build_main.py:371-388` chain. Notebook cell shrinks to import + call + `.show()`.
2. **Make the RDD path produce a consumed result.** Extend `demand.rdd_daily_order_count` (or a sibling) so the RDD pass also counts **malformed/unparseable rows** (lines failing the field-count check) and returns that tally. Wire the tally into the §2 cleaning-audit table so the RDD output is genuinely used downstream. Keep the fragile-`split` caveat as one honest markdown line ("manual parse for the RDD demo; production reads via the typed loader").
3. **E — lazy-eval cell.** One markdown + ~3-line code cell near the RDD section: build a transformation chain (no job), fire an action, call out transformation-vs-action + lazy DAG (`.explain()` / printed note). Add `check_lazy_eval_documented` to `checks.py`, register in `CHECKS`.
4. Rebuild + re-execute demand section; confirm cache reran demand and downstream behaved.
5. Commit: `feat(demand): load-bearing RDD malformed-row count + lazy-eval callout; move reconciliation to src`.

## Phase 2 — Outlier treatment
Addresses D. Recomputes demand + risk index.

1. **Winsorise `delivery_delay_days`** in `demand.py` feature prep using `approxQuantile` bounds (clip ~[p1, p99]); record clipped-row count. Reuses the `approxQuantile` already computed for EDA — now it drives a transform.
2. **Stabilise the normalisation range** in `convergence.compute_normalisation_ranges`: compute min/max on the winsorised columns (or clamp at percentile bounds) so one extreme seller no longer compresses everyone's `demand_norm`.
3. Log to `decisions_log.md`: bounds, rows clipped, rationale.
4. Rebuild + re-execute; commit: `feat(preprocess): winsorise delivery delay; stabilise risk normalisation range`.

## Phase 3 — Graph rework: add graph-unique signal (core of the pass)
Addresses A. Recomputes the network module.

Keep every existing op (degree, bipartite PageRank, CC, motif, BFS, delayed-subgraph PageRank) — they
satisfy `check_graphframe_ops ≥ 6` and rubric breadth. **Add a seller↔seller co-customer projection
layer**, where the genuinely graph-unique value lives.

1. **Co-customer projection graph** (`network.build_cocustomer_graph`): vertices = sellers (~3k), edges from the cached `network_motifs.parquet` `(seller_a, seller_b, n_shared_customers)`, thresholded `n_shared_customers ≥ 2`. Small → cheap.
2. **Co-customer centrality** (`network.compute_cocustomer_centrality`): PageRank weighted by shared-customer count on the projection = "embeddedness in the substitution network". **Report `corr(cocustomer_centrality, in_degree)` in the notebook to prove it is not a degree proxy.**
3. **Community detection** (`network.compute_substitution_communities`): `gf.labelPropagation(maxIter=5)` on the projection → substitution communities. Replaces the dead "0 isolated sellers" CC finding with informative segmentation. Add `community_id` + community size.
4. **Backup-for-ALL-sellers** (`network.compute_backup_map`): every seller's backup = co-customer partner with max `n_shared_customers`; `backup_strength` = that count. Persist for all sellers (not just top-10 hubs) — makes the motif **load-bearing**. Keep the top-10 BFS hop-count as the "how far is the nearest substitute" traversal finding.
5. **Substitutability-deficit feature:** high `in_degree` (failure impact) × low `backup_strength`/few partners = single point of failure. True degree×neighbourhood interaction, correlates poorly with raw degree — the new graph-unique risk signal.
6. **Extend `nb3_seller_network_scores.parquet`:** add `backup_seller_id` (all sellers), `backup_strength`, `cocustomer_centrality`, `n_cocustomer_partners`, `community_id`, `substitutability_deficit`. Existing columns retained.
7. New safety-log IDs for any small `.toPandas()`/collect feeding the new viz; register in `safety.py` + `big_data_safety_log.md`.
8. Commit: `feat(network): co-customer projection — centrality, communities, backup-for-all, substitutability deficit`.

## Phase 4 — Convergence integration
Addresses A (decision-impact half). Recomputes risk index.

**Open decision (recommendation in bold):** how the network axis enters the risk score.
- **Recommended: redefine `network_norm` as a 50/50 blend of contagion (existing delayed-PageRank `network_risk_score`) and the new `substitutability_deficit`** — graph-unique, non-degenerate across the population (fixes the 68%-zero problem), and genuinely distinct from demand's delay signal.
- Alt A: keep `network_risk_score` as-is, add `substitutability_deficit` only as a recommendation gate (smaller change, weaker fix).
- Alt B: replace contagion entirely with deficit (cleanest story, drops the delayed-subgraph link).

1. Implement the chosen network-axis definition in `convergence.build_seller_risk_index`; carry `backup_seller_id`, `backup_strength`, `community_id`, `substitutability_deficit` into `seller_risk_index.parquet`.
2. **Backup-gated recommendation:** flag "CRITICAL/WARNING seller with weak/no backup (`backup_strength` below threshold) → escalate; else route demand to backup". Surface in §7 — converts the graph insight into an action.
3. Commit: `feat(convergence): graph-unique network axis + backup-gated recommendations`.

## Phase 5 — Notebook narrative, viz, checks, docs
1. **§5 rewrite** in `build_main.py`: co-customer projection / centrality / community / substitutability narrative; **bipartite-PageRank≈degree honesty callout with the measured r**; reframe BFS/motif as load-bearing. Update §5 Key Takeaways and §6/§7 for the new network axis + backup recommendations.
2. **viz.py helpers** (small-DF only): substitutability-deficit scatter (impact×backup), community-size bar, centrality-vs-degree scatter (the proof chart).
3. **checks.py:** add `check_cocustomer_graph`, `check_outlier_treatment`, `check_lazy_eval_documented`; register in `CHECKS`. Update `check_parquet_artefacts` if paths change.
4. **Docs:** CLAUDE.md §5 (graph contract), §7 (new `nb3` + `seller_risk_index` schemas), §8 + §8.1 (graph + new checks). Append `decisions_log.md` per phase. Tick `grading_checklist.md`.
5. Commit: `docs+nb: graph-unique narrative, proof charts, new compliance checks`.

## Phase 6 — Full rerun, reconcile, verify
1. `./run.sh` end-to-end (clean `outputs/_gf_checkpoints/` first). Expect network + convergence + demand to recompute.
2. **Reconcile metrics into CLAUDE.md §7 and `big_data_safety_log.md`:** new risk-band counts, new `corr` values, and **fix F** (LSTM AUC → actual executed value).
3. `checks.run_all(spark)` all green. `scripts/assert_notebook_outputs.py` — every cell has output.
4. `./submission/build_zip.sh` up to the assert step (dry run).
5. Final commit + `git push origin main`.

---

## Risk / rollback
- Each phase is its own commit → revert granularly.
- `outputs/*.parquet` committed; `diff_parquets.py` compares pre/post where schemas are stable.
- Heaviest risk = Phase 3 `labelPropagation` cost — mitigated by running on the ~3k-seller projection (not the 100k bipartite graph) + edge-thresholding.
- If `labelPropagation`/projection misbehaves after two fix attempts → stop and ask (§9).

## Open decisions for sign-off
1. **Network-axis definition (Phase 4):** approve the recommended 50/50 deficit+contagion blend, or pick Alt A / Alt B?
2. **Scope/sequencing:** all phases now, or land 0–2 first and review before the Phase 3 graph rework?
