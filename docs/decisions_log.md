# Decisions log

Append-only record of autonomous choices made during the build. Used as the source of truth for the oral defence ("why did you choose X?"). Keep entries short — one heading + 2–4 lines.

## Format
```
### YYYY-MM-DD — short title
**Choice:** what was decided.
**Why:** the reason in one sentence.
**Impact:** what this affects (which notebook / output / metric).
```

---

### 2026-04-20 — Project scaffold and CLAUDE.md rewrite
**Choice:** Replaced 600-line CLAUDE.md with ~180-line contract-style version. Added `src/olist/` package (spark_session, schemas, loaders, transforms), `requirements.txt`, `docs/grading_checklist.md`, `submission/build_zip.sh`, `presentation/outline.md`.
**Why:** Original spec had a path bug (`data/raw/` vs. `data/`), prescribed section orders that limited adaptability, and burned tokens on a manual checkbox tracker that duplicates parquet-existence ground truth.
**Impact:** All future sessions; reduces per-session token load by ~70% while preserving every functional requirement.

### 2026-04-20 — Deep learning consolidated to NB2 only; LSTM picked over BERT
**Choice:** No LSTM in NB1 (GBT vs. RandomForest is enough for the MLlib rubric line). NB2 uses an LSTM in PyTorch, not multilingual DistilBERT.
**Why:** Rubric rewards the "Deep Learning" line once, not twice. BERT is ~500 MB and slow without a GPU; LSTM trains in minutes on Portuguese review text and answers the same rubric item.
**Impact:** Cuts model-debugging surface; frees time for NB1 streaming bonus and a tighter NB3.

### 2026-04-20 — Graph vertices use customer_unique_id, not customer_id
**Choice:** NB3 builds the GraphFrame with `customer_unique_id` for the customer-side vertices.
**Why:** `customer_id` is one-per-order in the Olist schema; PageRank and shared-customer motifs need the actual person.
**Impact:** NB3 motif counts and PageRank reflect repeat-customer behaviour correctly.

### 2026-04-22 — NB1 executed; dataset and model choices logged
**Choice:** Dropped 2,965 of 99,441 orders where `order_delivered_customer_date IS NULL` (≈3%). Chose `delay_risk_flag = (avg_delay_days > 3)` — p75 of delay is −8 days (most orders arrive early), so >3 cleanly isolates true late outliers. Selected `RandomForestRegressor` over `GBTRegressor` (test RMSE 5.06 vs 5.09 on the 80/20 time-aware split). CV used 3 folds, small param grids (GBT: maxDepth∈{3,5}; RF: maxDepth∈{5,10}).
**Why:** Required by notebook contract. Timeline-aware split avoids leakage; RF won by a hair and is also faster to re-score.
**Impact:** `outputs/nb1_seller_demand_scores.parquet` uses RF predictions. `delay_risk_flag` distribution is sparse (most sellers ship early) — this is the signal NB3 will fuse with network risk.

### 2026-04-22 — NB3 executed; graph size forced 6g driver memory + GraphX CC
**Choice:** Bumped `spark.driver.memory` from the 2g default to **6g** in `src/olist/spark_session.py` (applies to all three notebooks). `connectedComponents(algorithm="graphx")` swapped in for NB3 — the default message-passing variant OOMed the JVM heap on the ~100k-vertex bidirectional graph. Checkpoint dir at `outputs/_gf_checkpoints/` (gitignore-worthy, not committed to the deliverable).
**Why:** First NB3 run failed with `java.lang.OutOfMemoryError: Java heap space` inside `ConnectedComponents.skewedJoin`. GraphX CC is more memory-efficient on small-medium graphs and completes in under a minute.
**Impact:** NB3 now runs end-to-end. Isolated sellers = 0 (every seller has at least one delivered-order edge), so `is_isolated` is always 0 in the output parquet — still present per the contract.

### 2026-04-22 — Risk index: 0 CRITICAL, 68 WARNING, 2899 SAFE (threshold band kept)
**Choice:** Kept the contract-specified bands (CRITICAL > 0.75, SAFE < 0.40, WARNING otherwise) even though the max observed `risk_score` is 0.71 (so 0 sellers land in CRITICAL). Top WARNING seller has `avg_delay_days = 167`, `avg_sentiment_score = 1.0` — the index correctly isolates the extreme-risk tail.
**Why:** Thresholds were specified up-front; adjusting them post-hoc to the observed distribution would be data-snooping. The zero-CRITICAL finding is itself informative for the presentation narrative ("Olist's tail-risk sellers cluster at WARNING, not CRITICAL").
**Impact:** Presentation slide 7 (risk quadrant) should call this out — "no CRITICAL class populated; 68 WARNING sellers concentrated in X states".

### 2026-04-22 — NB2 executed; sentiment labels, LSTM fix, lead-indicator finding
**Choice:** Labels = `score ≥ 4 → 1`, `score ≤ 2 → 0`; neutrals (score == 3) dropped. Class balance 84,840 positive / 18,109 negative (≈82/18) out of 102,949 labelled review-seller rows; 43,337 have non-null comment text. LSTM uses masked mean-pool over time (right-padding bug caused AUC=0.50 on first run — fix is in-notebook). Lead-indicator cross-correlation between weekly sentiment change and weekly volume change is ~0 across lags 0–8; peak |corr|≈0.015 at lag=7w — reported as a weak signal, not a strong lead indicator.
**Why:** Dropping neutrals keeps the classifier bimodal; masked mean-pool is the minimal fix for fixed-length padding; the lead-indicator finding is honest rather than overclaimed.
**Impact:** LSTM test AUC = 0.9630 (beats LogReg baseline 0.9580). `outputs/nb2_seller_sentiment_scores.parquet` contains 3,090 sellers; 491 flagged as `sentiment_declining = 1`.
