# Presentation outline — slide ↔ output mapping

10-slide hard cap, NOVA IMS template, exported as PDF. Each slide is paired with the notebook artefact that powers it so nothing graded is left out.

| # | Slide | Powered by |
|---|---|---|
| 1 | Title — *Olist Supply Chain Risk Intelligence* | — |
| 2 | The client's problem (one number: % of late deliveries; one anecdote) | NB1 SparkSQL Q3 (sellers ranked by avg delay) |
| 3 | Why Spark — 4 V's framed for Olist (volume of orders, velocity of reviews, variety of joins, veracity of seller behaviour) | — (narrative) |
| 4 | Approach — three angles converging into one Risk Index | architecture diagram (one image) |
| 5 | Demand forecast — chart of weekly volume + forecast for one CRITICAL seller | NB1 forecast output, screenshotted |
| 6 | Customer sentiment as a leading indicator — chart of sentiment trend leading volume drop | NB2 lead-indicator section |
| 7 | Network view — top 10 PageRank sellers + isolated-cluster map | NB3 PageRank top-N, state bar chart |
| 8 | Unified Seller Risk Index — quadrant plot + tier counts | Convergence layer quadrant + bar chart |
| 9 | What the client does next — 3 concrete actions per tier (CRITICAL / WARNING / SAFE) | `outputs/seller_risk_index.parquet` top 10 |
| 10 | Q&A / references | — |

## Rules
- Non-technical language. No model names, no hyperparameters.
- Every chart that appears here must be reproducible from a parquet in `outputs/` — the export step copies the inline notebook image.
- Cap of 10 slides is hard; cut slide 3 first if needed (it is narrative, not data).
