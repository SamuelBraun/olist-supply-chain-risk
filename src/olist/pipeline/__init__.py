"""Public pipeline API used by the four notebooks.

Each submodule exposes pure-ish Spark functions that the notebooks call.
Functions whose computation is expensive are wrapped with the @step
decorator from `olist.cache` so that reruns on unchanged inputs + code
skip the compute and re-read the cached parquet.

Layout:
    pipeline/
        demand.py        — NB1: RDD warm-up, SparkSQL, feature pipeline,
                           GBT+RF cross-validation, per-seller demand scores
        sentiment.py     — NB2: TF-IDF+LR, LSTM, weekly rolling sentiment,
                           lead-indicator cross-correlation
        network.py       — NB3: GraphFrame, PageRank, connected components,
                           motifs, BFS backups, high-delay subgraph
        convergence.py   — final convergence layer used by 00_main.ipynb
"""
