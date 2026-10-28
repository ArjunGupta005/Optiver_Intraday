# Streaming benchmark

Measured by `optiver-tc benchmark` on 5 holdout days (275 batches, ~194 stocks each).
Each timing is one `StreamingPredictor.predict(batch)` call end to end: input coercion, feature computation on
(bounded history + batch), model inference (LightGBM, 1 thread, + ridge), post-processing.

| measurement | p50 ms | p95 ms | p99 ms | max ms | n |
|---|---|---|---|---|---|
| warm (batch with intraday history) | 25.8 | 29.7 | 30.7 | 32.0 | 270 |
| first batch of a day (state just reset) | 21.5 | 23.9 | 24.2 | 24.3 | 4 |

* First call in a warmed process: 20.8 ms; bundle load 0.02 s.
* Cold start: 2.08 s imports + 0.02 s bundle load + 25 ms first batch (mean of 3 fresh processes).
* Process RSS 420 MB before bundle load (imports + 5 days of raw data) -> 402 MB after replay;
  model files 1.2 MB; streaming state at the end: 1358 rows.
* Environment: Intel64 Family 6 Model 140 Stepping 1, GenuineIntel (4 cores / 8 threads), 16.87 GB RAM,
  Windows-10-10.0.26200-SP0, Python 3.11.9, polars 1.44.1, lightgbm 4.7.0.

Single machine, single run, other applications running. Not a latency guarantee.
