# Streaming benchmark

Measured by `optiver-tc benchmark` on 5 holdout days (275 batches, ~200 stocks each).
Each timing is one `StreamingPredictor.predict(batch)` call end to end: input coercion, feature computation on
(bounded history + batch), model inference (LightGBM, 1 thread, + ridge), post-processing.

| measurement | p50 ms | p95 ms | p99 ms | max ms | n |
|---|---|---|---|---|---|
| warm (batch with intraday history) | 33.3 | 37.5 | 42.3 | 70.1 | 270 |
| first batch of a day (state just reset) | 31.9 | 34.3 | 34.5 | 34.5 | 4 |

* First call in a warmed process: 42.6 ms; bundle load 0.04 s.
* Cold start: 1.78 s imports + 0.04 s bundle load + 43 ms first batch (mean of 3 fresh processes).
* Process RSS 205 MB before bundle load (imports + 5 days of raw data) -> 233 MB after replay;
  model files 10.8 MB; streaming state at the end: 1400 rows.
* Environment: Intel64 Family 6 Model 140 Stepping 1, GenuineIntel (4 cores / 8 threads), 16.87 GB RAM,
  Windows-10-10.0.26200-SP0, Python 3.11.9, polars 1.44.1, lightgbm 4.7.0.

Single machine, single run, other applications running. Not a latency guarantee.
