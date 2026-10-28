# The independent PC2 line

This is the separate method discussed on report pages 52–53. It grew from
the challenge's Gaussian covariance baseline and factored cumulant equations,
rather than from Team 504aldo's V29 implementation. The main solution later
adopted its effective-edge birth idea; that implementation is now in
[`solution/module_04.py`](../../solution/module_04.py).

Both files below are complete, unchanged historical predictors. Their fitted
coefficients are stored inside the Python source; they need no additional
parameter files.

| Source | Local adjusted score | Networks | Mean FLOPs |
| --- | ---: | --- | ---: |
| [mixed54.py](mixed54.py) | 4.458518507631481e-9 | mini 40–63, 24 reused networks | 520.305B |
| [centered.py](centered.py) | 4.394732436024916e-9 | mini 16–39, 24 reused networks | 503.415B |

These are **broad statistical historical results**, measured on different
panels under FlopScope 0.12.0 / whestbench 0.16.0. Neither was submitted.
They are not independent holdouts, mini100 results or current-meter scores.
The report's approximate “4.39e-9” refers to the second row.

The predictors retain means and a full covariance, cubic sources with their
own transported factors, a covariance-weighted cubic trace repair, and a
fourth-order trace model. A 28-feature per-layer readout, fitted on public
networks 0–15, adjusts the means. Pooled Winograd multiplication and pruning
reduce the cost. `centered.py` additionally reuses centered-moment calculations.

For the code, begin at `Estimator.setup`, `Estimator.predict` and `_predict`.
`gaussian_third_factors` creates the cubic source; `weighted_third` and
`covariance_weighted_third` query it; `_adjusted_compact_activation_cp`
implements support pruning. The later classes implement the multiplication
plans. Some functions remain as research alternatives and are not active in
the frozen `PREDICTION_CONFIG`.

With the repository's current environment, these small contract checks pass:

```sh
whest validate --estimator research/pc2/mixed54.py --class Estimator --json
whest validate --estimator research/pc2/centered.py --class Estimator --json
```

They do not remeasure the historical scores. Repeating the recorded panels
requires the historical runtime and the dataset commit in [RESULTS.json](RESULTS.json),
with the same row selection. Using the current meter changes the accounting.
[scores.csv](scores.csv) contains all 48 recorded rows; `python evidence/verify.py`
from the repository root recomputes their scores and cohort means.

[reference_frontier.json](reference_frontier.json) preserves selected checkpoint
records behind page 52, including the eight- and sixteen-rule over-budget
cubature references. Their full implementations are **not included** here;
those records are historical evidence, not runnable benchmarks or eligible
competition scores.

The root MIT licence applies to our work. The original AIcrowd starter-kit
notice is retained in [LICENSE_AIcrowd](LICENSE_AIcrowd). This research folder
is separate from the final graded predictor in `solution/`.
