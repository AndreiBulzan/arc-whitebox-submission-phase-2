# Run the estimator

Use Python 3.10 on Linux or WSL. From this directory:

```sh
python3.10 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
sha256sum --check SHA256SUMS
python evidence/verify.py
whest validate --estimator solution/estimator.py --json
```

`validate` checks the estimator contract on a small network. To evaluate one
full-size network with the public dataset and the official local harness:

```sh
whest run --estimator solution/estimator.py \
  --dataset hf://aicrowd/arc-whestbench-public-2026@v2-phase2 \
  --split mini --n-mlps 1 --runner subprocess --max-threads 1 \
  --detail full --json > results-local.json
```

Set `--n-mlps 100` to evaluate the entire public mini split. That local bank
is not the same 50-network leaderboard panel. The dataset is downloaded on
first use and is not included here. The default Phase 2 limits remain in
force: 2**41 FLOPs, 120 seconds per prediction, 400 milliseconds of residual
time, and 5 seconds for setup. No challenge login or submission is needed.

The estimator selects exact multiplication policies using elapsed time.
Hardware therefore affects the selected policies, FLOP count and floating-point
rounding. Local timing and scores need not equal the published evaluation.
All numerical operations in the predictor use FlopScope; NumPy and SciPy in
the environment serve the evaluation dependencies, not direct predictor imports.

## Recorded results

The precise published score and resource measurements are in [RESULTS.json](RESULTS.json).
The public leaderboard mean covers 50 networks; the complete execution ran
100 networks. Each adjusted error is its own MSE multiplied by
`max(0.1, FLOPs / 2**41)`, averaged over the scored networks.

The [evidence guide](evidence/README.md) explains the saved historical tables.
