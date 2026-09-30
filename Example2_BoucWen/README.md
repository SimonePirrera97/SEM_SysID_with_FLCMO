# Example 2: Bouc-Wen

The [Bouc-Wen hysteretic benchmark](https://www.nonlinearbenchmark.org/benchmarks)
records are generated with the benchmark's own MATLAB generator and Newmark
integrator (`BenchmarkFiles/`), wrapped by `generate_boucwen_data.m`, which fixes
the random-phase seed and saves the noiseless record to `data/boucwen_data.mat`.
Regenerate it from the repository root with:

```matlab
addpath('Example2_BoucWen')
generate_boucwen_data
```

## Identification campaign

Entry point: `main_smart_search.m`. The paper's campaign uses the dynamical
order `n=3` and the two architectures `NNOE(8,8)` and `NNOE(6,6)` (145 and 97
parameters):

```matlab
addpath('Example2_BoucWen', 'nnoe', 'native')
main_smart_search(orders=3, architectures={[8 8],[6 6]})
```

The first 12,000 samples of the generated record are used for training and the
next 5,000 for validation; both are corrupted with white Gaussian measurement
noise of standard deviation `8e-3` mm (the sampling frequency is exactly twice
the noise bandwidth, so this is also the discrete-time bandwidth). The
noiseless multisine test record shipped with the benchmark
(`BenchmarkFiles/Test signals/Validation signals/{uval,yval}_multisine.mat`) is
the held-out test set.

For each of the 8 (architecture, `K`, regularization) settings, four 75-iteration
probes rank the candidate steps `tau in {5e-4, 1e-3, 2e-3, 5e-3}` by training BFR;
training then restarts from the common initialization with the best-ranked step.
Training stops when validation BFR (checked every 25 iterations) fails to
improve by `0.001` over 20 consecutive checks, and the best iterate is restored.
Model selection is based exclusively on validation BFR; test BFR is reported for
every completed setting but never used for selection.

Results land in `results/smart_search_12000/`:

- `model_results.csv` — one row per (architecture, `K`, regularization) with
  training/validation/test BFR, iteration count and training time.
- `tau_probes.csv`, `full_attempts.csv` — the step-size probing and restart log.
- `models/setting_XX_init_01.mat` — the trained model for each setting.

`main_smart_search` also accepts a wider `orders`/`architectures`/`gains`/
`regularizations` grid (its defaults) for further exploration beyond what the
paper reports.
