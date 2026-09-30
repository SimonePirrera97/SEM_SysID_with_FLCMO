# Example 1: Fluid Damper

## Files and responsibilities

The example folder intentionally contains only the dataset, the files needed to
run the current comparison, and this documentation:

| File | Task |
|---|---|
| `main_es1.m` | Top-level entry point. Applies a uniform one-minute fit limit, runs the five methods with full-sequence NNOE last, combines all run-level results, and saves the final summaries. |
| `es1_prepare.m` | Loads the dataset, creates the chronological train/validation/test split, fixes the order-4/eight-neuron architecture, generates the configured common parameter initializations, and creates the output directory. |
| `es1_log.m` | Writes every completed operation and its result to both the MATLAB terminal and `results/results.txt`. |
| `es1_metrics.m` | Computes the common simulation RMSE and FIT after excluding the four initial-condition samples. |
| `method_flcmo.m` | Validates the FLCMO integration rate and trains the constrained-manifold NNOE for the configured common starts. |
| `method_nnsysid_nnoe.m` | Maps the common weights into NNSYSID20 format, validates the initial LM damping, and executes one downloaded `nnoe()` call per configured initialization. |
| `method_pytorch_nnarx.m` | MATLAB launcher for the PyTorch one-step-ahead NNARX experiment. |
| `method_pytorch_nnoe.m` | MATLAB launcher for full-sequence PyTorch NNOE; passes the fixed 60-second limit to Python. |
| `method_pytorch_nnoe_batched.m` | MATLAB launcher for truncated PyTorch NNOE with candidate sequence lengths 32, 64, and 128. |
| `run_pytorch_es1.py` | Implements the shared neural model, common-weight mapping, training, validation selection, and temporary Python-to-MATLAB result exchange. |
| `data/data_es2.mat` | Original fluid-damper estimation and independent test records. |
| `README_REVISED.md` | Defines the experiment, tuning grids, fairness rules, stopping conditions, metrics, dependencies, and generated outputs. |

The MATLAB files also use shared implementations located one level above the
example: `flcmo_optimize.m`, `flcmo_step.m`, `nnoe/`, and `native/`. Python uses
the environment and dependencies declared at the repository root's `.venv` and
`requirements.txt`. NNSYSID20 remains an external dependency; see below. The
persistent outputs are `results/results.txt`, `results/results.csv`,
`results/summary.csv`, and `results/summary.tex`.

## Running the experiment

Run the complete comparison from MATLAB:

```matlab
cd Example1_FluidDamper
summary = main_es1;
```

`main_es1.m` prepares the common data and initializations, runs the five
identification methods and writes the configured-run comparison to `results/results.csv`.

The number of common initializations per method is read from `cfg.runs` in
`es1_prepare.m`, and runtime messages print that current value. The hyperparameter
candidate lists are independent of the run count and are left unchanged.
During execution, every completed preparation, candidate evaluation,
hyperparameter selection, fixed-initial-condition identification, aggregate
calculation, and result-saving task is printed and appended to
`results/results.txt`. The file is reset at the start of each complete run.
Intermediate files use a temporary directory that is deleted automatically;
model checkpoints are not saved.

The Python methods use `.venv/bin/python` (the repository-root environment,
see the top-level README) when it exists and otherwise fall back to `python3`.
They require NumPy, SciPy, and PyTorch. The MATLAB NNOE comparison requires
[NNSYSID20](https://www.mathworks.com/matlabcentral/fileexchange/61779-nnsysid);
by default `es1_prepare.m` looks for it at
`<userpath>/SupportPackages/NNSYSID20` (MATLAB's usual Add-On install
location). Point it elsewhere by setting the `NNSYSID20_PATH` environment
variable before launching MATLAB.

## Data protocol

The supplied 2,000-sample estimation record is divided chronologically:

- Samples 1--1600: estimation/training data used to fit each candidate.
- Samples 1601--2000: validation data used only to select algorithm
  hyperparameters.
- The separate 1,499-sample validation record supplied with the original data is
  treated as the final test record and is not used for hyperparameter selection.

## Fixed model and common initializations

Every method uses the same nonlinear model:

- Dynamical order: 4.
- Hidden layers: one.
- Hidden neurons: 8.
- Hidden activation: `tanh`.
- Output activation: linear.
- Regressors: four past outputs, plus the current and four past inputs.

The model therefore implements

```text
yhat(k) = W2*tanh(W1y*[yhat(k-1),...,yhat(k-4)]
                     + W1u*[u(k),...,u(k-4)] + b1) + b2.
```

`cfg.runs` canonical 89-parameter initial points are generated with MATLAB's
`twister` generator using seeds `1001` through `1000+cfg.runs` and scale `0.01*randn`. The canonical
ordering is `vec(W1y')`, `vec(W1u')`, `b1`, `W2`, `b2`. Each implementation maps
this vector into its native parameter representation without changing the
initial network. The vectors are passed to Python through temporary storage.

During hyperparameter selection, all candidates for a method start at common
initialization 1. After selection, each method is trained from all `cfg.runs`
common initializations. FLCMO has additional optimization variables representing
the complete training-output trajectory. They use deterministic `0.01*randn`
initial values, matching the successful default `identify_nnoe` behavior, while
the network parameters remain exactly the common parameters used by all methods.

Every method uses the same regularization coefficient, `1e-3`: FLCMO through
`regularization`, NNSYSID through `D`, and PyTorch Adam through `weight_decay`.
This was enabled after an unregularized FLCMO diagnostic with `stepSize=2e-3`,
`K=1`, and 1,000 iterations still gave poor FIT on the tested common starts. The
hyperparameter candidate lists below remain unchanged. Every validation
candidate and every final fit has the same 60-second wall-time budget. The limit
is checked at safe optimizer boundaries, so an in-progress iteration or epoch
may finish just after the nominal limit.

## Hyperparameter selection

For every method, each candidate is fitted on samples 1--1600 and recursively
simulated on samples 1601--2000. The first four validation samples supply the
simulation initial condition and are omitted from the score. The candidate with
the highest validation FIT is selected. Validation RMSE, FIT, runtime, and the
relevant stopping count are recorded where available. The final test record has
no role in selection.

### FLCMO NNOE

File: `method_flcmo.m`.

FLCMO gain is fixed at 1 and regularization is fixed at `1e-3`. Only the integration
step size (rate) is tuned:

| Candidate | Step size |
|---:|---:|
| 1 | 0.0001 |
| 2 | 0.0005 |
| 3 | 0.001 |

Candidates run for at most 250 iterations. The selected rate is used for all configured
final runs, which run for at most 2,000 iterations. The optimizer can also stop
on its numerical tolerance or when monitored training FIT stabilizes according
to the defaults in `identify_nnoe.m` and `flcmo_optimize.m`. Candidate results are
logged to `results/results.txt`; final metrics go to `results/results.csv`.

### NNSYSID20 NNOE

File: `method_nnsysid_nnoe.m`.

This comparison calls NNSYSID20's `nnoe()` Levenberg--Marquardt implementation.
Weight decay is fixed at `D=1e-3`. Only the initial LM damping parameter is tuned:

| Candidate | Initial lambda |
|---:|---:|
| 1 | 0.1 |
| 2 | 1 |
| 3 | 10 |

Candidates run for at most 100 LM iterations. The selected initial damping is
used for all configured final runs, which run for at most 500 iterations and otherwise
use NNSYSID's stopping rules. NNSYSID's `skip` value is set to 4. Candidate
results are logged; final metrics go to `results/results.csv`.

### PyTorch NNARX

Files: `method_pytorch_nnarx.m` and `run_pytorch_es1.py`.

NNARX minimizes one-step-ahead mean squared error, using measured past outputs
in its training regressors. Adam learning rate and mini-batch size are tuned:

| Candidate | Learning rate | Batch size |
|---:|---:|---:|
| 1 | 0.001 | 32 |
| 2 | 0.0005 | 32 |
| 3 | 0.001 | 64 |
| 4 | 0.0005 | 64 |

Candidates train for at most 300 epochs. Final runs train for at most 1,000
epochs. Training stops after 40 epochs without a full-training MSE improvement
greater than `1e-8`. Mini-batch permutations use a fixed generator seed so that
candidate comparisons and runs do not receive additional shuffle randomness.
Candidate results are logged; final metrics go to `results/results.csv`.

### PyTorch NNOE

Files: `method_pytorch_nnoe.m` and `run_pytorch_es1.py`.

This method recursively simulates the neural model and minimizes the simulation
mean squared error

```text
mean((y_measured(k) - yhat_simulated(k; theta))^2),  k = 5,...,N.
```

It therefore differentiates through the complete simulated-output recurrence,
rather than fitting one-step-ahead targets. It uses full-record Adam updates.
Only the learning rate is tuned:

| Candidate | Learning rate |
|---:|---:|
| 1 | 0.001 |
| 2 | 0.0005 |
| 3 | 0.0001 |

Candidates train for at most 250 epochs. Final runs train for at most 1,000
epochs. Gradients are clipped to norm 10. Training stops after 40 epochs without
a loss improvement greater than `1e-7`, and the best parameter state encountered
during training is restored.

Like every other validation candidate and final fit, each full-sequence fit has
a fixed 60-second wall-clock training limit, independent of FLCMO runtime, and
this method is executed last. The limit is
checked after every complete recursive
simulation/gradient epoch. Once exceeded, training stops and restores the best
state reached before termination. The per-run CSV records `timedOut` and the
applied `timeoutSeconds`. Candidate results are logged;
final metrics go to `results/results.csv`.

### PyTorch batched NNOE

Files: `method_pytorch_nnoe_batched.m` and `run_pytorch_es1.py`.

This additional comparison uses truncated simulation-error training. Each
training sequence starts from four measured outputs, then recursively simulates
the remaining samples in that sequence and backpropagates through that truncated
recurrence. Adjacent sequences overlap by the four initial-condition samples so
that all prediction targets are covered.

Sequence length and Adam learning rate are selected jointly on validation data:

| Candidate | Sequence length | Learning rate |
|---:|---:|---:|
| 1 | 32 | 0.001 |
| 2 | 32 | 0.0005 |
| 3 | 32 | 0.0001 |
| 4 | 64 | 0.001 |
| 5 | 64 | 0.0005 |
| 6 | 64 | 0.0001 |
| 7 | 128 | 0.001 |
| 8 | 128 | 0.0005 |
| 9 | 128 | 0.0001 |

Candidates train for at most 250 epochs, and final runs for at most 1,000 epochs.
Gradient norm is clipped to 10. Training stops after 40 epochs without a mean
sequence-loss improvement greater than `1e-7`, and the best state is restored.
The uniform timeout also applies to this batched comparison. Candidate results
are logged; final metrics go to
`results/results.csv`.

## Test metrics and reported results

All final models are recursively simulated on the independent 1,499-sample test
record. Its first four measured outputs provide the common initial condition and
are excluded from scoring. The reported metrics are

```text
RMSE = sqrt(mean((yhat - y)^2))
FIT  = 100*(1 - norm(yhat-y)/norm(y-mean(y))).
```

`results/results.csv` contains exactly `method`, initialization/run number,
`testRMSE`, `testFIT`, and `executionTimeSeconds` for all final runs
(`5*cfg.runs` rows).
`results/summary.csv` and `results/summary.tex` report the average, best, worst,
and sample standard deviation of FIT and execution time for every method. The
`.tex` file is a complete `tabular` environment ready for direct inclusion in a
LaTeX document. The complete experiment can take substantial time,
particularly PyTorch NNOE because every update differentiates through the full
recursive simulation.
