# System identification through FL-CMO optimization

Code for the constrained-optimization approach to nonlinear system identification
by simulation-error minimization (SEM), as proposed in "A constrained optimization
approach to nonlinear system identification through simulation-error minimization"
by Vito Cerone, Sophie M. Fosson, Simone Pirrera, and Diego Regruto.

FL-CMO casts SEM as a constrained optimization problem over the model parameters
and the full simulated trajectory, and solves it with a QR-based feasible-descent
iteration. Every example below uses the same generic solver
(`flcmo_optimize.m`, `flcmo_step.m`) and the same packed-Jacobian Householder MEX
solver (`native/`); only the model-specific constraint function changes.

## Layout

```text
flcmo_optimize.m, flcmo_step.m       generic FL-CMO solver
nnoe/                                 NNOE model and constraint utilities
native/                               packed/QR MEX sources and binaries
Example0_MotivatingExample/           scalar example of Sec. 3 (gradient/iteration figures)
Example1_FluidDamper/                 magneto-rheological damper benchmark
Example2_BoucWen/                     Bouc-Wen hysteresis benchmark
Example3_MIMOWienerHammerstein/       MIMO Wiener-Hammerstein system
Example4_GrayBoxMagneticLevitation/   gray-box magnetic levitator
tests/                                numerical regression tests
benchmarks/                           timing scripts and reports
```

Each example folder has its own README with the exact experimental protocol
(data split, architecture, hyperparameter grid, stopping rule) and points to the
`results/` files that reproduce the tables in the paper.

## Running the examples

From the repository root, in MATLAB:

```matlab
run('Example1_FluidDamper/main_es1.m')
run('Example2_BoucWen/main_smart_search.m')          % see its README for the exact options
run('Example3_MIMOWienerHammerstein/run_cmo_replicates.m')
addpath('Example4_GrayBoxMagneticLevitation', 'native')
run_optimized_tau(2e-3, "flcmo")                     % Example 4, FL-CMO
```

Python baselines (PyTorch / TensorFlow) live alongside the MATLAB drivers in
each example folder: `run_pytorch_es1.py` (Example 1: NNARX/NNOE/truncated
NNOE), `run_recurrent_l512.py` and `run_adam_replicates.py` (Example 3:
LSTM/GRU/Adam-trained NNOE), `run_adam_comparison.py` (Example 4: Adam BPTT).
Each example's own README gives the exact invocation and options. Create an
environment with:

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Generic solver interface

`flcmo_optimize` has no notion of NNOE or any other model; the model only enters
through a constraint function returning the constraint values and their packed
Jacobian:

```matlab
[constraints, jacobian] = constraintFunction(x);
gradient = objectiveGradient(x);

result = flcmo_optimize(x0, objectiveGradient, constraintFunction, ...
    stepSize=1e-3, maxIterations=1000, gain=1);
```

For NNOE models the compact entry point is:

```matlab
model = struct('order',4,'hidden',6);
result = identify_nnoe(input, output, model);
```

The default constraint implementation is `@nnoe_constraints_mex`. A readable
MATLAB implementation with the identical signature is available for inspection
or for platforms without the compiled binary:

```matlab
result = identify_nnoe(input, output, model, ...
    constraintFunction=@nnoe_constraints_matlab);
```

Example 4 demonstrates a non-NNOE constraint model (a gray-box ODE); its run
script selects the implementation through the same kind of handle
(`@levitator_constraints_mex` / `@levitator_constraints_matlab`).

## Native code

Compiled Apple-silicon (arm64) binaries are included:

- `nnoe/nnoe_constraints_mex.mexmaca64`
- `Example4_GrayBoxMagneticLevitation/levitator_constraints_mex.mexmaca64`
- `native/sidqr_mex.mexmaca64`

On another platform, use the `*_matlab` fallback implementations, or rebuild from the C sources in `native/` and each example folder with MATLAB's `mex`, or (on macOS, without a full Xcode install) with:

```sh
./build_mex_clang.sh
```

## Validation

```matlab
addpath tests benchmarks
test_mex_correctness
test_example4_providers
benchmark_examples
```

`test_mex_correctness` checks that the MATLAB and MEX constraint
implementations agree at machine precision; `benchmarks/` reports the
measured speedup of the packed/MEX path over the plain MATLAB one.
