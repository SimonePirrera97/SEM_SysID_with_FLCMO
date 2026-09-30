# Example 3: MIMO Wiener-Hammerstein

Two-input, two-output Hammerstein-Wiener system (static nonlinearities around
four SISO linear blocks). `data/dataset_training.mat` (3,500 samples) and
`data/dataset_test.mat` (5,000 samples) hold the generated staircase-input
records; regenerate them with the Simulink model under `data/` if needed.

Every model uses dynamical order `n=3`. Five NNOE architectures are compared:
`(10)`, `(15)`, `(5,5)`, `(7,7)`, `(5,5,5)` (172, 257, 117, 177, 147 parameters),
each trained with **five random initializations**.

## FL-CMO (this paper's algorithm)

- `run_cmo_common_tau.m` runs the first initialization for all five
  architectures with `K=1`, regularization `1e-3`, step `tau=1e-2`, on the full
  3,500-sample record; FIT-fatigue stopping restores the best training-FIT
  iterate. Output: `results/fatigue_common/cmo_common_tau_models.csv` (+
  `cmo_common_tau_warmup.csv/.mat` for the step-size warm-up).
- `run_cmo_replicates.m` repeats the same five architectures for four more
  initializations. Output: `results/replicates/cmo_replicates.csv` and the
  per-run `cmo_rep{1..4}_row_XX.mat` models.
- `results/replicates/nnoe10_full_timecap900.csv` is the full-sequence
  (non-truncated) NNOE(10,1) comparison run quoted in the text (900 s time
  cap; converges far more slowly and less accurately than FL-CMO).

Combining the base run with the four replicates gives the five-initialization
mean/std reported in the paper's table.

## Adam baselines (NNOE, LSTM, GRU)

All three families are trained with Adam, length-512 truncated sequences and
batches of 64, regularization `1e-3`:

- `run_recurrent_l512.py` trains LSTM and GRU on the same five
  (order, architecture) rows as the NNOE/FL-CMO comparison, and the NNOE
  architectures with Adam. Output: `results/fatigue_common/recurrent_l512.csv`
  (base initialization) and `nnoe_adam_horizons.csv` (NNOE-Adam at this and
  other candidate horizons).
- `run_adam_replicates.py` repeats LSTM, GRU and Adam-trained NNOE for four
  more initializations. Output: `results/replicates/adam_replicates.csv`
  (+ failures in `adam_failed_attempts.csv`) and the per-run model files.
- The learning rate `3e-3` (selected from `{3e-3, 1e-3, 3e-4}`) was chosen per
  family from one representative architecture on a chronological 2800/700
  split of the training data before running the campaigns above.

## Running

From the repository root:

```matlab
addpath('Example3_MIMOWienerHammerstein', 'nnoe', 'native')
run_cmo_common_tau
run_cmo_replicates
```

```sh
python3 Example3_MIMOWienerHammerstein/run_recurrent_l512.py
python3 Example3_MIMOWienerHammerstein/run_adam_replicates.py
```

`run_python_baselines.py`, `run_batched_nnoe_horizons.py` and
`run_overnight_campaign.py` are the underlying/orchestration utilities that the
scripts above import or extend for further exploration beyond the paper's grid.
