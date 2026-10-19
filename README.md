# Constraint-aware adversarial NIDS: replication package

Reproducible experiments for **Constraint-aware Adversarial Robustness and Cross-dataset Generalization for Network Intrusion Detection Systems**.

Public code and results: https://github.com/ictu-se/constraint-aware-adversarial-nids

Version 2.1.0 corrects integer-rounding overshoot by enforcing the final normalized perturbation budget in projected, top-ten and augmentation candidates. Protocol identifiers and source/destination ports are immutable where present. Affected experiments have been rerun; four publication figures are exported at publication size as RGB LZW TIFF and PNG. Infiltration repetitions are model seeds on one fixed partition; UNSW/KDD use subsampling within fixed partitions, and Bot-day uses repeated splits.

Version 2.0.0 added target-specific score-query attacks, unseen-attack defense evaluation, validation-only threshold sensitivity, repeated CPU timing, and a second CSE-CIC day. It corrects infiltration-sample imputation to use training medians after splitting. The earlier release remains at https://doi.org/10.5281/zenodo.22161883; that archive does not contain these additions.

## Install

The reported environment is Python 3.9.6 on an Apple M4 Max CPU (48 GB RAM). No GPU is needed.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproduce

Run commands from the repository root, sequentially. Experiments write into `results/`; they overwrite matching generated outputs. Use an isolated clone to preserve the published observations. Seeds and hyperparameters are explicit in each script.

```bash
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4

# Base benchmark; downloads UNSW-NB15 and fetches KDDCup99 if absent.
python scripts/run_full_experiment.py

# Restore the exact bounded infiltration sample and rerun the four main settings.
python scripts/sample_cse_cic_2018.py
python scripts/run_q1_upgrade_experiment.py

# Paired clean-model comparisons and within-/cross-dataset five-feature controls.
python scripts/run_revision_analyses.py

# Direct target queries and validation calibration.
python scripts/run_r2_checks.py --skip-latency

# Bounded benign/Bot sample from 2 March 2018, with three within-day splits.
python scripts/run_additional_day.py

# Time saved targets after all experiment workloads have finished.
python scripts/measure_r2_latency.py

# Recreate figure panels and the four assembled figures from result tables.
python scripts/regenerate_manuscript_figures.py
python scripts/build_submission_figures.py

# Check archived measurements against their underlying predictions and traces.
python scripts/test_attack_invariants.py
python scripts/audit_r2_results.py
python scripts/verify_package.py
```

The score-query study can be run with `--skip-latency` when other computation is active. Then, after all other jobs finish, run `python scripts/measure_r2_latency.py`. Models generated locally for this purpose are ignored by Git; they are not required as download artifacts. `--data-dir`, `--out-dir`, and `--queries` on the query script support alternate runs; changing them changes the experiment. The reported budget is 128 candidate queries plus the initial score query, with a checkpoint at 32. Queries inspect each target's probabilities; they do not require gradients or model parameters.

## Contents and measurement units

- `scripts/`: executable experiment, sampling, analysis, figure-generation, and verification workflows.
- `inputs/`: compressed copies of the exact two bounded external study samples, plus source attribution and hashes. These are study subsets of a public benchmark, not privately collected traffic.
- `results/full_experiment/`: primary model/attack/transfer summaries and metadata; clean and transfer scores are unchanged, while attack results reflect the corrected budget.
- `results/q1_upgrade/`: recomputed 240 clean and 2,160 attack-condition rows; 240 compressed per-flow evidence files retain labels, clean and attacked predictions, and probabilities for all conditions.
- `results/revision_analyses/`: three paired comparisons and ten feature-alignment rows, paired predictions, 2,000 bootstrap differences for each comparison, and within-dataset full/five-feature predictions.
- `results/r2_checks/`: 18 model/defense/seed evidence sets, 144 metric rows, 18 thresholds, query trajectories, original/final adversarial feature values, training/validation/test identities and constraint manifests, and 1,080 timing observations.
- `results/additional_day/`: 18 metric rows and nine per-flow evidence sets for three models and three seeds on the second day.
- `figures/`: data-derived panels and the four assembled figures.

The three seed values are 20260711, 20260712, and 20260713. The normal-form half-width `1.96 * sample_sd / sqrt(3)` in legacy-named `*_ci` summaries is **descriptive**, not a calibrated 95% confidence interval. The query and second-day results use mean and observed seed range. The paired bootstrap concerns a fixed shared 50,000-flow test partition and does not measure independent deployment or training uncertainty.

Each timing value is elapsed batch time divided by 2,000 flows. Five warm-ups precede 30 calls per fitted model/mode/seed; model-only and scaler-plus-model probability prediction are separate. Four tree workers and four numerical-library threads are used. Percentiles pool 90 technical repetitions per model/defense/mode, not individual request latencies. Parsing, imputation, feature extraction, attacks, and network I/O are excluded. Hardware scheduling affects exact timings; predictions are the primary deterministic checks.

## Data and preprocessing

UNSW-NB15: https://research.unsw.edu.au/projects/unsw-nb15-dataset

KDDCup99: https://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html (fetched through scikit-learn).

CSE-CIC-IDS2018: https://www.unb.ca/cic/datasets/ids-2018.html (public processed CICFlowMeter data from the CIC/AWS collaboration).

The external inputs are bounded, class-balanced subsets, not representative samples of full days. The infiltration input is the exact 29,988-row sample used in the original experiment; the added Bot input contains 6,000 rows per class selected from successive chunks until quotas were filled. The second input includes source-row indices. `inputs/provenance.json` records hashes, counts, source URLs, and sampling scope. Dataset rights remain with the original providers; the MIT license applies to author-generated code, not third-party data.

The scripts split before learning imputation and feature selection, then apply frozen training transforms. Training ranges, empirical integer membership, model scaling, and surrogate fitting exclude test data. The query test also excludes calibration rows from fitting. A frozen 0.35 cutoff is retained as the main experiment's uncalibrated alert heuristic; the additional threshold test uses a disjoint training-file validation set and a prespecified 5% benign-FPR rule. It does not retune against test labels.

The manifest is a feature-space approximation. Name-matched monotonicity and empirical integrality are not proofs of packet realizability, dependency consistency, or preserved attack functionality. All constrained attacks and augmentation enforce the same final quantile-span coordinate budget after rounding, with a normalized tolerance of 1e-5 for float32 arithmetic. Coordinates outside that interval revert to their observed values. Query search can move either way on non-monotonic coordinates; transferred attacks retain their surrogate sign. Protocol and endpoint fields remain fixed in both cases.

## Scope of release

Only study code, exact dependencies, the two attributed study samples, and evidence required to reproduce results and figures are included. Manuscript sources/PDFs, journal templates, response letters, review comments, local notes, caches, trained model binaries, build products, and private files are excluded.
