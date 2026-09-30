#!/usr/bin/env python3
"""Validate the completeness and cleanliness of the replication package."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FILES = (
    "LICENSE",
    "CITATION.cff",
    "README.md",
    "requirements.txt",
    "scripts/build_submission_figures.py",
    "results/full_experiment/adversarial_robustness.csv",
    "results/full_experiment/clean_performance.csv",
    "results/full_experiment/cross_dataset_transfer.csv",
    "results/full_experiment/latency_throughput.csv",
    "results/q1_upgrade/q1_adversarial_by_seed.csv",
    "results/q1_upgrade/q1_clean_by_seed.csv",
    "results/q1_upgrade/q1_latency_by_seed.csv",
    "results/revision_analyses/paired_model_tests.csv",
    "results/revision_analyses/feature_alignment_decomposition.csv",
)
FORBIDDEN_SUFFIXES = {".aux", ".bbl", ".blg", ".log", ".pdf", ".tex"}


def main() -> None:
    missing = [name for name in REQUIRED_FILES if not (ROOT / name).is_file()]
    forbidden = [
        str(path.relative_to(ROOT))
        for path in ROOT.rglob("*")
        if path.is_file() and path.suffix.lower() in FORBIDDEN_SUFFIXES
    ]
    if missing or forbidden:
        raise SystemExit(json.dumps({"missing": missing, "forbidden": forbidden}, indent=2))

    pairwise = pd.read_csv(ROOT / "results/revision_analyses/paired_model_tests.csv")
    decomposition = pd.read_csv(
        ROOT / "results/revision_analyses/feature_alignment_decomposition.csv"
    )
    if len(pairwise) != 3 or len(decomposition) != 10:
        raise SystemExit("Unexpected revision-analysis row counts")

    adversarial = pd.read_csv(ROOT / "results/full_experiment/adversarial_robustness.csv")
    positive_budget = adversarial[adversarial["epsilon"] > 0]
    projected = positive_budget[positive_budget["attack"] == "constraint_aware"]
    unconstrained = positive_budget[
        positive_budget["attack"] == "unconstrained_surrogate"
    ]
    if projected["constraint_violation_rate"].max() > 1e-12:
        raise SystemExit("Projected base-run candidates contain constraint violations")
    if unconstrained["constraint_violation_rate"].min() < 1.0 - 1e-12:
        raise SystemExit("Unexpected unconstrained base-run validity profile")

    q1 = pd.read_csv(ROOT / "results/q1_upgrade/q1_adversarial_by_seed.csv")
    q1_positive = q1[q1["epsilon"] > 0]
    q1_projected = q1_positive[q1_positive["attack"].isin(
        ["constraint_aware", "greedy_constraint_aware"]
    )]
    q1_unconstrained = q1_positive[
        q1_positive["attack"] == "unconstrained_surrogate"
    ]
    if q1_projected["constraint_violation_rate"].max() > 1e-12:
        raise SystemExit("Projected repeated-seed candidates contain violations")
    if q1_unconstrained["constraint_violation_rate"].min() < 1.0 - 1e-12:
        raise SystemExit("Unexpected repeated-seed unconstrained validity profile")

    spec = importlib.util.spec_from_file_location(
        "package_base_check", ROOT / "scripts/run_full_experiment.py"
    )
    base = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = base
    spec.loader.exec_module(base)
    synthetic = pd.DataFrame(
        {"count": [2.0, 3.0], "score": [0.2, 0.8]}, dtype=np.float32
    )
    labels = np.array([0, 1])
    constraints = base.infer_constraints(synthetic, list(synthetic.columns))
    candidate, violation, _, _ = base.make_adversarial(
        synthetic,
        labels,
        np.array([-1.0, -1.0], dtype=np.float32),
        constraints,
        0.15,
        constrained=True,
    )
    if not np.array_equal(candidate.iloc[0].to_numpy(), synthetic.iloc[0].to_numpy()):
        raise SystemExit("Projection changed a benign synthetic row")
    if violation > 1e-12:
        raise SystemExit("Projected synthetic malicious row violates constraints")

    print(
        json.dumps(
            {
                "status": "ok",
                "required_files": len(REQUIRED_FILES),
                "paired_test_rows": len(pairwise),
                "decomposition_rows": len(decomposition),
                "projected_violation_max": float(
                    projected["constraint_violation_rate"].max()
                ),
                "repeated_seed_projected_violation_max": float(
                    q1_projected["constraint_violation_rate"].max()
                ),
                "benign_projection_invariant": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
