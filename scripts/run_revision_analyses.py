#!/usr/bin/env python3
"""Targeted analyses requested during PLOS ONE review.

This script adds (1) paired flow-level inference for close model comparisons
on the UNSW-NB15 official test partition and (2) a decomposition of the
five-feature transfer experiment into within-dataset feature-loss and residual
cross-dataset shift components.
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.base import clone
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = PACKAGE_ROOT / "scripts" / "run_full_experiment.py"
DATA_DIR = PACKAGE_ROOT / "data"
OUT_DIR = PACKAGE_ROOT / "results" / "revision_analyses"
RNG_SEED = 20260826
DEFAULT_BOOTSTRAP_REPLICATES = 2000


def load_base():
    spec = importlib.util.spec_from_file_location("revision_base", BASE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = load_base()


def fit_predictions(split):
    predictions = {}
    scores = {}
    for name, estimator in BASE.make_models().items():
        model = Pipeline(
            [("scaler", StandardScaler()), ("model", clone(estimator))]
        )
        model.fit(split.X_train, split.y_train)
        pred, score = BASE.predict_score(model, split.X_test)
        predictions[name] = pred
        scores[name] = score
        (OUT_DIR / "predictions").mkdir(parents=True, exist_ok=True)
        np.savez_compressed(OUT_DIR / "predictions" / f"paired_{name}.npz", y=split.y_test, pred=pred, score=score)
    return predictions, scores


def holm_adjust(p_values):
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    m = len(p_values)
    for rank, idx in enumerate(order):
        candidate = min(1.0, (m - rank) * p_values[idx])
        running = max(running, candidate)
        adjusted[idx] = running
    return adjusted


def paired_model_tests(split, predictions, bootstrap_replicates):
    rng = np.random.default_rng(RNG_SEED)
    y = split.y_test
    pairs = [
        ("ExtraTrees", "RandomForest"),
        ("ExtraTrees", "HistGB"),
        ("RandomForest", "HistGB"),
    ]
    rows = []
    raw_p = []
    for model_a, model_b in pairs:
        pred_a = predictions[model_a]
        pred_b = predictions[model_b]
        correct_a = pred_a == y
        correct_b = pred_b == y
        b = int(np.sum(correct_a & ~correct_b))
        c = int(np.sum(~correct_a & correct_b))
        p_value = float(binomtest(min(b, c), n=b + c, p=0.5).pvalue) if b + c else 1.0
        raw_p.append(p_value)

        observed = float(f1_score(y, pred_a) - f1_score(y, pred_b))
        boot = np.empty(bootstrap_replicates, dtype=float)
        for i in range(bootstrap_replicates):
            idx = rng.integers(0, len(y), size=len(y))
            boot[i] = f1_score(y[idx], pred_a[idx]) - f1_score(y[idx], pred_b[idx])
        np.save(OUT_DIR / "predictions" / f"bootstrap_{model_a}_{model_b}.npy", boot)
        low, high = np.quantile(boot, [0.025, 0.975])
        rows.append(
            {
                "model_a": model_a,
                "model_b": model_b,
                "f1_a": f1_score(y, pred_a),
                "f1_b": f1_score(y, pred_b),
                "paired_f1_difference_a_minus_b": observed,
                "bootstrap_ci95_low": float(low),
                "bootstrap_ci95_high": float(high),
                "discordant_a_only_correct": b,
                "discordant_b_only_correct": c,
                "mcnemar_exact_p": p_value,
            }
        )
    adjusted = holm_adjust(raw_p)
    for row, value in zip(rows, adjusted):
        row["mcnemar_holm_p"] = float(value)
    return pd.DataFrame(rows)


def evaluate_feature_set(split, columns):
    rows = []
    for name, estimator in BASE.make_models().items():
        model = Pipeline(
            [("scaler", StandardScaler()), ("model", clone(estimator))]
        )
        model.fit(split.X_train[columns], split.y_train)
        pred, score = BASE.predict_score(model, split.X_test[columns])
        (OUT_DIR / "predictions").mkdir(parents=True, exist_ok=True)
        np.savez_compressed(OUT_DIR / "predictions" / f"alignment_{split.dataset}_{len(columns)}_{name}.npz", y=split.y_test, pred=pred, score=score)
        metric = BASE.metric_row(split.y_test, pred, score)
        rows.append({"dataset": split.dataset, "model": name, **metric})
    return pd.DataFrame(rows)


def feature_alignment_decomposition(unsw, kdd):
    canonical = ["dst_bytes", "dst_host_count", "duration", "src_bytes", "srv_count"]
    aligned = {}
    for split in [unsw, kdd]:
        train = BASE.semantic_rename(split.dataset, split.X_train)
        test = BASE.semantic_rename(split.dataset, split.X_test)
        aligned[split.dataset] = (train[canonical], test[canonical])

    full_parts = []
    aligned_parts = []
    for split in [unsw, kdd]:
        full = evaluate_feature_set(split, list(split.X_train.columns))
        full["setting"] = "within_full"
        full_parts.append(full)

        train_aligned, test_aligned = aligned[split.dataset]
        proxy = BASE.SplitData(
            split.dataset,
            split.split,
            train_aligned,
            split.y_train,
            test_aligned,
            split.y_test,
            canonical,
            split.raw_train,
            split.raw_test,
        )
        reduced = evaluate_feature_set(proxy, canonical)
        reduced["setting"] = "within_aligned_five"
        aligned_parts.append(reduced)

    cross = BASE.cross_dataset([unsw, kdd]).rename(
        columns={"train_dataset": "dataset", "test_dataset": "test_dataset"}
    )
    cross["setting"] = "cross_aligned_five"

    detail = pd.concat(full_parts + aligned_parts, ignore_index=True)
    rows = []
    for source, target in [(unsw.dataset, kdd.dataset), (kdd.dataset, unsw.dataset)]:
        for model in BASE.make_models():
            full_f1 = float(detail[(detail.dataset == source) & (detail.model == model) & (detail.setting == "within_full")].f1.iloc[0])
            aligned_f1 = float(detail[(detail.dataset == source) & (detail.model == model) & (detail.setting == "within_aligned_five")].f1.iloc[0])
            cross_f1 = float(cross[(cross.dataset == source) & (cross.test_dataset == target) & (cross.model == model)].f1.iloc[0])
            rows.append(
                {
                    "train_dataset": source,
                    "test_dataset": target,
                    "model": model,
                    "within_full_f1": full_f1,
                    "within_aligned_five_f1": aligned_f1,
                    "cross_aligned_five_f1": cross_f1,
                    "feature_reduction_drop": full_f1 - aligned_f1,
                    "residual_transfer_drop": aligned_f1 - cross_f1,
                }
            )
    return pd.DataFrame(rows)


def write_summary(pairwise, decomposition, out_dir, bootstrap_replicates):
    summary = decomposition.groupby(["train_dataset", "test_dataset"], as_index=False).mean(numeric_only=True)
    lines = [
        "# Revision analyses",
        "",
        f"Paired bootstrap replicates: {bootstrap_replicates}.",
        "McNemar p-values use the exact binomial form and Holm correction across the three planned tree-model comparisons.",
        "",
        "## Pairwise model comparisons",
        "",
        pairwise.to_string(index=False, float_format=lambda value: f"{value:.6f}"),
        "",
        "## Five-feature transfer decomposition (model means)",
        "",
        summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"),
        "",
    ]
    (out_dir / "revision_analysis_summary.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reproduce the paired tests and five-feature transfer decomposition."
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=DEFAULT_BOOTSTRAP_REPLICATES,
        help="Number of paired bootstrap resamples (default: 2000).",
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse existing CSV outputs instead of recomputing the analyses.",
    )
    args = parser.parse_args()
    if args.bootstrap_replicates < 1:
        parser.error("--bootstrap-replicates must be positive")
    return args


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pairwise_path = args.out_dir / "paired_model_tests.csv"
    decomposition_path = args.out_dir / "feature_alignment_decomposition.csv"
    if args.reuse_existing and pairwise_path.exists() and decomposition_path.exists():
        pairwise = pd.read_csv(pairwise_path)
        decomposition = pd.read_csv(decomposition_path)
        metadata_path = args.out_dir / "analysis_metadata.json"
        existing_metadata = (
            json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata_path.exists()
            else {}
        )
        n_test = existing_metadata.get("n_test")
    else:
        unsw = BASE.load_unsw(args.data_dir, "temporal_official")
        kdd = BASE.load_kdd(args.data_dir)
        predictions, _ = fit_predictions(unsw)
        pairwise = paired_model_tests(unsw, predictions, args.bootstrap_replicates)
        decomposition = feature_alignment_decomposition(unsw, kdd)
        pairwise.to_csv(pairwise_path, index=False)
        decomposition.to_csv(decomposition_path, index=False)
        n_test = len(unsw.y_test)
    write_summary(pairwise, decomposition, args.out_dir, args.bootstrap_replicates)
    (args.out_dir / "analysis_metadata.json").write_text(
        json.dumps(
            {
                "random_seed": RNG_SEED,
                "bootstrap_replicates": args.bootstrap_replicates,
                "reused_existing_results": bool(args.reuse_existing),
                "test_partition": "UNSW-NB15 official test partition",
                "n_test": n_test,
                "aligned_features": ["dst_bytes", "dst_host_count", "duration", "src_bytes", "srv_count"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(pairwise.to_string(index=False))
    print(decomposition.groupby(["train_dataset", "test_dataset"]).mean(numeric_only=True).to_string())


if __name__ == "__main__":
    main()
