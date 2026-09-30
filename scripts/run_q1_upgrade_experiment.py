#!/usr/bin/env python3
"""Q1-strengthening experiment for direction 03.

This script extends the initial benchmark with:
  * repeated stratified resampling seeds and bootstrap confidence intervals;
  * a stronger greedy constrained attack;
  * lightweight deployable defenses: feature squeezing, adversarial
    augmentation, and reject-option classification;
  * generic CSV loader support for CICIDS/CICIoT/UQ NetFlow-style files when
    raw CSVs are placed under data/external/.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
BASE_SCRIPT = ROOT / "scripts" / "run_full_experiment.py"
OUT_DIR = ROOT / "results" / "q1_upgrade"
DATA_DIR = ROOT / "data"
SEEDS = [20260711, 20260712, 20260713]
EPSILONS = [0.06, 0.10, 0.15]
MAX_TRAIN = 30000
MAX_TEST = 15000


def load_base_module():
    spec = importlib.util.spec_from_file_location("base_exp3", BASE_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["base_exp3"] = mod
    spec.loader.exec_module(mod)
    return mod


BASE = load_base_module()


def models(seed: int) -> Dict[str, object]:
    rf = RandomForestClassifier(
        n_estimators=180,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=seed,
    )
    et = ExtraTreesClassifier(
        n_estimators=220,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=seed,
    )
    hgb = HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_iter=150,
        l2_regularization=0.02,
        random_state=seed,
    )
    return {
        "LogReg": LogisticRegression(max_iter=900, class_weight="balanced", n_jobs=-1, random_state=seed),
        "RandomForest": rf,
        "ExtraTrees": et,
        "HistGB": hgb,
        "SoftVoteEnsemble": VotingClassifier(
            estimators=[
                ("lr", LogisticRegression(max_iter=900, class_weight="balanced", n_jobs=-1, random_state=seed)),
                ("rf", rf),
                ("et", et),
                ("hgb", hgb),
            ],
            voting="soft",
            n_jobs=-1,
            weights=[0.7, 1.0, 1.2, 1.0],
        ),
    }


def metric_row(y_true: np.ndarray, pred: np.ndarray, score: np.ndarray | None = None) -> Dict[str, float]:
    out = {
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
    }
    if score is not None and len(np.unique(y_true)) == 2:
        try:
            out["roc_auc"] = roc_auc_score(y_true, score)
        except Exception:
            out["roc_auc"] = np.nan
    else:
        out["roc_auc"] = np.nan
    return out


def predict_score(pipe: Pipeline, X: pd.DataFrame, reject_option: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    if hasattr(pipe[-1], "predict_proba"):
        score = pipe.predict_proba(X)[:, 1]
    elif hasattr(pipe[-1], "decision_function"):
        raw = pipe.decision_function(X)
        score = 1 / (1 + np.exp(-raw))
    else:
        score = pipe.predict(X).astype(float)
    pred = (score >= 0.5).astype(int)
    if reject_option:
        # Security-oriented reject option: ambiguous benign decisions become alerts.
        pred[(score >= 0.35) & (score < 0.5)] = 1
    return pred, score


def squeeze(X: pd.DataFrame, constraints: Dict[str, object]) -> pd.DataFrame:
    Xv = X.to_numpy(np.float32).copy()
    Xv = np.minimum(np.maximum(Xv, constraints["low"]), constraints["high"])
    ints = constraints["integer_idx"]
    if len(ints):
        Xv[:, ints] = np.round(Xv[:, ints])
    return pd.DataFrame(Xv, columns=X.columns)


def greedy_attack(
    X: pd.DataFrame,
    y: np.ndarray,
    direction: np.ndarray,
    constraints: Dict[str, object],
    epsilon: float,
    constrained: bool = True,
    top_k: int = 10,
) -> Tuple[pd.DataFrame, float, float, float]:
    Xv = X.to_numpy(np.float32).copy()
    orig = Xv.copy()
    malicious = y == 1
    span = np.maximum(
        np.asarray(constraints["perturb_high"]) - np.asarray(constraints["perturb_low"]),
        1e-6,
    )
    mutable = np.ones(Xv.shape[1], dtype=bool)
    mutable[np.asarray(constraints["immutable_idx"], dtype=int)] = False
    ranked = np.argsort(-np.abs(direction * span))
    ranked = [i for i in ranked if mutable[i]][:top_k]
    for j in ranked:
        step = epsilon * span[j] * np.sign(direction[j] if direction[j] != 0 else 1)
        Xv[malicious, j] += step
    if constrained:
        Xv = BASE.project_adversarial(Xv, orig, malicious, direction, constraints)
    violation = BASE.constraint_violations(Xv, orig, constraints, malicious)
    delta_rows = Xv[malicious] - orig[malicious]
    l0 = np.mean(np.sum(np.abs(delta_rows) > 1e-8, axis=1)) if np.any(malicious) else 0.0
    linf = np.mean(np.max(np.abs(delta_rows), axis=1)) if np.any(malicious) else 0.0
    return pd.DataFrame(Xv, columns=X.columns), float(violation), float(l0), float(linf)


def resample_split(split, seed: int):
    Xtr, ytr = BASE.sample_stratified(split.X_train, split.y_train, MAX_TRAIN, seed)
    Xte, yte = BASE.sample_stratified(split.X_test, split.y_test, MAX_TEST, seed + 100)
    split.X_train, split.y_train, split.X_test, split.y_test = Xtr, ytr, Xte, yte
    return split


def load_external_csvs() -> List[object]:
    """Load raw CSVs placed by the user under data/external/.

    Expected label columns are one of: Label, label, Attack, attack, Class,
    class, binary_label, or category-level Bot-IoT fields. This is
    intentionally generic so CICIDS2017, CICIoT2023, Bot-IoT, and UQ
    NetFlow CSVs can be added without changing code.
    """
    out = []
    ext = DATA_DIR / "external"
    if not ext.exists():
        return out
    for csv_path in sorted(ext.glob("cse_cic_ids2018_thursday_2018_03_01_sample.csv")):
        try:
            df = pd.read_csv(csv_path, low_memory=False)
        except Exception:
            continue
        label_col = next(
            (
                c
                for c in [
                    "Label",
                    "label",
                    "Attack",
                    "attack",
                    "Class",
                    "class",
                    "binary_label",
                    "category",
                    "subcategory",
                ]
                if c in df
            ),
            None,
        )
        if not label_col:
            continue
        raw_y = df[label_col]
        if pd.api.types.is_numeric_dtype(raw_y):
            y = (raw_y.astype(float) > 0).astype(int).to_numpy()
        else:
            y = (
                ~raw_y.astype(str)
                .str.lower()
                .isin(["benign", "normal", "0", "none", "no", "false"])
            ).astype(int).to_numpy()
        X = df.drop(columns=[label_col], errors="ignore").select_dtypes(include=[np.number])
        if X.shape[1] < 5 or len(np.unique(y)) < 2:
            continue
        trX, teX, trY, teY = train_test_split(X, y, test_size=0.30, stratify=y, random_state=SEEDS[0])
        trX, teX = BASE.sanitize_train_test(trX, teX)
        trX, trY = BASE.sample_stratified(trX.reset_index(drop=True), trY, MAX_TRAIN, SEEDS[0])
        teX, teY = BASE.sample_stratified(teX.reset_index(drop=True), teY, MAX_TEST, SEEDS[0] + 1)
        name = csv_path.stem[:32]
        out.append(
            BASE.SplitData(
                name,
                "external_random",
                trX,
                trY,
                teX,
                teY,
                list(trX.columns),
                pd.DataFrame(),
                pd.DataFrame(),
            )
        )
    return out


def evaluate_one(split, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    constraints = BASE.infer_constraints(split.X_train, split.feature_names)
    direction = BASE.fit_surrogate_direction(split.X_train, split.y_train)
    clean_rows, adv_rows, latency_rows = [], [], []

    for model_name, estimator in models(seed).items():
        for defense in ["standard", "feature_squeeze", "adv_aug", "reject_option"]:
            X_train = split.X_train
            y_train = split.y_train
            X_test = split.X_test

            if defense == "feature_squeeze":
                X_train = squeeze(X_train, constraints)
                X_test = squeeze(X_test, constraints)
            elif defense == "adv_aug":
                X_aug, _, _, _ = BASE.make_adversarial(
                    split.X_train, split.y_train, direction, constraints, 0.06, constrained=True
                )
                # Keep augmentation moderate to avoid changing class priors too aggressively.
                aug_idx = np.where(split.y_train == 1)[0][: min(12000, np.sum(split.y_train == 1))]
                X_train = pd.concat([split.X_train, X_aug.iloc[aug_idx]], ignore_index=True)
                y_train = np.concatenate([split.y_train, split.y_train[aug_idx]])

            pipe = Pipeline([("scaler", StandardScaler()), ("model", clone(estimator))])
            t0 = time.perf_counter()
            pipe.fit(X_train, y_train)
            train_seconds = time.perf_counter() - t0
            pred, score = predict_score(pipe, X_test, reject_option=(defense == "reject_option"))
            row = metric_row(split.y_test, pred, score)
            row.update(
                dataset=split.dataset,
                split=split.split,
                seed=seed,
                model=model_name,
                defense=defense,
                train_seconds=train_seconds,
                n_train=len(X_train),
                n_test=len(X_test),
                n_features=len(split.feature_names),
            )
            clean_rows.append(row)
            evidence_dir = OUT_DIR / "predictions"
            evidence_dir.mkdir(parents=True, exist_ok=True)
            evidence = dict(y=split.y_test, clean_pred=pred, clean_score=score)


            reps = min(8000, len(X_test))
            t1 = time.perf_counter()
            _ = predict_score(pipe, X_test.iloc[:reps], reject_option=(defense == "reject_option"))
            elapsed = time.perf_counter() - t1
            latency_rows.append(
                {
                    "dataset": split.dataset,
                    "split": split.split,
                    "seed": seed,
                    "model": model_name,
                    "defense": defense,
                    "latency_mean_ms_per_flow": elapsed / max(reps, 1) * 1000,
                }
            )

            for eps in EPSILONS:
                attacks = [
                    ("unconstrained_surrogate", lambda: BASE.make_adversarial(split.X_test, split.y_test, direction, constraints, eps, False)),
                    ("constraint_aware", lambda: BASE.make_adversarial(split.X_test, split.y_test, direction, constraints, eps, True)),
                    ("constraint_greedy", lambda: greedy_attack(split.X_test, split.y_test, direction, constraints, eps, True)),
                ]
                for attack_name, attack_fn in attacks:
                    X_adv, violation, l0, linf = attack_fn()
                    if defense == "feature_squeeze":
                        X_adv = squeeze(X_adv, constraints)
                    adv_pred, adv_score = predict_score(pipe, X_adv, reject_option=(defense == "reject_option"))
                    adv = metric_row(split.y_test, adv_pred, adv_score)
                    adv.update(
                        dataset=split.dataset,
                        split=split.split,
                        seed=seed,
                        model=model_name,
                        defense=defense,
                        epsilon=eps,
                        attack=attack_name,
                        clean_f1=row["f1"],
                        f1_drop=row["f1"] - adv["f1"],
                        constraint_violation_rate=violation,
                        mean_l0=l0,
                        mean_linf=linf,
                    )
                    adv_rows.append(adv)
                    key = f"{attack_name}_{eps:.2f}"
                    evidence[key + "_pred"] = adv_pred
                    evidence[key + "_score"] = adv_score
            np.savez_compressed(evidence_dir / f"{split.dataset}_{split.split}_{seed}_{model_name}_{defense}.npz", **evidence)
            print(f"  completed {model_name}/{defense}", flush=True)

    return pd.DataFrame(clean_rows), pd.DataFrame(adv_rows), pd.DataFrame(latency_rows)


def aggregate_ci(df: pd.DataFrame, group_cols: List[str], value_cols: List[str]) -> pd.DataFrame:
    rows = []
    for key, part in df.groupby(group_cols):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(group_cols, key))
        for c in value_cols:
            vals = part[c].dropna().to_numpy()
            row[f"{c}_mean"] = float(np.mean(vals)) if len(vals) else np.nan
            row[f"{c}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            row[f"{c}_ci95"] = float(1.96 * row[f"{c}_std"] / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
        row["n"] = len(part)
        rows.append(row)
    return pd.DataFrame(rows)


def plot_outputs(clean: pd.DataFrame, adv: pd.DataFrame, latency: pd.DataFrame) -> None:
    fig_dir = OUT_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    eps = max(EPSILONS)

    robust = adv[(adv["epsilon"] == eps) & (adv["attack"].isin(["constraint_aware", "constraint_greedy"]))]
    summary = robust.groupby(["defense", "attack"])["f1"].mean().unstack("attack")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    summary.plot(kind="bar", ax=ax)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Robust F1")
    ax.set_title(f"Defense comparison under constrained attacks, epsilon={eps}")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "defense_constrained_f1.png", dpi=220)
    plt.close(fig)

    viol = adv[adv["epsilon"] == eps].groupby(["attack", "defense"])["constraint_violation_rate"].mean().unstack("defense")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    viol.plot(kind="bar", ax=ax)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Constraint violation rate")
    ax.set_title(f"Attack validity by defense, epsilon={eps}")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "q1_attack_validity.png", dpi=220)
    plt.close(fig)

    lat = latency.groupby(["defense", "model"])["latency_mean_ms_per_flow"].mean().reset_index()
    perf = clean.groupby(["defense", "model"])["f1"].mean().reset_index()
    merged = perf.merge(lat, on=["defense", "model"])
    fig, ax = plt.subplots(figsize=(8, 5))
    for defense, part in merged.groupby("defense"):
        ax.scatter(part["latency_mean_ms_per_flow"], part["f1"], label=defense, s=65)
    ax.set_xscale("log")
    ax.set_xlabel("Mean latency per flow (ms, log)")
    ax.set_ylabel("Clean F1")
    ax.set_title("Defense accuracy-latency trade-off")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / "q1_defense_latency_pareto.png", dpi=220)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    splits = [
        BASE.load_unsw(DATA_DIR, "temporal_official"),
        BASE.load_unsw(DATA_DIR, "random"),
        BASE.load_kdd(DATA_DIR),
    ] + load_external_csvs()

    clean_parts, adv_parts, latency_parts = [], [], []
    for seed in SEEDS:
        for base_split in splits:
            split = BASE.SplitData(
                base_split.dataset,
                base_split.split,
                base_split.X_train.copy(),
                base_split.y_train.copy(),
                base_split.X_test.copy(),
                base_split.y_test.copy(),
                list(base_split.feature_names),
                base_split.raw_train,
                base_split.raw_test,
            )
            split = resample_split(split, seed)
            print(f"Q1 eval seed={seed} {split.dataset}/{split.split}", flush=True)
            clean, adv, latency = evaluate_one(split, seed)
            clean_parts.append(clean)
            adv_parts.append(adv)
            latency_parts.append(latency)

    clean_df = pd.concat(clean_parts, ignore_index=True)
    adv_df = pd.concat(adv_parts, ignore_index=True)
    latency_df = pd.concat(latency_parts, ignore_index=True)
    clean_df.to_csv(OUT_DIR / "q1_clean_by_seed.csv", index=False)
    adv_df.to_csv(OUT_DIR / "q1_adversarial_by_seed.csv", index=False)
    latency_df.to_csv(OUT_DIR / "q1_latency_by_seed.csv", index=False)

    clean_ci = aggregate_ci(clean_df, ["dataset", "split", "model", "defense"], ["f1", "balanced_accuracy", "roc_auc"])
    robust_ci = aggregate_ci(
        adv_df[adv_df["epsilon"] == max(EPSILONS)],
        ["dataset", "split", "model", "defense", "attack"],
        ["f1", "f1_drop", "constraint_violation_rate"],
    )
    latency_ci = aggregate_ci(latency_df, ["dataset", "split", "model", "defense"], ["latency_mean_ms_per_flow"])
    clean_ci.to_csv(OUT_DIR / "q1_clean_ci.csv", index=False)
    robust_ci.to_csv(OUT_DIR / "q1_robustness_ci.csv", index=False)
    latency_ci.to_csv(OUT_DIR / "q1_latency_ci.csv", index=False)
    plot_outputs(clean_df, adv_df, latency_df)
    (OUT_DIR / "metadata.json").write_text(
        json.dumps({"seeds": SEEDS, "epsilons": EPSILONS, "max_train": MAX_TRAIN, "max_test": MAX_TEST}, indent=2),
        encoding="utf-8",
    )
    print(f"Done: {OUT_DIR}")


if __name__ == "__main__":
    main()
