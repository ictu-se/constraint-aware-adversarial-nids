#!/usr/bin/env python3
"""Full experiment harness for direction 03.

The script runs real-data NIDS experiments when public datasets are available:

* UNSW-NB15 official train/test CSV mirror.
* KDDCup99 from scikit-learn.

It reports clean performance, constrained vs unconstrained evasion robustness,
latency, and cross-dataset transfer on semantically aligned features.
"""

from __future__ import annotations

import argparse
import json
import math
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_kddcup99
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


UNSW_TRAIN_URL = (
    "https://raw.githubusercontent.com/Nir-J/ML-Projects/master/"
    "UNSW-Network_Packet_Classification/UNSW_NB15_training-set.csv"
)
UNSW_TEST_URL = (
    "https://raw.githubusercontent.com/Nir-J/ML-Projects/master/"
    "UNSW-Network_Packet_Classification/UNSW_NB15_testing-set.csv"
)

SEED = 20260711
EPSILONS = [0.0, 0.03, 0.06, 0.10, 0.15]
MAX_TRAIN = 80000
MAX_TEST = 50000


@dataclass
class SplitData:
    dataset: str
    split: str
    X_train: pd.DataFrame
    y_train: np.ndarray
    X_test: pd.DataFrame
    y_test: np.ndarray
    feature_names: List[str]
    raw_train: pd.DataFrame
    raw_test: pd.DataFrame


def download(url: str, path: Path) -> None:
    if path.exists() and path.stat().st_size > 1024:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {path}")
    urllib.request.urlretrieve(url, path)


def sanitize_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.replace([np.inf, -np.inf], np.nan).infer_objects(copy=False)
    out = out.select_dtypes(include=[np.number]).copy()
    out = out.dropna(axis=1, how="all")
    for col in out.columns:
        if out[col].isna().any():
            out[col] = out[col].fillna(out[col].median())
    nunique = out.nunique(dropna=True)
    keep = nunique[nunique > 1].index.tolist()
    return out[keep].astype(np.float32)


def sanitize_train_test(train, test):
    """Fit column selection and median imputation exclusively on training rows."""
    train = train.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
    keep = train.columns[(train.notna().any()) & (train.nunique(dropna=True) > 1)]
    train = train[keep]
    medians = train.median()
    test = test.reindex(columns=keep).replace([np.inf, -np.inf], np.nan)
    return train.fillna(medians).astype(np.float32), test.fillna(medians).astype(np.float32)


def sample_stratified(X: pd.DataFrame, y: np.ndarray, n: int, seed: int) -> Tuple[pd.DataFrame, np.ndarray]:
    if len(X) <= n:
        return X.reset_index(drop=True), y
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=n, random_state=seed)
    idx, _ = next(splitter.split(X, y))
    return X.iloc[idx].reset_index(drop=True), y[idx]


def load_unsw(data_dir: Path, split: str) -> SplitData:
    raw_dir = data_dir / "raw"
    train_path = raw_dir / "UNSW_NB15_training-set.csv"
    test_path = raw_dir / "UNSW_NB15_testing-set.csv"
    download(UNSW_TRAIN_URL, train_path)
    download(UNSW_TEST_URL, test_path)

    train_df = pd.read_csv(train_path, low_memory=False)
    test_df = pd.read_csv(test_path, low_memory=False)
    for df in (train_df, test_df):
        df["attack_cat"] = df["attack_cat"].fillna("Normal").astype(str).str.strip()
        df.loc[df["label"] == 0, "attack_cat"] = "Normal"

    if split == "temporal_official":
        tr_df, te_df = train_df.copy(), test_df.copy()
    elif split == "random":
        full = pd.concat([train_df, test_df], ignore_index=True)
        tr_df, te_df = train_test_split(
            full,
            test_size=0.30,
            stratify=full["label"].astype(int),
            random_state=SEED,
        )
    else:
        raise ValueError(split)

    y_train = tr_df["label"].astype(int).to_numpy()
    y_test = te_df["label"].astype(int).to_numpy()
    drop_cols = ["id", "attack_cat", "label", "proto", "service", "state"]
    X_train, X_test = sanitize_train_test(tr_df.drop(columns=drop_cols, errors="ignore"), te_df.drop(columns=drop_cols, errors="ignore"))
    common = sorted(set(X_train.columns) & set(X_test.columns))
    X_train, X_test = X_train[common], X_test[common]
    X_train, y_train = sample_stratified(X_train, y_train, MAX_TRAIN, SEED)
    X_test, y_test = sample_stratified(X_test, y_test, MAX_TEST, SEED + 1)
    return SplitData("UNSW-NB15", split, X_train, y_train, X_test, y_test, common, tr_df, te_df)


def load_kdd(data_dir: Path) -> SplitData:
    cache = data_dir / "processed" / "kddcup99_numeric.parquet"
    label_cache = data_dir / "processed" / "kddcup99_labels.npy"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists() and label_cache.exists():
        X = pd.read_parquet(cache)
        y = np.load(label_cache)
    else:
        bunch = fetch_kddcup99(percent10=True, shuffle=True, random_state=SEED, as_frame=True)
        df = bunch.frame.copy()
        y_raw = df["labels"].astype(str).str.replace("b'", "", regex=False).str.replace("'", "", regex=False)
        y = (y_raw != "normal.").astype(int).to_numpy()
        feature_cols = [c for c in df.columns if c != "labels"]
        X = df[feature_cols].infer_objects(copy=False).select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).copy()
        X.columns = [str(c) for c in X.columns]
        try:
            X.to_parquet(cache)
        except Exception:
            X.to_pickle(cache.with_suffix(".pkl"))
        np.save(label_cache, y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=SEED
    )
    X_train, X_test = sanitize_train_test(X_train, X_test)
    X_train, y_train = sample_stratified(X_train, y_train, MAX_TRAIN, SEED)
    X_test, y_test = sample_stratified(X_test, y_test, MAX_TEST, SEED + 1)
    return SplitData(
        "KDDCup99",
        "random",
        X_train.reset_index(drop=True),
        y_train,
        X_test.reset_index(drop=True),
        y_test,
        list(X_train.columns),
        pd.DataFrame(),
        pd.DataFrame(),
    )


def make_models() -> Dict[str, object]:
    return {
        "LogReg": LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1, random_state=SEED),
        "RandomForest": RandomForestClassifier(
            n_estimators=220,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=SEED,
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=260,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=SEED,
        ),
        "HistGB": HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_iter=180,
            l2_regularization=0.02,
            random_state=SEED,
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(96, 48),
            alpha=1e-4,
            learning_rate_init=1e-3,
            early_stopping=True,
            max_iter=60,
            random_state=SEED,
        ),
    }


def metric_row(y_true: np.ndarray, pred: np.ndarray, score: Optional[np.ndarray] = None) -> Dict[str, float]:
    row = {
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
    }
    if score is not None and len(np.unique(y_true)) == 2:
        try:
            row["roc_auc"] = roc_auc_score(y_true, score)
        except Exception:
            row["roc_auc"] = np.nan
    else:
        row["roc_auc"] = np.nan
    return row


def predict_score(model: Pipeline, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    pred = model.predict(X)
    if hasattr(model[-1], "predict_proba"):
        score = model.predict_proba(X)[:, 1]
    elif hasattr(model[-1], "decision_function"):
        raw = model.decision_function(X)
        score = 1 / (1 + np.exp(-raw))
    else:
        score = pred.astype(float)
    return pred, score


def infer_constraints(X_train: pd.DataFrame, feature_names: List[str]) -> Dict[str, object]:
    perturb_low = X_train.quantile(0.001).to_numpy(np.float32)
    perturb_high = X_train.quantile(0.999).to_numpy(np.float32)
    raw_min = X_train.min().to_numpy(np.float32)
    raw_max = X_train.max().to_numpy(np.float32)
    integer_idx = []
    immutable_idx = []
    monotone_non_decrease_idx = []
    for i, name in enumerate(feature_names):
        lname = name.lower()
        vals = X_train.iloc[:, i].dropna().to_numpy()
        if len(vals) and np.mean(np.isclose(vals, np.round(vals), atol=1e-6)) > 0.98:
            integer_idx.append(i)
        if any(p in lname for p in ["ttl", "sttl", "dttl", "win", "state", "is_", "ct_state_ttl"]):
            immutable_idx.append(i)
        if any(p in lname for p in ["byte", "bytes", "sbytes", "dbytes", "pkt", "pkts", "count", "dur"]):
            monotone_non_decrease_idx.append(i)
    integer_idx = np.array(sorted(set(integer_idx)), dtype=int)
    feasible_low = raw_min.copy()
    feasible_high = raw_max.copy()
    if len(integer_idx):
        # Integer projection must remain integer-valued at the feasibility
        # boundary, including for the small non-integer tail allowed by the
        # empirical integer-like rule.
        feasible_low[integer_idx] = np.floor(feasible_low[integer_idx])
        feasible_high[integer_idx] = np.ceil(feasible_high[integer_idx])
    return {
        "low": feasible_low,
        "high": feasible_high,
        "perturb_low": perturb_low,
        "perturb_high": perturb_high,
        "raw_min": raw_min,
        "raw_max": raw_max,
        "integer_idx": integer_idx,
        "immutable_idx": np.array(sorted(set(immutable_idx)), dtype=int),
        "monotone_non_decrease_idx": np.array(sorted(set(monotone_non_decrease_idx)), dtype=int),
    }


def fit_surrogate_direction(X_train: pd.DataFrame, y_train: np.ndarray) -> np.ndarray:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train)
    clf = LogisticRegression(max_iter=600, class_weight="balanced", random_state=SEED)
    clf.fit(Xs, y_train)
    # Decrease attack probability for malicious samples.
    coef_raw_space = clf.coef_[0] / np.maximum(scaler.scale_, 1e-8)
    direction = -np.sign(coef_raw_space)
    direction[direction == 0] = 1
    return direction.astype(np.float32)


def project_adversarial(
    Xv: np.ndarray,
    orig: np.ndarray,
    malicious: np.ndarray,
    direction: np.ndarray,
    constraints: Dict[str, object],
) -> np.ndarray:
    """Project attacked malicious rows while leaving benign rows unchanged."""
    if not np.any(malicious):
        return Xv
    attacked = Xv[malicious].copy()
    attacked_orig = orig[malicious]
    low = np.asarray(constraints["low"])
    high = np.asarray(constraints["high"])
    imm = np.asarray(constraints["immutable_idx"], dtype=int)
    mono = np.asarray(constraints["monotone_non_decrease_idx"], dtype=int)
    ints = np.asarray(constraints["integer_idx"], dtype=int)
    mutable_ints = np.setdiff1d(ints, imm, assume_unique=False)
    monotone_ints = np.intersect1d(mutable_ints, mono, assume_unique=False)
    other_ints = np.setdiff1d(mutable_ints, monotone_ints, assume_unique=False)

    if len(mono):
        attacked[:, mono] = np.maximum(attacked[:, mono], attacked_orig[:, mono])
    effective_low = np.minimum(low, attacked_orig)
    effective_high = np.maximum(high, attacked_orig)
    if len(mutable_ints):
        effective_low[:, mutable_ints] = np.floor(effective_low[:, mutable_ints])
        effective_high[:, mutable_ints] = np.ceil(effective_high[:, mutable_ints])
    attacked = np.minimum(np.maximum(attacked, effective_low), effective_high)
    if len(other_ints):
        attacked[:, other_ints] = np.round(attacked[:, other_ints])
    if len(monotone_ints):
        attacked[:, monotone_ints] = np.ceil(attacked[:, monotone_ints])
    if len(imm):
        attacked[:, imm] = attacked_orig[:, imm]
    # Projection and rounding must not reverse a benign-directed coordinate.
    # Revert any coordinate that moved opposite to the surrogate sign.
    reversed_direction = (attacked - attacked_orig) * direction.reshape(1, -1) < -1e-8
    attacked[reversed_direction] = attacked_orig[reversed_direction]
    Xv[malicious] = attacked
    return Xv


def make_adversarial(
    X: pd.DataFrame,
    y: np.ndarray,
    direction: np.ndarray,
    constraints: Dict[str, object],
    epsilon: float,
    constrained: bool,
) -> Tuple[pd.DataFrame, float, float, float]:
    if epsilon == 0:
        return X.copy(), 0.0, 0.0, 0.0
    Xv = X.to_numpy(np.float32).copy()
    orig = Xv.copy()
    malicious = y == 1
    span = np.maximum(
        np.asarray(constraints["perturb_high"]) - np.asarray(constraints["perturb_low"]),
        1e-6,
    )
    delta = epsilon * span * direction.reshape(1, -1)
    Xv[malicious] += delta

    if constrained:
        Xv = project_adversarial(Xv, orig, malicious, direction, constraints)

    violations = constraint_violations(Xv, orig, constraints, malicious)
    delta_rows = Xv[malicious] - orig[malicious]
    l0 = np.mean(np.sum(np.abs(delta_rows) > 1e-8, axis=1)) if np.any(malicious) else 0.0
    linf = np.mean(np.max(np.abs(delta_rows), axis=1)) if np.any(malicious) else 0.0
    return pd.DataFrame(Xv, columns=X.columns), float(violations), float(l0), float(linf)


def constraint_violations(
    Xv: np.ndarray,
    orig: np.ndarray,
    constraints: Dict[str, object],
    evaluated_mask: np.ndarray | None = None,
) -> float:
    if evaluated_mask is None:
        evaluated_mask = np.ones(len(Xv), dtype=bool)
    evaluated_mask = np.asarray(evaluated_mask, dtype=bool)
    if not np.any(evaluated_mask):
        return 0.0
    evaluated = Xv[evaluated_mask]
    evaluated_orig = orig[evaluated_mask]
    effective_low = np.minimum(np.asarray(constraints["low"]), evaluated_orig)
    effective_high = np.maximum(np.asarray(constraints["high"]), evaluated_orig)
    ints = constraints["integer_idx"]
    imm = constraints["immutable_idx"]
    mutable_ints = np.setdiff1d(ints, imm, assume_unique=False)
    if len(mutable_ints):
        effective_low[:, mutable_ints] = np.floor(effective_low[:, mutable_ints])
        effective_high[:, mutable_ints] = np.ceil(effective_high[:, mutable_ints])
    bad = np.zeros(len(evaluated), dtype=bool)
    bad |= np.any(evaluated < effective_low - 1e-5, axis=1)
    bad |= np.any(evaluated > effective_high + 1e-5, axis=1)
    if len(mutable_ints):
        modified_ints = (
            np.abs(evaluated[:, mutable_ints] - evaluated_orig[:, mutable_ints]) > 1e-5
        )
        bad |= np.any(
            modified_ints
            & (np.abs(evaluated[:, mutable_ints] - np.round(evaluated[:, mutable_ints])) > 1e-4),
            axis=1,
        )
    if len(imm):
        bad |= np.any(np.abs(evaluated[:, imm] - evaluated_orig[:, imm]) > 1e-5, axis=1)
    mono = constraints["monotone_non_decrease_idx"]
    if len(mono):
        bad |= np.any(evaluated[:, mono] < evaluated_orig[:, mono] - 1e-5, axis=1)
    return bad.mean()


def evaluate_split(split: SplitData, out_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    constraints = infer_constraints(split.X_train, split.feature_names)
    direction = fit_surrogate_direction(split.X_train, split.y_train)
    clean_rows, adv_rows, latency_rows = [], [], []

    for model_name, estimator in make_models().items():
        pipe = Pipeline([("scaler", StandardScaler()), ("model", estimator)])
        start = time.perf_counter()
        pipe.fit(split.X_train, split.y_train)
        train_seconds = time.perf_counter() - start
        pred, score = predict_score(pipe, split.X_test)
        row = metric_row(split.y_test, pred, score)
        row.update(
            dataset=split.dataset,
            split=split.split,
            model=model_name,
            train_seconds=train_seconds,
            n_train=len(split.X_train),
            n_test=len(split.X_test),
            n_features=len(split.feature_names),
            attack_ratio_train=float(np.mean(split.y_train)),
            attack_ratio_test=float(np.mean(split.y_test)),
        )
        clean_rows.append(row)

        reps = min(10000, len(split.X_test))
        latency_X = split.X_test.iloc[:reps]
        t0 = time.perf_counter()
        _ = pipe.predict(latency_X)
        elapsed = time.perf_counter() - t0
        per_flow_ms = elapsed / max(reps, 1) * 1000
        latency_rows.append(
            {
                "dataset": split.dataset,
                "split": split.split,
                "model": model_name,
                "flows": reps,
                "latency_mean_ms_per_flow": per_flow_ms,
                "throughput_flows_per_sec": 1000 / per_flow_ms if per_flow_ms > 0 else np.nan,
            }
        )

        clean_f1 = row["f1"]
        for eps in EPSILONS:
            for attack_type, constrained in [("unconstrained_surrogate", False), ("constraint_aware", True)]:
                X_adv, violation, l0, linf = make_adversarial(
                    split.X_test, split.y_test, direction, constraints, eps, constrained
                )
                adv_pred, adv_score = predict_score(pipe, X_adv)
                adv = metric_row(split.y_test, adv_pred, adv_score)
                eligible = (split.y_test == 1) & (pred == 1)
                attack_success = (
                    np.mean(adv_pred[eligible] == 0) if np.any(eligible) else 0.0
                )
                adv.update(
                    dataset=split.dataset,
                    split=split.split,
                    model=model_name,
                    epsilon=eps,
                    attack=attack_type,
                    clean_f1=clean_f1,
                    f1_drop=clean_f1 - adv["f1"],
                    evasion_success_rate=float(attack_success),
                    constraint_violation_rate=violation,
                    mean_l0=l0,
                    mean_linf=linf,
                )
                adv_rows.append(adv)

    return pd.DataFrame(clean_rows), pd.DataFrame(adv_rows), pd.DataFrame(latency_rows)


def semantic_rename(dataset: str, X: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "UNSW-NB15": {
            "dur": "duration",
            "sbytes": "src_bytes",
            "dbytes": "dst_bytes",
            "spkts": "src_pkts",
            "dpkts": "dst_pkts",
            "sload": "src_load",
            "dload": "dst_load",
            "smean": "src_mean_pkt",
            "dmean": "dst_mean_pkt",
            "ct_srv_src": "srv_count",
            "ct_dst_src_ltm": "dst_host_count",
        },
        "KDDCup99": {
            "duration": "duration",
            "src_bytes": "src_bytes",
            "dst_bytes": "dst_bytes",
            "count": "count",
            "srv_count": "srv_count",
            "serror_rate": "serror_rate",
            "srv_serror_rate": "srv_serror_rate",
            "rerror_rate": "rerror_rate",
            "srv_rerror_rate": "srv_rerror_rate",
            "same_srv_rate": "same_srv_rate",
            "diff_srv_rate": "diff_srv_rate",
            "dst_host_count": "dst_host_count",
            "dst_host_srv_count": "dst_host_srv_count",
        },
    }
    mapping = aliases.get(dataset, {})
    selected = {col: mapping[col] for col in X.columns if col in mapping}
    return X[list(selected.keys())].rename(columns=selected)


def cross_dataset(splits: List[SplitData]) -> pd.DataFrame:
    rows = []
    for src in splits:
        for dst in splits:
            if src.dataset == dst.dataset:
                continue
            Xs = semantic_rename(src.dataset, src.X_train)
            Xt = semantic_rename(dst.dataset, dst.X_test)
            common = sorted(set(Xs.columns) & set(Xt.columns))
            if len(common) < 3:
                continue
            Xs, Xt = Xs[common], Xt[common]
            for model_name, estimator in make_models().items():
                pipe = Pipeline([("scaler", StandardScaler()), ("model", estimator)])
                pipe.fit(Xs, src.y_train)
                pred, score = predict_score(pipe, Xt)
                row = metric_row(dst.y_test, pred, score)
                row.update(
                    train_dataset=src.dataset,
                    test_dataset=dst.dataset,
                    model=model_name,
                    n_common_features=len(common),
                    common_features=";".join(common),
                )
                rows.append(row)
    return pd.DataFrame(rows)


def save_plots(clean: pd.DataFrame, adv: pd.DataFrame, cross: pd.DataFrame, latency: pd.DataFrame, out_dir: Path) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    for dataset in adv["dataset"].unique():
        subset = adv[(adv["dataset"] == dataset) & (adv["attack"] != "unconstrained_surrogate")]
        fig, ax = plt.subplots(figsize=(8.0, 5.0))
        for model, part in subset.groupby("model"):
            means = part.groupby("epsilon")["f1"].mean()
            ax.plot(means.index, means.values, marker="o", label=model)
        ax.set_title(f"{dataset}: constraint-aware robustness curve")
        ax.set_xlabel("epsilon")
        ax.set_ylabel("F1 under attack")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.25)
        ax.legend(ncol=2, fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_dir / f"{dataset.lower().replace('-', '_')}_robustness_curve.png", dpi=220)
        plt.close(fig)

    eps = max(EPSILONS)
    viol = adv[adv["epsilon"] == eps].copy()
    viol["case"] = viol["dataset"] + "\n" + viol["attack"]
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    pivot = viol.pivot_table(index="case", columns="model", values="constraint_violation_rate", aggfunc="mean")
    pivot.plot(kind="bar", ax=ax)
    ax.set_title(f"Constraint violation rate at epsilon={eps}")
    ax.set_ylabel("Violation rate")
    ax.set_ylim(0, 1.02)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "constraint_violation_rate.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    merged = clean.merge(latency, on=["dataset", "split", "model"])
    for dataset, part in merged.groupby("dataset"):
        ax.scatter(part["latency_mean_ms_per_flow"], part["f1"], s=90, label=dataset)
        for _, r in part.iterrows():
            ax.annotate(r["model"], (r["latency_mean_ms_per_flow"], r["f1"]), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Mean latency per flow (ms, log scale)")
    ax.set_ylabel("Clean F1")
    ax.set_title("Accuracy-latency deployment trade-off")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "accuracy_latency_pareto.png", dpi=220)
    plt.close(fig)

    if not cross.empty:
        pivot = cross.pivot_table(index="train_dataset", columns="test_dataset", values="f1", aggfunc="mean")
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        im = ax.imshow(pivot.values, vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=35, ha="right")
        ax.set_yticks(range(len(pivot.index)), pivot.index)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                ax.text(j, i, f"{pivot.values[i, j]:.2f}", ha="center", va="center", color="white")
        ax.set_title("Cross-dataset transfer F1")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(fig_dir / "cross_dataset_heatmap.png", dpi=220)
        plt.close(fig)


def write_summary(
    splits: List[SplitData],
    clean: pd.DataFrame,
    adv: pd.DataFrame,
    cross: pd.DataFrame,
    latency: pd.DataFrame,
    out_dir: Path,
) -> None:
    def md_table(df: pd.DataFrame) -> str:
        if df.empty:
            return "_No rows._"
        text_df = df.copy()
        for col in text_df.columns:
            if pd.api.types.is_float_dtype(text_df[col]):
                text_df[col] = text_df[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
            else:
                text_df[col] = text_df[col].astype(str)
        header = "| " + " | ".join(text_df.columns) + " |"
        sep = "| " + " | ".join(["---"] * len(text_df.columns)) + " |"
        rows = ["| " + " | ".join(row) + " |" for row in text_df.to_numpy(dtype=str)]
        return "\n".join([header, sep] + rows)

    best_clean = clean.sort_values("f1", ascending=False).head(10)
    robust_eps = adv[adv["epsilon"] == max(EPSILONS)]
    best_robust = robust_eps.sort_values("f1", ascending=False).head(10)
    lines = [
        "# Direction 03 Full Experiment Report",
        "",
        "This report contains real-data experiments. UNSW-NB15 is downloaded from a public CSV mirror; KDDCup99 is fetched through scikit-learn. CICIDS2017/CICIoT2023 are not silently synthesized because their raw CSV files are not present in the workspace and common public mirrors are either very large or require manual download.",
        "",
        "## Datasets",
        "",
    ]
    for split in splits:
        lines.append(
            f"- {split.dataset} / {split.split}: train={len(split.X_train)}, test={len(split.X_test)}, "
            f"features={len(split.feature_names)}, attack ratio train={np.mean(split.y_train):.3f}, "
            f"attack ratio test={np.mean(split.y_test):.3f}."
        )
    lines.extend(
        [
            "",
            "## Top Clean Results",
            "",
            md_table(best_clean[["dataset", "split", "model", "f1", "balanced_accuracy", "roc_auc", "train_seconds"]]),
            "",
            f"## Top Robust Results at epsilon={max(EPSILONS)}",
            "",
            md_table(
                best_robust[
                    [
                        "dataset",
                        "split",
                        "model",
                        "attack",
                        "f1",
                        "f1_drop",
                        "evasion_success_rate",
                        "constraint_violation_rate",
                    ]
                ]
            ),
            "",
            "## Cross-Dataset Transfer",
            "",
            md_table(cross[["train_dataset", "test_dataset", "model", "n_common_features", "f1", "balanced_accuracy"]])
            if not cross.empty
            else "No cross-dataset run had at least three semantically aligned features.",
            "",
            "## Latency",
            "",
            md_table(latency[["dataset", "split", "model", "latency_mean_ms_per_flow", "throughput_flows_per_sec"]]),
            "",
            "## Generated Figures",
            "",
            "- `figures/unsw_nb15_robustness_curve.png`",
            "- `figures/kddcup99_robustness_curve.png`",
            "- `figures/constraint_violation_rate.png`",
            "- `figures/accuracy_latency_pareto.png`",
            "- `figures/cross_dataset_heatmap.png`",
        ]
    )
    (out_dir / "experiment_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    paper_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default=str(paper_dir / "results" / "full_experiment"))
    parser.add_argument("--data_dir", default=str(paper_dir / "data"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    data_dir = Path(args.data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    splits = [
        load_unsw(data_dir, "temporal_official"),
        load_unsw(data_dir, "random"),
        load_kdd(data_dir),
    ]

    clean_parts, adv_parts, latency_parts = [], [], []
    for split in splits:
        print(f"Evaluating {split.dataset} / {split.split}")
        clean, adv, latency = evaluate_split(split, out_dir)
        clean_parts.append(clean)
        adv_parts.append(adv)
        latency_parts.append(latency)

    clean_df = pd.concat(clean_parts, ignore_index=True)
    adv_df = pd.concat(adv_parts, ignore_index=True)
    latency_df = pd.concat(latency_parts, ignore_index=True)
    cross_df = cross_dataset([splits[0], splits[2]])

    clean_df.to_csv(out_dir / "clean_performance.csv", index=False)
    adv_df.to_csv(out_dir / "adversarial_robustness.csv", index=False)
    latency_df.to_csv(out_dir / "latency_throughput.csv", index=False)
    cross_df.to_csv(out_dir / "cross_dataset_transfer.csv", index=False)
    (out_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "seed": SEED,
                "epsilons": EPSILONS,
                "max_train": MAX_TRAIN,
                "max_test": MAX_TEST,
                "datasets": [s.dataset + "/" + s.split for s in splits],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    save_plots(clean_df, adv_df, cross_df, latency_df, out_dir)
    write_summary(splits, clean_df, adv_df, cross_df, latency_df, out_dir)
    print(f"Done. Results written to {out_dir}")


if __name__ == "__main__":
    main()
