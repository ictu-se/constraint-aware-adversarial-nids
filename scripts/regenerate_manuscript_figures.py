from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FULL = ROOT / "results" / "full_experiment"
Q1 = ROOT / "results" / "q1_upgrade"
OUT = ROOT / "figures"


COLORS = {
    "UNSW official": "#1f77b4",
    "UNSW random": "#4c78a8",
    "KDDCup99": "#f58518",
    "CSE-CIC-IDS2018": "#54a24b",
    "constraint_aware": "#2ca02c",
    "constraint_greedy": "#ff7f0e",
    "unconstrained_surrogate": "#d62728",
}


def clean_label(value: str) -> str:
    return (
        value.replace("temporal_official", "official")
        .replace("constraint_aware", "projected")
        .replace("constraint_greedy", "greedy")
        .replace("unconstrained_surrogate", "unconstrained")
        .replace("feature_squeeze", "squeeze")
        .replace("reject_option", "reject")
        .replace("adv_aug", "adv. aug.")
    )


def case_label(dataset: str, split: str) -> str:
    if dataset == "UNSW-NB15" and split == "temporal_official":
        return "UNSW official"
    if dataset == "UNSW-NB15" and split == "random":
        return "UNSW random"
    if dataset.startswith("cse_cic_ids2018"):
        return "CSE-CIC-IDS2018"
    return dataset


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_unsw_robustness(adv: pd.DataFrame) -> None:
    data = adv[
        (adv["dataset"] == "UNSW-NB15")
        & (adv["split"] == "temporal_official")
        & (adv["attack"] == "constraint_aware")
    ]
    pivot = data.pivot_table(index="epsilon", columns="model", values="f1", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for model in pivot.columns:
        ax.plot(pivot.index, pivot[model], marker="o", linewidth=2, label=model)
    ax.set_title("UNSW-NB15: constraint-aware robustness curve")
    ax.set_xlabel("epsilon")
    ax.set_ylabel("F1 under attack")
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    save(fig, "unsw_nb15_robustness_curve.png")


def plot_accuracy_latency(clean: pd.DataFrame, latency: pd.DataFrame) -> None:
    data = clean.merge(latency, on=["dataset", "split", "model"])
    data["case"] = [case_label(d, s) for d, s in zip(data["dataset"], data["split"])]
    markers = {"UNSW official": "o", "UNSW random": "s", "KDDCup99": "s"}
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for case, part in data.groupby("case"):
        ax.scatter(
            part["latency_mean_ms_per_flow"],
            part["f1"],
            s=80,
            marker=markers.get(case, "o"),
            label=case,
            alpha=0.8,
        )
        for _, row in part.iterrows():
            ax.annotate(row["model"], (row["latency_mean_ms_per_flow"], row["f1"]), fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("Mean latency per flow (ms, log scale)")
    ax.set_ylabel("Clean F1")
    ax.set_title("Accuracy-latency deployment trade-off")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    save(fig, "accuracy_latency_pareto.png")


def plot_cross_dataset(cross: pd.DataFrame) -> None:
    pivot = cross.pivot_table(index="train_dataset", columns="test_dataset", values="f1", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    image = ax.imshow(pivot.values, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            if np.isfinite(pivot.values[i, j]):
                ax.text(j, i, f"{pivot.values[i, j]:.2f}", ha="center", va="center", color="white")
    ax.set_title("Cross-dataset transfer F1")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    save(fig, "cross_dataset_heatmap.png")


def plot_legacy_kdd_zoom(adv: pd.DataFrame) -> None:
    data = adv[
        (adv["dataset"] == "KDDCup99")
        & (adv["split"] == "random")
        & (adv["attack"] == "constraint_aware")
    ]
    pivot = data.pivot_table(index="epsilon", columns="model", values="f1", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for model in pivot.columns:
        ax.plot(pivot.index, pivot[model], marker="o", linewidth=2, label=model)
    ax.set_title("KDDCup99 constrained robustness (zoomed legacy saturation)")
    ax.set_xlabel("Perturbation budget epsilon")
    ax.set_ylabel("F1")
    ax.set_ylim(0.988, 1.001)
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=8, frameon=False)
    ax.text(
        0.01,
        0.989,
        "Zoomed y-axis: most nonlinear baselines are saturated.",
        fontsize=8,
        color="#333333",
    )
    save(fig, "kddcup99_robustness_curve.png")


def plot_violation_by_attack_dataset(adv: pd.DataFrame) -> None:
    eps = adv["epsilon"].max()
    data = adv[adv["epsilon"] == eps].copy()
    data["case"] = [case_label(d, s) for d, s in zip(data["dataset"], data["split"])]
    grouped = (
        data.groupby(["case", "attack"], as_index=False)["constraint_violation_rate"]
        .mean()
        .sort_values(["case", "attack"])
    )
    cases = grouped["case"].drop_duplicates().tolist()
    attacks = ["constraint_aware", "unconstrained_surrogate"]
    x = np.arange(len(cases))
    width = 0.34
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    for i, attack in enumerate(attacks):
        vals = [
            grouped[(grouped["case"] == c) & (grouped["attack"] == attack)]["constraint_violation_rate"].iloc[0]
            for c in cases
        ]
        offset = (i - 0.5) * width
        ax.bar(x + offset, vals, width=width, label=clean_label(attack), color=COLORS[attack])
        for xi, v in zip(x + offset, vals):
            ax.text(xi, v + 0.015, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_title("Attack-validity gap at epsilon=0.15")
    ax.set_ylabel("Constraint violation rate")
    ax.set_xticks(x)
    ax.set_xticklabels(cases, rotation=15, ha="right")
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    save(fig, "constraint_violation_rate.png")


def plot_q1_validity_by_dataset(qadv: pd.DataFrame) -> None:
    eps = qadv["epsilon"].max()
    data = qadv[qadv["epsilon"] == eps].copy()
    data["case"] = [case_label(d, s) for d, s in zip(data["dataset"], data["split"])]
    grouped = (
        data.groupby(["case", "attack"], as_index=False)["constraint_violation_rate"]
        .agg(["mean", "std"])
        .reset_index()
    )
    cases = ["UNSW official", "UNSW random", "KDDCup99", "CSE-CIC-IDS2018"]
    attacks = ["constraint_aware", "constraint_greedy", "unconstrained_surrogate"]
    x = np.arange(len(cases))
    width = 0.25
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    for i, attack in enumerate(attacks):
        vals, errs = [], []
        for case in cases:
            row = grouped[(grouped["case"] == case) & (grouped["attack"] == attack)]
            vals.append(row["mean"].iloc[0])
            errs.append(0 if np.isnan(row["std"].iloc[0]) else row["std"].iloc[0])
        offset = (i - 1) * width
        ax.bar(
            x + offset,
            vals,
            width=width,
            yerr=errs,
            capsize=3,
            label=clean_label(attack),
            color=COLORS[attack],
        )
    ax.set_title("Attack validity by dataset at epsilon=0.15")
    ax.set_ylabel("Constraint violation rate")
    ax.set_xticks(x)
    ax.set_xticklabels(cases, rotation=15, ha="right")
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=1)
    save(fig, "q1_attack_validity.png")


def plot_defense_constrained_f1(qadv: pd.DataFrame) -> None:
    eps = qadv["epsilon"].max()
    data = qadv[
        (qadv["epsilon"] == eps)
        & (qadv["attack"].isin(["constraint_aware", "constraint_greedy"]))
    ].copy()
    data["case"] = [case_label(d, s) for d, s in zip(data["dataset"], data["split"])]
    grouped = (
        data.groupby(["case", "defense"], as_index=False)["f1"]
        .mean()
        .sort_values(["case", "f1"], ascending=[True, False])
    )
    cases = ["UNSW official", "UNSW random", "KDDCup99", "CSE-CIC-IDS2018"]
    defenses = ["standard", "feature_squeeze", "reject_option", "adv_aug"]
    x = np.arange(len(cases))
    width = 0.19
    palette = {
        "standard": "#4c78a8",
        "feature_squeeze": "#f58518",
        "reject_option": "#b279a2",
        "adv_aug": "#54a24b",
    }
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    for i, defense in enumerate(defenses):
        vals = []
        for case in cases:
            row = grouped[(grouped["case"] == case) & (grouped["defense"] == defense)]
            vals.append(row["f1"].iloc[0])
        ax.bar(
            x + (i - 1.5) * width,
            vals,
            width=width,
            label=clean_label(defense),
            color=palette[defense],
        )
    ax.set_title("Mean constrained robust F1 by dataset and defense")
    ax.set_ylabel("Robust F1")
    ax.set_xticks(x)
    ax.set_xticklabels(cases, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    save(fig, "defense_constrained_f1.png")


def plot_q1_latency_pareto(qclean: pd.DataFrame, qlat: pd.DataFrame) -> None:
    clean = qclean.groupby(["dataset", "split", "model", "defense"], as_index=False)["f1"].mean()
    lat = qlat.groupby(["dataset", "split", "model", "defense"], as_index=False)[
        "latency_mean_ms_per_flow"
    ].mean()
    data = clean.merge(lat, on=["dataset", "split", "model", "defense"])
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    markers = {"standard": "o", "feature_squeeze": "s", "reject_option": "^", "adv_aug": "D"}
    for defense, sub in data.groupby("defense"):
        ax.scatter(
            sub["latency_mean_ms_per_flow"],
            sub["f1"],
            label=clean_label(defense),
            marker=markers[defense],
            s=np.where(sub["model"].eq("SoftVoteEnsemble"), 80, 45),
            alpha=0.78,
        )
    ens = data[data["model"] == "SoftVoteEnsemble"]
    for _, row in ens.iterrows():
        if row["defense"] == "adv_aug":
            ax.annotate("ensemble", (row["latency_mean_ms_per_flow"], row["f1"]), fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("Mean latency (ms/flow, log scale)")
    ax.set_ylabel("Clean F1")
    ax.set_title("Defense accuracy-latency trade-off")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    save(fig, "q1_defense_latency_pareto.png")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    adv = pd.read_csv(FULL / "adversarial_robustness.csv")
    clean = pd.read_csv(FULL / "clean_performance.csv")
    cross = pd.read_csv(FULL / "cross_dataset_transfer.csv")
    latency = pd.read_csv(FULL / "latency_throughput.csv")
    qadv = pd.read_csv(Q1 / "q1_adversarial_by_seed.csv")
    qclean = pd.read_csv(Q1 / "q1_clean_by_seed.csv")
    qlat = pd.read_csv(Q1 / "q1_latency_by_seed.csv")
    plot_legacy_kdd_zoom(adv)
    plot_unsw_robustness(adv)
    plot_accuracy_latency(clean, latency)
    plot_cross_dataset(cross)
    plot_violation_by_attack_dataset(adv)
    plot_q1_validity_by_dataset(qadv)
    plot_defense_constrained_f1(qadv)
    plot_q1_latency_pareto(qclean, qlat)


if __name__ == "__main__":
    main()
