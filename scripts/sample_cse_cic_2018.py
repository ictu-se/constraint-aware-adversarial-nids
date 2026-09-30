#!/usr/bin/env python3
"""Create a bounded CSE-CIC-IDS2018 sample from the public AWS bucket."""

from __future__ import annotations

from pathlib import Path
import gzip
import shutil
from urllib.parse import quote

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "external" / "cse_cic_ids2018_thursday_2018_03_01_sample.csv"
KEY = "Processed Traffic Data for ML Algorithms/Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv"
URL = "https://cse-cic-ids2018.s3.ca-central-1.amazonaws.com/" + quote(KEY)
TARGET_PER_CLASS = 15000
CHUNKSIZE = 50000


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    archived = ROOT / "inputs" / (OUT.name + ".gz")
    if archived.exists():
        with gzip.open(archived, "rb") as source, OUT.open("wb") as target:
            shutil.copyfileobj(source, target)
        print(f"Restored exact study sample: {OUT}")
        return
    benign_parts = []
    attack_parts = []
    seen = {"benign": 0, "attack": 0}

    for chunk in pd.read_csv(URL, chunksize=CHUNKSIZE, low_memory=False):
        chunk.columns = [c.strip() for c in chunk.columns]
        if "Label" not in chunk.columns:
            raise RuntimeError(f"Label column not found; columns={chunk.columns.tolist()[:10]}")
        labels = chunk["Label"].astype(str).str.strip().str.lower()
        chunk = chunk[~labels.eq("label")]
        labels = chunk["Label"].astype(str).str.strip().str.lower()
        benign = chunk[labels.eq("benign")]
        attack = chunk[~labels.eq("benign")]

        if seen["benign"] < TARGET_PER_CLASS and not benign.empty:
            take = min(TARGET_PER_CLASS - seen["benign"], len(benign))
            benign_parts.append(benign.sample(n=take, random_state=20260711))
            seen["benign"] += take
        if seen["attack"] < TARGET_PER_CLASS and not attack.empty:
            take = min(TARGET_PER_CLASS - seen["attack"], len(attack))
            attack_parts.append(attack.sample(n=take, random_state=20260712))
            seen["attack"] += take

        if seen["benign"] >= TARGET_PER_CLASS and seen["attack"] >= TARGET_PER_CLASS:
            break

    if seen["attack"] == 0:
        raise RuntimeError("No attack rows were found in the selected CSE-CIC-IDS2018 file.")
    sample = pd.concat(benign_parts + attack_parts, ignore_index=True)
    sample = sample.sample(frac=1.0, random_state=20260713).reset_index(drop=True)
    sample.to_csv(OUT, index=False)
    print(f"Wrote {OUT}")
    print(sample["Label"].value_counts().to_string())


if __name__ == "__main__":
    main()
