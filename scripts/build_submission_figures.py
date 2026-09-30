#!/usr/bin/env python3
"""Assemble the four journal figure files from regenerated data-derived panels."""

from __future__ import annotations

import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"


def panel_font(size: int = 52) -> ImageFont.ImageFont:
    candidates = (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    )
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def combine(left_name: str, right_name: str, output_name: str) -> None:
    panels = [Image.open(FIGURES / name).convert("RGB") for name in (left_name, right_name)]
    target_height = max(panel.height for panel in panels)
    resized = []
    for panel in panels:
        width = round(panel.width * target_height / panel.height)
        resized.append(panel.resize((width, target_height), Image.Resampling.LANCZOS))
    label_band = 72
    gap = 24
    canvas = Image.new(
        "RGB",
        (sum(panel.width for panel in resized) + gap, target_height + label_band),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    font = panel_font()
    x = 0
    for label, panel in zip(("(a)", "(b)"), resized):
        draw.text((x + 8, 4), label, fill="black", font=font)
        canvas.paste(panel, (x, label_band))
        x += panel.width + gap
    canvas.save(FIGURES / output_name, dpi=(300, 300))


def main() -> None:
    combine("unsw_nb15_robustness_curve.png", "constraint_violation_rate.png", "Fig1.png")
    shutil.copyfile(FIGURES / "cross_dataset_heatmap.png", FIGURES / "Fig2.png")
    shutil.copyfile(FIGURES / "accuracy_latency_pareto.png", FIGURES / "Fig3.png")
    combine("defense_constrained_f1.png", "q1_attack_validity.png", "Fig4.png")


if __name__ == "__main__":
    main()
