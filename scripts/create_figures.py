"""Regenerate original vector diagrams and a plot of the released synthetic run."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs/assets"
ASSETS.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none",
                     "svg.hashsalt": "worldledger-toolkit-v0.1", "font.size": 11})
NAVY, TEAL, AMBER, INK, MUTED = "#101D30", "#0EA5A3", "#F4B942", "#172A42", "#60738A"


def save(fig, name):
    fig.savefig(ASSETS / (name + ".svg"), bbox_inches="tight", pad_inches=0.12,
                metadata={"Date": None, "Creator": "WorldLedger Toolkit"})
    vector = ASSETS / (name + ".svg")
    vector.write_bytes(("\n".join(line.rstrip() for line in vector.read_text(encoding="utf-8").splitlines()) + "\n").encode("utf-8"))
    fig.savefig(ASSETS / (name + ".png"), dpi=160, bbox_inches="tight", pad_inches=0.12,
                metadata={"Software": "WorldLedger Toolkit"})
    plt.close(fig)


def banner():
    fig, ax = plt.subplots(figsize=(12.8, 4.25))
    fig.patch.set_facecolor(NAVY)
    ax.set(xlim=(0, 12.8), ylim=(0, 4.25)); ax.axis("off")
    ax.text(.35, 3.62, "WORLD MODELS  /  DECISIONS  /  EVIDENCE", color="#83BFC7", fontsize=10, weight="bold")
    ax.text(.35, 2.68, "WorldLedger", color="white", fontsize=46, weight="bold")
    ax.text(.38, 1.91, "TOOLKIT", color=AMBER, fontsize=29, weight="bold", stretch="expanded")
    ax.text(.4, 1.11, "Executable worlds. Exact controls. Auditable experiments.", color="#CFDCE8", fontsize=13)
    for x, text in ((.4, "OFFLINE FIRST"), (2.75, "FINITE + EXACT"), (5.1, "OPEN SOURCE")):
        ax.add_patch(FancyBboxPatch((x, .32), 2.1, .35, boxstyle="round,pad=0.06,rounding_size=.08", facecolor="#20334A", edgecolor="none"))
        ax.text(x + .12, .49, text, color="#D6E6F0", va="center", fontsize=8, weight="bold")
    nodes = [(9.1, 3.25), (11.2, 2.8), (9.25, 1.5), (11.5, .95)]
    for start, end in ((0, 1), (0, 2), (1, 3), (2, 3), (2, 1)):
        ax.add_patch(FancyArrowPatch(nodes[start], nodes[end], arrowstyle="-", color="#34546D", linewidth=2, zorder=1))
    for i, (x, y) in enumerate(nodes):
        ax.add_patch(Circle((x, y), .43, color=TEAL if i < 3 else AMBER, alpha=.13))
        ax.add_patch(Circle((x, y), .21, facecolor=TEAL if i < 3 else AMBER, edgecolor="white", linewidth=1.3))
        ax.text(x, y, str(i + 1), color=NAVY, weight="bold", ha="center", va="center", fontsize=12)
    save(fig, "banner")


def architecture():
    fig, ax = plt.subplots(figsize=(12.8, 6.0))
    fig.patch.set_facecolor("#F4F7FB")
    ax.set(xlim=(0, 12.8), ylim=(0, 6)); ax.axis("off")
    ax.text(.2, 5.65, "A trace you can inspect. A boundary you can test.", fontsize=19, color=INK, weight="bold")
    ax.text(.22, 5.23, "Online decisions use public observations; counterfactual truth stays with the evaluator.", fontsize=10.5, color=MUTED)

    def box(x, y, w, h, title, body, color=TEAL):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=.14,rounding_size=.12", facecolor="white", edgecolor="#CFDAE5", linewidth=1.2))
        ax.plot([x + .1, x + .68], [y + h - .21] * 2, color=color, linewidth=3)
        ax.text(x + .1, y + h - .58, title, fontsize=12, color=INK, weight="bold")
        ax.text(x + .1, y + h - .92, body, fontsize=9.5, color=MUTED, va="top", linespacing=1.65)

    box(.35, 2.65, 3.25, 1.65, "FINITE WORLDS", "Grid / Inventory / Queue\nMachine Maintenance\nExplicit kernels and public histories")
    box(4.65, 2.65, 3.2, 1.65, "CANDIDATE DECISIONS", "Coherent committees and action sets\nShared observation-contingent policies\nExecuted branch only")
    box(9.0, 2.65, 3.1, 1.65, "SEALED EVALUATION", "Support, equivalence and regret\nTrusted counterfactual fork\nPost-episode truth access", AMBER)
    for start, end, label in (((3.78, 3.5), (4.44, 3.5), "public"), ((8.04, 3.5), (8.79, 3.5), "seal")):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14, color=MUTED, linewidth=1.5))
        ax.text((start[0] + end[0]) / 2, 3.77, label, ha="center", fontsize=8.5, color=MUTED)
    ax.plot([8.49, 8.49], [2.3, 4.65], linestyle=(0, (4, 4)), color=AMBER, linewidth=1.7)
    ax.text(8.49, 4.62, "information boundary", ha="center", fontsize=8.5, color="#916B14")
    box(.35, .43, 11.75, 1.25, "RESEARCH AUDIT LEDGER", "Input-file hashes  /  raw-before-derived records  /  linked events  /  unknown cost retained  /  replay", TEAL)
    for x in (1.98, 6.2, 10.55):
        ax.add_patch(FancyArrowPatch((x, 2.43), (x, 1.87), arrowstyle="-|>", mutation_scale=12, color="#8BA3B5", linewidth=1.2))
    ax.text(.36, .05, "Research controls in one Python process; see the documented isolation and single-writer limits.", color=MUTED, fontsize=8)
    save(fig, "architecture")


def results():
    data = json.loads((ROOT / "docs/demo_results/sprinkler_summary.json").read_text(encoding="utf-8"))
    keys = ["MAP", "BMA", "Robust", "GlobalIG", "RandomProbe", "FalsifyWM", "Oracle"]
    labels = ["MAP", "BMA", "Robust", "Global IG", "Random\nprobe", "Decision-\ndirected", "Oracle"]
    values = [data["summary"][key]["total_utility_mean"] for key in keys]
    fig, ax = plt.subplots(figsize=(10.8, 4.7))
    fig.patch.set_facecolor("#F4F7FB"); ax.set_facecolor("#F4F7FB")
    colors = ["#9BADBF"] * 3 + ["#D39B35", "#6AA2C5", TEAL, INK]
    bars = ax.bar(labels, values, color=colors, width=.62)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + .025, f"{value:.3f}", ha="center", fontsize=10, color=INK, weight="bold")
    ax.set_ylim(0, 1.16)
    ax.set_ylabel("Mean net utility", color=INK)
    ax.set_title("Decision-relevant probing in a constructed finite world", loc="left", pad=23, fontsize=16, weight="bold", color=INK)
    ax.text(0, 1.04, "Sprinkler synthetic control · 400 balanced seeds · noiseless probes · cost 0.05", transform=ax.transAxes, fontsize=9, color=MUTED)
    ax.set_axisbelow(True); ax.grid(axis="y", color="#DCE4EC", linewidth=.8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#BCCAD8")
    ax.tick_params(axis="both", length=0, colors=MUTED)
    fig.text(.12, .015, "Illustrative mechanism demonstration; no real-task, learned-model or cross-system ranking claim.", fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, .04, 1, 1))
    save(fig, "sprinkler_results")


if __name__ == "__main__":
    banner()
    architecture()
    results()
