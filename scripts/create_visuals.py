"""Render the original visual system from bundled fonts and published demo data."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, PathPatch, Polygon, Rectangle
from matplotlib.path import Path as MplPath

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs/assets"
FONTS = ASSETS / "fonts"
INK, MUTED, PAPER, RULE = "#14212B", "#64727C", "#F5F4EF", "#D9DDD9"
DARK, DIM, WHITE, LIME, BLUE = "#0C141D", "#748693", "#F3F5ED", "#C6EF83", "#639ED5"
plt.rcParams.update({"svg.fonttype": "path", "svg.hashsalt": "worldledger-editorial-v1",
                     "axes.unicode_minus": False, "path.simplify": False})


def face(size, weight="regular", mono=False):
    filename = "JetBrainsMono-Regular.ttf" if mono else f"InstrumentSans-{weight.title()}.ttf"
    return FontProperties(fname=FONTS / filename, size=size)


def canvas(width, height, color=PAPER):
    fig = plt.figure(figsize=(width / 100, height / 100), facecolor=color)
    ax = fig.add_axes((0, 0, 1, 1), xlim=(0, width), ylim=(height, 0))
    ax.axis("off")
    return fig, ax


def text(ax, x, y, label, size=16, color=INK, weight="regular", mono=False, **kwargs):
    return ax.text(x, y, label, fontproperties=face(size, weight, mono), color=color,
                   va="top", linespacing=1.25, **kwargs)


def line(ax, points, color=RULE, width=1, **kwargs):
    return ax.plot([p[0] for p in points], [p[1] for p in points], color=color, lw=width, **kwargs)


def curve(ax, start, end, color, bend=0, width=1.8, alpha=1):
    path = MplPath([start, (start[0] + (end[0] - start[0]) * .42, start[1] + bend),
                   (end[0] - (end[0] - start[0]) * .42, end[1] - bend), end],
                  [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])
    ax.add_patch(PathPatch(path, fill=False, edgecolor=color, lw=width, alpha=alpha))


def panel(ax, x, y, w, h, fill="white", stroke=RULE, radius=10):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
                              linewidth=1, facecolor=fill, edgecolor=stroke))


def mark(ax, x, y, scale=1, color=LIME):
    for offset, length in ((0, 30), (11, 23), (22, 30)):
        line(ax, [(x, y + offset * scale), (x + length * scale, y + offset * scale)], color, 3.2 * scale)
    line(ax, [(x + 30 * scale, y), (x + 38 * scale, y - 8 * scale)], color, 3.2 * scale)
    line(ax, [(x + 30 * scale, y + 22 * scale), (x + 38 * scale, y + 30 * scale)], color, 3.2 * scale)


def save(fig, name):
    fig.savefig(ASSETS / (name + ".png"), dpi=150, facecolor=fig.get_facecolor(),
                metadata={"Software": "WorldLedger Toolkit"})
    fig.savefig(ASSETS / (name + ".svg"), facecolor=fig.get_facecolor(),
                metadata={"Date": None, "Creator": "WorldLedger Toolkit"})
    path = ASSETS / (name + ".svg")
    path.write_bytes(("\n".join(row.rstrip() for row in path.read_text(encoding="utf-8").splitlines()) + "\n").encode())
    plt.close(fig)


def banner():
    fig, ax = canvas(1600, 580, DARK)
    mark(ax, 78, 69, .85)
    text(ax, 133, 58, "WORLDLEDGER / OPEN RESEARCH TOOLKIT", 11, DIM, mono=True)
    text(ax, 73, 155, "WorldLedger", 66, WHITE, "bold")
    text(ax, 79, 285, "Inspect the worlds\nbehind a decision.", 27, "#DCE4DF")
    line(ax, [(80, 435), (710, 435)], "#293642")
    for x, label in ((80, "01  EVALUATE"), (301, "02  DECIDE"), (515, "03  RECORD")):
        text(ax, x, 467, label, 11, DIM, mono=True)
    # Projected model sheets and a consistent path into one shared decision.
    for j in range(3):
        y, x = 222 - j * 53, 876 + j * 18
        a, b, d = (x, y), (x + 293, y - 55), (x + 204, y + 137)
        c = (b[0] + 204, b[1] + 137)
        ax.add_patch(Polygon([a, b, c, d], closed=True,
                             facecolor=["#13202B", "#172631", "#1B2D38"][j], edgecolor="#425561", lw=.85))
        def point(u, v):
            return (a[0] + u * (b[0] - a[0]) + v * (d[0] - a[0]),
                    a[1] + u * (b[1] - a[1]) + v * (d[1] - a[1]))
        for n in range(1, 7):
            line(ax, [point(n / 7, 0), point(n / 7, 1)], "#344752", .55)
        for n in range(1, 5):
            line(ax, [point(0, n / 5), point(1, n / 5)], "#344752", .55)
        route = [(0.12, .20), (.40, .20), (.40, .60), (.73, .60), (.73, .81), (.92, .81)]
        color = LIME if j == 2 else "#6B8996"
        line(ax, [point(*p) for p in route], color, 2.2 if j == 2 else 1.15, alpha=1 if j == 2 else .65)
        for u, v in (route[0], route[-1]):
            px, py = point(u, v)
            ax.scatter([px], [py], s=18, color=color, zorder=4)
    for x in range(883, 1465, 35):
        line(ax, [(x, 431), (x, 434)], "#40515D", .7)
    text(ax, 887, 455, "MODEL SUPPORT", 10, DIM, mono=True)
    curve(ax, (1190, 338), (1450, 385), LIME, width=1.8)
    ax.scatter([1450], [385], s=30, color=LIME)
    text(ax, 1290, 455, "SHARED DECISION", 10, LIME, mono=True)
    save(fig, "banner")


def architecture():
    fig, ax = canvas(1600, 780)
    text(ax, 72, 48, "SYSTEM MAP / WORLDLEDGER", 11, MUTED, mono=True)
    text(ax, 69, 100, "Public observations in. Inspectable evidence out.", 31, INK, "bold")
    text(ax, 72, 170, "A shared online policy, a trusted evaluator, and a persistent record of what happened.", 17, MUTED)
    cols = [72, 604, 1136]
    for x, num, name, package in ((cols[0], "01", "Finite worlds", "worldledger.benchmark"),
                                 (cols[1], "02", "Candidate decisions", "sbcc / worldledger.evaluator"),
                                 (cols[2], "03", "Sealed evaluation", "worldledger.evaluator")):
        text(ax, x, 245, num, 12, MUTED, mono=True)
        text(ax, x, 281, name, 24, INK, "bold")
        text(ax, x, 326, package, 10, MUTED, mono=True)
    for row in range(3):
        for col in range(4):
            fill = "#DDE6E4" if (row, col) in {(0, 2), (1, 2)} else "#EAECE7"
            ax.add_patch(Rectangle((72 + col * 33, 389 + row * 33), 28, 28, fc=fill, ec="none"))
    line(ax, [(85, 468), (118, 468), (118, 435), (151, 435), (151, 402), (184, 402)], "#3C748B", 2.6)
    text(ax, 234, 395, "Grid · Inventory · Queue", 15)
    text(ax, 234, 432, "Public history + contract", 15, MUTED)
    for row, color in enumerate(("#B3C6D4", "#7496AA", "#C5D2C2")):
        yy = 387 + row * 36
        ax.add_patch(Rectangle((604, yy), 7, 23, fc=color, ec="none"))
        line(ax, [(627, yy + 5), (768 - row * 12, yy + 5)], "#A2B0B5", 1.4)
        line(ax, [(627, yy + 17), (735 + row * 5, yy + 17)], "#CED5D4", 1)
    text(ax, 792, 395, "One shared policy", 15)
    text(ax, 792, 432, "Executed branch only", 15, MUTED)
    for yy, width in ((392, 48), (425, 85), (458, 118)):
        ax.add_patch(Rectangle((1136, yy), width, 13, facecolor="#A5B99A", edgecolor="none"))
    text(ax, 1290, 395, "Support + regret", 15)
    text(ax, 1290, 432, "Truth after seal", 15, MUTED)
    for x in (476, 1008):
        ax.add_patch(FancyArrowPatch((x, 416), (x + 75, 416), arrowstyle="-|>",
                                    mutation_scale=14, lw=1.5, color="#6F8790"))
    line(ax, [(1080, 258), (1080, 518)], "#91A685", 1.1, linestyle=(0, (4, 5)))
    text(ax, 1077, 534, "EPISODE SEAL", 9.5, "#55734C", mono=True, ha="center")
    line(ax, [(72, 592), (1528, 592)], "#BBC5C0", 1)
    text(ax, 72, 624, "AUDIT LEDGER", 12, "#496254", mono=True)
    text(ax, 72, 660, "research_audit", 10.5, MUTED, mono=True)
    for x, label, detail in ((380, "Freeze identity", "Hash the real input files"),
                             (760, "Preserve raw records", "Derive metrics after persistence"),
                             (1190, "Replay and reconcile", "Retain unresolved cost")):
        text(ax, x, 623, label, 17, INK, "bold")
        text(ax, x, 663, detail, 13, MUTED)
    text(ax, 72, 736, "In-process research controls. Isolation, single-writer and integrity limits are documented separately.", 11.5, MUTED)
    save(fig, "architecture")


def results():
    summary = json.loads((ROOT / "docs/demo_results/sprinkler_summary.json").read_text())["summary"]
    fig, ax = canvas(1600, 800)
    text(ax, 72, 47, "EXAMPLE 01 / DECISION-RELEVANT INFORMATION", 11, MUTED, mono=True)
    text(ax, 69, 98, "Information can matter differently to a decision.", 31, INK, "bold")
    text(ax, 72, 167, "Sprinkler synthetic control · 400 balanced seeds · noiseless probes · probe cost 0.05", 16, MUTED)
    line(ax, [(885, 249), (885, 671)], RULE)
    text(ax, 72, 245, "A / Net utility", 18, INK, "bold")
    keys = ["MAP", "BMA", "Robust", "GlobalIG", "RandomProbe", "FalsifyWM", "Oracle"]
    names = ["MAP", "BMA", "Robust", "Global information gain", "Random probe", "Decision-directed", "Oracle"]
    x0, x1 = 353, 759
    for value in (0, .25, .5, .75, 1):
        xx = x0 + value * (x1 - x0)
        line(ax, [(xx, 310), (xx, 663)], "#E1E4DE", .8)
        text(ax, xx, 682, f"{value:g}", 11, MUTED, mono=True, ha="center")
    for i, (key, name) in enumerate(zip(keys, names)):
        yy, value = 325 + i * 51, summary[key]["total_utility_mean"]
        color = "#436C42" if key == "FalsifyWM" else BLUE if key == "GlobalIG" else "#8A9A9F"
        text(ax, 72, yy - 10, name, 15, INK if key == "FalsifyWM" else MUTED,
             "bold" if key == "FalsifyWM" else "regular")
        line(ax, [(x0, yy + 4), (x0 + value * (x1 - x0), yy + 4)], color, 2.5)
        ax.scatter([x0 + value * (x1 - x0)], [yy + 4], s=63 if key == "FalsifyWM" else 36,
                   facecolor=color, edgecolor=PAPER, linewidth=1.2, zorder=4)
        text(ax, 855, yy - 8, f"{value:.3f}", 12, color, mono=True, ha="right")
    text(ax, 944, 245, "B / What remains uncertain?", 18, INK, "bold")
    chart = fig.add_axes((.618, .209, .287, .342), facecolor=PAPER)
    chart.set(xlim=(0, 5), ylim=(.65, 2.4), xticks=(0, 1, 2, 3, 4, 5), yticks=(1, 2))
    chart.grid(color="#DFE3DC", lw=.7)
    chart.set_axisbelow(True)
    for side in ("top", "right", "left"):
        chart.spines[side].set_visible(False)
    chart.spines["bottom"].set_color("#B7C3BC")
    chart.tick_params(length=0, labelsize=11, colors=MUTED, pad=9)
    for label in chart.get_xticklabels() + chart.get_yticklabels():
        label.set_fontproperties(face(11, mono=True))
    for key, label, color, offset in (("GlobalIG", "Global IG", BLUE, (10, 12)),
                                       ("FalsifyWM", "Decision-directed", "#436C42", (-128, -28))):
        x, y = summary[key]["entropy_after_mean"], summary[key]["decision_classes_after_mean"]
        chart.scatter([x], [y], s=120, color=color, edgecolor=PAPER, lw=1.5, zorder=5)
        chart.annotate(label, (x, y), xytext=offset, textcoords="offset points",
                       fontproperties=face(12), color=color)
    text(ax, 972, 681, "Remaining model entropy / bits", 12, MUTED)
    text(ax, 944, 300, "Decision classes", 11.5, MUTED)
    text(ax, 944, 719, "Different worlds can still support the same action.", 13.5, INK)
    text(ax, 72, 752, "Constructed finite-task outcomes; no inference across real tasks or independently learned models. Means shown without error bars.", 11.5, MUTED)
    save(fig, "sprinkler_results")


def coherence():
    import sys
    sys.path.insert(0, str(ROOT))
    from sbcc.hybrid_ensemble import audit_componentwise_mixing, make_crossed_reward_counterexample
    saved = json.loads((ROOT / "docs/demo_results/hybrid_summary.json").read_text())
    worlds = make_crossed_reward_counterexample()
    audit = audit_componentwise_mixing(worlds)
    assert saved["candidate_count"] == audit.candidate_count == 2
    assert saved["component_combinations"] == audit.combination_count == 4
    pairs = {(p.transition_candidate_id, p.reward_candidate_id): p for p in audit.combinations}
    fig, ax = canvas(1600, 700)
    text(ax, 72, 47, "EXAMPLE 02 / CANDIDATE COHERENCE", 11, MUTED, mono=True)
    text(ax, 69, 98, "Keep transition and reward in the same world.", 31, INK, "bold")
    text(ax, 72, 181, "Two complete worlds", 22, INK, "bold")
    text(ax, 72, 229, "Each prefers the safe action.\nCrossing their components creates\na different ranking.", 20, MUTED)
    for world, yy, name, color in zip(worlds, (389, 487), ("World A", "World B"), (BLUE, "#7F9F76")):
        ax.add_patch(Rectangle((72, yy), 7, 61, fc=color, ec="none"))
        text(ax, 99, yy - 2, name, 18, INK, "bold")
        values = world.action_values()
        text(ax, 99, yy + 35, f"safe {values['safe']:.1f}    risky {values['risky']:.1f}", 13, MUTED, mono=True)
    text(ax, 782, 181, "Transition", 12, MUTED, mono=True)
    text(ax, 1084, 225, "Reward A", 14, MUTED, mono=True)
    text(ax, 1345, 225, "Reward B", 14, MUTED, mono=True)
    for r, row in enumerate(("A", "B")):
        text(ax, 782, 318 + r * 166, row, 24, INK, mono=True)
        for c in range(2):
            pair = pairs[(worlds[r].candidate_id, worlds[c].candidate_id)]
            coherent = pair.is_complete_candidate
            x, y = 988 + c * 261, 278 + r * 166
            panel(ax, x, y, 235, 142, "#E9EEE5" if coherent else DARK,
                  "#C9D3C2" if coherent else DARK, radius=6)
            fg = INK if coherent else WHITE
            text(ax, x + 22, y + 20, "COMPLETE WORLD" if coherent else "CROSSED PAIR", 9.5,
                 MUTED if coherent else LIME, mono=True)
            action = pair.ranking[0]
            text(ax, x + 22, y + 57, action, 25, fg, "bold")
            text(ax, x + 213, y + 67, f"{pair.action_values[action]:.1f}", 15, fg, mono=True, ha="right")
            text(ax, x + 22, y + 108, "preferred action", 10, MUTED if coherent else DIM)
    text(ax, 72, 635, "Exact two-world counterexample. Shows a possible failure under this interface; does not estimate prevalence in external systems.", 11.5, MUTED)
    save(fig, "coherence")


if __name__ == "__main__":
    banner()
    architecture()
    results()
    coherence()
