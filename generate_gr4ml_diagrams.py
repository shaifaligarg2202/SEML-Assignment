"""
Generate GR4ML diagrams aligned with Session-3 lecture notation.

Notation mapping (from Session 3.pdf):
  Business View   : Actor (stick-figure), Strategic/Decision(D)/Question(Q) Goals
                    (ovals), Insight (doc-rectangle), Indicator (traffic-light box)
                    Arrows: desires, supports, answers, evaluates, generates
  Analytics View  : PredictionGoal (split-pill oval), Algorithm (oval),
                    Softgoal (cloud/scalloped), influence symbols ++/+/-/--
                    Arrows: achieves, ++ / + / - / --
  Data Prep View  : Entity (table), Operator (rectangle), PrepTask (double-line rect)
                    Arrows: solid data-flow, dashed input/output
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "diagrams")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── palette ─────────────────────────────────────────────────────────
C_GOAL     = "#1565C0"
C_SOFTGOAL = "#E65100"
C_TASK     = "#2E7D32"
C_INSIGHT  = "#6A1B9A"
C_ACTOR    = "#37474F"
C_RESOURCE = "#4E342E"
C_INDIC    = "#00695C"
C_ALGO     = "#1B5E20"
C_MERGE    = "#455A64"
C_BG       = "#FAFAFA"
ARROW_CLR  = "#424242"

FS_TITLE = 15
FS_NODE  = 9.5
FS_LABEL = 8.5


# ═══════════════════════════════════════════════════════════════════
#  SHAPE HELPERS
# ═══════════════════════════════════════════════════════════════════
def _box(ax, xy, w, h, txt, color, fs=FS_NODE, tc="white", rad=0.25,
         ls="-", lw=1.3):
    b = FancyBboxPatch((xy[0]-w/2, xy[1]-h/2), w, h,
                       boxstyle=f"round,pad=0.08,rounding_size={rad}",
                       fc=color, ec="black", lw=lw, ls=ls)
    ax.add_patch(b)
    ax.text(xy[0], xy[1], txt, fontsize=fs, ha="center", va="center",
            color=tc, fontweight="bold")


def _oval(ax, xy, w, h, txt, color, fs=FS_NODE, tc="white", ls="-",
          lw=1.3):
    e = mpatches.Ellipse(xy, w, h, fc=color, ec="black", lw=lw, ls=ls)
    ax.add_patch(e)
    ax.text(xy[0], xy[1], txt, fontsize=fs, ha="center", va="center",
            color=tc, fontweight="bold")


def _cloud(ax, xy, w, h, txt, color=C_SOFTGOAL, fs=FS_NODE):
    """Softgoal – dashed-border ellipse (cloud proxy)."""
    e = mpatches.Ellipse(xy, w, h, fc=color, ec="black", lw=1.4,
                         ls=(0, (4, 3)), alpha=0.92)
    ax.add_patch(e)
    ax.text(xy[0], xy[1], txt, fontsize=fs, ha="center", va="center",
            color="white", fontweight="bold")


def _edge_oval(cx, cy, w, h, tx, ty):
    """Point on ellipse(cx,cy,w,h) border closest toward (tx,ty)."""
    import math
    dx, dy = tx - cx, ty - cy
    if dx == 0 and dy == 0:
        return (cx + w/2, cy)
    a, b = w/2, h/2
    angle = math.atan2(dy, dx)
    return (cx + a * math.cos(angle), cy + b * math.sin(angle))


def _edge_box(cx, cy, w, h, tx, ty, pad=0.08):
    """Point on rectangle(cx,cy,w,h) border closest toward (tx,ty)."""
    import math
    dx, dy = tx - cx, ty - cy
    if dx == 0 and dy == 0:
        return (cx, cy + (h/2 + pad))
    hw, hh = w/2 + pad, h/2 + pad
    sx = hw / abs(dx) if dx != 0 else 1e9
    sy = hh / abs(dy) if dy != 0 else 1e9
    s = min(sx, sy)
    return (cx + dx * s, cy + dy * s)


def _arrow(ax, fr, to, label="", color=ARROW_CLR, lw=1.4, rad=0.0,
           fs=FS_LABEL, offset=(0, 0), ls="-", ha="center"):
    style = f"arc3,rad={rad}"
    ax.annotate("", xy=to, xytext=fr,
                arrowprops=dict(arrowstyle="->", color=color, lw=lw,
                                connectionstyle=style, linestyle=ls,
                                shrinkA=0, shrinkB=0))
    if label:
        mx = (fr[0]+to[0])/2 + offset[0]
        my = (fr[1]+to[1])/2 + offset[1]
        ax.text(mx, my, label, fontsize=fs, ha=ha, va="center",
                color=color, fontstyle="italic",
                bbox=dict(boxstyle="round,pad=0.15", fc="white",
                          ec="none", alpha=0.9))


# ═══════════════════════════════════════════════════════════════════
#  1.  BUSINESS  VIEW
# ═══════════════════════════════════════════════════════════════════
def business_view():
    fig, ax = plt.subplots(figsize=(20, 14))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 14)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor(C_BG)
    ax.set_title("GR4ML  --  Business View\n"
                 "(Automated Loan Underwriting)",
                 fontsize=FS_TITLE, fontweight="bold", pad=20)

    # Column x-anchors and node sizes (chosen to keep every edge clear of
    # the boxes it is not attached to — no overlaps, no crossings).
    ax_x, sg_x, dg_x, rt_x = 2.3, 6.7, 11.0, 16.9
    ins_xs = [8.0, 11.5, 15.0]
    ins_y, sgl_y = 6.1, 3.2

    # ── Actors (far left) ───────────────────────────────────────
    _oval(ax, (ax_x, 11.8), 2.9, 1.15, "Actor:\nCredit Officer", C_ACTOR, 9)
    _oval(ax, (ax_x, 9.2),  2.9, 1.15, "Actor:\nLoan Applicant", C_ACTOR, 9)

    # ── Strategic Goals ─────────────────────────────────────────
    _oval(ax, (sg_x, 11.8), 3.6, 1.2,
          "Strategic Goal:\nMinimize Loan\nDefaults", C_GOAL, 8.5)
    _oval(ax, (sg_x, 9.2), 3.6, 1.2,
          "Strategic Goal:\nAutomate\nUnderwriting", C_GOAL, 8.5)

    # ── Decision Goal (centre) ──────────────────────────────────
    dg_y = 10.5
    _oval(ax, (dg_x, dg_y), 4.2, 1.35,
          "[D] Decision Goal:\nApprove / Deny / Flag", C_TASK, 8.5)

    # ── Indicators (top-right) + Question Goal (mid-right) ──────
    _box(ax, (rt_x, 11.6), 3.7, 1.7,
         "Indicators:\nDefault < 2.5%\nAuto-appr > 80%\nLatency < 150ms",
         C_INDIC, 7.5)
    for i, c in enumerate(["#4CAF50", "#FFC107", "#F44336"]):
        ax.add_patch(mpatches.Circle((rt_x - 1.4 + i*0.35, 12.55), 0.12,
                                     fc=c, ec="black", lw=0.6))
    _oval(ax, (rt_x, 8.4), 3.4, 1.3,
          "[Q] Question:\nCredit risk\nacceptable?", C_GOAL, 8)

    # ── Insights (row below Decision Goal) ──────────────────────
    ins_labels = [
        "Insight:\nDefault Probability\nScore",
        "Insight:\nRisk Tier\n(LOW / MED / HIGH)",
        "Insight:\nComputed\nNet Worth",
    ]
    for x, lbl in zip(ins_xs, ins_labels):
        _box(ax, (x, ins_y), 3.4, 1.15, lbl, C_INSIGHT, 8.5)

    # ── Softgoals (bottom row) ──────────────────────────────────
    for x, lbl in zip(ins_xs, ["Softgoal:\nAccuracy",
                               "Softgoal:\nPerformance",
                               "Softgoal:\nExplainability"]):
        _cloud(ax, (x, sgl_y), 3.4, 1.15, lbl)

    # ══════════ ARROWS ══════════
    # Actors -> Strategic Goals  (desires)
    for y in (11.8, 9.2):
        _arrow(ax, _edge_oval(ax_x, y, 2.9, 1.15, sg_x, y),
               _edge_oval(sg_x, y, 3.6, 1.2, ax_x, y), "desires")

    # Strategic Goals -> Decision Goal  (supports)
    for y, off in ((11.8, 0.32), (9.2, -0.32)):
        _arrow(ax, _edge_oval(sg_x, y, 3.6, 1.2, dg_x, dg_y),
               _edge_oval(dg_x, dg_y, 4.2, 1.35, sg_x, y),
               "supports", offset=(0, off))

    # Decision Goal -> Indicators  (evaluates)
    _arrow(ax, _edge_oval(dg_x, dg_y, 4.2, 1.35, rt_x, 11.6),
           _edge_box(rt_x, 11.6, 3.7, 1.7, dg_x, dg_y),
           "evaluates", offset=(0.1, 0.32))
    # Decision Goal -> Question Goal  (refines)
    _arrow(ax, _edge_oval(dg_x, dg_y, 4.2, 1.35, rt_x, 8.4),
           _edge_oval(rt_x, 8.4, 3.4, 1.3, dg_x, dg_y),
           "refines", offset=(0.1, -0.3))

    # Decision Goal -> Insights  (generates)
    for x, rad, loff in ((ins_xs[0], 0.0, (-0.9, 0.0)),
                         (ins_xs[1], 0.0, (0.55, 0.0)),
                         (ins_xs[2], 0.0, (0.9, 0.0))):
        _arrow(ax, _edge_oval(dg_x, dg_y, 4.2, 1.35, x, ins_y),
               _edge_box(x, ins_y, 3.4, 1.15, dg_x, dg_y),
               "generates", rad=rad, offset=loff)

    # Insights -> Softgoals  (contributes +)
    for x in ins_xs:
        _arrow(ax, _edge_box(x, ins_y, 3.4, 1.15, x, sgl_y),
               _edge_oval(x, sgl_y, 3.4, 1.15, x, ins_y),
               "+", offset=(0.35, 0))

    # Net-Worth Insight answers the Question Goal  (answers – dashed, local)
    _arrow(ax, _edge_box(ins_xs[2], ins_y, 3.4, 1.15, rt_x, 8.4),
           _edge_oval(rt_x, 8.4, 3.4, 1.3, ins_xs[2], ins_y),
           "answers", rad=-0.15, ls="--", offset=(0.35, -0.15))

    # ── Legend ──
    legend = [
        mpatches.Patch(fc=C_ACTOR,   ec="k", label="Actor"),
        mpatches.Patch(fc=C_GOAL,    ec="k", label="Goal / Question [Q]"),
        mpatches.Patch(fc=C_TASK,    ec="k", label="Decision Goal [D]"),
        mpatches.Patch(fc=C_INSIGHT, ec="k", label="Insight"),
        mpatches.Patch(fc=C_SOFTGOAL,ec="k", label="Softgoal"),
        mpatches.Patch(fc=C_INDIC,   ec="k", label="Indicator"),
    ]
    ax.legend(handles=legend, loc="lower left", fontsize=9,
              framealpha=0.9, title="GR4ML Legend", title_fontsize=10)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "gr4ml_business_view.png"),
                dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("[OK] Business View saved.")


# ═══════════════════════════════════════════════════════════════════
#  2.  ANALYTICS  DESIGN  VIEW
#     Two candidate algorithms (Random Forest = chosen, Logistic
#     Regression = baseline) scored against the quality softgoals.
#     RF wins the dominant Accuracy softgoal -> selected for serving.
#     Influence links are colour-coded: RF (solid green), LogReg (dashed blue).
# ═══════════════════════════════════════════════════════════════════
def analytics_design_view():
    C_LOGREG = "#1565C0"
    fig, ax = plt.subplots(figsize=(17, 12.5))
    ax.set_xlim(0, 17)
    ax.set_ylim(0, 12.5)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor(C_BG)
    ax.set_title("GR4ML  --  Analytics Design View\n"
                 "(Algorithm Trade-off vs Quality Softgoals)",
                 fontsize=FS_TITLE, fontweight="bold", pad=18)

    # ── PredictionGoal (top centre) ─────────────────────────────
    pg = (8.5, 10.8)
    _oval(ax, pg, 6.6, 1.4,
          "PredictionGoal:\nPredict Default Risk (Binary)", C_GOAL, 10)

    # ── Two candidate Algorithms (row 2) ────────────────────────
    rf = (4.3, 7.8)
    lr = (12.7, 7.8)
    _oval(ax, rf, 4.6, 1.5,
          "Algorithm:\nRandom Forest\n(n=150, depth=10)", C_ALGO, 9)
    _oval(ax, lr, 4.6, 1.5,
          "Algorithm:\nLogistic Regression\n(scaled, baseline)", C_LOGREG, 9)
    # "CHOSEN" badge on the Random Forest
    ax.text(rf[0], rf[0]*0 + 6.75, "★ CHOSEN", fontsize=8.5,
            ha="center", va="center", color="#1B5E20", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.22", fc="#C8E6C9",
                      ec="#1B5E20", lw=1.2))

    # ── Softgoals (row 3):  Accuracy | Explainability | Performance | Reliability
    sg_y = 3.7
    sg = {"acc": 2.7, "exp": 6.6, "perf": 10.5, "rel": 14.4}
    _cloud(ax, (sg["acc"], sg_y), 3.5, 1.35, "Softgoal:\nAccuracy\n(RF 94.7% / LR 90.3%)", C_SOFTGOAL, 7.5)
    _cloud(ax, (sg["exp"], sg_y), 3.5, 1.35, "Softgoal:\nExplainability\n(feature imp. / coeffs)", C_SOFTGOAL, 7.5)
    _cloud(ax, (sg["perf"], sg_y), 3.5, 1.35, "Softgoal:\nPerformance\n(< 150 ms)", C_SOFTGOAL, 7.5)
    _cloud(ax, (sg["rel"], sg_y), 3.5, 1.35, "Softgoal:\nReliability\n(Pydantic)", C_SOFTGOAL, 7.5)

    # ── Pydantic boundary resource feeding Reliability ──────────
    _box(ax, (sg["rel"], 6.2), 3.2, 1.0,
         "Pydantic Boundary\n(schemas.py)", C_MERGE, 7.5)

    # ── Insight (bottom) ────────────────────────────────────────
    ins = (6.6, 1.4)
    _box(ax, ins, 5.2, 1.0, "Insight:  Feature Importance Ranking", C_INSIGHT, 9)

    # ══════════ ARROWS ══════════
    # Both algorithms -> PredictionGoal  (achieves)
    _arrow(ax, _edge_oval(*rf, 4.6, 1.5, *pg), _edge_oval(*pg, 6.6, 1.4, *rf),
           "achieves", color=C_ALGO, offset=(-0.6, 0.15))
    _arrow(ax, _edge_oval(*lr, 4.6, 1.5, *pg), _edge_oval(*pg, 6.6, 1.4, *lr),
           "achieves", color=C_LOGREG, ls="--", offset=(0.6, 0.15))

    # RF influence links (solid green)
    _arrow(ax, _edge_oval(*rf, 4.6, 1.5, sg["acc"], sg_y),
           _edge_oval(sg["acc"], sg_y, 3.5, 1.35, *rf),
           "++", color=C_ALGO, lw=1.8, offset=(-0.35, 0.25))
    _arrow(ax, _edge_oval(*rf, 4.6, 1.5, sg["exp"], sg_y),
           _edge_oval(sg["exp"], sg_y, 3.5, 1.35, *rf),
           "+", color=C_ALGO, offset=(-0.35, 0.25))

    # LogReg influence links (dashed blue)
    _arrow(ax, _edge_oval(*lr, 4.6, 1.5, sg["perf"], sg_y),
           _edge_oval(sg["perf"], sg_y, 3.5, 1.35, *lr),
           "++", color=C_LOGREG, ls="--", lw=1.8, offset=(0.35, 0.25))
    _arrow(ax, _edge_oval(*lr, 4.6, 1.5, sg["exp"], sg_y),
           _edge_oval(sg["exp"], sg_y, 3.5, 1.35, *lr),
           "++", color=C_LOGREG, ls="--", lw=1.8, offset=(0.55, 0.25))

    # Pydantic boundary -> Reliability  (operationalizes +)
    _arrow(ax, _edge_box(sg["rel"], 6.2, 3.2, 1.0, sg["rel"], sg_y),
           _edge_oval(sg["rel"], sg_y, 3.5, 1.35, sg["rel"], 6.2),
           "+", color=C_MERGE, offset=(0.35, 0))

    # Explainability -> Insight  (generates)
    _arrow(ax, _edge_oval(sg["exp"], sg_y, 3.5, 1.35, *ins),
           _edge_box(*ins, 5.2, 1.0, sg["exp"], sg_y),
           "generates", offset=(-0.55, 0.1))

    # ── Softgoal scorecard (top-left, empty space) ──────────────
    ax.text(2.55, 10.75,
            "Softgoal scorecard\n"
            "                RF     LR\n"
            "Accuracy        ++     +\n"
            "Explainability  +      ++\n"
            "Performance     +      ++\n"
            "Reliability     +      +",
            fontsize=7.6, ha="center", va="center", family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", fc="white",
                      ec=ARROW_CLR, lw=1))

    # ── Legend + influence key ──────────────────────────────────
    legend = [
        mpatches.Patch(fc=C_GOAL,    ec="k", label="PredictionGoal"),
        mpatches.Patch(fc=C_ALGO,    ec="k", label="Algorithm: RF (chosen)"),
        mpatches.Patch(fc=C_LOGREG,  ec="k", label="Algorithm: LogReg (baseline)"),
        mpatches.Patch(fc=C_SOFTGOAL,ec="k", label="Softgoal"),
        mpatches.Patch(fc=C_INSIGHT, ec="k", label="Insight"),
    ]
    ax.legend(handles=legend, loc="lower left", fontsize=8.5,
              framealpha=0.9, title="GR4ML Legend", title_fontsize=9.5)

    ax.text(12.6, 0.75, "Influence key:  ++ Make  |  + Help  |  - Hurt  |  -- Break",
            fontsize=8, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.3", fc="#FFF3E0",
                      ec=C_SOFTGOAL, lw=1))

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "gr4ml_analytics_design_view.png"),
                dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("[OK] Analytics Design View saved.")


# ═══════════════════════════════════════════════════════════════════
#  3.  DATA  PREPARATION  VIEW
#     FIXED: Matches actual prepare_data.py implementation.
#     Single CSV source (Loan.csv), 5 actual prep tasks:
#       1. Drop non-predictive columns (36 -> 30)
#       2. Remove redundant features via correlation (30 -> 27)
#       3. Remove low-importance features (27 -> 21)
#       4. Categorical encoding (dict mapping)
#       5. Feature engineering (LoanToIncomeRatio, SavingsToLoanRatio)
#     Output: pd.DataFrame [1, 22]
# ═══════════════════════════════════════════════════════════════════
def data_preparation_view():
    fig, ax = plt.subplots(figsize=(17, 20))
    ax.set_xlim(0, 17)
    ax.set_ylim(0, 20)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor(C_BG)
    ax.set_title("GR4ML  --  Data Preparation View\n"
                 "(Pipeline Flow with Data Quality Goals)",
                 fontsize=FS_TITLE, fontweight="bold", pad=22)

    pipe_x = 5.0          # pipeline centre
    sg_x   = 13.0         # softgoal column centre

    # y-positions  (top = high)
    y_src   = 18.0
    y_t1    = 15.5
    y_t2    = 13.2
    y_t3    = 10.9
    y_t4    = 8.6
    y_t5    = 6.3
    y_out   = 3.8

    # ── START banner ────────────────────────────────────────────
    ax.text(pipe_x, 19.2, "DATA  FLOW  START",
            fontsize=11, ha="center", va="center", fontweight="bold",
            color="#1565C0",
            bbox=dict(boxstyle="round,pad=0.3", fc="#E3F2FD",
                      ec="#1565C0", lw=1.5))
    ax.text(sg_x, 19.2, "DATA  QUALITY  GOALS",
            fontsize=11, ha="center", va="center", fontweight="bold",
            color="#E65100",
            bbox=dict(boxstyle="round,pad=0.3", fc="#FFF3E0",
                      ec="#E65100", lw=1.5))

    # ── Data Source (single CSV) ───────────────────────────────
    _box(ax, (pipe_x, y_src), 5.4, 1.3,
         "Data Source:\nKaggle Loan.csv\n(20,000 records, 36 features)",
         C_RESOURCE, 8.5)

    # ── Pipeline Tasks ─────────────────────────────────────────
    tasks = [
        (y_t1, "Prep Task 1:\nDrop Non-Predictive Columns\n(ApplicationDate, RiskScore, InterestRate)\n36 → 30 features"),
        (y_t2, "Prep Task 2:\nRemove Redundant Features\n(Correlation > 0.95: MonthlyIncome,\nExperience, TotalAssets)  30 → 27"),
        (y_t3, "Prep Task 3:\nRemove Low-Importance Features\n(importance < 0.005: MaritalStatus,\nHomeOwnership, etc.)  27 → 21"),
        (y_t4, "Prep Task 4:\nCategorical Encoding\n(Dict mapping: EmploymentStatus,\nEducationLevel, LoanPurpose)"),
        (y_t5, "Prep Task 5:\nFeature Engineering\n(LoanToIncomeRatio,\nSavingsToLoanRatio)  21 → 22 + target"),
    ]
    tw, th = 5.8, 1.4
    for y, lbl in tasks:
        _box(ax, (pipe_x, y), tw, th, lbl, C_TASK, 7.5)
        # double bottom line for "Data Preparation Task" notation
        ax.plot([pipe_x - tw/2 + 0.15, pipe_x + tw/2 - 0.15],
                [y - th/2 + 0.08, y - th/2 + 0.08],
                color="white", lw=1.2, alpha=0.7)

    # Sequential arrows: source -> t1 -> t2 -> ... -> t5
    # Source -> T1
    src_out = _edge_box(pipe_x, y_src, 5.4, 1.3, pipe_x, y_t1)
    t1_in = _edge_box(pipe_x, y_t1, tw, th, pipe_x, y_src)
    _arrow(ax, src_out, t1_in, "", lw=2.2, color="#1565C0")

    # T1 -> T2 -> T3 -> T4 -> T5
    task_ys = [y_t1, y_t2, y_t3, y_t4, y_t5]
    for i in range(len(task_ys) - 1):
        fr_pt = _edge_box(pipe_x, task_ys[i], tw, th, pipe_x, task_ys[i+1])
        to_pt = _edge_box(pipe_x, task_ys[i+1], tw, th, pipe_x, task_ys[i])
        _arrow(ax, fr_pt, to_pt, "", lw=2.2, color="#1565C0")

    # ── Output ─────────────────────────────────────────────────
    out_w, out_h = 5.4, 1.2
    _oval(ax, (pipe_x, y_out), out_w, out_h,
          "Output:\nloan_data_processed.csv\n(pd.DataFrame  [20000, 22+1])",
          C_INDIC, 8.5)
    last_out = _edge_box(pipe_x, y_t5, tw, th, pipe_x, y_out)
    out_in   = _edge_oval(pipe_x, y_out, out_w, out_h, pipe_x, y_t5)
    _arrow(ax, last_out, out_in, "", lw=2.2, color="#1565C0")

    ax.text(pipe_x, 2.6, "DATA  FLOW  END",
            fontsize=11, ha="center", va="center", fontweight="bold",
            color="#2E7D32",
            bbox=dict(boxstyle="round,pad=0.3", fc="#E8F5E9",
                      ec="#2E7D32", lw=1.5))

    # ── Quality Softgoals (right column) ───────────────────────
    softgoals = [
        (y_t1, "Quality Goal:\nRelevant Features\n(no noise)"),
        (y_t2, "Quality Goal:\nNo Multicollinearity\n(independent features)"),
        (y_t3, "Quality Goal:\nParsimonious Model\n(no redundancy)"),
        (y_t4, "Quality Goal:\nHomogeneous\nEncoding"),
        (y_t5, "Quality Goal:\nRich Feature\nRepresentation"),
    ]
    sg_rels = [
        "operationalizes",
        "++",
        "+",
        "operationalizes",
        "++",
    ]
    sg_w, sg_h = 3.8, 1.2
    for (y, txt), rel in zip(softgoals, sg_rels):
        _cloud(ax, (sg_x, y), sg_w, sg_h, txt, C_SOFTGOAL, 8)
        task_out = _edge_box(pipe_x, y, tw, th, sg_x, y)
        sg_in    = _edge_oval(sg_x, y, sg_w, sg_h, pipe_x, y)
        _arrow(ax, task_out, sg_in,
               rel, color="#BF360C", lw=1.3,
               offset=(0, 0.35), ls="--")

    # ── Legend ──────────────────────────────────────────────────
    legend = [
        mpatches.Patch(fc=C_RESOURCE, ec="k",
                       label="Data Source (start)"),
        mpatches.Patch(fc=C_TASK,     ec="k",
                       label="Data Prep Task"),
        mpatches.Patch(fc=C_SOFTGOAL, ec="k",
                       label="Data Quality Softgoal"),
        mpatches.Patch(fc=C_INDIC,    ec="k",
                       label="Output Dataset (end)"),
    ]
    ax.legend(handles=legend, loc="lower left", fontsize=9,
              framealpha=0.9, title="GR4ML Legend", title_fontsize=10)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "gr4ml_data_prep_view.png"),
                dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("[OK] Data Preparation View saved.")


# ═══════════════════════════════════════════════════════════════════
#  4.  SYSTEM  ARCHITECTURE  DIAGRAM
# ═══════════════════════════════════════════════════════════════════
def system_architecture():
    fig, ax = plt.subplots(figsize=(22, 18))
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 18)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor(C_BG)
    ax.set_title("System Architecture Diagram\n"
                 "Loan Approval Risk Service  (Pipe-and-Filter + Microservice)",
                 fontsize=16, fontweight="bold", pad=22)

    # ── colours for layers ──
    C_CLIENT  = "#1565C0"
    C_PIPE    = "#2E7D32"
    C_ML      = "#E65100"
    C_CFG     = "#37474F"
    C_LOG     = "#546E7A"
    C_DATA    = "#4E342E"

    bw, bh = 4.2, 1.1

    # ════════════════════════════════════════════════════════════
    #  LAYER BACKGROUNDS
    # ════════════════════════════════════════════════════════════
    r1 = mpatches.FancyBboxPatch((0.3, 0.5), 5.4, 16.5,
            boxstyle="round,pad=0.2", fc="#E3F2FD", ec=C_CLIENT,
            lw=1.5, alpha=0.3)
    ax.add_patch(r1)
    ax.text(3.0, 17.3, "NON-ML  COMPONENTS", fontsize=10,
            ha="center", fontweight="bold", color=C_CLIENT)

    r2 = mpatches.FancyBboxPatch((6.0, 0.5), 9.5, 16.5,
            boxstyle="round,pad=0.2", fc="#E8F5E9", ec=C_PIPE,
            lw=1.5, alpha=0.3)
    ax.add_patch(r2)
    ax.text(10.75, 17.3, "FastAPI  INFERENCE  SERVICE  (Pipe-and-Filter)",
            fontsize=10, ha="center", fontweight="bold", color=C_PIPE)

    r3 = mpatches.FancyBboxPatch((15.8, 0.5), 5.9, 16.5,
            boxstyle="round,pad=0.2", fc="#FFF3E0", ec=C_ML,
            lw=1.5, alpha=0.3)
    ax.add_patch(r3)
    ax.text(18.75, 17.3, "ML  COMPONENTS", fontsize=10,
            ha="center", fontweight="bold", color=C_ML)

    # ════════════════════════════════════════════════════════════
    #  NON-ML COLUMN (left)
    # ════════════════════════════════════════════════════════════
    _box(ax, (3.0, 15.5), bw, bh,
         "API Clients\n(Swagger UI / Web Portal)", C_CLIENT, 8.5)
    _box(ax, (3.0, 12.5), bw, bh,
         "Config Management\nconfigs/config.yaml\napp/config.py", C_CFG, 8)
    _box(ax, (3.0, 9.5), bw, bh,
         "GET /health\nLiveness + Readiness\nCheck", C_CLIENT, 8.5)
    _box(ax, (3.0, 6.5), bw, bh,
         "Pydantic Schemas\napp/schemas.py\n20 validated fields", C_CFG, 8)
    _box(ax, (3.0, 3.0), bw, 1.3,
         "Structured JSON Logger\napp/logger.py\nJSON to stdout\n(latency, decision, features)",
         C_LOG, 7.5)

    # ════════════════════════════════════════════════════════════
    #  INFERENCE PIPELINE (centre)
    # ════════════════════════════════════════════════════════════
    pw, ph = 5.0, 1.2
    pipe_x = 10.75

    _box(ax, (pipe_x, 15.5), pw, bh,
         "POST /predict\napp/main.py\n(FastAPI Route)", C_PIPE, 8.5)

    f1_y = 13.0
    _box(ax, (pipe_x, f1_y), pw, ph,
         "Filter 1: validate_input()\napp/pipeline.py\n"
         "Business rules + Pydantic  (Robustness)", C_PIPE, 7.5)

    f2_y = 10.5
    _box(ax, (pipe_x, f2_y), pw, ph,
         "Filter 2: extract_features()\nFeature engineering\n"
         "LoanToIncomeRatio, SavingsToLoanRatio\nBuilds DataFrame [1, 22]",
         C_PIPE, 7.5)

    f3_y = 8.0
    _box(ax, (pipe_x, f3_y), pw, ph,
         "Filter 3: run_model()\nRandomForest.predict_proba()\n"
         "threshold = 0.50  (from config)", C_PIPE, 7.5)

    f4_y = 5.5
    _box(ax, (pipe_x, f4_y), pw, ph,
         "Filter 4: format_response()\nRisk tier: LOW/MED/HIGH\n"
         "LoanApprovalResult JSON", C_PIPE, 7.5)

    _box(ax, (pipe_x, 3.0), pw, bh,
         "JSON Response\n{is_approved, probability,\n"
         "risk_tier, net_worth, latency_ms}", C_PIPE, 7.5)

    # Pipeline vertical arrows
    pipe_nodes = [15.5, f1_y, f2_y, f3_y, f4_y, 3.0]
    pipe_heights = [bh, ph, ph, ph, ph, bh]
    for i in range(len(pipe_nodes) - 1):
        fr = _edge_box(pipe_x, pipe_nodes[i], pw, pipe_heights[i],
                       pipe_x, pipe_nodes[i+1])
        to = _edge_box(pipe_x, pipe_nodes[i+1], pw, pipe_heights[i+1],
                       pipe_x, pipe_nodes[i])
        _arrow(ax, fr, to, "", lw=2.5, color=C_PIPE)

    # "pipe" labels
    for i, y_mid in enumerate([(f1_y+f2_y)/2, (f2_y+f3_y)/2, (f3_y+f4_y)/2]):
        ax.text(pipe_x + 2.8, y_mid, "pipe", fontsize=7.5,
                ha="center", color=C_PIPE, fontstyle="italic",
                bbox=dict(boxstyle="round,pad=0.1", fc="white",
                          ec=C_PIPE, alpha=0.7, lw=0.8))

    # ════════════════════════════════════════════════════════════
    #  ML COLUMN (right)
    # ════════════════════════════════════════════════════════════
    ml_x = 18.75

    _box(ax, (ml_x, 15.5), bw, bh,
         "Raw Data\ndata/Loan.csv\n(Kaggle, 36 columns)", C_DATA, 8.5)
    _box(ax, (ml_x, 13.0), bw, ph,
         "Data Preparation\ndata/prepare_data.py\nDrop 15 cols, Encode\n"
         "Engineer 2 ratios -> 22 features", C_ML, 7.5)
    _box(ax, (ml_x, 10.5), bw, ph,
         "Model Training\ntraining/step1_notebook.py\nRandomForest(n=150,\n"
         "depth=10)  Acc: 94.7%", C_ML, 7.5)
    _box(ax, (ml_x, 8.0), bw, bh,
         "Model Artifact\napp/model.pkl\n(joblib serialized)", C_ML, 8.5)
    _box(ax, (ml_x, 5.5), bw, bh,
         "ModelStore\napp/main.py\nLoads model at startup\n"
         "via joblib.load()", C_ML, 7.5)

    # ML pipeline vertical arrows
    ml_nodes = [15.5, 13.0, 10.5, 8.0, 5.5]
    ml_heights = [bh, ph, ph, bh, bh]
    ml_labels = ["", "feeds", "trains", "saves"]
    for i in range(len(ml_nodes) - 1):
        fr = _edge_box(ml_x, ml_nodes[i], bw, ml_heights[i],
                       ml_x, ml_nodes[i+1])
        to = _edge_box(ml_x, ml_nodes[i+1], bw, ml_heights[i+1],
                       ml_x, ml_nodes[i])
        _arrow(ax, fr, to, ml_labels[i], lw=1.8, color=C_ML,
               offset=(0.7, 0))

    # ════════════════════════════════════════════════════════════
    #  CROSS-LAYER ARROWS
    # ════════════════════════════════════════════════════════════
    fr = _edge_box(3.0, 15.5, bw, bh, pipe_x, 15.5)
    to = _edge_box(pipe_x, 15.5, pw, bh, 3.0, 15.5)
    _arrow(ax, fr, to, "POST /predict", color=C_CLIENT, lw=1.3,
           ls="--", offset=(0, 0.3))

    fr = _edge_box(3.0, 12.5, bw, bh, pipe_x, f1_y)
    to = _edge_box(pipe_x, f1_y, pw, ph, 3.0, 12.5)
    _arrow(ax, fr, to, "injects settings", color=C_CFG, lw=1.2,
           ls="--", offset=(0, 0.3))

    fr = _edge_box(3.0, 9.5, bw, bh, pipe_x, f2_y)
    to = _edge_box(pipe_x, f2_y, pw, ph, 3.0, 9.5)
    _arrow(ax, fr, to, "monitors", color=C_CLIENT, lw=1.2,
           ls="--", offset=(0, 0.3))

    fr = _edge_box(3.0, 6.5, bw, bh, pipe_x, f4_y)
    to = _edge_box(pipe_x, f4_y, pw, ph, 3.0, 6.5)
    _arrow(ax, fr, to, "schema\nvalidation", color=C_CFG, lw=1.2,
           ls="--", offset=(0, 0.3))

    fr = _edge_box(pipe_x, 3.0, pw, bh, 3.0, 3.0)
    to = _edge_box(3.0, 3.0, bw, 1.3, pipe_x, 3.0)
    _arrow(ax, fr, to, "logs\ntransaction", color=C_LOG, lw=1.2,
           ls="--", offset=(0, 0.35))

    fr = _edge_box(ml_x, 5.5, bw, bh, pipe_x, f3_y)
    to = _edge_box(pipe_x, f3_y, pw, ph, ml_x, 5.5)
    _arrow(ax, fr, to, "loads\nmodel", color=C_ML, lw=1.3,
           ls="--", offset=(0, -0.35))

    # ════════════════════════════════════════════════════════════
    #  LEGEND
    # ════════════════════════════════════════════════════════════
    legend = [
        mpatches.Patch(fc="#E3F2FD", ec=C_CLIENT, lw=1.5,
                       label="Non-ML (API, Config, Logging)"),
        mpatches.Patch(fc="#E8F5E9", ec=C_PIPE, lw=1.5,
                       label="Inference Service (Pipe-and-Filter)"),
        mpatches.Patch(fc="#FFF3E0", ec=C_ML, lw=1.5,
                       label="ML Components (Data, Training, Model)"),
    ]
    ax.legend(handles=legend, loc="lower center", fontsize=9,
              framealpha=0.95, title="Architecture Layers",
              title_fontsize=10, ncol=3,
              bbox_to_anchor=(0.5, -0.01))

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "system_architecture.png"),
                dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("[OK] System Architecture saved.")


# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    business_view()
    analytics_design_view()
    data_preparation_view()
    system_architecture()
    print("\nAll diagrams regenerated in:", OUTPUT_DIR)
