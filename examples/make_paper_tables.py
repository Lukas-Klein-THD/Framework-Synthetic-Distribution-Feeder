"""Write paper_tables.tex for a results folder from gin_seeds.json and numbers.json.

    python examples/make_paper_tables.py <results folder>

Tables I and IV come from ``gin_seeds.json`` (examples/gin_seed_sweep.py),
Tables II and III from ``numbers.json`` (examples/full_results.py), Table V from
both via ``summary_scores.py``, so no value is copied by hand.

Each generator appears with its post-hoc phase-repaired variant ("+ phase
repair": the same feeders, every segment clamped to its parent's phases) where
the comparison is meaningful: realism, physical consistency, operational
performance and the summary scores.  Bold marks the better raw generator; a
repaired row is never bold, since its validity comes from the repair.
"""
import json, os, sys

sys.path[:0] = [os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                os.path.dirname(os.path.abspath(__file__))]
from gin_seed_sweep import add_realism_scores                              # noqa: E402
from summary_scores import summary_scores                                  # noqa: E402

out_dir = sys.argv[1]
seeds = json.load(open(os.path.join(out_dir, "gin_seeds.json")))
S, n_main = add_realism_scores(seeds["summary"]), seeds["n"]
N = json.load(open(os.path.join(out_dir, "numbers.json")))
E, P = N["electrical"], N["power_flow"]
SC = summary_scores(S, N)
json.dump(SC, open(os.path.join(out_dir, "summary_scores.json"), "w"), indent=2)
n_pf = 64

# ── formatting ───────────────────────────────────────────────────────────────
def pr(v):            # precision / recall
    return f"${v['mean']:.2f} \\pm {v['std']:.2f}$"

def fd(v):
    m, s = v["mean"], v["std"]
    if m >= 10000:
        return f"${round(m, -2):,.0f} \\pm {round(s, -2):,.0f}$".replace(",", "{,}")
    if m >= 10:
        return f"${m:,.0f} \\pm {s:,.0f}$".replace(",", "{,}")
    return f"${m:.1f} \\pm {s:.1f}$"

def score(v):         # 0-10 score with std over seeds
    return f"${v['mean']:.1f} \\pm {v['std']:.1f}$"

def plain(x):
    return f"${x:.1f}$"

def pct(x, d=1):
    return f"{100 * x:.{d}f}"

def bold(cell):
    if cell.startswith("$"):
        return "$\\mathbf{" + cell.strip("$") + "}$"
    return "\\textbf{" + cell + "}"

# ── rows: every generator followed by its repaired variant ───────────────────
REP = "\\quad + phase repair"
#: (label, gin_seeds table1 key, numbers electrical key, power-flow arm, summary key)
GEN = [("Staged", "staged", "staged (full)", "staged", "staged"),
       (REP, "staged + phase repair", "staged + phase repair", "staged + phase repair",
        "staged + phase repair"),
       ("Improved FeederGAN", "feedergan-improved", "feedergan-improved",
        "feedergan-improved", "feedergan-improved"),
       (REP, "feedergan-improved + phase repair", "feedergan-improved + phase repair",
        "feedergan-improved + phase repair", "feedergan-improved + phase repair")]
RAW = {"staged", "feedergan-improved"}             # gin / summary keys eligible for bold
REPAIR_NOTE = ("``+ phase repair'' denotes the same generated feeders after post-hoc "
               "repair, which clamps every segment to the phases of its parent, root "
               "outward; topology, lengths, and ratings are unchanged. Bold marks the "
               "better raw generator.")


def bold_best(cells, values, keys, higher=True):
    """Bold the best value among the raw generators only."""
    raw = [i for i, k in enumerate(keys) if k in RAW]
    best = (max if higher else min)(raw, key=lambda i: values[i])
    cells[best] = bold(cells[best])
    return cells


def line(label, cs):
    half = len(cs) // 2
    return (f"{label:<20}& " + " & ".join(cs[:half]) + "\n" + " " * 20 + "& "
            + " & ".join(cs[half:]) + " \\\\")


# ── Table I: realism ─────────────────────────────────────────────────────────
SK = ("gin.prdc.precision", "gin.prdc.recall", "gin.fid.value")
AK = ("gin-attributed.prdc.precision", "gin-attributed.prdc.recall", "gin-attributed.fid.value")
t1 = S["table1"]
keys1 = [g for _, g, _, _, _ in GEN]


def family(keys, score_key):
    """Columns P, R, FD, score of one metric family for all generator rows."""
    cols = [[pr(t1[k][keys[0]]) for k in keys1],
            bold_best([pr(t1[k][keys[1]]) for k in keys1],
                      [t1[k][keys[1]]["mean"] for k in keys1], keys1),
            bold_best([fd(t1[k][keys[2]]) for k in keys1],
                      [t1[k][keys[2]]["mean"] for k in keys1], keys1, higher=False),
            bold_best([score(t1[k][score_key]) for k in keys1],
                      [t1[k][score_key]["mean"] for k in keys1], keys1)]
    return [list(r) for r in zip(*cols)]


struct_rows, attr_rows = family(SK, "score.structure"), family(AK, "score.attributes")
rv, rt = t1["real vs real"], t1["uniform random tree"]
rows1 = "\n".join(line(label, s + a) for (label, *_), s, a in zip(GEN, struct_rows, attr_rows))

table1 = f"""% ── Table I: realism, structure and attributes (n = {n_main}) ───────────────────
% Mean +- sample standard deviation over five GIN initialisations (gin_seeds.json).
% Real sets are the two halves of the held-out test split; models never saw them.
\\begin{{table*}}[t]
\\centering
\\caption{{Realism scorecard. Precision (P) and recall (R) in a random graph
neural network embedding, and Fr\\'echet distance (FD) between Gaussians fitted
to the embeddings, for structure alone and for structure with attributes;
$n = {n_main}$ held-out feeders. Mean $\\pm$ standard deviation over five random
initialisations of the embedding (seeds 0--4); the generated feeders are the same
under every seed. Fr\\'echet distances scale with each initialisation, but the
ranking of the raw generators is identical under every seed. Real vs.\\ real
compares the held-out reference set with a second, disjoint held-out sample of
real feeders; it gives the values a perfect generator would reach at this sample
size. The score is $10 \\cdot F_1(P, R) / F_1^{{\\mathrm{{real}}}}$, the harmonic
mean of precision and recall relative to real vs.\\ real, computed per
initialisation: 10 corresponds to an independent real sample and 0 to an
untrained model. Phase repair leaves the topology unchanged, so it affects only
the attributed columns. {REPAIR_NOTE} The untrained random tree has no
attributes.}}
\\label{{tab:realism}}
\\setlength{{\\tabcolsep}}{{3.5pt}}
\\begin{{tabular}}{{@{{}}lcccccccc@{{}}}}
\\toprule
 & \\multicolumn{{4}}{{c}}{{Structure}} & \\multicolumn{{4}}{{c}}{{Structure and attributes}} \\\\
\\cmidrule(lr){{2-5}}\\cmidrule(l){{6-9}}
 & P $\\uparrow$ & R $\\uparrow$ & FD $\\downarrow$ & Score $\\uparrow$ & P $\\uparrow$ & R $\\uparrow$ & FD $\\downarrow$ & Score $\\uparrow$ \\\\
\\midrule
{line("Real vs.\\ real", [pr(rv[SK[0]]), pr(rv[SK[1]]), fd(rv[SK[2]]), score(rv["score.structure"]), pr(rv[AK[0]]), pr(rv[AK[1]]), fd(rv[AK[2]]), score(rv["score.attributes"])])}
\\midrule
{rows1}
\\midrule
{line("Random tree", [pr(rt[SK[0]]), pr(rt[SK[1]]), fd(rt[SK[2]]), score(rt["score.structure"]), "--", "--", "--", "--"])}
\\bottomrule
\\end{{tabular}}
\\end{{table*}}
"""

# ── Table II: physical consistency ───────────────────────────────────────────
ref = E["real (reference)"]
cols2 = [[pct(E[e]["phase-subset.edges"]) for _, _, e, _, _ in GEN],
         [pct(E[e]["phase-subset.feeders"]) for _, _, e, _, _ in GEN]]
for c, key in zip(cols2, ("phase-subset.edges", "phase-subset.feeders")):
    bold_best(c, [E[e][key] for _, _, e, _, _ in GEN], keys1)
rows2 = "\n".join(f"{label:<20}& {a} & {b} \\\\" for (label, *_), a, b in zip(GEN, *cols2))

table2 = f"""% ── Table II: physical consistency ───────────────────────────────────────────
\\begin{{table}}[t]
\\centering
\\caption{{Physical consistency of generated feeders: share of edges and of
feeders satisfying the phase-subset rule ($n = {n_main}$). Reference values from
held-out real feeders. All values in percent; the target is 100. After repair the
phase rule holds by construction. {REPAIR_NOTE}}}
\\label{{tab:physical}}
\\footnotesize
\\setlength{{\\tabcolsep}}{{4pt}}
\\begin{{tabular}}{{@{{}}lcc@{{}}}}
\\toprule
 & Phase-valid edges & Phase-valid feeders \\\\
\\midrule
Reference feeders   & {pct(ref['phase-subset.edges'])} & {pct(ref['phase-subset.feeders'])} \\\\
\\midrule
{rows2}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""

# ── Table III: operational performance ──────────────────────────────────────
def flow(arm):
    return P[f"50 kW | {arm}"]

pr_r, pr_m = flow("real"), flow("feedergan-improved")
metrics3 = [lambda f: f["energised"], lambda f: f["ansi_a"],
            lambda f: 1 - f["overload"], lambda f: f["unbalance_ok"]]
cols3 = []
for j, get in enumerate(metrics3):
    col = [pct(get(flow(arm))) for _, _, _, arm, _ in GEN]
    # the raw improved FeederGAN's V, W, U are measured on its energised part
    if j > 0:
        col = [c + "$^{\\dagger}$" if k == "feedergan-improved" else c
               for c, k in zip(col, keys1)]
    if j == 0:      # only energisation is comparable for bold across raw models
        bold_best(col, [get(flow(arm)) for _, _, _, arm, _ in GEN], keys1)
    cols3.append(col)
rows3 = "\n".join(f"{label:<20}& " + " & ".join(cs) + " \\\\"
                  for (label, *_), cs in zip(GEN, zip(*cols3)))

table3 = f"""% ── Table III: operational performance ──────────────────────────────────────
\\begin{{table}}[t]
\\centering
\\caption{{Operational performance of generated feeders in an OpenDSS snapshot
power flow at 50~kW per node and 1.03~pu ($n = {n_pf}$): energized phases;
energized phases with a voltage inside the band 0.95--1.05~pu derived from ANSI
C84.1 Range A; lines within their conductor rating; and energized three-phase
buses with a voltage unbalance of at most 3\\,\\%. Reference values from held-out
real feeders. All values in percent; the target is 100. {REPAIR_NOTE}}}
\\label{{tab:operational}}
\\footnotesize
\\setlength{{\\tabcolsep}}{{3pt}}
\\begin{{tabular}}{{@{{}}lcccc@{{}}}}
\\toprule
 & Energized & Voltage & Lines within & Unbalance \\\\
 & phases & in band & rating & within limit \\\\
\\midrule
Reference feeders   & {pct(pr_r['energised'])} & {pct(pr_r['ansi_a'])} & {pct(1 - pr_r['overload'])} & {pct(pr_r['unbalance_ok'])} \\\\
\\midrule
{rows3}
\\bottomrule
\\multicolumn{{5}}{{@{{}}p{{\\columnwidth}}@{{}}}}{{\\footnotesize $^{{\\dagger}}$Measured on
the energized part only: with {pct(1 - pr_m['energised'])}\\,\\% of phases de-energized, the
remaining network carries little load, so these values do not indicate good
operation.}}
\\end{{tabular}}
\\end{{table}}
"""

# ── Table IV: error localisation (component swap; repair does not apply) ────
def cells(row, keys):
    return [pr(row[keys[0]]), pr(row[keys[1]]), fd(row[keys[2]])]

t3 = S["table3"]
swaps = [("GAN", "Trained model (staged)", "GAN + MDN (staged)"),
         ("GAN", "Independent", "GAN + i.i.d. echo"),
         ("Random tree", "Trained model", "random tree + MDN"),
         ("Random tree", "Independent", "random tree + echo")]
n3 = min(256, n_main)
body3 = "\n".join(
    f"{topo:<11} & {att:<22}\n  & " + " & ".join(cells(t3[key], SK)) + "\n  & "
    + " & ".join(cells(t3[key], AK)) + " \\\\" for topo, att, key in swaps)
table4 = f"""% ── Table IV: error localisation by component swap (n = {n3}) ──────────────
% Alternative to the figure baseline_localisation.pdf; use one of the two.
% Full text width (table*): mean +- std over five GIN initialisations (gin_seeds.json).
\\begin{{table*}}[t]
\\centering
\\caption{{Error localisation in a modular pipeline. Each stage of the staged
model is replaced by an untrained component: a uniform random tree for the
topology, and attributes sampled independently per node from the marginals of
the training feeders for the attribute model. Each stage is diagnosed by its own
metric family: the topology swap moves the structural metrics, the attribute swap
moves the attributed recall. Precision (P), recall (R) and Fr\\'echet distance
(FD); $n = {n3}$ held-out feeders; mean $\\pm$ standard deviation over five
embedding initialisations. The improved FeederGAN generates jointly and cannot be
decomposed; it is shown for reference.}}
\\label{{tab:localisation}}
\\setlength{{\\tabcolsep}}{{4pt}}
\\begin{{tabular}}{{@{{}}llcccccc@{{}}}}
\\toprule
 & & \\multicolumn{{3}}{{c}}{{Structure}} & \\multicolumn{{3}}{{c}}{{Structure and attributes}} \\\\
\\cmidrule(lr){{3-5}}\\cmidrule(l){{6-8}}
Topology & Attributes & P $\\uparrow$ & R $\\uparrow$ & FD $\\downarrow$
                      & P $\\uparrow$ & R $\\uparrow$ & FD $\\downarrow$ \\\\
\\midrule
\\multicolumn{{2}}{{@{{}}l}}{{Real vs.\\ real}}
  & {" & ".join(cells(t3["real vs real"], SK))}
  & {" & ".join(cells(t3["real vs real"], AK))} \\\\
\\midrule
{body3}
\\midrule
\\multicolumn{{2}}{{@{{}}l}}{{Improved FeederGAN (joint)}}
  & {" & ".join(cells(t3["feedergan-improved"], SK))}
  & {" & ".join(cells(t3["feedergan-improved"], AK))} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table*}}
"""

# ── Table V: summary scores per evaluation dimension ─────────────────────────
keys5 = [s for *_, s in GEN]
spec5 = [("realism", lambda s: s["realism"]["mean"], lambda s: score(s["realism"])),
         ("physical", lambda s: s["physical"], lambda s: plain(s["physical"])),
         ("operational", lambda s: s["operational"], lambda s: plain(s["operational"])),
         ("overall", lambda s: s["overall"]["mean"], lambda s: score(s["overall"]))]
cols5 = [bold_best([fmt(SC[k]) for k in keys5], [get(SC[k]) for k in keys5], keys5)
         for _, get, fmt in spec5]
rows5 = "\n".join(f"{label:<20}& " + " & ".join(cs) + " \\\\"
                  for (label, *_), cs in zip(GEN, zip(*cols5)))
real5 = [plain(SC["real"]["realism"]["mean"]), plain(SC["real"]["physical"]),
         plain(SC["real"]["operational"]), plain(SC["real"]["overall"]["mean"])]

table5 = f"""% ── Table V: summary scores per evaluation dimension ───────────────────────
% summary_scores.json (examples/summary_scores.py); realism and overall carry the
% std over the five GIN seeds, physical and operational are single evaluations.
\\begin{{table}}[t]
\\centering
\\caption{{Summary scores per evaluation dimension (0--10). Realism is the mean
of the structural and attributed realism scores of Table~\\ref{{tab:realism}},
relative to real vs.\\ real. Physical: $10 \\cdot$ mean of the phase-valid edge
and feeder shares (Table~\\ref{{tab:physical}}). Operational: $10 \\cdot E \\cdot$
mean of voltage in band, lines within rating, and unbalance within limit, where
$E$ is the share of energized phases (Table~\\ref{{tab:operational}}). Physical
and operational scores use absolute targets of 100\\,\\%, so real feeders need
not reach 10. Overall: mean of the three dimensions, weighted equally. Mean
$\\pm$ standard deviation over five embedding initializations where the
embedding is involved. A repaired variant reaches a physical score of 10 by
construction. {REPAIR_NOTE}}}
\\label{{tab:summary_scores}}
\\setlength{{\\tabcolsep}}{{4pt}}
\\begin{{tabular}}{{@{{}}lcccc@{{}}}}
\\toprule
 & Realism & Physical & Operational & Overall \\\\
\\midrule
Reference feeders   & {" & ".join(real5)} \\\\
\\midrule
{rows5}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""

header = """% Scorecard tables for the results section (framework as the contribution).
% Held-out protocol: models trained on the training split; every real set is
% held out. Generated by make_paper_tables.py from gin_seeds.json and numbers.json.
% Needs \\usepackage{booktabs}. Captions are drafts; reword freely.

"""
open(os.path.join(out_dir, "paper_tables.tex"), "w", encoding="utf-8").write(
    header + "\n".join([table1, table2, table3, table4, table5]))
print("written")
