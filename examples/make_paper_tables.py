"""Write paper_tables.tex for a results folder from gin_seeds.json and numbers.json.

    python examples/make_paper_tables.py <results folder>

Tables I and III come from ``gin_seeds.json`` (examples/gin_seed_sweep.py), Table
II from ``numbers.json`` (examples/full_results.py), so no value is copied by hand.
"""
import json, os, sys

sys.path[:0] = [os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                os.path.dirname(os.path.abspath(__file__))]
from gin_seed_sweep import add_realism_scores                              # noqa: E402

out_dir = sys.argv[1]
seeds = json.load(open(os.path.join(out_dir, "gin_seeds.json")))
S, n_main = add_realism_scores(seeds["summary"]), seeds["n"]
N = json.load(open(os.path.join(out_dir, "numbers.json")))

def pr(v):            # precision / recall
    return f"${v['mean']:.2f} \\pm {v['std']:.2f}$"

def fd(v):
    m, s = v["mean"], v["std"]
    if m >= 10000:
        return f"${round(m, -2):,.0f} \\pm {round(s, -2):,.0f}$".replace(",", "{,}")
    if m >= 10:
        return f"${m:.0f} \\pm {s:.0f}$"
    return f"${m:.1f} \\pm {s:.1f}$"

def score(v):         # 0-10 realism score
    return f"${v['mean']:.1f} \\pm {v['std']:.1f}$"

def bold(cell):
    return "$\\mathbf{" + cell.strip("$") + "}$"

SK = ("gin.prdc.precision", "gin.prdc.recall", "gin.fid.value")
AK = ("gin-attributed.prdc.precision", "gin-attributed.prdc.recall", "gin-attributed.fid.value")

def cells(row, keys):
    return [pr(row[keys[0]]), pr(row[keys[1]]), fd(row[keys[2]])]

t1 = S["table1"]
rv, st, m1, rt = (t1["real vs real"], t1["staged"], t1["feedergan-improved"], t1["uniform random tree"])
st_s, st_a, m1_s, m1_a = cells(st, SK), cells(st, AK), cells(m1, SK), cells(m1, AK)
st_s.append(score(st["score.structure"])); m1_s.append(score(m1["score.structure"]))
st_a.append(score(st["score.attributes"])); m1_a.append(score(m1["score.attributes"]))
for a, b, k in ((st_s, m1_s, "score.structure"), (st_a, m1_a, "score.attributes")):
    if st[k]["mean"] >= m1[k]["mean"]: a[3] = bold(a[3])
    else: b[3] = bold(b[3])
# best model per column (recall and FD): bold, as in the earlier tables
if m1["gin.prdc.recall"]["mean"] > st["gin.prdc.recall"]["mean"]: m1_s[1] = bold(m1_s[1])
else: st_s[1] = bold(st_s[1])
if m1["gin.fid.value"]["mean"] < st["gin.fid.value"]["mean"]: m1_s[2] = bold(m1_s[2])
else: st_s[2] = bold(st_s[2])
if st[AK[1]]["mean"] > m1[AK[1]]["mean"]: st_a[1] = bold(st_a[1])
else: m1_a[1] = bold(m1_a[1])
if st[AK[2]]["mean"] < m1[AK[2]]["mean"]: st_a[2] = bold(st_a[2])
else: m1_a[2] = bold(m1_a[2])

def line(label, cs, indent="  "):
    half = len(cs) // 2
    return f"{label:<19}& " + " & ".join(cs[:half]) + "\n" + " " * 19 + "& " + " & ".join(cs[half:]) + " \\\\"

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
ranking of the models is identical under every seed. Real vs.\\ real compares the
held-out reference set with a second, disjoint held-out sample of real feeders;
it gives the values a perfect generator would reach at this sample size.
The score is $10 \\cdot F_1(P, R) / F_1^{{\\mathrm{{real}}}}$, the harmonic mean of
precision and recall relative to real vs.\\ real, computed per initialisation:
10 corresponds to an independent real sample and 0 to an untrained model. The
untrained random tree has no attributes.}}
\\label{{tab:realism}}
\\setlength{{\\tabcolsep}}{{3.5pt}}
\\begin{{tabular}}{{@{{}}lcccccccc@{{}}}}
\\toprule
 & \\multicolumn{{4}}{{c}}{{Structure}} & \\multicolumn{{4}}{{c}}{{Structure and attributes}} \\\\
\\cmidrule(lr){{2-5}}\\cmidrule(l){{6-9}}
 & P $\\uparrow$ & R $\\uparrow$ & FD $\\downarrow$ & Score $\\uparrow$ & P $\\uparrow$ & R $\\uparrow$ & FD $\\downarrow$ & Score $\\uparrow$ \\\\
\\midrule
{line("Real vs.\\ real", cells(rv, SK) + [score(rv["score.structure"])] + cells(rv, AK) + [score(rv["score.attributes"])])}
\\midrule
{line("Staged", st_s + st_a)}
{line("Improved FeederGAN", m1_s + m1_a)}
{line("Random tree", cells(rt, SK) + [score(rt["score.structure"])] + ["--", "--", "--", "--"])}
\\bottomrule
\\end{{tabular}}
\\end{{table*}}
"""

E, P = N["electrical"], N["power_flow"]
def pct(x, d=1): return f"{100 * x:.{d}f}"
ref, stg, mg = E["real (reference)"], E["staged (full)"], E["feedergan-improved"]
pr_r, pr_s, pr_m = P["50 kW | real"], P["50 kW | staged"], P["50 kW | feedergan-improved"]
n_pf = 64
table2 = f"""% ── Table II: physical consistency ───────────────────────────────────────────
\\begin{{table}}[t]
\\centering
\\caption{{Physical consistency of generated feeders: share of edges and of
feeders satisfying the phase-subset rule ($n = {n_main}$). Reference values from
held-out real feeders. All values in percent; the target is 100.}}
\\label{{tab:physical}}
\\footnotesize
\\setlength{{\\tabcolsep}}{{4pt}}
\\begin{{tabular}}{{@{{}}lcc@{{}}}}
\\toprule
 & Phase-valid edges & Phase-valid feeders \\\\
\\midrule
Reference feeders  & {pct(ref['phase-subset.edges'])} & {pct(ref['phase-subset.feeders'])} \\\\
\\midrule
Staged             & {pct(stg['phase-subset.edges'])} & {pct(stg['phase-subset.feeders'])} \\\\
Improved FeederGAN & {pct(mg['phase-subset.edges'])} & {pct(mg['phase-subset.feeders'])} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}

% ── Table III: operational performance ──────────────────────────────────────
\\begin{{table}}[t]
\\centering
\\caption{{Operational performance of generated feeders in an OpenDSS snapshot
power flow at 50~kW per node and 1.03~pu ($n = {n_pf}$): energized phases;
energized phases with a voltage inside the band 0.95--1.05~pu derived from ANSI
C84.1 Range A; lines within their conductor rating; and energized three-phase
buses with a voltage unbalance of at most 3\\,\\%. Reference values from held-out
real feeders. All values in percent; the target is 100.}}
\\label{{tab:operational}}
\\footnotesize
\\setlength{{\\tabcolsep}}{{3pt}}
\\begin{{tabular}}{{@{{}}lcccc@{{}}}}
\\toprule
 & Energized & Voltage & Lines within & Unbalance \\\\
 & phases & in band & rating & within limit \\\\
\\midrule
Reference feeders  & {pct(pr_r['energised'])} & {pct(pr_r['ansi_a'])} & {pct(1 - pr_r['overload'])} & {pct(pr_r['unbalance_ok'])} \\\\
\\midrule
Staged             & {pct(pr_s['energised'])} & {pct(pr_s['ansi_a'])} & {pct(1 - pr_s['overload'])} & {pct(pr_s['unbalance_ok'])} \\\\
Improved FeederGAN & {pct(pr_m['energised'])} & {pct(pr_m['ansi_a'])}$^{{\\dagger}}$ & {pct(1 - pr_m['overload'])}$^{{\\dagger}}$ & {pct(pr_m['unbalance_ok'])}$^{{\\dagger}}$ \\\\
\\bottomrule
\\multicolumn{{5}}{{@{{}}p{{\\columnwidth}}@{{}}}}{{\\footnotesize $^{{\\dagger}}$Measured on
the energized part only: with {pct(1 - pr_m['energised'])}\\,\\% of phases de-energized, the
remaining network carries little load, so these values do not indicate good
operation.}}
\\end{{tabular}}
\\end{{table}}
"""

t3 = S["table3"]
rows3 = [("GAN", "Trained model (staged)", "GAN + MDN (staged)"),
         ("GAN", "Independent", "GAN + i.i.d. echo"),
         ("Random tree", "Trained model", "random tree + MDN"),
         ("Random tree", "Independent", "random tree + echo")]
n3 = min(256, n_main)
body3 = "\n".join(
    f"{topo:<11} & {att:<22}\n  & " + " & ".join(cells(t3[key], SK)) + "\n  & "
    + " & ".join(cells(t3[key], AK)) + " \\\\" for topo, att, key in rows3)
table3 = f"""% ── Table IV: error localisation by component swap (n = {n3}) ──────────────
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

header = """% Minimal scorecard for the results section (framework as the contribution).
% Held-out protocol: models trained on the training split; every real set is
% held out. Generated by make_paper_tables.py from gin_seeds.json and numbers.json.
% Needs \\usepackage{booktabs}. Captions are drafts; reword freely.

"""
# ── Table V: summary scores per dimension (summary_scores.py) ───────────────
from summary_scores import summary_scores                                  # noqa: E402

SC = summary_scores(S, N)
json.dump(SC, open(os.path.join(out_dir, "summary_scores.json"), "w"), indent=2)

def plain(x):
    return f"${x:.1f}$"

def best_of(cells_by_model, values_by_model):
    best = max(values_by_model, key=values_by_model.get)
    cells_by_model[best] = bold(cells_by_model[best])
    return cells_by_model

models4 = ("staged", "feedergan-improved")
cols4 = {}
for col, get, fmt in (("realism", lambda s: s["realism"]["mean"], lambda s: score(s["realism"])),
                      ("physical", lambda s: s["physical"], lambda s: plain(s["physical"])),
                      ("operational", lambda s: s["operational"], lambda s: plain(s["operational"])),
                      ("overall", lambda s: s["overall"]["mean"], lambda s: score(s["overall"]))):
    cols4[col] = best_of({m: fmt(SC[m]) for m in models4}, {m: get(SC[m]) for m in models4})
row4 = {m: [cols4[c][m] for c in ("realism", "physical", "operational", "overall")] for m in models4}
real4 = [plain(SC["real"]["realism"]["mean"]), plain(SC["real"]["physical"]),
         plain(SC["real"]["operational"]), plain(SC["real"]["overall"]["mean"])]
table4 = f"""% ── Table V: summary scores per evaluation dimension ───────────────────────
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
embedding is involved.}}
\\label{{tab:summary_scores}}
\\setlength{{\\tabcolsep}}{{4pt}}
\\begin{{tabular}}{{@{{}}lcccc@{{}}}}
\\toprule
 & Realism & Physical & Operational & Overall \\\\
\\midrule
Reference feeders  & {" & ".join(real4)} \\\\
\\midrule
Staged             & {" & ".join(row4["staged"])} \\\\
Improved FeederGAN & {" & ".join(row4["feedergan-improved"])} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""

open(os.path.join(out_dir, "paper_tables.tex"), "w", encoding="utf-8").write(
    header + table1 + "\n" + table2 + "\n" + table3 + "\n" + table4)
print("written")
