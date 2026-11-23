"""Report of the Experiment 1 runs: results with intervals over weeks, paired differences, and the sensitivity analysis."""
import os, time, platform, datetime, numpy as np, pandas as pd
from scipy import stats
import warnings; warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
from evaluation import STRATS, NAMES

def ci(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]; n = len(x)
    if n == 0: return np.nan, np.nan, np.nan, 0
    m = x.mean()
    if n < 2: return m, np.nan, np.nan, n
    h = stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n); return m, m - h, m + h, n
def f(x, d=2, scale=1.0): return "n/a" if x != x else f"{x * scale:.{d}f}"
def mci(x, d=2, scale=1.0, ub=None):
    m, lo, hi, n = ci(x)
    if m != m: return "n/a"
    if lo != lo: return f"{m * scale:.{d}f}"
    if scale == 100.0: lo, hi = max(lo, 0.0), min(hi, 1.0)          # proportions shown as percentages stay inside 0 to 100
    elif ub is not None: hi = min(hi, ub)                          # AUROC stays at or below 1
    return f"{m * scale:.{d}f} [{lo * scale:.{d}f}, {hi * scale:.{d}f}]"
def md(cols, rows):
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    return "\n".join(out + ["| " + " | ".join(str(v) for v in r) + " |" for r in rows])
def wilson(k, n, z=1.96):
    if n == 0: return np.nan, np.nan
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d; return c - h, c + h
def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); adj = np.empty_like(p); run = 0.0
    for r, i in enumerate(o): run = max(run, (len(p) - r) * p[i]); adj[i] = min(1.0, run)
    return adj

CLAIMS = [("C1", "The aligned interpretation removes fewer valid observations than the strict one", lambda r: r["ev_loss_aligned"] < r["ev_loss_strict"]),
          ("C2", "The aligned interpretation identifies at least 90% of the apparent conflicts", lambda r: r["ev_apparent_recall"] >= 0.9),
          ("C3", "Preserve has fewer wrong recommendations than Repair", lambda r: r["wrong_S3_0.9"] < r["wrong_S2_0.9"]),
          ("C4", "Preserve has fewer wrong recommendations than statistical fusion (S1)", lambda r: r["wrong_S3_0.9"] < r["wrong_S1_0.9"]),
          ("C5", "The full framework has fewer wrong recommendations than Repair", lambda r: r["wrong_S4_0.9"] < r["wrong_S2_0.9"]),
          ("C6", "The full framework has fewer wrong recommendations than statistical fusion (S1)", lambda r: r["wrong_S4_0.9"] < r["wrong_S1_0.9"]),
          ("C7", "Preserve has fewer wrong recommendations than every conventional baseline (S1, S5, S6)", lambda r: r["wrong_S3_0.9"] < min(r["wrong_S1_0.9"], r["wrong_S5_0.9"], r["wrong_S6_0.9"])),
          ("C8", "Disputes are rare (below 2% of bay decisions) and the value retained by repair is wrong for most of them (above 50%)", lambda r: r["disp_S3_share"] < 0.02 and r["disp_S3_repairwrong"] > 0.5)]

def seed_section(d, title, with_auc):
    L = [f"## {title}", "", (f"Single run (seed {int(d.seed.iloc[0])}). No interval is computed." if len(d) == 1 else f"{len(d)} weeks. Values are means over weeks with 95% confidence intervals (t distribution over weeks) in brackets."), ""]
    rows = []
    for S in STRATS:
        r = [f"{S} {NAMES[S]}", mci(d[f"wrong_{S}_0.9"], 2, 100), mci(d[f"recall_{S}_0.9"], 1, 100), mci(d[f"brier_{S}"], 4), mci(d[f"bayauc_{S}"], 4, ub=1.0)]
        if with_auc: r += [mci(d[f"auc_{S}_{h}"], 3, ub=1.0) if f"auc_{S}_{h}" in d else "n/a" for h in (15, 30, 60)]
        rows.append(r)
    if with_auc: rows.append(["True state", "", "", "", ""] + [mci(d[f"auc_ORACLE_{h}"], 3, ub=1.0) if f"auc_ORACLE_{h}" in d else "n/a" for h in (15, 30, 60)])
    L += ["### Strategies", "", md(["Strategy", "Wrong recommendations at 0.9 (%)", "Recall at 0.9 (%)", "Brier score", "Bay AUROC"] + (["AUROC 15 min", "AUROC 30 min", "AUROC 60 min"] if with_auc else []), rows), ""]
    pairs = [] if len(d) < 3 else [("S3", "S1"), ("S4", "S1"), ("S2", "S1"), ("S5", "S1"), ("S6", "S1"), ("S3", "S5"), ("S4", "S5"), ("S3", "S2"), ("S4", "S2"), ("S3", "S4")]; out = []; pv = []; recs = []
    for a, b in pairs:
        diff = (d[f"wrong_{a}_0.9"] - d[f"wrong_{b}_0.9"]).values * 100; m, lo, hi, n = ci(diff)
        p = stats.ttest_rel(d[f"wrong_{a}_0.9"], d[f"wrong_{b}_0.9"]).pvalue if n > 2 and np.nanstd(diff) > 0 else np.nan; pv.append(1.0 if p != p else p)
        db = (d[f"brier_{a}"] - d[f"brier_{b}"]).values; da = (d[f"bayauc_{a}"] - d[f"bayauc_{b}"]).values
        recs.append([f"{a} minus {b}", f"{NAMES[a]} against {NAMES[b]}", mci(diff, 2), f"{int((diff < 0).sum())} of {n}", mci(db * 1000, 2), mci(da * 1000, 2)])
    adj = holm(pv)
    for r, p, q in zip(recs, pv, adj): out.append(r[:4] + [f"{p:.3g}", f"{q:.3g}"] + r[4:])
    if out: L += ["### Paired differences between strategies (same weeks)", "", "Wrong recommendations: difference in percentage points (negative means the first strategy is better). Weeks with a lower share for the first strategy are counted. p values are from paired t tests, adjusted for the ten comparisons with the Holm method. Brier and bay AUROC differences are multiplied by 1,000.", "",
          md(["Pair", "Meaning", "Difference in wrong recommendations (points)", "Weeks first better", "p", "p (Holm)", "Brier difference (x1000)", "Bay AUROC difference (x1000)"], out), ""]
    ev = [("Valid observations removed, strict (%)", "ev_loss_strict", 100, 2), ("Valid observations removed, aligned (%)", "ev_loss_aligned", 100, 2), ("Erroneous observations removed, strict (%)", "ev_err_removed_strict", 100, 1), ("Erroneous observations removed, aligned (%)", "ev_err_removed_aligned", 100, 1),
          ("Apparent conflicts identified (%)", "ev_apparent_recall", 100, 1), ("Observations classed as apparent that are labelled so (%)", "ev_apparent_precision", 100, 1), ("Disputed bay decisions, Preserve (%)", "disp_S3_share", 100, 2), ("Value retained by repair wrong among disputed, Preserve (%)", "disp_S3_repairwrong", 100, 1),
          ("Disputed bay decisions, full framework (%)", "disp_S4_share", 100, 3), ("Confirmation requests per hour", "nconf_per_hour", 1, 3), ("Explanation coverage, strict (%)", "expl_cover_s", 100, 1), ("Explanation coverage, aligned (%)", "expl_cover_a", 100, 1),
          ("Decision time, strict (s)", "lat_s", 1, 2), ("Decision time, aligned (s)", "lat_a", 1, 2), ("Consistency check rounds, strict", "rounds_s", 1, 2), ("Consistency check rounds, aligned", "rounds_a", 1, 2)]
    L += ["### Evidence and conflict measures", "", md(["Measure", "Mean [95% CI]"], [[n, mci(d[c], dd, sc)] for n, c, sc, dd in ev if c in d]), "",
          f"Tile problems not solved to optimality (all seeds): strict {int(d['nonopt_tiles_s'].sum())}, aligned {int(d['nonopt_tiles_a'].sum())}. Consistency checks not verified: strict {int(d['unverified_s'].sum())}, aligned {int(d['unverified_a'].sum())}. Decision times depend on the machine and on how many runs share it.", ""]
    return L

def cond_table(g, label):
    r = g.mean(numeric_only=True); return r

def oat_section(d, a):
    L = ["## Sensitivity to the assumptions of the synthetic world, one at a time", "",
         f"Each assumption is multiplied by a low and a high factor while the others keep their default value, with {int(d.seed.nunique())} seed(s) per condition (the same seeds in every condition) and every {a.get('sens_step', 4)}th decision point. Each cell is the mean over seeds. Differences are in percentage points of wrong recommendations at the threshold 0.9 (negative means the first strategy is better).", ""]
    base = d[d["lv_condition"] == "base"].mean(numeric_only=True); rows = []; crows = []
    def line(label, lvl, r):
        sd = lambda x, y: f(100 * (r[f"wrong_{x}_0.9"] - r[f"wrong_{y}_0.9"]), 2)
        ok = [c[2](r) for c in CLAIMS]; return [label, lvl, f(100 * (r["ev_loss_strict"] - r["ev_loss_aligned"]), 2), sd("S2", "S1"), sd("S3", "S1"), sd("S4", "S1"), sd("S5", "S1"), sd("S6", "S1"), f(100 * r["disp_S3_share"], 2), f(100 * r["disp_S3_repairwrong"], 0), f"{sum(ok)} of {len(ok)}"], ok
    rr, ok = line("Default values", "1", base); rows.append(rr); crows.append(["Default values", "1"] + ["yes" if o else "no" for o in ok])
    from study import PARAMS
    for p in PARAMS:
        for lv in ("low", "high"):
            g = d[(d["lv_condition"] == p["name"]) & (d["lv_level"] == lv)]
            if g.empty: continue
            mult = p["lo"] if lv == "low" else p["hi"]; rr, ok = line(p["label"], f"{mult:g}", g.mean(numeric_only=True)); rows.append(rr); crows.append([p["label"], f"{mult:g}"] + ["yes" if o else "no" for o in ok])
    L += [md(["Assumption", "Factor", "Valid loss strict minus aligned (points)", "Repair minus S1", "Preserve minus S1", "Full minus S1", "Bayesian filter minus S1", "Evidential minus S1", "Disputed bays, Preserve (%)", "Repair wrong among disputed (%)", "Claims holding"], rows), "",
          "### Which claims hold in each condition", "", "\n".join(f"{c[0]}: {c[1]}." for c in CLAIMS), "", md(["Assumption", "Factor"] + [c[0] for c in CLAIMS], crows), ""]
    return L

def lhs_section(d, a):
    L = ["## Sensitivity to the assumptions of the synthetic world, all at once", "",
         f"{int(d['lv_draw'].nunique())} random combinations of all assumptions (Latin hypercube over the same ranges as above, log scale), {int(d.groupby('lv_draw').size().iloc[0])} seed(s) per combination, averaged over the seeds of a combination.", ""]
    g = d.groupby("lv_draw").mean(numeric_only=True).reset_index(); n = len(g); rows = []
    for cid, txt, fn in CLAIMS:
        k = int(sum(fn(r) for _, r in g.iterrows())); lo, hi = wilson(k, n); rows.append([cid, txt, f"{k} of {n}", f"{100 * k / n:.0f}% [{100 * lo:.0f}%, {100 * hi:.0f}%]"])
    L += [md(["Claim", "Statement", "Combinations in which it holds", "Share [95% Wilson interval]"], rows), ""]
    from study import PARAMS
    outs = [("Preserve minus S1", lambda r: r["wrong_S3_0.9"] - r["wrong_S1_0.9"]), ("Full minus S1", lambda r: r["wrong_S4_0.9"] - r["wrong_S1_0.9"]), ("Repair minus S1", lambda r: r["wrong_S2_0.9"] - r["wrong_S1_0.9"]), ("Valid loss strict minus aligned", lambda r: r["ev_loss_strict"] - r["ev_loss_aligned"])]
    Y = {o: np.array([fn(r) for _, r in g.iterrows()]) for o, fn in outs}; crow = []
    for p in PARAMS:
        x = g[f"lv_{p['name']}"].values; row = [p["label"]]
        for o, _ in outs:
            rho, pv = stats.spearmanr(x, Y[o]); row.append(f"{rho:+.2f}" + (" *" if pv < 0.05 else ""))
        crow.append(row)
    L += ["### Rank correlation between each assumption and the differences", "", "Spearman correlation across combinations. A star marks p below 0.05 without correction for the number of tests, and with this number of combinations only large effects can be detected.", "", md(["Assumption"] + [o for o, _ in outs], crow), ""]
    return L

def build(t_session, args):
    L = ["# Experiment 1 and sensitivity analysis", "",
         f"Generated {datetime.datetime.now():%Y-%m-%d %H:%M} on {platform.system()} {platform.release()} ({os.cpu_count()} cores), Python {platform.python_version()}. Time of this session {(time.time() - t_session) / 3600:.1f} h." if False else f"Generated {datetime.datetime.now():%Y-%m-%d %H:%M} on {platform.system()} {platform.release()} ({os.cpu_count()} cores), Python {platform.python_version()}.", "",
         "Strategies. S1 statistical fusion: the aggregator of the framework applied to all observations (log odds sum of calibrated, time discounted weights, no conflict handling). S2 Repair, S3 Preserve and S4 Full framework: the three configurations of the framework, which use the same aggregator after conflict resolution. "
         "S5 Bayesian filter: a hidden Markov model per bay over the same window, with the same calibrated source reliabilities and the bay state transitions estimated on the calibration day. S6 evidential fusion: Dempster Shafer combination of the same weights of evidence, with the same priors. "
         "No baseline is tuned on anything but the calibration day, and the weights, priors and transitions come from the same audited calibration day as the framework.", ""]
    for st, title, auc in (("main", "Normal faults (333 decision points per week)", True), ("stress", "Stress test (three times the faults and a surge day, 93 decision points)", False)):
        fn = f"runs_{st}.csv"
        if os.path.exists(fn): L += seed_section(pd.read_csv(fn), title, auc and True)
    if os.path.exists("runs_oat.csv"): L += oat_section(pd.read_csv("runs_oat.csv"), args)
    if os.path.exists("runs_lhs.csv"): L += lhs_section(pd.read_csv("runs_lhs.csv"), args)
    open("report.md", "w", encoding="utf-8").write("\n".join(L)); print("report.md written", flush=True)
