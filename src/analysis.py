"""Evidence level metrics, dispute metrics and the downstream prediction of the evaluation.
The conflict handling configurations and the baselines share one aggregator, so differences between them
come from how conflicts are handled and reported."""
import numpy as np, pandas as pd
from sim import *
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

W, STEP = 24, 6                      # window of two hours and decision interval of 30 minutes, in 5 minute slots
TAU = 0.9; NCONF = 2; WCONF = 3.48   # recommendation threshold, confirmation requests per decision, weight of a confirmation
sig = lambda x: 1 / (1 + np.exp(-x)); logit = lambda p: np.log(p / (1 - p))
BAYK = (K_SOC, K_IT, K_CO)

def priors(w):
    L = w["L"]; po = np.zeros(L.n_bays); pb = np.zeros(L.n_bays)
    for z in range(len(L.zones)):
        b = L.bays_of[z]; po[b] = logit(np.clip(w["occ"][b, :SLOTS_DAY].mean(), .02, .98)); pb[b] = logit(np.clip(w["blk"][b, :SLOTS_DAY].mean(), .01, .5))
    return po, pb

def fuse(A, mask, t0, nb):
    d = 0.5 ** ((t0 - A["t"]) / 2.0); Wt = A["w"] / 100.0 * d
    co = A["kind"] == K_CO; v = A["v"]
    occ_c = np.where(co, np.where(v == 2, Wt, np.where(v == 0, -Wt, 0.0)), np.where(v == 1, Wt, -Wt))
    blk_c = np.where(co, np.where(v == 1, Wt, np.where(v == 0, -Wt, 0.0)), 0.0)
    m = mask
    so = np.bincount(A["s"][m], occ_c[m], minlength=nb); sb = np.bincount(A["s"][m], blk_c[m], minlength=nb)
    n = np.bincount(A["s"][m], minlength=nb); return so, sb, n

def pav(so, sb, po, pb): return (1 - sig(po + so)) * (1 - sig(pb + sb))

def rec_set(P1, P2, n, tau, tolerant):
    r = P1 >= tau
    if tolerant: r = r & (P2 >= tau)
    return r

def evidence_metrics(df, res):
    """Information loss and conflict interpretation, evaluated once per estimate on settled evidence
    (estimates 6 to 11 slots old at the decision time) and on online evidence (newest 6 slots)."""
    lab = df.set_index("id").lab; t_by_id = df.set_index("id").t
    t_arr = df.t.values; ids_all = df.id.values; labs_all = df.lab.values; rows = []
    for rec in res:
        t0 = rec["t0"]
        for phase, (lo, hi) in (("settled", (t0 - 2 * STEP + 1, t0 - STEP)), ("online", (t0 - STEP + 1, t0))):
            i0, i1 = np.searchsorted(t_arr, lo, "left"), np.searchsorted(t_arr, hi, "right")
            ids = ids_all[i0:i1]; lb = labs_all[i0:i1]
            fs = np.isin(ids, list(rec["R_s"])); fa = np.isin(ids, list(rec["R_a"]))
            for l, f_s, f_a in zip(lb, fs, fa): rows.append((phase, l, f_s, f_a))
    d = pd.DataFrame(rows, columns=["phase", "lab", "fs", "fa"]); return d

def summarise_evidence(d):
    out = {}
    for phase, g in d.groupby("phase"):
        valid = g[g.lab != LAB_ERR]; err = g[g.lab == LAB_ERR]; app = g[g.lab.isin([LAB_TIME, LAB_SEM, LAB_GRAN])]
        fl = g[g.fs]
        pred_app = fl[~fl.fa]
        out[phase] = dict(
            n=len(g), n_valid=len(valid), n_err=len(err), n_apparent=len(app),
            loss_strict=float(valid.fs.mean()), loss_aligned=float(valid.fa.mean()),
            apparent_removed_strict=float(app.fs.mean()), apparent_removed_aligned=float(app.fa.mean()),
            err_removed_strict=float(err.fs.mean()), err_removed_aligned=float(err.fa.mean()),
            flagged_strict=int(g.fs.sum()), flagged_aligned=int(g.fa.sum()),
            flagged_valid_share_strict=float((fl.lab != LAB_ERR).mean()) if len(fl) else np.nan,
            flagged_valid_share_aligned=float((g[g.fa].lab != LAB_ERR).mean()) if g.fa.sum() else np.nan,
            apparent_precision=float(pred_app.lab.isin([LAB_TIME, LAB_SEM, LAB_GRAN]).mean()) if len(pred_app) else np.nan,
            apparent_recall=float((~app[app.fs].fa).mean()) if app.fs.sum() else np.nan,
            genuine_retained_flag=float(err[err.fs].fa.mean()) if err.fs.sum() else np.nan)
    return out

def dispute_metrics(out, tol=("S3", "S4")):
    rows = []
    for S in tol:
        tot = disp = disp_repair_wrong = 0; unknown = 0
        for o in out:
            P1, P2 = o[S]["P1"], o[S]["P2"]; d = (P1 >= TAU) != (P2 >= TAU); a = o["avail"]
            tot += len(a); disp += d.sum(); disp_repair_wrong += (d & ((P1 >= TAU) != a)).sum(); unknown += (o[S]["n"] == 0).sum()
        rows.append(dict(strategy=S, disputed_share=disp / tot, repair_would_be_wrong=disp_repair_wrong / max(1, disp), unknown_share=unknown / tot))
    return pd.DataFrame(rows)

def features(out, w, S, oracle=False, hmax=12):
    L = w["L"]; T = w["T"]; rows = []; prev = None
    av = ((w["occ"] == 0) & (w["blk"] == 0)); maj = L.bays_of[1]; oth = [b for z in L.zone_order if z != 1 for b in L.bays_of[z]]
    for o in out:
        t0 = o["t0"]; hr = (t0 % SLOTS_DAY) / SLOTS_DAY * 2 * np.pi
        if oracle:
            fm = av[maj, t0].sum(); fo = av[oth, t0].sum(); fq = w["q"][t0]; fd = 0
        else:
            P = o[S]["P1"] if S in ("S1", "S2") else 0.5 * (o[S]["P1"] + o[S]["P2"])
            fm = P[maj].sum(); fo = P[oth].sum(); fq = o[S]["q"] if o[S]["q"] == o[S]["q"] else 0.0
            fd = ((o[S]["P1"] >= TAU) != (o[S]["P2"] >= TAU))[maj].sum() if S in ("S3", "S4") else 0
        dm = 0.0 if prev is None else fm - prev; prev = fm
        rows.append([fm, fo, fq, np.sin(hr), np.cos(hr), dm] + ([fd] if S in ("S3", "S4") and not oracle else []))
    return np.array(rows)

def predict(out, w, S, h, oracle=False):
    L = w["L"]; av = ((w["occ"] == 0) & (w["blk"] == 0)); maj = L.bays_of[1]
    X = features(out, w, S, oracle); t0s = np.array([o["t0"] for o in out]); ok = t0s + h < w["T"]
    y = np.array([(av[maj, t + h].sum() == 0) if t + h < w["T"] else False for t in t0s]).astype(int)
    day = t0s // SLOTS_DAY; p = np.full(len(out), np.nan)
    for d in np.unique(day):
        tr = (day != d) & ok; te = (day == d) & ok
        if len(np.unique(y[tr])) < 2 or te.sum() == 0: continue
        m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=500)).fit(X[tr], y[tr]); p[te] = m.predict_proba(X[te])[:, 1]
    return y, p, ok
