"""Fusion strategies, one run of the evaluation for a given seed and set of assumptions, and the resulting metrics.
S1 applies the aggregator of the framework to all observations. S2 (repair), S3 (preserve) and S4 (full framework) are the three configurations.
S5 is a Bayesian filter and S6 evidential fusion; neither removes observations."""
import time, numpy as np, pandas as pd
import sim, model
from sim import *
from resolve import *
import analysis as an
from analysis import fuse, pav, priors, TAU, NCONF, WCONF, BAYK, W, STEP

STRATS = ("S1", "S2", "S3", "S4", "S5", "S6")
NAMES = {"S1": "Statistical fusion", "S2": "Repair", "S3": "Preserve", "S4": "Full framework", "S5": "Bayesian filter (HMM)", "S6": "Evidential fusion (Dempster Shafer)"}
TAUS = (0.8, 0.9, 0.95, 0.98)
ALLOW_KEYS = ("lag_it", "resv", "lag_co", "lag_q", "census_c")
DEFAULT_ASSUME = dict(sim.ASSUME); DEFAULT_ALLOW = {k: model.P[k] for k in ALLOW_KEYS}

def set_assumptions(assume=None, allow_scale=1.0):
    sim.ASSUME.update(DEFAULT_ASSUME)
    if assume: sim.ASSUME.update(assume)
    for k in ALLOW_KEYS: model.P[k] = max(0, int(round(DEFAULT_ALLOW[k] * allow_scale)))

def setup_run(seed, days=8, fs=1.0, surge=(), assume=None, allow_scale=1.0, drift=None):
    set_assumptions(assume, allow_scale)
    w = simulate(days, seed=seed, lam0=0.5, surge_days=surge); kw = dict(seed=seed + 1, fs=fs)
    if drift is not None: kw["drift_per_day"] = drift
    df = gen_sources(w, **kw); tab = calibrate(df[df.t < SLOTS_DAY], np.random.default_rng(seed + 2)); df = apply_weights(df, tab)
    return w, df, tab

def run_decisions(w, df, step_sub=1):
    L = w["L"]; res = []
    for t0 in list(range(SLOTS_DAY + W - 1, w["T"], STEP))[::step_sub]:
        a = t0 - W + 1; sub = df[(df.t >= a) & (df.t <= t0)].reset_index(drop=True); rec = dict(t0=t0); tid = sub.set_index("id").t
        for name, al in (("s", False), ("a", True)):
            tt = time.time(); R, st = resolve_local(L, sub, a, t0, al, ks=0, kt=3, bt=6)
            recent = [e for e in R if tid.loc[e] >= t0 - STEP + 1][:20]
            t1 = time.time(); disp, expl = dispute_analysis(L, sub, a, t0, al, R, recent)
            rec.update({f"R_{name}": R, f"disp_{name}": disp, f"expl_{name}": expl, f"stats_{name}": st, f"t_res_{name}": t1 - tt, f"t_disp_{name}": time.time() - t1})
        res.append(rec)
    return res

# ---------------------------------------------------------------- conventional fusion baselines
def _contrib(A, t0):
    d = 0.5 ** ((t0 - A["t"]) / 2.0); Wt = A["w"] / 100.0 * d; co = A["kind"] == K_CO; v = A["v"]
    occ_c = np.where(co, np.where(v == 2, Wt, np.where(v == 0, -Wt, 0.0)), np.where(v == 1, Wt, -Wt))
    blk_c = np.where(co, np.where(v == 1, Wt, np.where(v == 0, -Wt, 0.0)), 0.0)
    return occ_c, blk_c, Wt

_POP = np.array([bin(i).count("1") for i in range(8)], float)
def _ds_add(m, F, wt):
    s = 1.0 - np.exp(-wt); new = (1.0 - s) * m; K = 0.0
    for i in range(1, 8):
        mi = m[i]
        if mi <= 0.0: continue
        c = i & F
        if c == 0: K += mi * s
        else: new[c] += mi * s
    den = 1.0 - K
    return new / den if den > 1e-12 else new

def fuse_ds(A, t0, nb, po, pb):
    """Evidential fusion (Dempster Shafer) on the frame {available, occupied, blocked}. Each observation is a simple support function whose weight of evidence is the
    calibrated, time discounted weight of S1; the occupancy and blocking priors of S1 enter as simple support functions; Dempster's rule combines them
    (conflict is normalised away); the pignistic probability of 'available' is returned."""
    _, _, Wt = _contrib(A, t0); k = A["kind"]; v = A["v"]
    F = np.where(k == K_CO, np.where(v == 2, 2, np.where(v == 0, 1, 4)), np.where(v == 1, 2, 5))
    order = np.argsort(A["s"], kind="stable"); s_ = A["s"][order]; F = F[order]; Wt = Wt[order]
    bounds = np.searchsorted(s_, np.arange(nb + 1)); p = np.zeros(nb)
    for b in range(nb):
        m = np.zeros(8); m[7] = 1.0
        m = _ds_add(m, 2 if po[b] > 0 else 5, abs(po[b])); m = _ds_add(m, 4 if pb[b] > 0 else 3, abs(pb[b]))
        for j in range(bounds[b], bounds[b + 1]): m = _ds_add(m, int(F[j]), float(Wt[j]))
        p[b] = float(sum(m[i] / _POP[i] for i in range(1, 8) if i & 1))
    return p

def _bay_obs(df_cols, t_arr, lo, hi):
    i0, i1 = np.searchsorted(t_arr, lo, "left"), np.searchsorted(t_arr, hi, "right"); sl = slice(i0, i1); bay = np.isin(df_cols["kind"][sl], BAYK)
    return dict(id=df_cols["id"][sl][bay], kind=df_cols["kind"][sl][bay], t=df_cols["t"][sl][bay], s=df_cols["s"][sl][bay], v=df_cols["v"][sl][bay], w=df_cols["w_s"][sl][bay])

def hmm_setup(w, df):
    """Bayesian filtering baseline. Transition probabilities between the three bay states (available, blocked, occupied) and the stationary distribution are estimated on the
    calibration day (day 0, whose truth is the audited day of the other methods). Observation likelihoods use the same calibrated source reliabilities as S1."""
    nb = w["L"].n_bays; T = w["T"]; st = np.where(w["occ"] == 1, 2, np.where(w["blk"] == 1, 1, 0))[:, :SLOTS_DAY]; C = np.full((3, 3), 0.5)
    for i in range(3):
        for j in range(3): C[i, j] += ((st[:, :-1] == i) & (st[:, 1:] == j)).sum()
    trans = C / C.sum(1, keepdims=True); pi = np.bincount(st.ravel(), minlength=3) + 1.0; pi = pi / pi.sum()
    d = df[np.isin(df.kind.values, BAYK)]; t_ = d.t.values; s_ = d.s.values; v = d.v.values.astype(int); e = 1.0 / (1.0 + np.exp(d.w_s.values / 100.0)); co = d.kind.values == K_CO
    LL = np.zeros((T, nb, 3))
    for state in range(3):
        lik = np.where(v == 1, np.where(state == 2, 1 - e, e), np.where(state == 2, e, 1 - e)); lik_co = np.where(v == state, 1 - e, e / 2.0)
        np.add.at(LL[:, :, state], (t_, s_), np.log(np.where(co, lik_co, lik)))
    return trans, pi, LL

def hmm_filter(hmm, t0, nb):
    trans, pi, LL = hmm; a = t0 - W + 1; alpha = np.tile(pi, (nb, 1))
    for t in range(a, t0 + 1):
        if t > a: alpha = alpha @ trans
        alpha = alpha * np.exp(LL[t] - LL[t].max(1, keepdims=True)); alpha /= alpha.sum(1, keepdims=True)
    return alpha[:, 0]

def run_strategies_ext(w, df, res, seed=5):
    """Same as analysis.run_strategies (S1 to S4, unchanged), plus S5 (Bayesian filter) and S6 (evidential fusion)."""
    L = w["L"]; nb = L.n_bays; po, pb = priors(w); rng = np.random.default_rng(seed); t_arr = df.t.values; hmm = hmm_setup(w, df)
    cols = {c: df[c].values for c in ("id", "kind", "t", "s", "v", "w_s", "w_a", "lab")}
    truth_state = np.where(w["occ"] == 1, 2, np.where(w["blk"] == 1, 1, 0)); avail = (truth_state == 0); out = []
    for rec in res:
        t0 = rec["t0"]; i1 = np.searchsorted(t_arr, t0, "right"); A = _bay_obs(cols, t_arr, t0 - STEP + 1, t0)
        q0 = np.searchsorted(t_arr, t0 - 2, "left"); qs = slice(q0, i1); qk = np.isin(cols["kind"][qs], (K_SQ, K_IQ))
        Q = dict(id=cols["id"][qs][qk], v=cols["v"][qs][qk], w=cols["w_s"][qs][qk]); R_s, R_a = rec["R_s"], rec["R_a"]; ds, da = rec["disp_s"], rec["disp_a"]
        ones = np.ones(len(A["id"]), bool); o = dict(t0=t0, avail=avail[:, t0].copy())
        def qfeat(removed):
            k = ~np.isin(Q["id"], list(removed)) if removed else np.ones(len(Q["id"]), bool)
            return float(np.average(Q["v"][k], weights=Q["w"][k])) if k.any() else np.nan
        so, sb, n = fuse(A, ones, t0, nb); p = pav(so, sb, po, pb); o["S1"] = dict(P1=p, P2=p, n=n, q=qfeat(set()), conf=0)
        m2 = ~np.isin(A["id"], list(R_s)); so, sb, n = fuse(A, m2, t0, nb); p = pav(so, sb, po, pb); o["S2"] = dict(P1=p, P2=p, n=n, q=qfeat(R_s), conf=0)
        for name, R, dsp in (("S3", R_s, ds), ("S4", R_a, da)):
            ret = ~np.isin(A["id"], list(R)); isd = np.isin(A["id"], list(dsp)); m1 = ret; m2b = (ret & ~isd) | (~ret & isd)
            so1, sb1, n1 = fuse(A, m1, t0, nb); so2, sb2, n2 = fuse(A, m2b, t0, nb); n = n1; conf = 0
            if name == "S4":
                P1 = pav(so1, sb1, po, pb); P2 = pav(so2, sb2, po, pb); disp_b = (P1 >= TAU) != (P2 >= TAU); qlen = qfeat(R)
                cand = [b for b in np.where(disp_b)[0] if L.queue_zone[L.zone_of[b]]] if (qlen == qlen and qlen >= 1) else []
                cand = sorted(cand, key=lambda b: -abs(P1[b] - P2[b]))[:NCONF]
                for b in cand:
                    r = truth_state[b, t0] if rng.random() < 0.97 else int(rng.choice([x for x in (0, 1, 2) if x != truth_state[b, t0]]))
                    for so_, sb_ in ((so1, sb1), (so2, sb2)):
                        if r == 0: so_[b] -= WCONF; sb_[b] -= WCONF
                        elif r == 1: sb_[b] += WCONF
                        else: so_[b] += WCONF
                    conf += 1
            o[name] = dict(P1=pav(so1, sb1, po, pb), P2=pav(so2, sb2, po, pb), n=n, q=qfeat(R), conf=conf)
        p = hmm_filter(hmm, t0, nb); o["S5"] = dict(P1=p, P2=p, n=np.bincount(A["s"], minlength=nb), q=qfeat(set()), conf=0)
        p = fuse_ds(A, t0, nb, po, pb); o["S6"] = dict(P1=p, P2=p, n=np.bincount(A["s"], minlength=nb), q=qfeat(set()), conf=0)
        o["nconf"] = o["S4"]["conf"]; out.append(o)
    return out, avail, truth_state

def risk_row(out):
    r = {}
    for S in STRATS:
        for tau in TAUS:
            tp = fp = fn = 0
            for o in out:
                rr = an.rec_set(o[S]["P1"], o[S]["P2"], o[S]["n"], tau, S in ("S3", "S4")); a = o["avail"]; tp += (rr & a).sum(); fp += (rr & ~a).sum(); fn += (~rr & a).sum()
            r[f"wrong_{S}_{tau}"] = fp / max(1, tp + fp); r[f"recall_{S}_{tau}"] = tp / max(1, tp + fn); r[f"recn_{S}_{tau}"] = int(tp + fp)
    return r

def ranking_row(out):
    """Threshold free measures per strategy: Brier score and AUROC of the probability that a bay is available, over all bay decisions."""
    from sklearn.metrics import roc_auc_score
    r = {}; A = np.concatenate([o["avail"] for o in out])
    for S in STRATS:
        P = np.concatenate([np.minimum(o[S]["P1"], o[S]["P2"]) if S in ("S3", "S4") else o[S]["P1"] for o in out])
        r[f"brier_{S}"] = float(((P - A) ** 2).mean()); r[f"bayauc_{S}"] = float(roc_auc_score(A, P))
    return r

def metrics(w, df, res, with_pred, seed=5):
    out, avail, ts = run_strategies_ext(w, df, res, seed=seed); row = risk_row(out); row.update(ranking_row(out))
    ev = an.summarise_evidence(an.evidence_metrics(df, res))["settled"]
    for k in ("loss_strict", "loss_aligned", "err_removed_strict", "err_removed_aligned", "apparent_removed_strict", "apparent_removed_aligned", "apparent_recall", "apparent_precision", "flagged_valid_share_strict", "flagged_valid_share_aligned"): row["ev_" + k] = ev[k]
    dm = an.dispute_metrics(out).set_index("strategy")
    for S in ("S3", "S4"): row[f"disp_{S}_share"] = float(dm.loc[S, "disputed_share"]); row[f"disp_{S}_repairwrong"] = float(dm.loc[S, "repair_would_be_wrong"]); row[f"disp_{S}_unknown"] = float(dm.loc[S, "unknown_share"])
    row["nconf_per_hour"] = sum(o["nconf"] for o in out) / (len(out) * STEP / 12)
    for k in ("s", "a"):
        row[f"lat_{k}"] = float(np.mean([r[f"t_res_{k}"] + r[f"t_disp_{k}"] for r in res])); row[f"rounds_{k}"] = float(np.mean([r[f"stats_{k}"]["rounds"] for r in res]))
        row[f"nonopt_tiles_{k}"] = int(sum(r[f"stats_{k}"].get("tiles_nonopt", 0) for r in res)); row[f"unverified_{k}"] = int(sum(not r[f"stats_{k}"].get("verified", True) for r in res))
        sizes = [1 + len(v[1]) for r in res for v in r[f"expl_{k}"].values()]; row[f"expl_cover_{k}"] = float(np.mean([s > 1 for s in sizes])) if sizes else np.nan
    row["n_decisions"] = len(res)
    if with_pred:
        from sklearn.metrics import roc_auc_score
        for h in (3, 6, 12):
            ys, ps = {}, {}
            for S in STRATS + ("ORACLE",):
                y, p, ok = an.predict(out, w, S if S != "ORACLE" else "S1", h, oracle=(S == "ORACLE")); ys[S] = y; ps[S] = p
            okm = ok.copy()
            for S in ps: okm &= ~np.isnan(ps[S])
            y = ys["S1"][okm]
            if len(np.unique(y)) < 2: continue
            for S in ps: row[f"auc_{S}_{h * 5}"] = float(roc_auc_score(y, ps[S][okm]))
            row[f"event_{h * 5}"] = int(y.sum())
    return row

def run_task(t):
    """One run: a world (seed, assumptions), the decisions, the strategies, the metrics. Returns a flat record."""
    t0 = time.time(); w, df, tab = setup_run(t["seed"], days=t.get("days", 8), fs=t.get("fs", 1.0), surge=tuple(t.get("surge", ())), assume=t.get("assume"), allow_scale=t.get("allow_scale", 1.0), drift=t.get("drift"))
    res = run_decisions(w, df, step_sub=t.get("step_sub", 1)); ndays = len({r["t0"] // SLOTS_DAY for r in res})
    row = metrics(w, df, res, with_pred=bool(t.get("pred")) and ndays >= 4); row.update(key=t["key"], stage=t["stage"], seed=t["seed"], run_seconds=time.time() - t0)
    for k, v in (t.get("levels") or {}).items(): row["lv_" + k] = v
    set_assumptions(); return row
