"""Space time local conflict resolution with global verification, plus dispute analysis.
Evidence is resolved inside small tiles (a zone with its waiting queue, over a time block with a margin).
A tile only uses constraints that the full model implies, so a tile is a relaxation: it can miss conflicts
that span several tiles. A single global consistency check finds what was missed and returns it as a new
conflict, so the verified result is consistent whatever the tile size."""
import time, numpy as np
from ortools.sat.python import cp_model
from sim import *; from model import build, solve

QKINDS = (K_SQ, K_IQ)

def est_tuples(df, wcol):
    return list(zip(df.id.values, df.kind.values, df.t.values, df.s.values, df.z.values, df.v.values, df[wcol].values.astype(int)))

def calibrate(df, rng, audit_frac=0.10):
    """Estimate source error rates from an audited sample of a calibration day, per kind and context.
    Strict interpretation counts timing, meaning and granularity differences as errors; aligned interpretation does not."""
    aud = df[rng.random(len(df)) < audit_frac]; tab = {}
    for (kind, ctx), g in aud.groupby(["kind", "ctx"]):
        n = len(g); es = (g.lab != LAB_OK).sum(); ea = (g.lab == LAB_ERR).sum()
        def w(e): 
            p = float(np.clip((e + 1) / (n + 2), 0.01, 0.49)); return int(round(100 * np.log((1 - p) / p)))
        tab[(kind, ctx)] = (w(es), w(ea))
    return tab

def apply_weights(df, tab):
    ws, wa = [], []
    for k, c in zip(df.kind.values, df.ctx.values):
        r = tab.get((k, c)) or tab.get((k, 0)) or tab.get((k, 1)); ws.append(r[0]); wa.append(r[1])
    df = df.copy(); df["w_s"] = ws; df["w_a"] = wa; return df

def _zones_for(L, z, ks):
    if not L.queue_zone[z]: return [z]
    order = L.zone_order; i = order.index(z)
    others = sorted([o for o in order if o != z], key=lambda o: abs(order.index(o) - i))[:ks]
    return [z] + others

def tiles(L, a, b, ks, kt, bt):
    for z in range(len(L.zones)):
        zl = _zones_for(L, z, ks); qz = L.queue_zone[z]
        full = qz and all(o in zl for o in L.zone_order)
        bays = [s for zz in zl for s in L.bays_of[zz]]
        for c0 in range(a, b + 1, bt):
            c1 = min(b, c0 + bt - 1)
            yield z, zl, qz, full, bays, c0, c1, max(a, c0 - kt), min(b, c1 + kt)

def tile_est(L, df, z, zl, qz, c0, c1, e0, e1):
    t = df.t.values; ts = (t >= e0) & (t <= e1)
    zone_s = np.where(df.s.values >= 0, L.zone_of[np.clip(df.s.values, 0, None)], -9)
    kind = df.kind.values
    inzone = np.isin(zone_s, zl); qk = np.isin(kind, QKINDS) & qz; cen = (kind == K_CEN) & np.isin(df.z.values, zl)
    sel = ts & (inzone | qk | cen)
    sub = df[sel]
    zs = np.where(sub.s.values >= 0, L.zone_of[np.clip(sub.s.values, 0, None)], -9)
    owned = (sub.t.values >= c0) & (sub.t.values <= c1) & (
        (zs == z) | ((sub.kind.values == K_CEN) & (sub.z.values == z)) |
        (np.isin(sub.kind.values, QKINDS) & (z == L.zone_order[0])))
    return sub, set(sub.id.values[owned])

def resolve_local(L, df, a, b, aligned, ks=0, kt=3, bt=6, tmax=5.0, verify=True, max_rounds=40):
    wcol = "w_a" if aligned else "w_s"; stats = dict(tiles=0, rounds=0, missed=0, local_removed=0, t_local=0, t_verify=0, tiles_nonopt=0)
    R = set(); t0 = time.time()
    for z, zl, qz, full, bays, c0, c1, e0, e1 in tiles(L, a, b, ks, kt, bt):
        sub, owned = tile_est(L, df, z, zl, qz, c0, c1, e0, e1)
        if not len(sub): continue
        bd = build(L, bays, e0, e1, est_tuples(sub, wcol), aligned, queue=qz, queue_full=full)
        st, rem, cost, _ = solve(bd, tmax)
        stats["tiles"] += 1
        if st != cp_model.OPTIMAL: stats["tiles_nonopt"] += 1
        if rem: R |= (rem & owned)
    stats["t_local"] = time.time() - t0; stats["local_removed"] = len(R)
    if verify:
        t1 = time.time(); R, vs = verify_loop(L, df, a, b, aligned, R, wcol); stats.update(vs); stats["t_verify"] = time.time() - t1
    return R, stats

def verify_loop(L, df, a, b, aligned, R, wcol, max_rounds=40, tmax=20.0):
    conflicts = []; rounds = 0; allbays = list(range(L.n_bays))
    w = dict(zip(df.id.values, df[wcol].values.astype(int)))
    while rounds < max_rounds:
        rounds += 1
        sub = df[~df.id.isin(R)]
        bd = build(L, allbays, a, b, est_tuples(sub, wcol), aligned, queue=True, queue_full=True, objective=False, assume=True)
        sv = cp_model.CpSolver(); sv.parameters.num_workers = 1; sv.parameters.max_time_in_seconds = tmax
        st = sv.Solve(bd["m"])
        if st in (cp_model.OPTIMAL, cp_model.FEASIBLE): return R, dict(rounds=rounds, missed=len(conflicts), verified=True)
        if st != cp_model.INFEASIBLE: return R, dict(rounds=rounds, missed=len(conflicts), verified=False)
        idx2e = {k.Index(): e for e, k in bd["keep"].items()}
        core = [idx2e[i] for i in sv.SufficientAssumptionsForInfeasibility() if i in idx2e]
        conflicts.append(core)
        R |= hitting_set(conflicts, w, R)
    return R, dict(rounds=rounds, missed=len(conflicts), verified=False)

def hitting_set(conflicts, w, R):
    m = cp_model.CpModel(); ids = sorted({e for c in conflicts for e in c}); r = {e: m.NewBoolVar("") for e in ids}
    for c in conflicts:
        if not any(e in R for e in c): m.AddBoolOr([r[e] for e in c])
    m.Minimize(sum(w[e] * r[e] for e in ids)); sv = cp_model.CpSolver(); sv.Solve(m)
    return {e for e in ids if sv.Value(r[e])}

def resolve_global(L, df, a, b, aligned, tmax=100.0):
    wcol = "w_a" if aligned else "w_s"
    bd = build(L, list(range(L.n_bays)), a, b, est_tuples(df, wcol), aligned, queue=True, queue_full=True)
    st, rem, cost, sv = solve(bd, tmax)
    return (rem or set()), cost, sv.StatusName(st)

def cost_of(df, R, aligned):
    wcol = "w_a" if aligned else "w_s"; return int(df[df.id.isin(R)][wcol].sum())

def dispute_analysis(L, df, a, b, aligned, R, targets, theta=150, tmax=3.0, kt=6):
    """For each removed target, ask how much extra weight must be given up to keep it. A small loss means the
    data do not decide between the alternatives: the estimate, and the evidence it displaces, are disputed."""
    wcol = "w_a" if aligned else "w_s"; disputed = set(); expl = {}; cache = {}
    ids = df.set_index("id")
    for e in targets:
        r = ids.loc[e]
        if r.kind in QKINDS: z = L.zone_order[0]
        elif r.kind == K_CEN: z = int(r.z)
        else: z = int(L.zone_of[int(r.s)])
        zl = [z]; qz = L.queue_zone[z]; e0 = max(a, int(r.t) - kt); e1 = min(b, int(r.t) + kt)
        sub, _ = tile_est(L, df, z, zl, qz, e0, e1, e0, e1)
        tup = est_tuples(sub, wcol); bays = list(L.bays_of[z])
        key = (z, e0, e1)
        if key not in cache:
            bd = build(L, bays, e0, e1, tup, aligned, queue=qz, queue_full=False)
            _, rem0, c0, _ = solve(bd, tmax); cache[key] = (rem0 or set(), c0 if c0 is not None else 0)
        rem0, c0 = cache[key]
        bd = build(L, bays, e0, e1, tup, aligned, queue=qz, queue_full=False)
        if e not in bd["keep"]: continue
        _, rem1, c1, _ = solve(bd, tmax, force_keep=[e])
        if rem1 is None: continue
        loss = c1 - c0
        if loss <= theta:
            partners = rem1 - rem0
            disputed |= {e} | partners; expl[e] = (loss, sorted(partners))
        else:
            expl[e] = (loss, sorted(rem1 - rem0))
    return disputed, expl
