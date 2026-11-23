"""Simulation of the ED (5 minute slots) and error models of the data sources.
All patient related quantities are synthetic."""
import numpy as np, pandas as pd
from layout import make_layout, ZONES

K_SOC, K_IT, K_CO, K_SQ, K_IQ, K_CEN = range(6)   # evidence kinds
LAB_OK, LAB_TIME, LAB_SEM, LAB_GRAN, LAB_ERR = range(5)
SLOTS_DAY = 288

# Assumptions of the synthetic world, as multipliers of their default values (1.0 gives the default simulator)
ASSUME = dict(arrival=1.0, los=1.0, closure=1.0, drop=1.0, itmiss=1.0, coord_rate=1.0, copy=1.0, delay=1.0)

# Arrival profile: share of arrivals by hour, and weekday factors (Monday first)
HSH = np.array([.012,.009,.010,.009,.007,.020,.032,.056,.075,.084,.089,.070,.072,.071,.061,.057,.061,.044,.045,.032,.032,.019,.022,.011]); HSH = HSH / HSH.sum()
WK = np.array([1.13, 1.14, 0.95, 0.98, 0.90, 0.92, 0.97])
DAILY = 76.0

def simulate(days, scale=1, seed=0, lam0=None, surge_days=(), board_p=0.3):
    rng = np.random.default_rng(seed)
    L = make_layout(scale)
    T = days * SLOTS_DAY
    occ = np.zeros((L.n_bays, T), np.int8); blk = np.zeros((L.n_bays, T), np.int8)
    q = np.zeros(T, np.int32); corr = np.zeros((len(L.zones), T), np.int32)
    stays = []                                   # (bay, start, end_exclusive)
    free_at = np.zeros(L.n_bays, int)            # earliest slot the bay can be occupied
    end_at = np.full(L.n_bays, -1); blk_until = np.zeros(L.n_bays, int)
    queue = []                                   # (entry_slot, zone)
    resus_wait = 0
    def stay(zi):
        d = int(rng.lognormal(np.log(39 * ASSUME["los"]), 0.4))
        if rng.random() < L.params[zi]['admit']: d += int(min(150, rng.lognormal(np.log(11.6), 1.18)))
        return max(3, d)
    qz_list = L.zone_order; cum = np.cumsum([L.params[z]['share'] for z in qz_list])
    resus_bays = [s for z in range(len(L.zones)) if not L.queue_zone[z] for s in L.bays_of[z]]
    for t in range(1, T):
        # departures and cleaning
        for s in np.where(end_at == t)[0]:
            end_at[s] = -1; blk_until[s] = t + 2
        # closures start on free unblocked bays
        free_now = (occ[:, t-1] == 0) & (end_at < 0) & (blk_until <= t)
        starts = free_now & (rng.random(L.n_bays) < 0.0015 * ASSUME["closure"])
        for s in np.where(starts)[0]:
            blk_until[s] = t + int(rng.integers(6, 31))
        # arrivals at slot t (enter the queue at t+1 to guarantee at least one slot of waiting)
        day = t // SLOTS_DAY
        mult = (1.6 if day in surge_days else 1.0) * scale
        n_arr = min(int(rng.poisson(DAILY * ASSUME["arrival"] * mult * HSH[(t % SLOTS_DAY) // 12] * WK[day % 7] / 12.0)), 8 * scale)
        for _ in range(n_arr):
            u = rng.random()
            if u < 0.02: resus_wait += 1
            else:
                z = qz_list[int(np.searchsorted(cum, rng.random() * cum[-1]))]
                queue.append((t + 1, z))
        # assignment (patients with entry <= t-1 are eligible at t, i.e. were in the queue at t-1)
        elig = [p for p in queue if p[0] <= t - 1]
        keep_q = [p for p in queue if p[0] > t - 1]
        left = []
        for zi in qz_list:
            zq = [p for p in elig if p[1] == zi]
            for s in L.bays_of[zi]:
                if not zq: break
                if occ[s, t-1] == 0 and end_at[s] < 0 and blk_until[s] <= t:
                    p = zq.pop(0); d = stay(zi)
                    end_at[s] = t + d; free_at[s] = t; stays.append((s, t, t + d)); occ[s, t:t+d] = 1
            left += zq
        # resus direct
        for s in resus_bays:
            if resus_wait <= 0: break
            if occ[s, t-1] == 0 and end_at[s] < 0 and blk_until[s] <= t:
                d = stay(0)
                end_at[s] = t + d; stays.append((s, t, t + d)); occ[s, t:t+d] = 1; resus_wait -= 1
        queue = keep_q + left
        # walkouts without being seen (at most 2 per slot)
        for _ in range(min(2, len(queue))):
            if rng.random() < 0.01: queue.pop(int(rng.integers(len(queue))))
        q[t] = len(queue)
        blk[:, t] = ((blk_until > t) & (occ[:, t] == 0)).astype(np.int8)
        zocc = np.array([occ[L.bays_of[z], t].mean() for z in range(len(L.zones))])
        corr[:, t] = np.minimum(4, rng.poisson(0.3 + 1.2 * zocc))
    # cleaning slots after a stay: make sure blk marks them (blk_until set at departure)
    return dict(L=L, T=T, occ=occ, blk=blk, q=q, corr=corr, stays=stays, days=days, seed=seed)

def _crowd(world):
    L, occ = world["L"], world["occ"]
    zocc = np.zeros((len(L.zones), world["T"]))
    for z in range(len(L.zones)): zocc[z] = occ[L.bays_of[z]].mean(0)
    return zocc

def gen_sources(world, seed=1, drift_zone=3, drift_per_day=0.012, fs=1.0):
    """Return a DataFrame of estimates with fault labels. Error models are deliberately richer than
    the independence assumptions used later for the fusion weights (bursts, drift, copying)."""
    rng = np.random.default_rng(seed)
    L, occ, blk, q, corr, T = world["L"], world["occ"], world["blk"], world["q"], world["corr"], world["T"]
    zocc = _crowd(world); rows = []
    nb = L.n_bays; zone_of = L.zone_of
    # --- bay presence sensors -------------------------------------------------------------
    for s in range(nb):
        z = zone_of[s]; stuck_left = 0; held = 0
        u = rng.random(T); u2 = rng.random(T); ustk = rng.random(T)
        for t in range(1, T):
            day = t // SLOTS_DAY
            if stuck_left == 0 and ustk[t] < 0.001 * fs: stuck_left = int(rng.integers(6, 21)); held = occ[s, t-1]
            if u[t] < 0.03 * ASSUME["drop"]: 
                if stuck_left: stuck_left -= 1
                continue                              # dropout
            if stuck_left > 0:
                v = held; stuck_left -= 1
            else:
                fp = fs * (0.01 + 0.04 * zocc[z, t]) + (drift_per_day * day if z % len(ZONES) == drift_zone else 0)
                v = occ[s, t]
                if v == 1 and u2[t] < 0.02 * fs: v = 0
                elif v == 0 and u2[t] < fp: v = 1
            ctx = int(zocc[z, t] >= 0.8)
            rows.append((K_SOC, t, s, -1, v, LAB_OK if v == occ[s, t] else LAB_ERR, ctx, 0))
    # --- IT tracking system: reservation lead, late discharge update, mis entry, stale copy forward
    it = np.zeros((nb, T), np.int8); cause = np.zeros((nb, T), np.int8)   # 0 none, 1 lag, 2 reservation, 3 err
    for (s, a, b) in world["stays"]:
        rho = int(rng.choice([0, 1, 2], p=[.5, .3, .2])); dl = int(rng.choice([0, 1, 2, 3], p=[.3, .3, .25, .15]))
        rho = int(round(rho * ASSUME["delay"])); dl = int(round(dl * ASSUME["delay"]))
        bay = s; err = False
        if rng.random() < 0.015 * fs and L.neighbours[s]:
            bay = int(rng.choice(L.neighbours[s])); err = True
        if rng.random() < 0.01 * fs: dl += int(rng.integers(8, 21)); err = True
        lo, hi = max(1, a - rho), min(T, b + dl)
        it[bay, lo:hi] = 1
        if err: cause[bay, lo:hi] = 3
        else:
            cause[bay, lo:a] = np.where(cause[bay, lo:a] == 3, 3, 2); cause[bay, b:hi] = np.where(cause[bay, b:hi] == 3, 3, 1)
    chg = np.zeros((nb, T), bool)                     # observable context: the IT record changed in the last 4 slots
    for k in range(1, 5): chg[:, k:] |= (it[:, k:] != it[:, :-k])
    for s in range(nb):
        z = zone_of[s]; um = rng.random(T)
        for t in range(1, T):
            if um[t] < 0.10 * ASSUME["itmiss"]: continue
            v = int(it[s, t]); ctx = int(chg[s, t])
            if v == occ[s, t]: lab = LAB_OK
            else:
                c = cause[s, t]; lab = LAB_SEM if (c == 2 and v == 1) else (LAB_TIME if (c == 1 and v == 1) else LAB_ERR)
            rows.append((K_IT, t, s, -1, v, lab, ctx, 0))
    # --- coordinator reports (sparse, delayed, partly copied from IT) --------------------------
    state = np.where(occ == 1, 2, np.where(blk == 1, 1, 0))     # 0 available, 1 closed, 2 occupied
    for t in range(2, T):
        pick = np.where(rng.random(nb) < 0.06 * ASSUME["coord_rate"])[0]
        for s in pick:
            d = int(rng.integers(0, 3)); truth_d = state[s, t-d]; dep = 0
            if rng.random() < min(0.5, 0.25 * fs * ASSUME["copy"]):                    # copies the IT view (source dependence)
                v = 2 if it[s, t] == 1 else 0; dep = 1
            elif rng.random() < 0.04 * fs: v = int(rng.choice([x for x in (0, 1, 2) if x != truth_d]))
            else: v = int(truth_d)
            lab = LAB_OK if v == state[s, t] else (LAB_TIME if v == truth_d and d > 0 else LAB_ERR)
            rows.append((K_CO, t, s, -1, v, lab, 0, dep))
    # --- queue counts ------------------------------------------------------------------------
    for t in range(2, T):
        n = max(0, int(q[t] + np.rint(rng.normal(0, 1.0)))); lab = LAB_OK if abs(n - q[t]) <= 1 else LAB_ERR
        rows.append((K_SQ, t, -1, -1, n, lab, 0, 0))
        lag = int(rng.integers(0, 3)); n2 = max(0, int(q[t - lag] + rng.poisson(0.7)))
        lab = LAB_OK if abs(n2 - q[t]) <= 1 else (LAB_TIME if abs(n2 - q[t - lag]) <= 1 else LAB_ERR)
        rows.append((K_IQ, t, -1, -1, n2, lab, 0, 0))
    # --- IT zone census every 3 slots (includes corridor patients) ---------------------------
    for t in range(3, T, 3):
        for z in range(len(L.zones)):
            base = int(occ[L.bays_of[z], t - 1].sum()); n = base + int(corr[z, t]) + (int(rng.choice([-1, 1])) if rng.random() < 0.1 else 0)
            truth = int(occ[L.bays_of[z], t].sum())
            lab = LAB_OK if abs(n - truth) <= 1 else (LAB_GRAN if n - truth >= 2 and corr[z, t] >= 2 else LAB_ERR)
            rows.append((K_CEN, t, -1, z, max(0, n), lab, 0, 0))
    df = pd.DataFrame(rows, columns=["kind", "t", "s", "z", "v", "lab", "ctx", "dep"])
    df = df.sort_values(["t", "kind"]).reset_index(drop=True); df["id"] = np.arange(len(df))
    return df
