"""Global against local resolution under the same time budget per instance (no fixed limit per tile).
Local resolution: the budget covers the tile problems and the consistency check. Each tile problem receives a fair share of the time
that remains (unused time passes to later tiles), and 20% of the budget is kept for the consistency check.
Global resolution: the budget is the time limit of the whole solve.
Every solve is single threaded. Results are saved after each instance, and a stopped run continues where it stopped."""
import argparse, os, sys, time, platform, datetime
import numpy as np, pandas as pd, ortools
from ortools.sat.python import cp_model
from sim import *; from resolve import *; from model import build, solve

def make(scale, seed=21):
    """World of `scale` times the bays of the base department (two days: a calibration day and a surge day) with calibrated weights."""
    w = simulate(2, scale=scale, seed=seed, surge_days=(1,)); df = gen_sources(w, seed=seed + 1)
    tab = calibrate(df[df.t < SLOTS_DAY], np.random.default_rng(seed + 2)); return w, apply_weights(df, tab)

# ------------------------------------------------------------------ local resolution under a budget
def verify_budget(L, df, a, b, aligned, R, wcol, deadline, max_rounds=40):
    conflicts = []; rounds = 0; allbays = list(range(L.n_bays)); w = dict(zip(df.id.values, df[wcol].values.astype(int)))
    while rounds < max_rounds:
        remaining = deadline - time.time()
        if remaining <= 0: return R, dict(rounds=rounds, missed=len(conflicts), verified=False, reason="budget")
        rounds += 1; sub = df[~df.id.isin(R)]
        bd = build(L, allbays, a, b, est_tuples(sub, wcol), aligned, queue=True, queue_full=True, objective=False, assume=True)
        sv = cp_model.CpSolver(); sv.parameters.num_workers = 1; sv.parameters.max_time_in_seconds = max(1.0, deadline - time.time()); st = sv.Solve(bd["m"])
        if st in (cp_model.OPTIMAL, cp_model.FEASIBLE): return R, dict(rounds=rounds, missed=len(conflicts), verified=True, reason="")
        if st != cp_model.INFEASIBLE: return R, dict(rounds=rounds, missed=len(conflicts), verified=False, reason="budget")
        idx2e = {k.Index(): e for e, k in bd["keep"].items()}; conflicts.append([idx2e[i] for i in sv.SufficientAssumptionsForInfeasibility() if i in idx2e]); R |= hitting_set(conflicts, w, R)
    return R, dict(rounds=rounds, missed=len(conflicts), verified=False, reason="rounds")

def local_budget(L, df, a, b, aligned, kt, bt=6, ks=0, budget=1800.0, reserve=0.20, max_rounds=40):
    wcol = "w_a" if aligned else "w_s"; t_start = time.time(); deadline = t_start + budget; tile_deadline = t_start + budget * (1 - reserve); items = []
    for z, zl, qz, full, bays, c0, c1, e0, e1 in tiles(L, a, b, ks, kt, bt):
        sub, owned = tile_est(L, df, z, zl, qz, c0, c1, e0, e1)
        if len(sub): items.append((qz, full, bays, e0, e1, sub, owned))
    n = len(items); R = set(); nonopt = 0
    for i, (qz, full, bays, e0, e1, sub, owned) in enumerate(items):
        share = max(0.5, (tile_deadline - time.time()) / (n - i))
        bd = build(L, bays, e0, e1, est_tuples(sub, wcol), aligned, queue=qz, queue_full=full); st, rem, cost, _ = solve(bd, share)
        if st != cp_model.OPTIMAL: nonopt += 1
        if rem: R |= (rem & owned)
    t_tiles = time.time() - t_start; R, vs = verify_budget(L, df, a, b, aligned, R, wcol, deadline, max_rounds=max_rounds)
    return R, dict(tiles=n, tiles_nonopt=nonopt, t_tiles=t_tiles, time=time.time() - t_start, **vs)

# ------------------------------------------------------------------ global resolution with a time limit
class Log(cp_model.CpSolverSolutionCallback):
    def __init__(self, total): super().__init__(); self.total = total; self.rows = []
    def on_solution_callback(self): self.rows.append((self.WallTime(), self.total - self.ObjectiveValue()))

def global_run(L, sub, a, b, aligned, budget):
    wcol = "w_a" if aligned else "w_s"; bd = build(L, list(range(L.n_bays)), a, b, est_tuples(sub, wcol), aligned, queue=True, queue_full=True)
    total = sum(bd["weights"].values()); sv = cp_model.CpSolver(); sv.parameters.max_time_in_seconds = budget; sv.parameters.num_workers = 1; cb = Log(total)
    t1 = time.time(); st = sv.Solve(bd["m"], cb); dt = time.time() - t1
    return dict(status=sv.StatusName(st), time=dt, cost=int(round(total - sv.ObjectiveValue())), bound=int(np.ceil(total - sv.BestObjectiveBound() - 1e-6))), cb.rows

# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--scales", default="1,2,4,6"); ap.add_argument("--margins", default="3,6,9"); ap.add_argument("--budgets", default="1800")
    ap.add_argument("--windows-small", type=int, default=1); ap.add_argument("--windows-large", type=int, default=1); ap.add_argument("--report-only", action="store_true"); args = ap.parse_args()
    scales = [int(x) for x in args.scales.split(",")]; margins = [int(x) for x in args.margins.split(",")]; budgets = sorted(float(x) for x in args.budgets.split(",")); bmax = budgets[-1]
    res = pd.read_csv("scalability_results.csv").to_dict("records") if os.path.exists("scalability_results.csv") else []
    any_ = pd.read_csv("scalability_anytime.csv") if os.path.exists("scalability_anytime.csv") else pd.DataFrame(columns=["bays", "aligned", "t0", "time", "weight"])
    def find(**kw):
        for r in res:
            if all(r[k] == v for k, v in kw.items()): return r
    def save(): pd.DataFrame(res).to_csv("scalability_results.csv", index=False); any_.to_csv("scalability_anytime.csv", index=False)
    t_session = time.time(); WIN = [190, 150, 230]
    if not args.report_only:
        for f in scales:
            w, df = make(f); L = w["L"]; nwin = args.windows_small if f <= 2 else args.windows_large
            for off in WIN[:nwin]:
                T0 = SLOTS_DAY + off; a = T0 - 23; sub = df[(df.t >= a) & (df.t <= T0)].reset_index(drop=True)
                for al in (False, True):
                    key = dict(bays=L.n_bays, aligned=al, t0=T0)
                    if find(method="global", **key) is None:
                        g, rows = global_run(L, sub, a, T0, al, bmax); res.append(dict(method="global", margin=0, budget=bmax, **key, **g, complete=g["status"] == "OPTIMAL", verified=True, tiles=0, tiles_nonopt=0, rounds=0, missed=0, reused=False))
                        any_ = pd.concat([any_, pd.DataFrame([dict(bays=L.n_bays, aligned=al, t0=T0, time=t_, weight=int(round(c_))) for t_, c_ in rows])], ignore_index=True); save()
                        print(f"[{datetime.datetime.now():%H:%M:%S}] bays {L.n_bays} {'aligned' if al else 'strict'} window {off} GLOBAL {g['status']} {g['time']:.0f}s weight {g['cost']} bound {g['bound']}", flush=True)
                    for kt in margins:
                        prev = None
                        for B in budgets:
                            if find(method="local", margin=kt, budget=B, **key) is not None: prev = find(method="local", margin=kt, budget=B, **key); continue
                            if prev is not None and prev["complete"]:
                                res.append(dict(prev, budget=B, reused=True)); save(); prev = res[-1]; continue
                            R, st = local_budget(L, sub, a, T0, al, kt=kt, budget=B)
                            row = dict(method="local", margin=kt, budget=B, **key, status="", time=st["time"], cost=int(cost_of(sub, R, al)), bound=0, complete=bool(st["tiles_nonopt"] == 0 and st["verified"]), verified=bool(st["verified"]),
                                       tiles=st["tiles"], tiles_nonopt=st["tiles_nonopt"], rounds=st["rounds"], missed=st["missed"], reused=False); res.append(row); save(); prev = row
                            print(f"[{datetime.datetime.now():%H:%M:%S}] bays {L.n_bays} {'aligned' if al else 'strict'} window {off} LOCAL margin {kt} budget {B:.0f}s -> {row['time']:.0f}s weight {row['cost']} tiles not optimal {row['tiles_nonopt']}/{row['tiles']} verified {row['verified']}", flush=True)
    report(res, any_, budgets, margins, t_session)

# ------------------------------------------------------------------ report
def report(res, any_, budgets, margins, t_session):
    R = pd.DataFrame(res); T_MAIN = SLOTS_DAY + 190
    def md(df):
        cols = [str(c) for c in df.columns]; out = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
        for _, r in df.iterrows(): out.append("| " + " | ".join(str(v) for v in r.values) + " |")
        return "\n".join(out)
    def ip(al): return "Aligned" if al else "Strict"
    def glob_at(bays, al, t0, B):
        g = any_[(any_.bays == bays) & (any_.aligned == al) & (any_.t0 == t0) & (any_.time <= B)]; return None if g.empty else int(g.weight.min())
    def loc(bays, al, t0, kt, B):
        r = R[(R.method == "local") & (R.bays == bays) & (R.aligned == al) & (R.t0 == t0) & (R.margin == kt) & (R.budget == B)]; return None if r.empty else r.iloc[0]
    def flag(r): return "unverified" if not r.verified else ("complete" if r.complete else "limited")
    def best_local(bays, al, t0, B):
        c = [(loc(bays, al, t0, kt, B), kt) for kt in margins]; c = [(r, kt) for r, kt in c if r is not None and r.verified]; return min(c, key=lambda x: x[0].cost) if c else (None, None)
    def verdict(gw, lw):
        if gw is None or lw is None: return "n/a"
        return "local" if lw < gw else ("global" if gw < lw else "tie")
    sizes = sorted(R.bays.unique()); L_ = ["# Global against local resolution under a common time budget", "", 
         f"Generated {datetime.datetime.now():%Y-%m-%d %H:%M} on {platform.system()} {platform.release()} ({os.cpu_count()} cores), Python {platform.python_version()}, OR-Tools {ortools.__version__}. Time of this session {(time.time() - t_session) / 3600:.1f} h.", "",
         f"Budgets: {', '.join(f'{b:.0f}' for b in budgets)} s per instance. Local margins: {', '.join(map(str, margins))} slots, no neighbouring zones. Every solve is single threaded. Weights are the total weight of the removed observations (lower is better).", "",
         "How to read: for local resolution the budget covers the tile problems and the consistency check, with no fixed limit per tile (each tile receives a fair share of the remaining time, and 20% of the budget is kept for the consistency check). "
         "Status of a local result: complete = every tile problem solved to optimality and the result verified; limited = some tile problems stopped at their share of the budget (result verified); unverified = the consistency check did not finish within the budget, so the weight is not comparable and is excluded from the comparisons. "
         "Global weights at smaller budgets are the best solution found by that time in the long global run. A reused result means that a smaller budget already solved every tile to optimality, so the same result holds for larger budgets.", ""]
    # table 1 main comparison at the largest budget
    B = budgets[-1]; rows = []
    for al in (False, True):
        for n in sizes:
            g = R[(R.method == "global") & (R.bays == n) & (R.aligned == al) & (R.t0 == T_MAIN)]
            if g.empty: continue
            g = g.iloc[0]; row = [ip(al), n, ("optimal" if g.status == "OPTIMAL" else "limit reached"), f"{g.time:.0f}", f"{int(g.cost):,}", (f"{int(g.bound):,}" if g.bound > 0 else "n/a")]
            for kt in margins:
                r = loc(n, al, T_MAIN, kt, B)
                row.append("not run" if r is None else f"{int(r.cost):,} ({100 * (r.cost / g.cost - 1):+.1f}%), {r.time:.0f} s, {flag(r)}")
            rows.append(row)
    L_ += [f"## 1. Main window, budget {B:.0f} s", "", md(pd.DataFrame(rows, columns=["Interpretation", "Bays", "Global end", "Global time (s)", "Global weight", "Global lower bound"] + [f"Local margin {kt}: weight (gap to global), time, status" for kt in margins])), ""]
    # table 2 budgets
    rows = []
    for al in (False, True):
        for n in sizes:
            for B2 in budgets:
                gw = glob_at(n, al, T_MAIN, B2); cells = []; valid = []
                for kt in margins:
                    r = loc(n, al, T_MAIN, kt, B2)
                    if r is None: cells.append("not run")
                    else: cells.append(f"{int(r.cost):,}" + ("" if r.verified else " (unverified)")); (valid.append((r.cost, f"margin {kt}")) if r.verified else None)
                bl = min(valid) if valid else None
                rows.append([ip(al), n, f"{B2:.0f}", ("none" if gw is None else f"{gw:,}")] + cells + [("n/a" if bl is None else f"{verdict(gw, bl[0])} ({bl[1]})")])
    L_ += ["## 2. Weight reached within each budget (main window)", "", "Best = the lower of the global weight and the best verified local weight.", "", md(pd.DataFrame(rows, columns=["Interpretation", "Bays", "Budget (s)", "Global"] + [f"Local margin {kt}" for kt in margins] + ["Better approach"])), ""]
    # table 3 other windows
    rows = []
    for al in (False, True):
        for n in sizes:
            for t0 in sorted(R[(R.bays == n)].t0.unique()):
                g = R[(R.method == "global") & (R.bays == n) & (R.aligned == al) & (R.t0 == t0)]
                if g.empty: continue
                g = g.iloc[0]; bl, kt = best_local(n, al, t0, budgets[-1])
                rows.append([ip(al), n, f"{t0 - SLOTS_DAY}", f"{int(g.cost):,} ({'optimal' if g.status == 'OPTIMAL' else 'limit reached'})", ("n/a" if bl is None else f"{int(bl.cost):,} (margin {kt}, {bl.time:.0f} s)"), verdict(int(g.cost), None if bl is None else bl.cost)])
    L_ += [f"## 3. All windows, budget {budgets[-1]:.0f} s", "", "Window = start offset of the two hour window on the surge day (190 is the default window). Local = best verified margin.", "", md(pd.DataFrame(rows, columns=["Interpretation", "Bays", "Window", "Global weight", "Best local weight", "Better approach"])), ""]
    # summary
    summ = []
    for al in (False, True):
        for B2 in budgets:
            for label, kts in (("any margin", margins), ("margin 3", [3])):
                wins, ties, losses = [], [], []
                for n in sizes:
                    gw = glob_at(n, al, T_MAIN, B2)
                    lw = [loc(n, al, T_MAIN, kt, B2) for kt in kts]; lw = [r.cost for r in lw if r is not None and r.verified]
                    v = verdict(gw, min(lw) if lw else None)
                    (wins if v == "local" else ties if v == "tie" else losses if v == "global" else []).append(n)
                summ.append([ip(al), f"{B2:.0f}", label, ", ".join(map(str, wins)) or "none", ", ".join(map(str, ties)) or "none", ", ".join(map(str, losses)) or "none"])
    L_ += ["## 4. Summary: sizes at which each approach is better (main window)", "", md(pd.DataFrame(summ, columns=["Interpretation", "Budget (s)", "Local setting", "Local better at (bays)", "Tie at (bays)", "Global better at (bays)"])), ""]
    open("scalability_report.md", "w", encoding="utf-8").write("\n".join(L_)); print("scalability_report.md written", flush=True)

if __name__ == "__main__": main()
