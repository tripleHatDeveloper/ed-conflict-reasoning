"""Scaling beyond 186 bays, global against local resolution under the same time budget.
Two ways of enlarging the department:
  bays  : every zone has `scale` times as many bays (tiles get larger)
  zones : the whole set of zones is copied `scale` times (more tiles of the base size, one shared waiting queue)
Global resolution: time limit = budget. Local resolution: the budget covers tiles and consistency check (fair share per tile, 20% reserve).
Everything is single threaded. Results are saved after each instance; a stopped run continues where it stopped."""
import argparse, os, sys, time, platform, datetime
ap = argparse.ArgumentParser(); ap.add_argument("--mode", default="bays"); ap.add_argument("--scales", default="12,24"); ap.add_argument("--margins", default="3,6"); ap.add_argument("--budget", type=float, default=1800.0)
ap.add_argument("--report-only", action="store_true"); args = ap.parse_args(); os.environ["ED_SCALE_MODE"] = args.mode
import numpy as np, pandas as pd, ortools
from sim import *; from resolve import *; from scalability import local_budget, global_run, make

def main():
    scales = [int(x) for x in args.scales.split(",")]; margins = [int(x) for x in args.margins.split(",")]; fn = f"large_{args.mode}.csv"; fa = f"large_{args.mode}_anytime.csv"
    res = pd.read_csv(fn).to_dict("records") if os.path.exists(fn) else []; any_ = pd.read_csv(fa) if os.path.exists(fa) else pd.DataFrame(columns=["scale", "aligned", "time", "weight"])
    def done(**kw): return any(all(r[k] == v for k, v in kw.items()) for r in res)
    def save(): pd.DataFrame(res).to_csv(fn, index=False); any_.to_csv(fa, index=False)
    if not args.report_only:
        for f in scales:
            w, df = make(f); L = w["L"]; T0 = SLOTS_DAY + 190; a = T0 - 23; sub = df[(df.t >= a) & (df.t <= T0)].reset_index(drop=True)
            print(f"[{datetime.datetime.now():%H:%M:%S}] mode {args.mode} scale {f}: {L.n_bays} bays, {len(L.zones)} zones, {len(sub):,} observations in the window", flush=True)
            for al in (False, True):
                key = dict(mode=args.mode, scale=f, bays=L.n_bays, aligned=al)
                if not done(method="global", **key):
                    g, rows = global_run(L, sub, a, T0, al, args.budget)
                    res.append(dict(method="global", margin=0, budget=args.budget, **key, status=g["status"], time=g["time"], t_tiles=0, t_check=0, cost=g["cost"], bound=g["bound"], tiles=0, tiles_nonopt=0, rounds=0, missed=0, verified=True))
                    any_ = pd.concat([any_, pd.DataFrame([dict(scale=f, aligned=al, time=t_, weight=int(round(c_))) for t_, c_ in rows])], ignore_index=True); save()
                    print(f"[{datetime.datetime.now():%H:%M:%S}]   {'aligned' if al else 'strict'} GLOBAL {g['status']} {g['time']:.0f}s weight {g['cost']:,} bound {g['bound']:,}", flush=True)
                for kt in margins:
                    if done(method="local", margin=kt, **key): continue
                    R, st = local_budget(L, sub, a, T0, al, kt=kt, budget=args.budget, max_rounds=400)
                    res.append(dict(method="local", margin=kt, budget=args.budget, **key, status="", time=st["time"], t_tiles=st["t_tiles"], t_check=st["time"] - st["t_tiles"], cost=int(cost_of(sub, R, al)), bound=0, tiles=st["tiles"], tiles_nonopt=st["tiles_nonopt"], rounds=st["rounds"], missed=st["missed"], verified=bool(st["verified"])))
                    save(); r = res[-1]
                    print(f"[{datetime.datetime.now():%H:%M:%S}]   {'aligned' if al else 'strict'} LOCAL margin {kt}: {r['time']:.0f}s (tiles {r['t_tiles']:.0f}s, check {r['t_check']:.0f}s) weight {r['cost']:,} tiles not optimal {r['tiles_nonopt']}/{r['tiles']} rounds {r['rounds']} verified {r['verified']}", flush=True)
    report()

def report():
    R = []
    for m in ("bays", "zones"):
        if os.path.exists(f"large_{m}.csv"): R.append(pd.read_csv(f"large_{m}.csv"))
    if not R: return
    R = pd.concat(R, ignore_index=True)
    def md(df):
        cols = [str(c) for c in df.columns]; out = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
        for _, r in df.iterrows(): out.append("| " + " | ".join(str(v) for v in r.values) + " |")
        return "\n".join(out)
    margins = sorted(R[R.method == "local"].margin.unique()); budget = R.budget.max()
    L_ = ["# Global against local resolution for 372 and 744 bays", "", 
          f"Generated {datetime.datetime.now():%Y-%m-%d %H:%M} on {platform.system()} {platform.release()} ({os.cpu_count()} cores), Python {platform.python_version()}, OR-Tools {ortools.__version__}.", "",
          f"Budget per instance: {budget:.0f} s for both approaches. Every solve is single threaded. Weights are the total weight of the removed observations (lower is better). The queue growth bound of the rules grows with the size of the department beyond six times the base department.", "",
          "Modes: bays = every zone has more bays (tiles get larger); zones = the set of zones is copied (more tiles of the base size, one shared waiting queue).", "",
          "Local status: complete = every tile problem solved to optimality and verified; limited = some tile problems stopped at their share of the budget, result verified; unverified = the consistency check did not finish, so the weight is not comparable. Time is split into tile problems and consistency check.", ""]
    for mode, title in (("bays", "More bays per zone"), ("zones", "More zones of the base size")):
        d = R[R["mode"] == mode]
        if d.empty: continue
        rows = []
        for (n, al), g in d.groupby(["bays", "aligned"]):
            gl = g[g.method == "global"]
            if gl.empty: continue
            gl = gl.iloc[0]; row = ["Aligned" if al else "Strict", n, f"{int(gl.cost):,} ({'optimal' if gl.status == 'OPTIMAL' else 'limit reached'}{', bound ' + format(int(gl.bound), ',') if gl.bound > 0 and gl.status != 'OPTIMAL' else ''}), {gl.time:.0f} s"]
            for kt in margins:
                l = g[(g.method == "local") & (g.margin == kt)]
                if l.empty: row.append("not run"); continue
                l = l.iloc[0]; st = "unverified" if not l.verified else ("complete" if l.tiles_nonopt == 0 else "limited")
                row.append(f"{int(l.cost):,} ({100 * (l.cost / gl.cost - 1):+.1f}%), {l.time:.0f} s (tiles {l.t_tiles:.0f}, check {l.t_check:.0f}), {l.tiles_nonopt}/{l.tiles} tiles not optimal, {l.rounds} rounds, {st}")
            rows.append(row)
        L_ += [f"## {title}", "", md(pd.DataFrame(rows, columns=["Interpretation", "Bays", "Global: weight, time"] + [f"Local margin {kt}: weight (gap to global), time, tiles, rounds, status" for kt in margins])), ""]
    open("scalability_large_report.md", "w", encoding="utf-8").write("\n".join(L_)); print("scalability_large_report.md written", flush=True)

if __name__ == "__main__": main()
