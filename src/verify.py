"""Re-runs one simulated week and one comparison of global and local resolution, and compares them with the stored results."""
import os, sys, time
import numpy as np, pandas as pd

def close(a, b, tol=1e-9): return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(b)))

def check_week(results):
    import evaluation as R
    stored = pd.read_csv(os.path.join(results, "runs_main.csv")); seed = int(stored.seed.min()); ref = stored[stored.seed == seed].iloc[0]
    print(f"Re-running the simulated week with seed {seed} (333 decision points) ...", flush=True); t = time.time()
    row = R.run_task(dict(key="verify", stage="verify", seed=seed, days=8, pred=True, step_sub=1))
    cols = [c for c in ref.index if c.startswith(("wrong_", "recall_", "recn_", "brier_", "bayauc_", "auc_", "ev_", "disp_")) or c in ("nconf_per_hour", "n_decisions", "nonopt_tiles_s", "nonopt_tiles_a", "unverified_s", "unverified_a")]
    bad = [c for c in cols if c in row and not close(row[c], ref[c])]
    print(f"  {len(cols) - len(bad)} of {len(cols)} values equal the stored ones ({time.time() - t:.0f} s)")
    for c in bad[:10]: print(f"  differs: {c}: {row[c]} against {ref[c]}")
    return not bad

def check_small_comparison(results):
    import scalability as S
    from sim import SLOTS_DAY
    from resolve import resolve_global, cost_of
    stored = pd.read_csv(os.path.join(results, "scalability_results.csv")); stored = stored[(stored.bays == 31) & (stored.t0 == SLOTS_DAY + 190)]
    w, df = S.make(1); L = w["L"]; T0 = SLOTS_DAY + 190; a = T0 - 23; sub = df[(df.t >= a) & (df.t <= T0)].reset_index(drop=True); ok = True; n = 0
    print("Re-running the comparison of global and local resolution for 31 bays ...", flush=True)
    for al in (False, True):
        g = stored[(stored.method == "global") & (stored.aligned == al)].iloc[0]; R_, cg, st = resolve_global(L, sub, a, T0, al, tmax=300.0)
        n += 1; ok &= close(cg, g.cost)
        for m in (3, 6, 9):
            l = stored[(stored.method == "local") & (stored.aligned == al) & (stored.margin == m)].iloc[0]; Rl, stats = S.local_budget(L, sub, a, T0, al, kt=m, budget=300.0)
            n += 1; same = close(cost_of(sub, Rl, al), l.cost); ok &= same
            if not same: print(f"  differs: aligned={al} margin={m}: {cost_of(sub, Rl, al)} against {l.cost}")
    print(f"  {n} total weights compared")
    return ok

if __name__ == "__main__":
    results = sys.argv[1] if len(sys.argv) > 1 else "results"; a = check_small_comparison(results); b = check_week(results)
    print("\nAll checked values agree with the stored results." if (a and b) else "\nSome values differ from the stored results.")
    sys.exit(0 if (a and b) else 1)
