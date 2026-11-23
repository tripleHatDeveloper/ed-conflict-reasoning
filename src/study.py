"""Experiment 1 and the sensitivity analysis. Stages: main (20 weeks, normal faults), stress (three times the faults, a surge day), oat (assumptions varied one at a time)
and lhs (assumptions varied together, Latin hypercube). Each run is a complete world (seed, assumptions) with all decisions, six fusion strategies and the metrics of evaluation.py.
Results are appended to runs_<stage>.csv after every run, and a stopped run continues where it stopped."""
import argparse, os, sys, time, platform, datetime, json
import numpy as np, pandas as pd

PARAMS = [  # name, label, low and high multiplier of the default value, and how it enters the world
    dict(name="fault_scale", label="Overall fault rates", lo=0.5, hi=2.0, how="fs"),
    dict(name="sensor_dropout", label="Sensor dropout", lo=0.5, hi=2.0, how="drop"),
    dict(name="record_missing", label="Missing record updates", lo=0.5, hi=2.0, how="itmiss"),
    dict(name="coordinator_rate", label="Coordinator report rate", lo=0.5, hi=2.0, how="coord_rate"),
    dict(name="coordinator_copying", label="Coordinator copying of the record", lo=0.4, hi=2.0, how="copy"),
    dict(name="sensor_drift", label="Sensor drift", lo=0.0, hi=2.5, how="drift"),
    dict(name="record_delays", label="Record delays (reservation lead, discharge lag)", lo=0.5, hi=1.5, how="delay"),
    dict(name="arrival_intensity", label="Arrival intensity", lo=0.7, hi=1.3, how="arrival"),
    dict(name="length_of_stay", label="Length of stay", lo=0.8, hi=1.25, how="los"),
    dict(name="closure_rate", label="Bay closure rate", lo=0.5, hi=2.0, how="closure"),
    dict(name="allowances", label="Alignment allowances of the framework", lo=0.5, hi=2.0, how="allow")]
DRIFT0 = 0.012

def world_args(levels):
    fs, assume, drift, allow = 1.0, {}, None, 1.0
    for p in PARAMS:
        m = levels.get(p["name"], 1.0)
        if p["how"] == "fs": fs = m
        elif p["how"] == "drift": drift = DRIFT0 * m if p["name"] in levels else None
        elif p["how"] == "allow": allow = m
        else: assume[p["how"]] = m
    return dict(fs=fs, assume=assume, drift=drift, allow_scale=allow)

def build_tasks(a):
    T = {"main": [], "stress": [], "oat": [], "lhs": []}
    for i in range(a.seeds_main): s = 11 + i; T["main"].append(dict(key=f"main_{s}", stage="main", seed=s, days=8, pred=True, step_sub=a.main_step))
    for i in range(a.seeds_stress): s = 31 + i; T["stress"].append(dict(key=f"stress_{s}", stage="stress", seed=s, days=3, fs=3.0, surge=(2,), pred=False, step_sub=a.stress_step))
    plist = PARAMS[:a.oat_params]
    conds = [("base", None, {})] + [(p["name"], lv, {p["name"]: (p["lo"] if lv == "low" else p["hi"])}) for p in plist for lv in ("low", "high")]
    for cname, lv, levels in conds:
        for i in range(a.sens_seeds):
            s = 101 + i; T["oat"].append(dict(key=f"oat_{cname}_{lv}_{s}", stage="oat", seed=s, days=8, pred=False, step_sub=a.sens_step, levels=dict(levels, condition=cname, level=lv), **world_args(levels)))
    if a.lhs_draws > 0:
        from scipy.stats import qmc
        U = qmc.LatinHypercube(d=len(PARAMS), seed=2027).random(a.lhs_draws)
        for d in range(a.lhs_draws):
            levels = {p["name"]: (p["lo"] + (p["hi"] - p["lo"]) * U[d, j] if p["lo"] <= 0 else p["lo"] * (p["hi"] / p["lo"]) ** U[d, j]) for j, p in enumerate(PARAMS)}
            for i in range(a.lhs_seeds):
                s = 201 + i; T["lhs"].append(dict(key=f"lhs_{d}_{s}", stage="lhs", seed=s, days=8, pred=False, step_sub=a.sens_step, levels=dict(levels, draw=d), **world_args(levels)))
    return T

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="main,stress,oat,lhs"); ap.add_argument("--seeds-main", type=int, default=20); ap.add_argument("--seeds-stress", type=int, default=1)
    ap.add_argument("--sens-seeds", type=int, default=3); ap.add_argument("--oat-params", type=int, default=len(PARAMS)); ap.add_argument("--lhs-draws", type=int, default=24); ap.add_argument("--lhs-seeds", type=int, default=2)
    ap.add_argument("--main-step", type=int, default=1); ap.add_argument("--stress-step", type=int, default=1); ap.add_argument("--sens-step", type=int, default=4)
    ap.add_argument("--workers", type=int, default=1); ap.add_argument("--report-only", action="store_true"); a = ap.parse_args()
    t_session = time.time()
    if not a.report_only:
        T = build_tasks(a); stages = [s for s in a.stages.split(",") if s]
        todo = []
        for st in stages:
            fn = f"runs_{st}.csv"; done = set(pd.read_csv(fn).key) if os.path.exists(fn) else set()
            todo += [t for t in T[st] if t["key"] not in done]
        print(f"[{datetime.datetime.now():%H:%M:%S}] {len(todo)} runs to do ({', '.join(f'{st}: {sum(1 for t in todo if t['stage'] == st)}' for st in stages)})", flush=True)
        rows = {st: (pd.read_csv(f"runs_{st}.csv").to_dict("records") if os.path.exists(f"runs_{st}.csv") else []) for st in stages}
        def record(row, k):
            rows[row["stage"]].append(row); pd.DataFrame(rows[row["stage"]]).to_csv(f"runs_{row['stage']}.csv", index=False)
            el = time.time() - t_session; eta = el / k * (len(todo) - k)
            print(f"[{datetime.datetime.now():%H:%M:%S}] {row['key']} done in {row['run_seconds']:.0f}s ({k}/{len(todo)}, elapsed {el / 60:.0f} min, remaining about {eta / 60:.0f} min)", flush=True)
        if a.workers <= 1:
            import evaluation as R
            for k, t in enumerate(todo, 1): record(R.run_task(t), k)
        else:
            from concurrent.futures import ProcessPoolExecutor, as_completed
            import evaluation as R
            with ProcessPoolExecutor(max_workers=a.workers) as ex:
                futs = [ex.submit(R.run_task, t) for t in todo]
                for k, f in enumerate(as_completed(futs), 1): record(f.result(), k)
    import report; report.build(t_session, vars(a))

if __name__ == "__main__": main()
