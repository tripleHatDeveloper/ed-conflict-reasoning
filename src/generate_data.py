"""Writes the synthetic data of one simulated week to compressed CSV files: the observations of the data sources with their fault labels, and the state of every bay."""
import argparse, os
import numpy as np, pandas as pd
from sim import simulate, gen_sources, SLOTS_DAY

KINDS = ["bay_sensor", "hospital_record", "coordinator_report", "queue_sensor", "record_queue", "zone_census"]
LABELS = ["valid", "timing", "meaning", "granularity", "error"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=11); ap.add_argument("--days", type=int, default=8); ap.add_argument("--scale", type=int, default=1); ap.add_argument("--out", default="data")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    w = simulate(a.days, scale=a.scale, seed=a.seed, lam0=0.5); df = gen_sources(w, seed=a.seed + 1)
    obs = pd.DataFrame(dict(id=df.id, source=[KINDS[k] for k in df.kind], slot=df.t, bay=df.s, zone=df.z, value=df.v, label=[LABELS[k] for k in df.lab], context=df.ctx, copied_from_record=df.dep))
    L = w["L"]; nb, T = w["occ"].shape
    states = pd.DataFrame(dict(bay=np.repeat(np.arange(nb), T), slot=np.tile(np.arange(T), nb), zone=np.repeat(L.zone_of, T), occupied=w["occ"].ravel(), blocked=w["blk"].ravel()))
    obs.to_csv(os.path.join(a.out, "observations.csv.gz"), index=False); states.to_csv(os.path.join(a.out, "bay_states.csv.gz"), index=False)
    print(f"{len(obs):,} observations and {len(states):,} bay states written to {a.out} (slot length 5 minutes, {SLOTS_DAY} slots per day, day 0 is the calibration day)")
