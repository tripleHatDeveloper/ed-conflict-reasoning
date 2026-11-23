"""Prints the results from a folder of run files: the report of Experiment 1 with the sensitivity analysis, and the comparison of global and local resolution."""
import argparse, os, time
import numpy as np, pandas as pd

def _t(v): return f"{v:,.0f} s" if v >= 10 else f"{v:.1f} s"
def _x(v): return f"{v:,.0f}x" if v >= 10 else f"{v:.1f}x"

def scalability_table(folder):
    """Quality gain and speed-up of local resolution over global resolution, one row per size and interpretation."""
    def read(name): 
        p = os.path.join(folder, name); return pd.read_csv(p) if os.path.exists(p) else None
    parts = []
    b = read("scalability_results.csv")
    if b is not None:
        b = b[(b.t0 == 288 + 190) & (b.budget == b.budget.max())].copy(); b["enlargement"] = "bays"; parts.append(b)
    for mode in ("zones", "bays"):
        d = read(f"large_{mode}.csv")
        if d is not None: d = d.copy(); d["enlargement"] = mode; parts.append(d[[c for c in d.columns if c not in ("scale", "t_tiles", "t_check", "mode")]])
    if not parts: return "No scalability results found."
    R = pd.concat(parts, ignore_index=True); margins = sorted(int(m) for m in R[R.method == "local"].margin.unique()); rows = []
    for (al, enl, n), g in sorted(R.groupby(["aligned", "enlargement", "bays"]), key=lambda x: (x[0][0], x[0][2], x[0][1])):
        gl = g[g.method == "global"]
        if gl.empty: continue
        gl = gl.iloc[0]; gcell = f"{int(gl.cost):,}" + (" (optimal)" if gl.status == "OPTIMAL" else (f" (bound {int(gl.bound):,})" if gl.bound > 0.5 * gl.cost else "")) + f", {_t(gl.time)}"
        row = ["Aligned" if al else "Strict", int(n), enl, gcell]
        for m in margins:
            l = g[(g.method == "local") & (g.margin == m)]
            if l.empty: row.append("not run"); continue
            l = l.iloc[0]; qg = 100 * (gl.cost - l.cost) / gl.cost; sign = "+" if qg > 0.05 else ("-" if qg < -0.05 else "")
            row.append(f"{int(l.cost):,} ({sign}{abs(qg):.1f}%), {_t(l.time)} ({_x(gl.time / l.time)})" + ("*" if l.tiles_nonopt > 0 else ""))
        rows.append(row)
    cols = ["Interpretation", "Bays", "Enlargement", "Global: weight, time"] + [f"Local, margin {m}: weight (quality gain), time (speed-up)" for m in margins]
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"] + ["| " + " | ".join(str(v) for v in r) + " |" for r in rows]
    note = ("Weights are the total weight of the removed observations (lower is better). The quality gain is the weight of global resolution minus the weight of local resolution, "
            "divided by the weight of global resolution. The speed-up is the time of global resolution divided by the time of local resolution. "
            "An asterisk marks results in which some tile problems stopped at their share of the time budget. "
            "Enlargement: bays = every zone has proportionally more bays, zones = the set of zones is repeated with one shared waiting queue.")
    return "\n".join(out) + "\n\n" + note

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", default="results"); args = ap.parse_args(); folder = os.path.abspath(args.dir)
    if any(os.path.exists(os.path.join(folder, f"runs_{s}.csv")) for s in ("main", "stress", "oat", "lhs")):
        import report; here = os.getcwd(); os.chdir(folder)
        try: report.build(time.time(), dict(sens_step=4))
        finally: os.chdir(here)
        print(open(os.path.join(folder, "report.md"), encoding="utf-8").read())
    print("\n# Global against local resolution\n"); print(scalability_table(folder))
