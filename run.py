#!/usr/bin/env python3
"""Command line entry point.

    python run.py experiment1     normal faults over 20 weeks and the stress test
    python run.py sensitivity     sensitivity to the assumptions of the simulated world
    python run.py scalability     global against local resolution from 31 to 744 bays
    python run.py summary         tables from the stored results (or from outputs/)
    python run.py data            synthetic data of one week as CSV files
    python run.py verify          re-runs one week and a small comparison and checks them against the stored results

New runs are written to outputs/. Add --quick to the first three commands for a short test run."""
import argparse, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__)); SRC = os.path.join(HERE, "src"); OUT = os.path.join(HERE, "outputs")

def run(script, args, log, cwd=OUT):
    os.makedirs(cwd, exist_ok=True); os.makedirs(os.path.join(OUT, "logs"), exist_ok=True)
    env = dict(os.environ, PYTHONPATH=SRC + os.pathsep + os.environ.get("PYTHONPATH", ""), PYTHONUNBUFFERED="1")
    with open(os.path.join(OUT, "logs", log), "a", encoding="utf-8") as lg:
        lg.write(f"\n--- {time.strftime('%Y-%m-%d %H:%M:%S')} {script} {' '.join(args)}\n")
        p = subprocess.Popen([sys.executable, os.path.join(SRC, script)] + args, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in p.stdout: print(line, end="", flush=True); lg.write(line)
            p.wait()
        except KeyboardInterrupt:
            p.terminate(); p.wait(); print("\nStopped. Finished runs are saved; run the same command again to continue.", flush=True); sys.exit(130)
    return p.returncode

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("experiment1", "sensitivity"):
        p = sub.add_parser(name); p.add_argument("--quick", action="store_true"); p.add_argument("--workers", type=int, default=1, help="independent runs in parallel")
        if name == "experiment1": p.add_argument("--weeks", type=int, default=20, help="simulated weeks under the normal faults")
    sub.add_parser("scalability").add_argument("--quick", action="store_true")
    p = sub.add_parser("summary"); p.add_argument("--dir", default="results")
    p = sub.add_parser("data"); p.add_argument("--seed", type=int, default=11); p.add_argument("--out", default="data")
    sub.add_parser("verify")
    a = ap.parse_args()
    if a.cmd == "experiment1":
        args = ["--stages", "main,stress", "--workers", str(a.workers)] + (["--seeds-main", "2", "--main-step", "40", "--seeds-stress", "1", "--stress-step", "12"] if a.quick else ["--seeds-main", str(a.weeks), "--seeds-stress", "1"])
        sys.exit(run("study.py", args, "experiment1.log"))
    if a.cmd == "sensitivity":
        args = ["--stages", "oat,lhs", "--workers", str(a.workers)] + (["--sens-seeds", "1", "--oat-params", "3", "--sens-step", "40", "--lhs-draws", "4", "--lhs-seeds", "1"] if a.quick else [])
        sys.exit(run("study.py", args, "sensitivity.log"))
    if a.cmd == "scalability":
        steps = ([("scalability.py", ["--scales", "1,2", "--margins", "3,6", "--budgets", "20"]), ("scalability_large.py", ["--mode", "zones", "--scales", "2,3", "--margins", "3", "--budget", "20"]), ("scalability_large.py", ["--mode", "bays", "--scales", "2,3", "--margins", "3", "--budget", "20"])] if a.quick else
                 [("scalability.py", []), ("scalability_large.py", ["--mode", "zones", "--scales", "12,24", "--margins", "3,6,9"]), ("scalability_large.py", ["--mode", "bays", "--scales", "12,24", "--margins", "3,6"])])
        for script, args in steps:
            if run(script, args, "scalability.log"): sys.exit(1)
        return
    if a.cmd == "summary": sys.exit(run("summary.py", ["--dir", os.path.abspath(a.dir)], "summary.log", cwd=HERE))
    if a.cmd == "data": sys.exit(run("generate_data.py", ["--seed", str(a.seed), "--out", os.path.abspath(a.out)], "data.log", cwd=HERE))
    if a.cmd == "verify": sys.exit(run("verify.py", [os.path.join(HERE, "results")], "verify.log", cwd=HERE))

if __name__ == "__main__": main()
