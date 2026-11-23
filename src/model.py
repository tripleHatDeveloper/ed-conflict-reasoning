"""Constraint model of the ED (hard constraints) and encoding of evidence as removable assertions.
Variables per bay and slot: occ (a patient is in the bay) and blk (bay blocked: cleaning or closed).
Hard constraints come from the BIM zoning and from operational physics (minimum stay, cleaning,
waiting queue coupling). Evidence items are removable and carry a cost (weight)."""
from ortools.sat.python import cp_model
from sim import K_SOC, K_IT, K_CO, K_SQ, K_IQ, K_CEN

P = dict(Amax=8, Lw=2, lag_it=3, resv=2, lag_co=2, lag_q=2, census_c=4)

def build(L, bays, a, b, est, aligned, queue=True, queue_full=False, objective=True, assume=False):
    """est: iterable of tuples (id, kind, t, s, z, v, w). Returns dict with model and handles.
    An evidence item whose assertion cannot be falsified inside the tile is left out (kept)."""
    m = cp_model.CpModel(); occ, blk = {}, {}
    bset = set(bays)
    for s in bays:
        for t in range(a, b + 1):
            occ[s, t] = m.NewBoolVar(""); blk[s, t] = m.NewBoolVar("")
    for s in bays:
        for t in range(a, b + 1):
            m.AddBoolOr([occ[s, t].Not(), blk[s, t].Not()])
        for t in range(a, b - 1):
            m.AddBoolOr([occ[s, t], occ[s, t+1].Not(), occ[s, t+2]])                 # minimum stay (not 0 1 0)
            m.AddBoolOr([occ[s, t].Not(), occ[s, t+1], occ[s, t+2].Not()])           # turnaround (not 1 0 1)
            m.AddBoolOr([blk[s, t], blk[s, t+1].Not(), blk[s, t+2]])                 # blocked at least two slots
        for t in range(a, b - 2):
            m.AddBoolOr([occ[s, t], occ[s, t+1].Not(), occ[s, t+2].Not(), occ[s, t+3]])   # minimum stay of three slots
        for t in range(a, b):                                                        # cleaning after departure
            m.AddBoolOr([occ[s, t].Not(), occ[s, t+1], blk[s, t+1]])
            if t + 2 <= b: m.AddBoolOr([occ[s, t].Not(), occ[s, t+1], blk[s, t+2]])
    q = {}; newocc = {}
    if queue:
        qb = [s for s in bays if L.queue_zone[L.zone_of[s]]]
        for t in range(a, b + 1): q[t] = m.NewIntVar(0, 400, "")
        for t in range(a, b):
            terms = []
            for s in qb:
                v = m.NewBoolVar(""); m.AddBoolOr([v, occ[s, t+1].Not(), occ[s, t]]); newocc[s, t] = v; terms.append(v)
            if terms: m.Add(sum(terms) <= q[t])                                       # intake bounded by queue (valid relaxation)
            m.Add(q[t+1] - q[t] <= max(P["Amax"], int(round(P["Amax"] * getattr(L, "scale", 1) / 6))))
            if queue_full and terms: m.Add(q[t] - q[t+1] <= sum(terms) + P["Lw"])      # drop needs full intake sum
    keep = {}; weights = {}
    def lit(kind, s, t, v):
        x = occ[s, t] if kind == "occ" else blk[s, t]
        return x if v else x.Not()
    for (eid, kind, t, s, z, v, w) in est:
        if not (a <= t <= b): continue
        k = m.NewBoolVar(""); ok = True
        if kind in (K_SOC, K_IT, K_CO) and (s, t) not in occ: continue
        if kind == K_SOC:
            m.AddBoolOr([k.Not(), lit("occ", s, t, v)])
        elif kind == K_IT:
            if not aligned: m.AddBoolOr([k.Not(), lit("occ", s, t, v)])
            else:
                if v == 1: alts = [(s, t + i) for i in range(0, P["resv"] + 1)] + [(s, t - i) for i in range(1, P["lag_it"] + 1)]
                else: alts = [(s, t - i) for i in range(0, P["lag_it"] + 1)]
                if any(x not in occ for x in alts): continue
                m.AddBoolOr([k.Not()] + [lit("occ", x[0], x[1], v) for x in alts])
        elif kind == K_CO:
            def state_lit(tt):                                   # literal group for the reported state at slot tt
                if v == 2: return [lit("occ", s, tt, 1)]
                if v == 1: return [lit("blk", s, tt, 1)]
                return [lit("occ", s, tt, 0), lit("blk", s, tt, 0)]
            if not aligned:
                for l in state_lit(t): m.AddBoolOr([k.Not(), l])
            else:
                if any((s, t - i) not in occ for i in range(P["lag_co"] + 1)): continue
                aux = []
                for i in range(P["lag_co"] + 1):
                    x = m.NewBoolVar("")
                    for l in state_lit(t - i): m.AddBoolOr([x.Not(), l])
                    aux.append(x)
                m.AddBoolOr([k.Not()] + aux)
        elif kind in (K_SQ, K_IQ):
            if not queue or t not in q: continue
            if kind == K_SQ or not aligned:
                m.Add(q[t] <= v + 1).OnlyEnforceIf(k); m.Add(q[t] >= v - 1).OnlyEnforceIf(k)
            else:
                if any((t - i) not in q for i in range(P["lag_q"] + 1)): continue
                sel = []
                for i in range(P["lag_q"] + 1):
                    x = m.NewBoolVar(""); m.Add(q[t - i] <= v + 1).OnlyEnforceIf(x); m.Add(q[t - i] >= v - 1).OnlyEnforceIf(x); sel.append(x)
                m.AddBoolOr([k.Not()] + sel)
        elif kind == K_CEN:
            zb = L.bays_of[z]
            if any(x not in bset for x in zb): continue
            tot = sum(occ[x, t] for x in zb)
            m.Add(tot <= v + 1).OnlyEnforceIf(k)
            m.Add(tot >= v - 1 - (P["census_c"] if aligned else 0)).OnlyEnforceIf(k)
        keep[eid] = k; weights[eid] = w
    if objective: m.Maximize(sum(int(weights[e]) * keep[e] for e in keep))
    elif assume: m.AddAssumptions(list(keep.values()))
    return dict(m=m, keep=keep, weights=weights, occ=occ, blk=blk, q=q)

def solve(bd, tmax=10.0, force_keep=(), force_remove=(), workers=1):
    m = bd["m"]
    for e in force_keep:
        if e in bd["keep"]: m.Add(bd["keep"][e] == 1)
    for e in force_remove:
        if e in bd["keep"]: m.Add(bd["keep"][e] == 0)
    sv = cp_model.CpSolver(); sv.parameters.max_time_in_seconds = tmax; sv.parameters.num_workers = workers
    st = sv.Solve(m)
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE): return st, None, None, sv
    removed = {e for e, k in bd["keep"].items() if sv.Value(k) == 0}
    cost = sum(bd["weights"][e] for e in removed)
    return st, removed, cost, sv
