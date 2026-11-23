"""Layout of the emergency department: five zones with 31 bays, based on the department floor plan.
The number of resuscitation bays (4) and of trolley spaces in the main monitoring area (12) are assumptions, since the plan does not show them.
Each treatment space is one bay in the model."""
from dataclasses import dataclass
import os
import numpy as np
MODE = os.environ.get("ED_SCALE_MODE", "bays")      # "bays": scale multiplies the bays of every zone; "zones": scale copies the whole set of zones

# name, spaces, served by the shared waiting queue, share of queue arrivals, probability of admission
ZONES = [
    dict(name="RESUS",      n=4,  q=False, share=0.000, admit=0.41, note="Resus room, 4 bays"),
    dict(name="MONITORING", n=12, q=True,  share=0.444, admit=0.30, note="Main monitoring area, 12 trolley spaces"),
    dict(name="EXAM_AMB",   n=6,  q=True,  share=0.222, admit=0.15, note="examination rooms 1 to 6 on the ambulance side"),
    dict(name="EXAM_CTR",   n=4,  q=True,  share=0.148, admit=0.15, note="examination room beside Resus and 3 examination rooms near the nurse station"),
    dict(name="ASSESS",     n=5,  q=True,  share=0.186, admit=0.15, note="Assessment Suite, 5 cubicles"),
]

def space_names(scale=1):
    names = []
    for z in ZONES:
        n = z["n"] * scale
        if z["name"] == "RESUS": names += [f"Resus bay {i+1}" for i in range(n)]
        elif z["name"] == "MONITORING": names += [f"Monitoring space {i+1}" for i in range(n)]
        elif z["name"] == "EXAM_AMB": names += [f"Examination room {i+1}" for i in range(6 * scale)]
        elif z["name"] == "EXAM_CTR": names += [f"Examination room {7+i}" for i in range(n)]
        else: names += [f"Assessment cubicle {i+1}" for i in range(n)]
    return names

@dataclass
class Layout:
    zones: list; zone_of: np.ndarray; bays_of: list; queue_zone: list; neighbours: dict
    zone_order: list; n_bays: int; params: list; names: list; scale: int = 1

def make_layout(scale=1, mode=None):
    mode = mode or MODE
    zones, zone_of, bays_of, qz, nb, params = [], [], [], [], {}, []
    zlist = [dict(z, name=f"{z['name']}_{r + 1}") for r in range(scale) for z in ZONES] if mode == "zones" else ZONES; mult = 1 if mode == "zones" else scale
    b = 0
    for zi, z in enumerate(zlist):
        n = z["n"] * mult
        zones.append(z["name"]); qz.append(z["q"]); params.append(z)
        idx = list(range(b, b + n)); bays_of.append(idx); zone_of += [zi] * n
        for j, s in enumerate(idx):
            nb[s] = [x for x in (idx[j-1] if j > 0 else None, idx[j+1] if j < n-1 else None) if x is not None]
        b += n
    order = [i for i, q in enumerate(qz) if q]
    names = [f"Bay {i + 1}" for i in range(b)] if mode == "zones" else space_names(scale)
    return Layout(zones, np.array(zone_of), bays_of, qz, nb, order, b, params, names, scale)
