"""Control-mechanism verification matrix (Table 1 in the manuscript).

Each row isolates one control configuration and reports time-averaged,
phase-averaged metrics so the numbers in the paper are reproducible rather
than drawn from a single snapshot (the prototype's flaw).
"""
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "model"))
from rod import GrowingRoot, GranularMedium, Params

N_STEPS, DT, WINDOW, SEEDS = 40, 100.0, 16, np.array([0.0, 2.1, 4.2])


def run(kg, kc, n_elem=48):
    med = GranularMedium()
    rows = []
    for seed in SEEDS:
        p = Params(n_elem=n_elem, K_grav=kg, K_circ=kc, circ_phase=seed,
                   F_max=1.0)
        r = GrowingRoot(med, p)
        r._last_dt = DT
        prev = None
        eff, exc = [], []
        for i in range(N_STEPS):
            r.step(DT)
            if i >= N_STEPS - WINDOW:
                tip = np.array(r.tip_position())
                if prev is not None:
                    seg = tip - prev
                    ns = float(np.linalg.norm(seg))
                    if ns > 0:
                        eff.append(-seg[2] / ns)
                exc.append(r.lateral_excursion() / max(r.L, 1e-12))
                prev = tip
        rows.append(dict(
            eff=float(np.mean(eff)), lat=float(np.mean(exc)),
            depth=float(-r.tip_position()[2] / max(r.L, 1e-12)),
            label=r.morphology(), L=float(r.L)))
    e = np.array([x["eff"] for x in rows]); l = np.array([x["lat"] for x in rows])
    d = np.array([x["depth"] for x in rows])
    return dict(k_grav=kg, k_circ=kc,
                eff=float(e.mean()), eff_lo=float(e.min()), eff_hi=float(e.max()),
                lat=float(l.mean()), lat_lo=float(l.min()), lat_hi=float(l.max()),
                depth=float(d.mean()),
                label=rows[0]["label"], L_mm=float(rows[0]["L"] * 1000))


if __name__ == "__main__":
    cases = [
        ("none", dict(K_grav=0.0, K_circ=0.0)),
        ("gravitropism only", dict(K_grav=150.0, K_circ=0.0)),
        ("circumnutation only", dict(K_grav=0.0, K_circ=80.0)),
        ("both, grav dominant", dict(K_grav=150.0, K_circ=40.0)),
        ("both, nominal", dict(K_grav=150.0, K_circ=80.0)),
        ("circ dominant", dict(K_grav=40.0, K_circ=200.0)),
    ]
    out = []
    print(f"  {'case':23s} {'eff':>7} {'lat':>7} {'depth':>7} {'label':9s} L[mm]")
    for name, kw in cases:
        d = run(kw["K_grav"], kw["K_circ"])
        d["case"] = name
        out.append(d)
        print(f"  {name:23s} {d['eff']:+7.4f} {d['lat']:7.4f} "
              f"{d['depth']:+7.3f} {d['label']:9s} {d['L_mm']:6.1f}",
              flush=True)
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "..", "results", "control_matrix.json"), "w") as f:
            json.dump(out, f, indent=1)
    print("DONE", flush=True)