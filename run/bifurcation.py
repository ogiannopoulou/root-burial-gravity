"""Time-averaged morphology transition scan.

The prototype classified a single snapshot of an oscillating trajectory, so its
morphology labels were arbitrary. Circumnutation makes the lateral excursion a
periodic function of time, so the transition is characterised here by a window
average and by the envelope over that window, both of which are reproducible.
"""
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model"))
from rod import GrowingRoot, GranularMedium, Params


def sweep(kcirc, kgrav, medium, n_steps=60, dt=100.0, window=20, n_elem=48):
    p = Params(n_elem=n_elem, K_grav=kgrav, K_circ=kcirc)
    r = GrowingRoot(medium, p)
    r._last_dt = dt
    hist = []
    for i in range(n_steps):
        r.step(dt)
        if i >= n_steps - window:
            exc = r.lateral_excursion() / max(r.L, 1e-12)
            dep = -float(r.tip_position()[2]) / max(r.L, 1e-12)
            hist.append((exc, dep, r.L))
    a = np.array(hist)
    return dict(
        k_circ=float(kcirc), k_grav=float(kgrav),
        lat_mean=float(a[:, 0].mean()), lat_max=float(a[:, 0].max()),
        lat_std=float(a[:, 0].std()),
        depth_mean=float(a[:, 1].mean()),
        L_final=float(a[-1, 2]),
    )


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    res = os.path.join(here, "..", "results", "bifurcation.json")
    med = GranularMedium()

    ks = [0, 100, 200, 400, 700, 1000, 1500, 2200, 3000, 4000, 5500, 7000]
    rows = []
    print(f"  {'K_circ':>7} {'lat_mean':>9} {'lat_max':>8} "
          f"{'lat_std':>8} {'depth':>7} {'L(mm)':>7}")
    for kc in ks:
        d = sweep(float(kc), 150.0, med)
        rows.append(d)
        print(f"  {d['k_circ']:7.0f} {d['lat_mean']:9.4f} {d['lat_max']:8.4f} "
              f"{d['lat_std']:8.4f} {d['depth_mean']:7.3f} "
              f"{d['L_final']*1000:7.2f}", flush=True)
        with open(res, "w") as f:
            json.dump(rows, f, indent=1)
    print("DONE", flush=True)
EOF