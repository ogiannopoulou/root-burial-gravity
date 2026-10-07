"""Core paper experiment: variable-gravity penetration.

For each target body the root grows with the statolith model (tropic gain
proportional to local g) and active circumnutation. The soil parameters are
fixed at Earth values so that the gravity enters through (i) the effective
pressure in the bearing and side resistance (denser at depth under higher g)
and (ii) the gravitropism signal. Each case is repeated over several
circumnutation phases; the reported quantities are window means with the range
across phases as a sensitivity band.

Order parameter: the vertical penetration efficiency, the time-averaged ratio
of gained burial depth to newly grown root length. It is 1 for a perfect
straight descent, 0 for horizontal growth, and negative for roots that rise.
It is bounded and does not drift when the tip path precesses, unlike the raw
tip depth.
"""
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "model"))
from rod import GrowingRoot, GranularMedium, Params

G_EARTH = 9.81
BODIES = [
    ("Europa", 0.134), ("Titan", 0.138), ("Moon", 0.1654),
    ("Mars", 0.376), ("Venus", 0.907), ("Earth", 1.000),
    ("2 Earth", 2.0), ("4 Earth", 4.0),
]
N_STEPS = 60
DT = 100.0
WINDOW = 20
SEEDS = np.array([0.0, 2.1, 4.2])


def run_body(medium, g_ratio, seed, n_elem=48):
    medium.gravity = G_EARTH * g_ratio
    p = Params(n_elem=n_elem, K_grav=150.0, K_circ=80.0, circ_phase=seed)
    r = GrowingRoot(medium, p)
    r._last_dt = DT
    prev = None
    eff, exc, depth, tipf, thrust = [], [], [], [], []
    for i in range(N_STEPS):
        r.step(DT)
        if i >= N_STEPS - WINDOW:
            tip = np.array(r.tip_position())
            if prev is not None:
                # vertical cosine of the growth step: bounded in [-1, 1]
                seg = tip - prev
                nseg = float(np.linalg.norm(seg))
                if nseg > 0:
                    eff.append(-seg[2] / nseg)
            exc.append(r.lateral_excursion() / max(r.L, 1e-12))
            d = max(-float(tip[2]) / max(r.L, 1e-12),
                    -float(tip[2]) / max(r.L, 1e-12))
            depth.append(-float(tip[2]))          # metres, not normalised
            tipf.append(float(np.linalg.norm(r._last["f_tip"])))
            thrust.append(float(r._last["thrust_limited"]))
            prev = tip
    return dict(
        g=g_ratio,
        efficiency=float(np.mean(eff)),
        efficiency_std=float(np.std(eff)),
        latent=float(np.mean(exc)),
        depth=float(np.max(depth)),          # deepest burial reached, in m
        depth_end=float(np.mean(depth)),
        tip_force=float(np.mean(tipf)),
        thrust_fraction=float(np.mean(thrust)),
        L=float(r.L),
    )


def analytic_zstar(medium, g_ratio, F_max):
    """Soil-only penetration limit z* = (F_max/A - c Nc) / (rho g Nq)."""
    phi = medium.phi_rad
    Nq = np.exp(np.pi * np.tan(phi)) * np.tan(np.pi / 4 + phi / 2) ** 2
    Nc = (Nq - 1) / np.tan(phi)
    A = np.pi * medium.tip_radius ** 2
    g = G_EARTH * g_ratio
    z = (F_max / A - medium.cohesion * Nc) / (medium.density * g * Nq)
    return max(float(z), 0.0)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(here, "..", "results", "gravity_sweep.json")
    med = GranularMedium()
    F_max = Params().F_max
    rows = []
    for name, g in BODIES:
        per_seed = [run_body(med, g, s) for s in SEEDS]
        effs = np.array([d["efficiency"] for d in per_seed])
        lat = np.array([d["latent"] for d in per_seed])
        dep = np.array([d["depth"] for d in per_seed])
        tipf = np.array([d["tip_force"] for d in per_seed])
        thr = np.array([d["thrust_fraction"] for d in per_seed])
        zstar = analytic_zstar(med, g, F_max)
        row = dict(
            body=name, g=g,
            eff_mean=float(effs.mean()), eff_lo=float(effs.min()),
            eff_hi=float(effs.max()),
            lat_mean=float(lat.mean()),
            depth_mean=float(dep.mean() * 1000),     # mm
            thrust_mean=float(thr.mean()),
            tip_force=float(tipf.mean()),
            zstar_mm=float(zstar * 1000),
            L_mm=float(per_seed[0]["L"] * 1000),
        )
        rows.append(row)
        print(f"  {name:8s} g={g:5.3f}  eff={row['eff_mean']:+.4f} "
              f"[{row['eff_lo']:+.3f},{row['eff_hi']:+.3f}]  "
              f"depth={row['depth_mean']:6.2f} mm (z*={row['zstar_mm']:6.2f}) "
              f"thrust_lim={row['thrust_mean']:5.2f} lat={row['lat_mean']:.3f}",
              flush=True)
        with open(out_path, "w") as f:
            json.dump(rows, f, indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()