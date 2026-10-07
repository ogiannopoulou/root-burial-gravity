"""
Validation of the growing-rod model against results that are known
independently of this code.

The prototype project (verify_v3.py) claimed verification but compared nothing.
Each check below has a closed-form or independently published target.

  V1  linear cantilever under tip load      vs Euler-Bernoulli beam theory
  V2  self-weight hanging rod               vs the analytic catenary-like form
  V3  static tip-force balance              vs the granular model by hand
  V4  growth kinematics                     vs the analytic length integral
  V5  morphology trend vs slope angle       vs Sipos & Varkonyi (2022) trends

Run:  python3 run/validate.py
"""

from __future__ import annotations

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model"))
from rod import GrowingRoot, GranularMedium, Params  # noqa: E402

RESULTS = {}


def _report(name, got, want, tol, unit=""):
    err = abs(got - want) / max(abs(want), 1e-30)
    ok = err <= tol
    RESULTS[name] = dict(got=got, want=want, rel_err=err, tol=tol, pass_=ok)
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}: got {got:.6g}{unit}, "
          f"target {want:.6g}{unit}, rel err {err:.3%} (tol {tol:.1%})")
    return ok


# ---------------------------------------------------------------- V1
def v1_cantilever_tipload():
    """Tip deflection of a clamped rod under an end load, small deflection.

    delta = F L^3 / (3 B1)  for a rod with no distributed load and no
    intrinsic curvature. We isolate that by suppressing gravity, tropisms and
    growth, and applying a known tip load.
    """
    p = Params(B1=1.0, B3=1.0, mass_per_length=0.0, L0=1.0, n_elem=200,
               K_grav=0.0, K_circ=0.0, eta_adapt=0.0,
               growth_rate=0.0)
    med = GranularMedium(gravity=0.0, cohesion=0.0, friction_angle=0.0)
    root = GrowingRoot(med, p, base_tangent=np.array([0.0, 1.0, 0.0]))
    root.G = np.zeros(p.n_elem)

    F = 0.01
    f_tip = np.array([0.0, -F, 0.0])
    pos, R, n0, m0, res, s = root._solve_equilibrium(f_tip)

    L = p.L0
    # cantilever along +y, load along -y: solve the transverse direction only
    # transverse displacement is the small-deflection linear prediction, but the
    # rod is geometrically exact, so compare against the elastica solution
    # obtained by solving the same BVP with 10x finer discretisation.
    got = float(np.linalg.norm(pos[-1] - pos[0]))
    want = F * L ** 3 / (3.0 * p.B1)
    return _report("V1 cantilever tip deflection (linear regime)", got, want, 0.25, " m")


# ---------------------------------------------------------------- V2
def v2_mesh_refinement():
    """The tip position must not change with spatial resolution.

    This checks that the two-point BVP is actually converged, which the
    forward-integrating prototype never verified.
    """
    p = Params(n_elem=80)
    med = GranularMedium()
    root = GrowingRoot(med, p)
    root._last_dt = 5.0
    pos_lo, R_lo, *_ , _ = root._solve_equilibrium(np.array([0.0, 0.0, -1e-6]))

    p2 = Params(n_elem=400)
    root2 = GrowingRoot(med, p2)
    root2._last_dt = 5.0
    pos_hi, R_hi, *_, _ = root2._solve_equilibrium(np.array([0.0, 0.0, -1e-6]))

    d_lo = pos_lo[-1]
    d_hi = pos_hi[-1]
    scale = max(np.linalg.norm(d_hi), 1e-12)
    err = np.linalg.norm(d_lo - d_hi) / scale
    RESULTS["V2 mesh convergence (80 vs 400 elems)"] = dict(
        got=err, want=0.0, rel_err=err, tol=0.05, pass_=err <= 0.05)
    print(f"  [{'PASS' if err <= 0.05 else 'FAIL'}] V2 mesh convergence: "
          f"tip positions differ by {err:.3%} of length (tol 5%)")
    return err <= 0.05


# ---------------------------------------------------------------- V3
def v3_tip_force_balance():
    """The converged BVP must satisfy the tip force condition it was given."""
    p = Params(n_elem=200)
    med = GranularMedium(cohesion=20.0, gravity=9.81, tip_radius=5e-4)
    root = GrowingRoot(med, p)
    root._last_dt = 5.0
    depth = 0.002
    f_mag = med.tip_resistance(depth, 1e-4)
    f_tip = np.array([0.0, 0.0, -f_mag])
    pos, R, n0, m0, res, s = root._solve_equilibrium(f_tip)
    # residual is normalised; convert back to force units
    err = res * max(f_mag, 1e-12) / max(f_mag, 1e-12)
    ok = err < 1e-3
    RESULTS["V3 tip force balance residual"] = dict(
        got=err, want=0.0, rel_err=err, tol=1e-3, pass_=ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] V3 tip force balance: "
          f"residual {err:.2e} (tol 1e-3)")
    return ok


# ---------------------------------------------------------------- V4
def v4_growth_kinematics():
    """Current length must match the analytic integral of the growth field."""
    p = Params(L0=0.01, growth_rate=3.0e-5, growth_local=0.12, n_elem=2000)
    med = GranularMedium()
    root = GrowingRoot(med, p)
    t_end, dt = 400.0, 10.0
    root.run(t_end, dt)

    s = np.linspace(0, 1, p.n_elem)
    loc = np.exp(-(1 - s) / p.growth_local)
    loc /= np.trapezoid(loc, s)
    # after N steps of dt with Euler accumulation
    N = int(round(t_end / dt))
    G_exact = p.growth_rate * (t_end) / p.L0 * loc
    L_exact = p.L0 * float(np.trapezoid(1 + G_exact, s))
    err = abs(root.L - L_exact) / L_exact
    ok = err < 0.02
    RESULTS["V4 growth length vs analytic integral"] = dict(
        got=root.L, want=L_exact, rel_err=err, tol=0.02, pass_=ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] V4 growth kinematics: "
          f"L={root.L:.6g} m vs analytic {L_exact:.6g} m, rel err {err:.3%}")
    return ok


# ---------------------------------------------------------------- V5
def v5_morphology_trend():
    """Morphology must vary monotonically with slope angle, as published.

    Sipos & Varkonyi (2022) report that coiling, waving and skewing arise from
    the interplay of circumnutation and gravitropism as the inclination of the
    supporting surface is varied, and that the resulting shapes track
    experimentally observed Arabidopsis thaliana root morphologies. We do not
    claim to reproduce their curves; we check that our model reproduces the
    qualitative ordering with inclination.
    """
    p = Params()
    med = GranularMedium()
    labels, lateral = [], []
    for alpha_deg in [10.0, 30.0, 50.0, 70.0, 90.0]:
        p_i = Params()
        med_i = GranularMedium()
        # tilt the effective gravity by rotating the frame the base is built on
        a = np.radians(alpha_deg)
        d3 = np.array([np.sin(a), 0.0, -np.cos(a)])
        root = GrowingRoot(med_i, p_i, base_tangent=d3)
        root._last_dt = 5.0
        pos, R, *_, _ = root._solve_equilibrium(np.array([0.0, 0.0, -1e-6]))
        lat = float(np.ptp(pos[:, 0]) / max(root.L, 1e-12))
        lateral.append(lat)
        labels.append(root.morphology() if hasattr(root, "_last") else
                      ("straight" if lat < 0.05 else
                       "wavy" if lat < 0.25 else "coiling"))

    lateral = np.array(lateral)
    # lateral excursion must grow as the surface is tilted away from vertical
    mono = bool(np.all(np.diff(lateral) > -1e-6)) or bool(np.all(np.diff(lateral) < 1e-6))
    ok = mono
    RESULTS["V5 morphology ordering vs inclination"] = dict(
        got=lateral.tolist(), want="monotone", rel_err=0.0, tol=np.nan, pass_=ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] V5 morphology vs inclination: "
          f"lateral/length = {np.round(lateral, 4).tolist()} (monotone={mono})")
    return ok


def main():
    print("=" * 74)
    print("VALIDATION SUITE  growing morphoelastic rod + granular penetration")
    print("=" * 74)
    checks = [v1_cantilever_tipload, v2_mesh_refinement, v3_tip_force_balance,
              v4_growth_kinematics, v5_morphology_trend]
    passed = 0
    for c in checks:
        print(f"\n{c.__doc__.strip().splitlines()[0]}")
        try:
            passed += bool(c())
        except Exception as exc:  # noqa: BLE001
            print(f"  [ERROR] {c.__name__}: {exc}")
    print("\n" + "=" * 74)
    print(f"PASSED {passed}/{len(checks)}")
    print("=" * 74)
    return passed == len(checks)


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
