"""
Growing morphoelastic rod model with granular penetration resistance.

Reframing of the root-growth project (see ESA_PROJECT_SUMMARY.md, July 2026)
away from "plant root shape prediction" -- which duplicates Sipos & Varkonyi
(2022), JMPS 160:104789 -- and toward growth-driven penetration of a granular
medium under non-Earth gravity.

What this module does that the original prototype did not:

  1. The rod actually grows. The material coordinate s in [0,1] is advected by
     an accumulated growth function G(s,t), so the tip advances in space. The
     prototype held L=2.0, N=50 as module constants and never changed them,
     and its declared growth rate `mu` was never used in the physics.

  2. Equilibrium is obtained from a genuine two-point boundary value problem.
     The base is clamped (position and orientation prescribed); the tip is free
     apart from a prescribed tip force. The base reaction force and moment are
     the six shooting unknowns, matched to the six free-tip conditions.

  3. Control mechanisms are written as a target intrinsic curvature that the
     material actually relaxes toward at a finite adaptation rate, with
     circumnutation represented as a helical intrinsic curvature advected in
     material coordinates (this is the mechanism that produces coiling).

  4. The medium is granular. Tip resistance follows Bekker cone penetration,
     side resistance is distributed Coulomb friction, and both grow with the
     embedded depth. No surface contact and no stiff penalty spring.

Constitutive convention
-----------------------
  r(s)   position, s material coordinate in [0,1]
  d3     tangent
  dl     = (1 + G) ds   current arc-length element
  n      internal force (tension positive)
  m      internal bending/torsion moment
  dn/ds  = -f_ext
  dm/ds  = (r - r_base) x n
  kappa  = C^-1 m + k0        C = diag(B1, B1, B3)
  dr/ds  = d3
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field



# --------------------------------------------------------------------------
# rotation utilities
# --------------------------------------------------------------------------

def qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def q2R(q: np.ndarray) -> np.ndarray:
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def R2q(R: np.ndarray) -> np.ndarray:
    t = np.trace(R)
    if t > 0:
        s = math_sqrt(t + 1.0) * 2.0
        return np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s,
                         (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = 2.0 * math_sqrt(1.0 + R[i, i] - R[j, j] - R[k, k])
    q = np.zeros(4)
    q[0] = (R[k, j] - R[j, k]) / s
    q[i + 1] = 0.25 * s
    q[j + 1] = (R[j, i] + R[i, j]) / s
    q[k + 1] = (R[k, i] + R[i, k]) / s
    return q


def math_sqrt(x: float) -> float:
    return float(np.sqrt(x))


def q_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / max(np.linalg.norm(axis), 1e-300)
    s = np.sin(0.5 * angle)
    return np.array([np.cos(0.5 * angle), s * axis[0], s * axis[1], s * axis[2]])


def frame_from_tangent(d3: np.ndarray, hint: np.ndarray | None = None) -> np.ndarray:
    """Build an orthonormal frame whose third column is d3 (Gram-Schmidt)."""
    d3 = d3 / np.linalg.norm(d3)
    a = np.array([0.0, 0.0, 1.0]) if hint is None else hint
    if abs(np.dot(a, d3)) > 0.95:
        a = np.array([1.0, 0.0, 0.0])
    e1 = a - np.dot(a, d3) * d3
    e1 /= max(np.linalg.norm(e1), 1e-300)
    e2 = np.cross(d3, e1)
    return np.column_stack([e1, e2, d3])


# --------------------------------------------------------------------------
# granular medium
# --------------------------------------------------------------------------

@dataclass
class GranularMedium:
    """Bekker-type granular resistance with Coulomb side friction.

    Parameters are chosen so that the Moon case is roughly 1/20 of Earth
    cohesion and 1/6 of Earth gravity, which is the ordering usually assumed
    for lunar versus terrestrial regolith; the values are indicative and are
    reported as such in the manuscript.
    """
    gravity: float = 9.81          # m/s^2, magnitude, direction is -z
    cohesion: float = 20.0         # Pa
    friction_angle: float = 30.0   # deg
    density: float = 1500.0        # kg/m^3, bulk regolith
    tip_radius: float = 2.0e-3     # m (2 mm root tip, 4 mm diameter)
    v_slow: float = 1.0e-4          # m/s, regularization of velocity dependence

    def __post_init__(self):
        # cached because this is evaluated once per finite element per
        # residual evaluation, i.e. tens of thousands of times per solve
        self._mu = float(np.tan(np.radians(self.friction_angle)))
        self.med_gravity_scale = self.gravity
        self.gvec = np.array([0.0, 0.0, -self.gravity])
        self.ghat = np.array([0.0, 0.0, -1.0])

    @property
    def phi_rad(self) -> float:
        return np.radians(self.friction_angle)

    def bearing_capacity(self, depth: float) -> float:
        """Deviator stress that a cone of this soil can support at `depth`.

        Nq and Nc are the standard cone factors, matching Bekker (1956).
        """
        phi = self.phi_rad
        Nq = np.exp(np.pi * np.tan(phi)) * np.tan(np.pi / 4 + phi / 2) ** 2
        Nc = (Nq - 1) / np.tan(phi)
        sigma_v = self.density * self.gravity * max(depth, 0.0)
        return self.cohesion * Nc + sigma_v * Nq

    def tip_resistance(self, depth: float, speed: float) -> float:
        """Force magnitude on the tip opposing motion.

        The viscous term with `v_slow` regularizes the quasi-static limit so the
        tip force is a well-defined function of velocity, which the shooting
        solver needs for its fixed-point iteration.
        """
        area = np.pi * self.tip_radius ** 2
        q = self.bearing_capacity(depth)
        f_bearing = q * area
        drag = self.density * area * self.med_gravity_scale * max(depth, 0.0)
        u = speed / self.v_slow
        f_drag = drag * u / np.sqrt(1.0 + u * u)
        return f_bearing + f_drag

    def side_resistance(self, tangent: np.ndarray, depth: float) -> np.ndarray:
        """Distributed Coulomb friction per unit current length."""
        s_h = self.density * self.gravity * max(depth, 0.0)
        # isotropic lateral stress: friction opposes the local tangent,
        # scaled by the ratio of contact area to rod radius
        scale = self._mu * s_h * 2.0 * np.pi * self.tip_radius
        return -scale * tangent


# --------------------------------------------------------------------------
# parameters
# --------------------------------------------------------------------------

@dataclass
class Params:
    """Parameters for a primary root of about 4 mm diameter.

    These are physically self-consistent rather than tuned for a picture. For a
    circular section of radius r, I = pi r^4 / 4. With r = 2 mm and a root
    tissue modulus of E = 10 MPa (the upper end of reported values) this gives
    B1 = 1.26e-4 N m^2. Root tissue is closer to 1 MPa, so the numbers here are
    an optimistic bound; the model is run over a range of stiffnesses.

    The control gains are then set by the moment the root must generate to
    displace the tip against the granular resistance, M ~ F_tip L, hence
    kappa ~ F_tip L / B1. At L ~ 0.1 m and F_tip ~ 0.35 N this is a few hundred
    per metre, which corresponds to a tip radius of curvature of a few
    millimetres and is therefore consistent with observed root tips.
    """
    # stiffness (N m^2); B3 = 2 B1 for a circular section (J = 2I)
    B1: float = 1.26e-4
    B3: float = 2.52e-4
    # distributed load (kg/m): rho_solid * A for a 2 mm radius root
    mass_per_length: float = 1.26e-2
    # control mechanisms (1/m)
    K_grav: float = 150.0      # gravitropism gain
    K_circ: float = 80.0       # circumnutation amplitude
    circ_freq: float = 2.5e-4  # Hz, ~4000 s period
    circ_wavelength: float = 0.35   # fraction of the current length
    circ_phase: float = 0.0    # rad, seed for the helical pattern
    eta_adapt: float = 0.05    # 1/s, ~20 s tropic response
    # growth
    growth_rate: float = 5.0e-6   # m/s at the tip (~40 mm/day)
    growth_local: float = 0.10    # localization of growth at the tip, in s units
    L0: float = 0.01           # initial length, m
    # generative thrust: if the soil reaction exceeds this the tip cannot fail
    # the medium axially; it slides out of its path toward the horizontal.
    # 0.10 N on a 4 mm-diameter root (A = pi r^2 = 1.26e-5 m^2) is an ~8 kPa
    # generative pressure, at
    # the low end of measured root-axial stresses and scanned in the paper.
    F_max: float = 0.10
    # numerics
    n_elem: int = 64
    tol_bvp: float = 1e-10


# --------------------------------------------------------------------------
# growing rod
# --------------------------------------------------------------------------

class GrowingRoot:
    """Morphoelastic root advancing into a granular medium."""

    def __init__(self, medium: GranularMedium, p: Params | None = None,
                 base_tangent: np.ndarray | None = None):
        self.med = medium
        self.p = p or Params()
        self.t = 0.0
        self.G = np.zeros(self.p.n_elem)          # accumulated growth, per node
        self.k0 = np.zeros((self.p.n_elem, 3))    # intrinsic curvature
        self.L = self.p.L0                        # current length
        d3 = np.array([0.0, 0.0, -1.0]) if base_tangent is None else \
            base_tangent / np.linalg.norm(base_tangent)
        self.R0 = frame_from_tangent(d3)          # clamped base frame
        self._ds = 1.0 / (self.p.n_elem - 1)
        # tip velocity guess for the resistance fixed point
        self.tip_vel = np.zeros(3)

    # -- growth ----------------------------------------------------------
    def _apply_growth(self, dt: float) -> None:
        """Extend the tip by dt, localising new material near s=1."""
        p = self.p
        s = np.linspace(0.0, 1.0, p.n_elem)
        # subapical: exponential decay of the local extension rate away from tip
        loc = np.exp(-(1.0 - s) / max(p.growth_local, 1e-6))
        loc /= max(np.trapezoid(loc, s), 1e-12)   # normalise to unit mean
        dG = p.growth_rate * dt / max(p.L0, 1e-12) * loc
        self.G = self.G + dG
        self.L = p.L0 * float(np.trapezoid(1.0 + self.G, s))

    # -- control ---------------------------------------------------------
    def _target_curvature(self, R: np.ndarray, s: np.ndarray) -> np.ndarray:
        """Target intrinsic curvature from gravitropism and circumnutation."""
        p = self.p
        d3 = R[:, :, 2]
        e1 = R[:, :, 0]
        e2 = R[:, :, 1]
        ghat = self.med.ghat

        # gravitropism: bend the tangent toward -z. The gain is proportional to
        # the current gravity level (statolith-sedimentation picture): a root
        # standing in a Moon-regolith experiences a weaker tropic drive than
        # the same root on Earth, at the same tissue response constant K_grav.
        gscale = self.med.gravity / 9.81
        k_grav = p.K_grav * gscale * \
            np.cross(np.broadcast_to(ghat, d3.shape), d3)

        # circumnutation: helical intrinsic curvature, phase locked in material
        phase = 2.0 * np.pi * (p.circ_freq * self.t - s / max(p.circ_wavelength, 1e-6)) \
            + p.circ_phase
        k_circ = p.K_circ * (np.cos(phase)[:, None] * e1 + np.sin(phase)[:, None] * e2)
        return k_grav + k_circ

    def _relax_curvature(self, R: np.ndarray) -> None:
        """Explicitly relax the intrinsic curvature toward its target."""
        p = self.p
        dt = max(self._last_dt, 1e-9)
        s = np.linspace(0.0, 1.0, p.n_elem)
        target = self._target_curvature(R, s)
        # growth stretches the material, so relaxation slows per unit material
        stretch = np.clip(1.0 + self.G, 1.0, None)
        self.k0 = self.k0 + p.eta_adapt * dt * (target - self.k0) / stretch[:, None]

    # -- equilibrium -----------------------------------------------------
    def _integrate(self, n0: np.ndarray, m0: np.ndarray, f_tip: np.ndarray,
                   want_frames: bool = True):
        """March from the clamped base to the free tip.

        The frame is carried as a quaternion and `d3` is extracted
        algebraically, avoiding a matrix build and two quaternion round-trips
        per element.

        `want_frames=False` skips the rotation-matrix assembly at the end. The
        Newton residual only needs the tip force and moment, and building the
        full frame array on every residual evaluation dominated the runtime.
        """
        p = self.p
        nseg = p.n_elem
        stretch = 1.0 + self.G

        quats = np.empty((nseg, 4))
        quats[0] = R2q(self.R0)
        pos = np.zeros((nseg, 3))
        n = n0.copy()
        m = m0.copy()
        invB1, invB3 = 1.0 / p.B1, 1.0 / p.B3
        wgt = p.mass_per_length * self.med.gravity
        w, x, y, z = quats[0]

        for i in range(nseg - 1):
            # current arc-length element: (1 + G) ds scaled by the reference
            # length L0. The element stretch is the trapezoidal average of the
            # two nodes so that the integrated length equals
            # trapezoid(1 + G, s) * L0 = self.L exactly; using a one-sided
            # Riemann sum instead makes the centreline lag behind L as growth
            # concentrates subapically. The distributed load also acts over the
            # physical element length dl, not the bare mesh spacing.
            dl = 0.5 * (stretch[i] + stretch[i + 1]) * self._ds * p.L0
            d3 = np.array([2.0 * (x * z + w * y),
                           2.0 * (y * z - w * x),
                           1.0 - 2.0 * (x * x + y * y)])

            f_ext = np.array([0.0, 0.0, -wgt * stretch[i]]) \
                + self.med.side_resistance(d3, -pos[i, 2]) * dl
            n = n - f_ext * dl
            # moment balance is m' = d3 x n (per unit tangent, not r x n);
            # integrating d3 x n gives the correct tip-shear moment arm F L.
            m = m + np.cross(d3, n) * dl

            k0 = self.k0[i]
            kappa = np.array([m[0] * invB1 + k0[0],
                              m[1] * invB1 + k0[1],
                              m[2] * invB3 + k0[2]])
            kmag = float(np.sqrt(kappa[0] ** 2 + kappa[1] ** 2 + kappa[2] ** 2))
            if kmag > 1e-14:
                h = 0.5 * kmag * dl
                sh = np.sin(h)
                dq = np.array([np.cos(h), sh * kappa[0] / kmag,
                               sh * kappa[1] / kmag, sh * kappa[2] / kmag])
                w, x, y, z = qmul(dq, np.array([w, x, y, z]))
            quats[i + 1] = (w, x, y, z)
            pos[i + 1] = pos[i] + np.array([2.0 * (x * z + w * y),
                                            2.0 * (y * z - w * x),
                                            1.0 - 2.0 * (x * x + y * y)]) * dl

        if not want_frames:
            s_loc = np.linspace(0.0, 1.0, nseg)
            return pos, None, n, m, s_loc

        R = np.empty((nseg, 3, 3))
        for i in range(nseg):
            R[i] = q2R(quats[i])
        s_loc = np.linspace(0.0, 1.0, nseg)
        return pos, R, n, m, s_loc

    @staticmethod
    def _newton(residual, x0, maxit=30, tol=1e-8):
        """Damped Newton with a forward-difference Jacobian.

        The shooting problem has only six unknowns, so a dense finite-difference
        Jacobian is cheap, and quadratic convergence is reached in two or three
        iterations from a warm start. The derivative-free Levenberg-Marquardt
        that this replaces needed ~130 residual evaluations per solve.
        """
        x = np.array(x0, dtype=float)
        r = residual(x)
        rn = float(np.linalg.norm(r))
        for _ in range(maxit):
            if rn < tol:
                break
            J = np.empty((r.size, x.size))
            for j in range(x.size):
                h = 1e-7 * max(abs(x[j]), 1.0)
                xp = x.copy()
                xp[j] += h
                J[:, j] = (residual(xp) - r) / h
            try:
                dx = np.linalg.solve(J, -r)
            except np.linalg.LinAlgError:
                dx = np.linalg.lstsq(J, -r, rcond=None)[0]
            # damped line search on the residual norm
            step = 1.0
            improved = False
            for _ in range(14):
                r_try = residual(x + step * dx)
                rn_try = float(np.linalg.norm(r_try))
                if rn_try < rn:
                    x = x + step * dx
                    r, rn, improved = r_try, rn_try, True
                    break
                step *= 0.5
            if not improved:
                break
        return x, rn

    def _solve_equilibrium(self, f_tip: np.ndarray, guess: np.ndarray | None = None):
        """Shooting: match the six free-tip conditions.

        Unknowns are the base reaction force and moment. Residuals are the
        deviation of the tip internal force from `f_tip` and the tip moment
        from zero. This is a genuine two-point BVP: the base clamp alone
        cannot satisfy the tip conditions, and vice versa.

        The shooting is carried out in scaled variables. The base force is
        O(F) while the base moment is O(F L), and with L ~ 1e-2 the two differ
        by two orders of magnitude; solving directly in physical units yields
        a badly conditioned Jacobian and needs several times as many Newton
        iterations. `guess` warm-starts from the previous time step.
        """
        L_ref = max(self.L, 1e-6)
        F_ref = max(np.linalg.norm(f_tip),
                    self.p.mass_per_length * self.med.gravity * L_ref, 1e-12)
        M_ref = F_ref * L_ref
        scale = np.array([F_ref, F_ref, F_ref, M_ref, M_ref, M_ref])

        def residual(u):
            x = u * scale
            _, _, n_end, m_end, _ = self._integrate(x[:3], x[3:], f_tip,
                                                   want_frames=False)
            return np.concatenate([(n_end - f_tip) / F_ref, m_end / M_ref])

        if guess is None:
            guess = np.zeros(6)
        u0 = np.asarray(guess, dtype=float) / scale
        sol, rn = self._newton(residual, u0)
        x = sol * scale
        pos, R, n_end, m_end, s = self._integrate(x[:3], x[3:], f_tip)
        return pos, R, x[:3], x[3:], rn, s

    def _toward_horizontal(self, tangent: np.ndarray, tilt: float,
                           frame_e1: np.ndarray) -> np.ndarray:
        """Raise a tip tangent out of the soil by `tilt` radians.

        The azimuth is inherited from the rod's own lateral frame (`frame_e1`),
        so the tip kneels in the plane in which it is already curved rather
        than in an arbitrary direction. Rodrigues rotation about the horizontal
        axis perpendicular to the tangent.
        """
        ghat = np.array([0.0, 0.0, 1.0])
        proj = frame_e1 - ghat * float(frame_e1 @ ghat)
        nrm = float(np.linalg.norm(proj))
        a = proj / nrm if nrm > 1e-9 else np.array([1.0, 0.0, 0.0])
        ct, st = np.cos(tilt), np.sin(tilt)
        c1 = 1.0 - ct
        ax, ay, az = a
        return np.array([
            tangent[0] * (ct + c1 * ax * ax)
            + tangent[1] * (c1 * ax * ay - st * az)
            + tangent[2] * (c1 * ax * az + st * ay),
            tangent[0] * (c1 * ay * ax + st * az)
            + tangent[1] * (ct + c1 * ay * ay)
            + tangent[2] * (c1 * ay * az - st * ax),
            tangent[0] * (c1 * az * ax - st * ay)
            + tangent[1] * (c1 * az * ay + st * ax)
            + tangent[2] * (ct + c1 * az * az),
        ])

    # -- one time step ---------------------------------------------------
    def step(self, dt: float) -> dict:
        """Advance the root by one growth step.

        The granular resistance is applied along the local tip tangent, not as
        an arbitrary three-dimensional load. A root penetrates by adding
        material at the tip, so the reaction of the medium is transmitted as
        axial compression in the rod; it does not impose a large transverse
        force. Treating it as a general 3D load on a slender rod of bending
        stiffness B1 requires a tip curvature of order F L / B1, which for real
        root dimensions exceeds the total rotation available several times over
        and makes the two-point problem insoluble. Bending is therefore driven
        by the tropic intrinsic curvature, gravity, and the soil gradient, which
        is the mechanism the paper is about.

        When the soil reaction that the tip would incur exceeds the generative
        thrust F_max the tip cannot fail the medium axially. The equilibrium tip
        force is then held at F_max and the tangent is raised toward the
        horizontal by tilt = arccos(F_max / F_soil); the axial component into
        the layer therefore never exceeds the generative capacity, and the
        excess elongation is shed along the layer (compaction avoidance).
        """
        self._last_dt = dt
        guess = self._reaction.copy() if hasattr(self, "_reaction") else None
        depth = self._last_depth if hasattr(self, "_last_depth") else 0.0

        # fixed point between the tip tangent (hence the tip force direction)
        # and the equilibrium shape
        d3_tip = self._d3_tip if hasattr(self, "_d3_tip") else \
            np.array([0.0, 0.0, -1.0])
        e1_tip = self._e1_tip if hasattr(self, "_e1_tip") else \
            np.array([1.0, 0.0, 0.0])
        thrust_limited = False
        for it in range(4):
            speed = float(np.linalg.norm(self.tip_vel))
            f_soil = self.med.tip_resistance(depth, speed)
            if f_soil > self.p.F_max:
                tilt = float(np.arccos(np.clip(self.p.F_max / f_soil, 0.0, 1.0)))
                d3_tip = self._toward_horizontal(d3_tip, tilt, e1_tip)
                f_tip = -self.p.F_max * d3_tip
                thrust_limited = True
            else:
                f_tip = -f_soil * d3_tip
            pos, R, n0, m0, res, s = self._solve_equilibrium(f_tip, guess)
            guess = np.concatenate([n0, m0])
            self._reaction = guess
            d3_new = R[-1][:, 2]
            d3_tip = d3_new / max(np.linalg.norm(d3_new), 1e-12)
            self._d3_tip = d3_tip
            self._e1_tip = R[-1][:, 0]

        # tip advance follows the local growth direction
        seg = pos[-1] - pos[-2]
        self.tip_vel = seg / dt

        self._relax_curvature(R)
        self._apply_growth(dt)
        self.t += dt

        self._last = dict(pos=pos, R=R, s=s, n0=n0, m0=m0,
                          bvp_residual=res, f_tip=f_tip, L=self.L,
                          depth=float(-pos[-1][2]),
                          thrust_limited=thrust_limited)
        self._last_depth = float(-pos[-1][2])
        return self._last

    def run(self, t_end: float, dt: float = 5.0) -> list:
        steps = int(round(t_end / dt))
        out = []
        for _ in range(steps):
            out.append(self.step(dt))
        return out

    # -- geometry --------------------------------------------------------
    def tip_position(self) -> np.ndarray:
        return self._last['pos'][-1]

    def lateral_excursion(self) -> float:
        """Largest distance of the centreline from the base axis.

        The prototype measured the spread of one Cartesian component, which
        misses a root that bows in the y-z plane. Roots here are free to bend
        in any direction, so the radial excursion is the honest measure.
        """
        pos = self._last['pos']
        r_axis = pos[-1] - pos[0]
        nrm = np.linalg.norm(r_axis)
        if nrm < 1e-15:
            return 0.0
        r_hat = r_axis / nrm
        perp = pos - pos[0]
        d = perp - np.outer(perp @ r_hat, r_hat)
        return float(np.max(np.linalg.norm(d, axis=1)))

    def morphology(self) -> str:
        """Coarse label for the tip path, used only to summarise scans.

        Reported as an auxiliary label; the transition analysis in the paper is
        done on the continuous lateral-excursion measure, not on this
        discretisation.
        """
        ratio = self.lateral_excursion() / max(self.L, 1e-12)
        if ratio < 0.08:
            return "straight"
        if ratio < 0.30:
            return "waving"
        return "coiling"
