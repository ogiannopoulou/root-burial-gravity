"""Render final root centreline shapes for four bodies."""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "model"))
from rod import GrowingRoot, GranularMedium, Params

BODIES = [("Europa", 0.134), ("Mars", 0.376), ("Earth", 1.0), ("4 Earth", 4.0)]
N_STEPS = 60
DT = 100.0

fig = plt.figure(figsize=(14, 10))
med = GranularMedium()
for idx, (name, g) in enumerate(BODIES):
    med.gravity = 9.81 * g
    p = Params(n_elem=64, K_grav=150.0, K_circ=80.0)
    r = GrowingRoot(med, p)
    r._last_dt = DT
    for _ in range(N_STEPS):
        r.step(DT)
    pos = r._last["pos"]
    ax = fig.add_subplot(2, 2, idx + 1, projection="3d")
    ax.plot(pos[:, 0] * 1000, pos[:, 1] * 1000, -pos[:, 2] * 1000,
            "b-", lw=1.6)
    ax.scatter(*([0, 0, 0]), color="k", s=40, label="base (seed)")
    tip = pos[-1]
    ax.scatter(tip[0] * 1000, tip[1] * 1000, -tip[2] * 1000, color="r",
               s=30, label="tip")
    ax.set_title(f"{name}  g={g:.3f}  L={r.L*1000:.1f} mm  "
                 f"depth={-tip[2]*1000:.1f} mm")
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)"); ax.set_zlabel("depth (mm)")
    ax.set_box_aspect((1, 1, 1)); ax.legend(fontsize=8)
    ax.view_init(elev=20, azim=-60)

fig.suptitle("Root growth into granular medium under body gravity "
             "(K_grav ~ g, circ=80 1/m, L_final ~ 40mm)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                   "figures", "gravity_trajectories.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=150)
print("WROTE", out)