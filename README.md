# Root burial under gravity
### Morphoelastic growing rod with gravitropism and circumnutation in a
### depth-strengthening granular medium

Data and code accompanying:

> O. Giannopoulou, "Penetration of a growing root into a depth-strengthening
> granular medium under variable gravity," submitted to the *European Journal
> of Mechanics – A/Solids* (2026).

## What this is

An IB-LBM-free, morphoelastic-rod model of a growing root. The organ is
treated as an inextensible growing filament clamped at the base, free at the
tip, and loaded by a granular reaction that (i) tracks the tip tangent,
(ii) strengthens linearly with depth through an overburden term, and (iii)
enters a finite generative-thrust ceiling. Two tropic controls operate
autonomously: a gravitropic target from the "statolith" model (gain
proportional to local gravity) and an active circumnutation loop around the
local gravity direction. The question asked is how deep a root can bury itself
on a given body, from Europa's low gravity up to `4 g_E`.

## Layout

| Path | Contents |
|------|----------|
| `model/rod.py` | Growing-root rod, granular medium, and parameter tables |
| `run/validate.py` | Closed-form / independent validation targets (V1–V5) |
| `run/control_matrix.py` | Control-mechanism verification matrix (Table 2) |
| `run/variable_gravity.py` | Core gravity sweep over 8 bodies (Table 3) |
| `run/bifurcation.py` | Time-averaged morphology transition scan |
| `run/render_shapes.py` | Figure rendering helpers |
| `results/*.json` | Output tables backing Tables 2–3 and the figures |
| `manuscript/root_paper.tex` | Manuscript source (EJM A/Solids, `elsarticle`) |
| `figures/gravity_trajectories.png` | Manuscript Figure 1 |

## Reproduce

```bash
# validation targets
python run/validate.py

# Table 2: control mechanics (none / gravitropism / circumnutation / both)
python run/control_matrix.py

# Table 3: gravity sweep, Europa gravity .. 4 g_E (three circumnutation seeds)
python run/variable_gravity.py
```

All reported numbers are phase-averaged window means (the metrics are bounded
and do not drift when the tip path precesses). Parameters and all equations are
in the manuscript; `results/` holds the JSON written by the scripts above.

## License

MIT — see `LICENSE`. The manuscript text is licensed separately under the
publisher's terms upon acceptance.