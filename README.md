# Suspended tantala grating coupler: inverse design (v2)

```
run.py              studies + entry point (check / optimize / report)
tgc/
  config.py         Study dataclass: every parameter, derived geometry
  materials.py      tantala from Tantala_tidy3d_Medium.pkl (fit of Ta2O5_Index_of_refraction.xlsx)
  fabrication.py    masks, filter+projection, connectivity (heat method), island removal, feature check
  source.py         GaussianBeam or CustomFieldSource from your wavefront
  simulation.py     tidy3d structures / monitors / Simulation (geometry unchanged)
  optimize.py       Adam loop with checkpointing, final report
data-v2/<study>/    all outputs of one study
```

## Usage

```bash
python run.py check    lmm40x         # local only: builds + validates, writes check_*.png
python run.py estimate lmm40x         # FlexCredit + time estimate (uploads, prices, deletes; runs nothing)
python run.py optimize lmm40x         # estimate -> "Submit to Tidy3D? [y/N]" -> run / resume + final report
python run.py report   lmm40x         # final report only (asks only if the final sim is not cached)
python run.py optimize lmm40x --yes   # skip the question
```

**Cost / time estimate.** Before anything is submitted, each distinct simulation is uploaded to the tidy3d folder
`tgc-estimates`, priced, and deleted. The table shows the remaining iterations (forward + adjoint, with the adjoint
assumed to cost the same as the forward), the final report and, if enabled, the source-normalisation run. tidy3d
gives a *maximum* (full `run_time`). Runs usually stop early, so the real charge is lower. tidy3d has no time
estimate, so every finished iteration records its wall time per cell x time step in `data-v2/_timings.json`. Later
estimates scale from those records. Until the first iteration is done a rough 4 min/sim default is shown, and the
table says so. Without an interactive terminal nothing is submitted unless `--yes` is given.

## New study

Add one entry to `STUDIES` in `run.py`. The key is what you type on the command line, and `name` is the output
folder `data-v2/<name>/`:

```python
"lmm40x-mfs150": Study(
    name="lmm40x-mfs150",
    source="custom", beam_function="lmm40x", source_half_width=7.0,
    beam_kwargs=dict(focus_depth=0.4),      # focus at mid-membrane
    min_feature_size=0.15, connectivity_weight=0.2,
    n_iter=20, beta_min=10, beta_max=40,
),
```

Then `python run.py check lmm40x-mfs150` and `python run.py optimize lmm40x-mfs150`. Every parameter, with
its default, is in `tgc/config.py` (`Study`). Anything not given keeps its default.
* Re-running `optimize` on the same study resumes from `history.pkl`. You may raise `n_iter` to continue. Changing
  anything physical (geometry, constraints, source, schedule...) on a study that already has results is refused:
  give it a new name, or delete its folder to restart.
* To start from an earlier result, set `init="study:<other study>"` (its last parameters, same grid), or a saved
  array (`.pkl`, any size, resized to the grid). Start beta near the other study's final beta.
* From a notebook: `from run import STUDIES; from tgc.optimize import Optimizer;
  opt = Optimizer(STUDIES["lmm40x-mfs150"]); opt.run(); opt.report(ask=False)`. The prompt appears as an input box.
* `Study.load("lmm40x")` rebuilds a study from its saved `config.json`.

Output of `data-v2/<study>/`:
`config.json`, `output.log`, `history.pkl` (checkpoint), `optimization_schedule.png`,
`init_design.png`, `iterations/iteration-XXX.png`, `source_profile.png` + `source_info.json` (custom beam),
`fabrication_check.png`, `final_results.png`, `final_design.gds`, `final_sim_data.hdf5`, `summary.json`.

## What changed vs. the notebooks

**Material.** The arms and waveguide use the dispersive `PoleResidue` fit (n = 2.0235 at 2.923 um; matches the xlsx
to 5e-9). The design region interpolates between air and n(lambda0)^2 (autograd needs a non-dispersive custom medium;
n changes by < 0.2 % over the band). The mode monitor target index uses the same n.

**Fabrication.** `params -> masks -> conic filter (R = min_feature_size) + tanh projection -> masks -> rho`.
* Masks: `masks=("border", "arm", "waveguide", "some_mask.pkl")`, any subset. They now force rho = 1 (the old code set
  eps_max ~ 4.1 into a [0, 1] parameter).
* Minimum feature size: the filter, plus tidy3d's erosion/dilation penalty (`feature_penalty_weight`).
  The weight can ramp up over the run (`feature_penalty_start` -> `feature_penalty_weight`, shape
  `t**feature_penalty_ramp_power`), so early iterations optimise coupling instead of just smoothing a random start.
  With `feature_penalty_trigger_fom` the ramp waits until the FoM first reaches that value, then takes
  `feature_penalty_ramp_iters` iterations (it starts anyway late enough to finish by the last iteration).
  `fabrication_check.png` / `summary.json` report the pixels that still violate the minimum solid/void width
  (morphological open/close with a disk of diameter `min_feature_size`).
* Projection schedule: `beta_hold_iters` keeps beta at `beta_min` for the first iterations, then
  `beta_ramp_power` shapes the rise to `beta_max` (0 and 1 = the plain linear ramp). The fab-penalty trigger never
  fires while beta is held, and the curvature penalty follows the same ramp as the feature penalty.
* Edge smoothness (`curvature_weight`, `min_curvature_radius`): the conic filter leaves boundaries that are chains of
  ~R_filter arcs (R ~ 0.05-0.15 um). The penalty measures the curvature of the boundary as the 0.5 level set of the
  conic-filtered field (before projection: smooth, exact boundary, gradient goes straight to the parameters; within
  4 % on disks) and charges every piece with radius < `min_curvature_radius` (0.2 um is a good start). Straight edges and gentle
  arcs cost nothing. `fabrication_check.png` (and `check_fabrication.png` from `check`) maps the boundary radius.
* Connectivity: heat method. Tantala makes heat and conducts it (k ~ rho^3, air k = 1e-8). The arm and waveguide
  interfaces are held at T = 0. A p-norm of T over solid, divided by T of an all-solid design, is ~3-5 for connected
  designs and > 10^3 with a floating island. The penalty `connectivity_weight * log10(max(measure/connectivity_tol, 1))`
  is zero for connected designs. The sparse solve has an exact adjoint (gradient checked against finite differences).
* Hard guarantee: before the final simulation and GDS export, every solid component that does not touch an arm or
  the waveguide is removed (`remove_islands_at_end`). The final efficiency is computed on that cleaned design.

**Source** (`source="custom"`). Your module exposes

```python
def field_on_plane(x, y, wavelength, **beam_kwargs):  # x, y 1D (um), centred on the coupler
    return Ex, Ey                                      # complex, shape (len(x), len(y)), at z = membrane top
```

Set `beam_function="your_module:field_on_plane"` and `beam_kwargs` in the Study (the module must be importable from
the repo root). Conventions: tidy3d `exp(-i w t)`, beam travelling -z. A focus a distance R below the plane has
phase `exp(-1j*k*(x**2+y**2)/(2R))`. `from_intensity_phase(I, phi, pol_angle)` helps if you have a scalar field.
The code then:
1. samples your field over the old source plane (8.2 x 8.2 um),
2. symmetrises it about y = 0 (the sim uses a y-mirror; it warns if the input is > 5 % asymmetric),
3. uses an angular-spectrum decomposition to move it up by `beam_focus_offset` to the source plane, compute Ez and
   the matching H = k x E / (w mu0), so the source radiates only downward,
4. normalises it to 1 W and builds a `td.CustomFieldSource`.

This was checked against tidy3d's own `GaussianBeamProfile`: flux within 0.1 %, propagated field within 1 %,
H within 3.6 % (tidy3d's analytic beam is paraxial, H = E/eta). For an exact power normalisation set
`exact_source_norm=True`: one extra empty simulation on the same grid, cached in `source_norm.json`.

**LMM40X objective beam** (`run.py` study `lmm40x`, `beam_function="lmm40x"`). This wraps `lmm40x_focus.py`
with y polarisation and a 4.4 mm beam. `beam_kwargs=dict(focus_depth=...)` sets how far (um) the best focus sits
below the membrane top. The objective frame (+z, metres, (ny, nx) arrays) is rotated 180 deg about x into the
simulation frame (-z). Checked: shifting the module's field 1 um with the angular spectrum matches the module's
own defocus to 0.3 %, against 9 % with the opposite sign.
* The spot has wide rings. The study uses `source_half_width=7.0`, which holds 88 % of the focused power and ends
  near a ring minimum (edge intensity 0.3 % of peak). The old 8.2 um plane held only 64 %. The simulation box and the
  mode monitor move out to fit it (14.6 um in y, monitor at x = 7.5 um, +15 % cells). The device is unchanged.
* `source_info.json` / `summary.json` record `objective_transmission` (0.435 of the 4.4 mm beam reaches the focus),
  `window_fraction` (0.877) and `discarded_by_y_symmetrisation` (0.013, spider asymmetry). The FoM is relative to
  the injected power. `efficiency_end_to_end_at_center` multiplies these factors in, so it is relative to the beam
  entering the objective.

**Bug fixes.** History entries no longer all share the same `params` array (it was updated in place). Resuming
from a checkpoint works (the old notebook loop never defined `params` / `opt_state` on resume). The design permittivity
samples are now at pixel centres.
