"""All knobs of a study live in one dataclass.

Geometry / wavelength defaults are copied from the previous `main.ipynb`
(square design region, arms, waveguide, 2.923 um). Derived quantities
(grid sizes, coordinates, simulation box) are properties so they always stay
consistent with the primary parameters.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "data-v2"


@dataclass
class Study:
    name: str = "default"

    # ---------------- wavelength ----------------
    wavelength: float = 2.923  # central wavelength (um)
    wavelength_width: float = 0.6  # source bandwidth (um)
    fom_wavelengths: tuple = (2.923,)  # wavelengths averaged in the FoM
    report_num_wavelengths: int = 61  # points in the final broadband report
    run_time: float = 1000e-15

    # ---------------- material ----------------
    tantala_medium_file: str = "Tantala_tidy3d_Medium.pkl"  # fitted from Ta2O5_Index_of_refraction.xlsx

    # ---------------- geometry (preserved) ----------------
    thickness: float = 0.8  # membrane thickness (um)
    wg_width: float = 1.7
    wg_length: float = 1.0
    gc_width: float = 8.2
    gc_length: float = 8.2
    border_buffer: float = 0.15
    arm_width: float = 1.0
    arm_separation: float = 2.0
    dr_grid_size: float = 0.03  # design pixel (um)
    min_steps_per_wvl: int = 15

    # ---------------- fixed-solid masks ----------------
    # Any subset of: "border", "arm", "waveguide", or a path to a .pkl 2D array
    # (resized to the design grid; 1 = forced tantala).
    masks: tuple = ("border", "arm")
    arm_buffer: float = 0.25
    wg_buffer_length: float = 0.2
    wg_buffer_width_frac: float = 0.6

    # ---------------- fabrication constraints ----------------
    min_feature_size: float = 0.12  # um; sets the conic filter radius
    feature_penalty_weight: float = 1.0  # erosion/dilation penalty weight (final value)
    # Ramp: weight goes from `feature_penalty_start` (iteration 0) to `feature_penalty_weight`
    # (last iteration) as start + (end - start) * t**power, t = 0..1. None = constant weight.
    feature_penalty_start: float | None = None
    feature_penalty_ramp_power: float = 2.0
    # Optional FoM trigger: hold the weight at `feature_penalty_start` until an iteration
    # reaches FoM >= `feature_penalty_trigger_fom`, then ramp to `feature_penalty_weight`
    # over `feature_penalty_ramp_iters` iterations. If the FoM never gets there, the ramp
    # starts anyway so that it ends on the last iteration.
    feature_penalty_trigger_fom: float | None = None
    feature_penalty_ramp_iters: int = 20
    connectivity_weight: float = 0.1  # heat-method connectivity penalty weight
    # Edge smoothness: penalises boundary pieces whose radius of curvature is below
    # `min_curvature_radius` (um). Removes the "chain of arcs" wiggle left by the conic
    # filter. 0 = off.
    curvature_weight: float = 0.0
    min_curvature_radius: float = 0.3
    connectivity_tol: float = 10.0  # allowed T_pnorm / T_reference before penalising
    remove_islands_at_end: bool = True  # hard guarantee on the final (binarised) design

    # ---------------- source ----------------
    # "gaussian" -> tidy3d GaussianBeam (old behaviour)
    # "custom"   -> CustomFieldSource built from `beam_module:beam_function`
    source: str = "gaussian"
    beam_waist_radius: float = 2.6210
    beam_focus_offset: float = 0.05  # source plane height above the membrane top (um)
    pol_angle: float = float(np.pi / 2)  # E along y -> TE0 of the output waveguide
    beam_function: str = ""  # e.g. "objective_beam:field_on_plane"
    beam_kwargs: dict = field(default_factory=dict)
    beam_backpropagate: bool = True  # field given at membrane top -> shift to source plane
    # half width (um) of the square source plane; 0 -> design region minus border (old).
    # The simulation box grows laterally if needed to hold it.
    source_half_width: float = 0.0
    exact_source_norm: bool = False  # custom source: 1 extra empty run to measure injected power

    # ---------------- optimisation ----------------
    init: str = "seed38-cleaned.pkl"  # path to a .pkl array or "random"
    random_seed: int = 0
    n_iter: int = 15
    beta_min: float = 10.0
    beta_max: float = 30.0
    # Delayed projection: hold beta = beta_min for `beta_hold_iters` iterations, then
    # beta_min + (beta_max - beta_min) * t**beta_ramp_power, t = 0..1 over the rest.
    # (0, 1.0) = the plain linear ramp. While beta is held the fab-penalty trigger waits.
    beta_hold_iters: int = 0
    beta_ramp_power: float = 1.0
    learning_rate: float = 0.2

    # =====================================================================
    # derived quantities
    # =====================================================================
    @property
    def folder(self) -> Path:
        return DATA_ROOT / self.name

    @property
    def freq0(self) -> float:
        import tidy3d as td

        return td.C_0 / self.wavelength

    @property
    def freq_width(self) -> float:
        import tidy3d as td

        lo, hi = self.wavelength - self.wavelength_width / 2, self.wavelength + self.wavelength_width / 2
        return 0.5 * (td.C_0 / lo - td.C_0 / hi)

    @property
    def fom_freqs(self) -> list:
        import tidy3d as td

        return [td.C_0 / wl for wl in self.fom_wavelengths]

    @property
    def report_wavelengths(self) -> np.ndarray:
        w = self.wavelength_width / 2
        return np.linspace(self.wavelength - w, self.wavelength + w, self.report_num_wavelengths)

    @property
    def beta_schedule(self) -> np.ndarray:
        i = np.arange(self.n_iter)
        t = np.clip((i - self.beta_hold_iters) / max(self.n_iter - 1 - self.beta_hold_iters, 1), 0.0, 1.0)
        return self.beta_min + (self.beta_max - self.beta_min) * t**self.beta_ramp_power

    @property
    def feature_weight_schedule(self) -> np.ndarray:
        """Planned weights (with a FoM trigger: the latest possible ramp)."""
        return np.array([self.feature_weight(i) for i in range(self.n_iter)])

    def feature_weight(self, i: int, foms=()) -> float:
        """Feature-penalty weight at iteration i; `foms` = FoM of iterations 0..i-1."""
        end, start = self.feature_penalty_weight, self.feature_penalty_start
        if start is None:
            return end
        if self.feature_penalty_trigger_fom is None:
            t = i / max(self.n_iter - 1, 1)
        else:
            hit = [j for j, f in enumerate(foms) if f >= self.feature_penalty_trigger_fom]
            k = hit[0] if hit else self.n_iter - 1 - self.feature_penalty_ramp_iters  # ramp starts after k
            k = max(k, self.beta_hold_iters - 1)  # not while beta is held
            t = min(max((i - k) / self.feature_penalty_ramp_iters, 0.0), 1.0)
        return float(start + (end - start) * t**self.feature_penalty_ramp_power)

    # design region (only y >= 0 half is parametrised, mirrored about y = 0)
    @property
    def nx(self) -> int:
        return int((self.gc_length + 2 * self.border_buffer) / self.dr_grid_size)

    @property
    def ny(self) -> int:
        return int((self.gc_width + 2 * self.border_buffer) / self.dr_grid_size / 2.0)

    @property
    def dr_size_x(self) -> float:
        return self.nx * self.dr_grid_size

    @property
    def dr_size_y(self) -> float:
        return 2 * self.ny * self.dr_grid_size

    @cached_property
    def xy_half(self) -> tuple[np.ndarray, np.ndarray]:
        """Pixel-centre coordinates of the parametrised half (indexing='ij')."""
        dl = self.dr_grid_size
        x = -self.dr_size_x / 2 + (np.arange(self.nx) + 0.5) * dl
        y = (np.arange(self.ny) + 0.5) * dl
        return np.meshgrid(x, y, indexing="ij")

    # simulation box
    @property
    def pml_spacing(self) -> float:
        return 0.6 * self.wavelength

    @property
    def source_plane_size(self) -> tuple:
        if self.source_half_width > 0:
            return (2 * self.source_half_width, 2 * self.source_half_width, 0)
        return (self.dr_size_x - 2 * self.border_buffer, self.dr_size_y - 2 * self.border_buffer, 0)

    @property
    def _sim_bounds_xy(self) -> tuple:
        """(xmin, xmax, ymax) of the simulation box (symmetric in y)."""
        xmin = -self.dr_size_x / 2 - self.pml_spacing
        xmax = self.mon_x + 0.25 * self.wg_length + self.pml_spacing
        ymax = self.dr_size_y / 2 + self.pml_spacing
        margin = 0.3  # keep the source plane away from the PML
        hx, hy = self.source_plane_size[0] / 2 + margin, self.source_plane_size[1] / 2 + margin
        return min(xmin, -hx), max(xmax, hx), max(ymax, hy)

    @property
    def sim_size(self) -> tuple:
        xmin, xmax, ymax = self._sim_bounds_xy
        return (xmax - xmin, 2 * ymax, self.thickness + 2 * self.pml_spacing)

    @property
    def sim_center(self) -> tuple:
        xmin, xmax, _ = self._sim_bounds_xy
        return ((xmin + xmax) / 2, 0.0, 0.0)

    @property
    def source_z(self) -> float:
        return self.thickness / 2 + self.beam_focus_offset

    @property
    def mon_x(self) -> float:
        """Mode monitor in the waveguide; kept 0.5 um beyond a wide source plane."""
        return max(self.dr_size_x / 2 + 0.75 * self.wg_length, self.source_plane_size[0] / 2 + 0.5)

    @property
    def mon_size(self) -> tuple:
        dl = self.dr_grid_size
        return (0, int(3 * self.wg_width / dl) * dl, int(5 * self.thickness / dl) * dl)

    # ---------------- io ----------------
    def save(self) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        d = {k: v for k, v in asdict(self).items()}
        (self.folder / "config.json").write_text(json.dumps(d, indent=2, default=list))

    @classmethod
    def load(cls, name: str) -> "Study":
        d = json.loads((DATA_ROOT / name / "config.json").read_text())
        for k in ("fom_wavelengths", "masks"):
            d[k] = tuple(d[k])
        return cls(**d)
