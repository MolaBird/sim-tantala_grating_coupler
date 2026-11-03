"""tidy3d simulation of the grating coupler (geometry unchanged from before)."""

from __future__ import annotations

import json

import autograd.numpy as anp
import numpy as np
import tidy3d as td

from .config import Study
from .materials import n_tantala, tantala
from .source import make_source

FOM_MONITOR = "fom"
INF = 1000.0


class Device:
    """Builds simulations for a study. One instance per study."""

    def __init__(self, s: Study, beam_function=None):
        self.s = s
        self.n0 = n_tantala(s.wavelength, s.tantala_medium_file)
        self.eps_min, self.eps_max = 1.0, self.n0**2
        self.medium = tantala(s.tantala_medium_file)
        self.source, self.source_info = make_source(s, beam_function)
        self.power_norm = 1.0  # set by `normalise_source`

    # ---------------- fixed structures ----------------
    def static_structures(self) -> list:
        s, t = self.s, self.s.thickness
        x0, x1 = -s.dr_size_x / 2, s.dr_size_x / 2
        y0 = s.arm_separation / 2
        boxes = [
            ((x1, -s.wg_width / 2), (INF, s.wg_width / 2)),  # output waveguide
            ((-INF, y0), (x0, y0 + s.arm_width)),  # arms
            ((-INF, -y0 - s.arm_width), (x0, -y0)),
        ]
        return [
            td.Structure(geometry=td.Box.from_bounds((a[0], a[1], -t / 2), (b[0], b[1], t / 2)), medium=self.medium)
            for a, b in boxes
        ]

    # ---------------- design region ----------------
    def design_structure(self, rho, unfold: bool = False) -> td.Structure:
        """rho: (nx, ny) density on the y >= 0 half."""
        s = self.s
        eps = self.eps_min + (self.eps_max - self.eps_min) * rho
        xx, yy = s.xy_half
        x, y = xx[:, 0], yy[0]
        if unfold:
            eps = anp.concatenate((eps[:, ::-1], eps), axis=1)
            y = np.concatenate((-y[::-1], y))
            center_y, size_y = 0.0, s.dr_size_y
        else:
            center_y, size_y = s.dr_size_y / 4, s.dr_size_y / 2
        eps = anp.reshape(eps, (len(x), len(y), 1))
        perm = td.SpatialDataArray(eps, coords=dict(x=x, y=y, z=[0.0]))
        box = td.Box(center=(0, center_y, 0), size=(s.dr_size_x, size_y, s.thickness))
        return td.Structure(geometry=box, medium=td.CustomMedium(permittivity=perm))

    # ---------------- monitors ----------------
    def mode_monitor(self, freqs, name=FOM_MONITOR) -> td.ModeMonitor:
        s = self.s
        return td.ModeMonitor(
            center=(s.mon_x, 0, 0),
            size=s.mon_size,
            freqs=list(freqs),
            mode_spec=td.ModeSpec(num_modes=1, target_neff=self.n0),
            name=name,
        )

    # ---------------- simulation ----------------
    def grid_spec(self) -> td.GridSpec:
        s = self.s
        mesh = td.MeshOverrideStructure(
            geometry=td.Box(center=(0, 0, 0), size=(s.dr_size_x, s.dr_size_y, s.thickness)),
            dl=[s.dr_grid_size] * 3,
            enforce=True,
        )
        return td.GridSpec.auto(
            wavelength=s.wavelength + s.wavelength_width / 2,
            min_steps_per_wvl=s.min_steps_per_wvl,
            override_structures=[mesh],
        )

    def simulation(self, rho, unfold=False, monitors=None) -> td.Simulation:
        s = self.s
        return td.Simulation(
            size=s.sim_size,
            center=s.sim_center,
            grid_spec=self.grid_spec(),
            symmetry=(0, -1, 0),
            structures=self.static_structures() + [self.design_structure(rho, unfold)],
            sources=[self.source],
            monitors=monitors if monitors is not None else [self.mode_monitor(s.fom_freqs)],
            run_time=s.run_time,
            subpixel=True,
        )

    def efficiency(self, sim_data, name=FOM_MONITOR):
        """|a+|^2 of the fundamental mode, averaged over the monitor frequencies."""
        amps = sim_data[name].amps.sel(direction="+", mode_index=0).values
        return anp.mean(anp.abs(amps) ** 2) / self.power_norm

    # ---------------- optional exact source normalisation ----------------
    def normalise_source(self, folder, run=False):
        """The custom source is normalised analytically to 1 W. For an exact value
        (as recommended by tidy3d for CustomFieldSource) an empty simulation on the
        same grid measures the injected flux. Cached in `<folder>/source_norm.json`.
        """
        f = folder / "source_norm.json"
        if f.exists():
            self.power_norm = json.loads(f.read_text())["power"]
            return self.power_norm
        if not run:
            return self.power_norm
        import tidy3d.web as web

        s = self.s
        sim = self.simulation(np.zeros((s.nx, s.ny)))
        grid = sim.grid.boundaries
        z_mon = s.source_z - 3 * s.dr_grid_size
        mon = td.FluxMonitor(
            center=(0, 0, z_mon), size=(td.inf, td.inf, 0), freqs=[s.freq0], normal_dir="-", name="flux"
        )
        empty = sim.updated_copy(
            structures=[],
            monitors=[mon],
            grid_spec=td.GridSpec(
                grid_x=td.CustomGridBoundaries(coords=grid.x),
                grid_y=td.CustomGridBoundaries(coords=grid.y),
                grid_z=td.CustomGridBoundaries(coords=grid.z),
            ),
        )
        data = web.run(empty, task_name=f"{s.name}_source_norm", verbose=False)
        self.power_norm = float(np.abs(data["flux"].flux.values[0]))
        f.write_text(json.dumps({"power": self.power_norm}))
        return self.power_norm
