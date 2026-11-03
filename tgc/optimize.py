"""Optimisation loop + final report. Everything is written to data-v2/<study>/."""

from __future__ import annotations

import json
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
import tidy3d as td
import tidy3d.web as web
from autograd import value_and_grad
from tidy3d.plugins.autograd.optimizers import adam, apply_updates

from . import cost
from . import fabrication as fab
from .config import ROOT, Study
from .simulation import Device


class Optimizer:
    def __init__(self, s: Study, beam_function=None):
        self.s = s
        self.folder = s.folder
        (self.folder / "iterations").mkdir(parents=True, exist_ok=True)
        self._check_existing_config()
        s.save()
        self.param = fab.Parametrization(s)
        self.dev = Device(s, beam_function)
        self.history_file = self.folder / "history.pkl"

    # fields that may change between runs of the same study without invalidating it
    _MUTABLE = {"n_iter", "report_num_wavelengths", "exact_source_norm", "remove_islands_at_end"}

    def _check_existing_config(self):
        """Refuse to resume a study whose physics / constraints were edited."""
        from dataclasses import asdict

        f = self.folder / "config.json"
        if not (f.exists() and (self.folder / "history.pkl").exists()):
            return
        old = json.loads(f.read_text())
        new = json.loads(json.dumps(asdict(self.s), default=list))
        # keys missing from an older config.json count as their default value
        dflt = json.loads(json.dumps(asdict(type(self.s)(name=self.s.name)), default=list))
        diff = sorted(
            k for k in set(old) | set(new) if k not in self._MUTABLE and old.get(k, dflt.get(k)) != new.get(k)
        )
        if diff:
            raise RuntimeError(
                f"Study '{self.s.name}' already has results with different {diff}. "
                f"Give the changed study a new name, or delete {self.folder} to restart it."
            )

    # ---------------- utils ----------------
    def log(self, *msg, echo=True):
        line = " ".join(str(m) for m in msg)
        if echo:
            print(line)
        with open(self.folder / "output.log", "a") as f:
            f.write(line + "\n")

    def initial_params(self) -> np.ndarray:
        s = self.s
        if s.init == "random":
            rng = np.random.default_rng(s.random_seed)
            return ndi.gaussian_filter(rng.uniform(0, 1, (s.nx, s.ny)), 2)
        if s.init.startswith("study:"):  # last params of another study (same grid)
            f = self.folder.parent / s.init[len("study:") :] / "history.pkl"
            with open(f, "rb") as fh:
                p = np.array(pickle.load(fh)["params"][-1], float)
            if p.shape != (s.nx, s.ny):
                raise ValueError(f"{f}: params {p.shape} do not match the design grid {(s.nx, s.ny)}")
            return p
        import cv2

        with open(ROOT / s.init, "rb") as f:
            arr = np.asarray(pickle.load(f), float)
        return np.clip(cv2.resize(arr, dsize=(s.ny, s.nx), interpolation=cv2.INTER_CUBIC), 0, 1)

    def load_history(self) -> dict:
        if self.history_file.exists():
            with open(self.history_file, "rb") as f:
                return pickle.load(f)
        return {k: [] for k in ("params", "beta", "objective", "fom", "penalties", "grad_norm", "opt_state")}

    def save_history(self, h: dict) -> None:
        with open(self.history_file, "wb") as f:
            pickle.dump(h, f)

    def plot_density(self, rho, fname, title=""):
        full = np.concatenate((rho[:, ::-1], rho), axis=1)
        s = self.s
        ext = [-s.dr_size_x / 2, s.dr_size_x / 2, -s.dr_size_y / 2, s.dr_size_y / 2]
        fig, ax = plt.subplots(figsize=(4, 4), tight_layout=True)
        ax.imshow(full.T, origin="lower", extent=ext, cmap="gray_r", vmin=0, vmax=1)
        ax.set_title(title)
        ax.set_xlabel("x (um)")
        ax.set_ylabel("y (um)")
        fig.savefig(fname)
        plt.close(fig)

    # ---------------- objective ----------------
    def objective(self, params, beta, info: dict, feature_weight=None):
        rho = self.param.density(params, beta)
        sim = self.dev.simulation(rho)
        data = web.run(sim, task_name=f"{self.s.name}", verbose=False)
        fom = self.dev.efficiency(data)
        pens = self.param.penalties(params, rho, feature_weight)
        info["fom"] = float(getattr(fom, "_value", fom))
        info["penalties"] = {k: float(getattr(v, "_value", v)) for k, v in pens.items()}
        return fom - sum(pens.values())

    # ---------------- cost estimate ----------------
    def _current_params(self, h=None):
        h = h or self.load_history()
        return np.array(h["params"][-1]) if h["params"] else self.initial_params()

    def plan(self, iterations: bool = True, final: bool = True) -> list[dict]:
        """Simulations that `run` (iterations) and `report` (final) would still submit."""
        s, dev = self.s, self.dev
        h = self.load_history()
        params = self._current_params(h)
        remaining = s.n_iter - len(h["params"])
        rows = []
        if iterations and remaining > 0:
            beta = s.beta_schedule[len(h["params"])]
            sim = dev.simulation(self.param.density(params, beta))
            rows.append(dict(label=f"{remaining} iterations (forward+adjoint)", n_sims=2 * remaining, sim=sim))
        if iterations and s.source == "custom" and s.exact_source_norm and not (self.folder / "source_norm.json").exists():
            rows.append(dict(label="source normalisation (empty sim)", n_sims=1, sim=dev.simulation(np.zeros((s.nx, s.ny)))))
        if final and not (self.folder / "final_sim_data.hdf5").exists():
            b = (self.param.density(params, s.beta_max) > 0.5) | self.param.mask
            rows.append(dict(label="final broadband report", n_sims=1, sim=self.final_simulation(b)))
        return rows

    def estimate(self, iterations: bool = True, final: bool = True) -> tuple[str, list]:
        rows = self.plan(iterations, final)
        if not rows:
            return "Nothing to submit (all results cached).", rows
        print(f"Estimating FlexCredits for {len(rows)} simulation type(s) (upload only, not run)...")
        for i, r in enumerate(rows):
            r["credits"] = cost.flex_credits(r["sim"], f"{self.s.name}_{i}")
        head = f"Study '{self.s.name}'  ({rows[0]['sim'].num_cells / 1e6:.1f} M cells, {rows[0]['sim'].num_time_steps} time steps max)"
        return head + "\n" + cost.summarize(rows), rows

    def ask(self, iterations: bool = True, final: bool = True, assume_yes: bool = False) -> bool:
        text, rows = self.estimate(iterations, final)
        if not rows:
            print(text)
            return True
        ok = cost.confirm(text, assume_yes)
        self.log(text + f"\nsubmitted: {ok}", echo=False)
        return ok

    # ---------------- setup figures (also written by `run.py check`) ----------------
    def plot_schedule(self, fname=None):
        s = self.s
        fname = fname or self.folder / "optimization_schedule.png"
        it = np.arange(s.n_iter)
        fig, ax = plt.subplots(figsize=(5, 3.2), tight_layout=True)
        ax.plot(it, s.beta_schedule, "o", ms=3, label="beta")
        ax.set(xlabel="iteration", ylabel="beta")
        ax2 = ax.twinx()
        ax2.plot(it, s.feature_weight_schedule, "s", color="C1", ms=3)
        ax2.set_ylabel("feature penalty weight", color="C1")
        title = "Projection / penalty schedule"
        if s.feature_penalty_trigger_fom is not None:
            title += f"\nramp starts when FoM >= {s.feature_penalty_trigger_fom} (latest case shown)"
        ax.set_title(title, fontsize=9)
        fig.savefig(fname)
        plt.close(fig)

    def plot_initial(self, params=None):
        s = self.s
        params = self.initial_params() if params is None else params
        rho = self.param.density(params, s.beta_schedule[0])
        self.plot_density(rho, self.folder / "init_design.png", "initial")
        self.dev.simulation(rho, unfold=True).plot_eps(z=0)
        plt.savefig(self.folder / "init_sim.png")
        plt.close("all")

    def write_setup(self):
        """Schedule plot and (custom beam) source profile + source_info.json."""
        s = self.s
        self.plot_schedule()
        if s.source == "custom":
            from .source import plot_source

            plot_source(s, fname=self.folder / "source_profile.png")
            plt.close("all")
            (self.folder / "source_info.json").write_text(json.dumps(self.dev.source_info, indent=2))

    # ---------------- loop ----------------
    def run(self, ask: bool = True, assume_yes: bool = False, include_report: bool = True):
        """Optimise (resumes from history). With `ask`, prints the cost / time estimate of
        everything left (iterations and, if `include_report`, the final report) and waits for y/N."""
        s = self.s
        if ask and not self.ask(iterations=True, final=include_report, assume_yes=assume_yes):
            self.log("Cancelled, nothing submitted.")
            return None
        self.write_setup()
        if s.source == "custom":
            self.dev.normalise_source(self.folder, run=s.exact_source_norm)
        betas = s.beta_schedule

        opt = adam(learning_rate=s.learning_rate)
        h = self.load_history()
        if h["params"]:
            params = np.array(h["params"][-1])
            state = h["opt_state"][-1]
            self.log(f"Resuming: {len(h['params'])}/{s.n_iter} iterations done.")
        else:
            params = self.initial_params()
            state = opt.init(params)
            self.plot_initial(params)

        vg = value_and_grad(self.objective)
        ref_sim = self.dev.simulation(self.param.density(params, betas[0]))  # for timing records
        for i in range(len(h["params"]), s.n_iter):
            info = {}
            t0 = time.time()
            w_feat = s.feature_weight(i, h["fom"])
            J, g = vg(params, betas[i], info, w_feat)
            cost.record_timing(ref_sim, (time.time() - t0) / 2, s.name)  # forward + adjoint
            updates, state = opt.update(-g, state, params)
            params = np.clip(apply_updates(params, updates), 0.0, 1.0)

            for k, v in dict(
                params=params.copy(), beta=betas[i], objective=float(J), fom=info["fom"],
                penalties=info["penalties"], grad_norm=float(np.linalg.norm(g)), opt_state=state,
            ).items():
                h[k].append(v)
            self.save_history(h)
            pen_str = " ".join(f"{k}={v:.4f}" for k, v in info["penalties"].items())
            self.log(f"iter {i:3d} | beta {betas[i]:6.2f} | w_feat {w_feat:.3f} | FoM {info['fom']:.4f} | {pen_str} | J {float(J):.4f}")
            self.plot_density(
                self.param.density(params, betas[i]),
                self.folder / "iterations" / f"iteration-{i:03d}.png",
                f"iter {i}, FoM {info['fom']:.3f}",
            )
        return h

    # ---------------- final report ----------------
    def final_design(self, h=None) -> tuple[np.ndarray, dict]:
        """Binarised, island-free density of the last iterate + fabrication report."""
        s = self.s
        h = h or self.load_history()
        rho = self.param.density(np.array(h["params"][-1]), h["beta"][-1])
        b = (rho > 0.5) | self.param.mask
        n_removed = 0
        if s.remove_islands_at_end:
            b, n_removed = fab.remove_islands(b, s)
        rep = fab.feature_size_report(b, s)
        rep["islands_removed"] = n_removed
        rep["connectivity_measure"] = float(self.param.connectivity.measure(b.astype(float)))
        rep.update(self.param.curvature.report(self.param.filtered(np.array(h["params"][-1]))))
        return b, rep

    def plot_fabrication(self, b, rep, fname):
        """Left: min-feature violations (red). Right: boundary radius of curvature."""
        s = self.s
        full = lambda a: np.concatenate((a[:, ::-1], a), axis=1).T
        ext = [-s.dr_size_x / 2, s.dr_size_x / 2, -s.dr_size_y / 2, s.dr_size_y / 2]
        fig, ax = plt.subplots(1, 2, figsize=(10, 4.5), tight_layout=True)
        ax[0].imshow(full(b), origin="lower", extent=ext, cmap="gray_r")
        bad = np.ma.masked_where(~full(rep["solid_violation"] | rep["void_violation"]), np.ones_like(full(b), float))
        ax[0].imshow(bad, origin="lower", extent=ext, cmap="autumn", alpha=0.9)
        ax[0].set_title(
            f"red: < {s.min_feature_size} um features ({rep['solid_violation_px']}+{rep['void_violation_px']} px)", fontsize=9
        )
        im = ax[1].imshow(
            full(rep["radius_map"]), origin="lower", extent=ext, cmap="jet_r", vmin=0, vmax=2 * s.min_curvature_radius
        )
        fig.colorbar(im, ax=ax[1], label="boundary radius of curvature (um)")
        ax[1].set_title(
            f"{100 * rep['boundary_frac_below_min_radius']:.0f} % of boundary with R < {s.min_curvature_radius} um",
            fontsize=9,
        )
        fig.savefig(fname)
        plt.close(fig)

    def final_simulation(self, b) -> td.Simulation:
        s, dev = self.s, self.dev
        mon_xy = td.FieldMonitor(size=(td.inf, td.inf, 0), freqs=[s.freq0], name="field_xy")
        mon_xz = td.FieldMonitor(size=(td.inf, 0, td.inf), freqs=[s.freq0], name="field_xz")
        freqs = td.C_0 / s.report_wavelengths
        return dev.simulation(b.astype(float), unfold=True, monitors=[mon_xy, mon_xz, dev.mode_monitor(freqs, "eff")])

    def report(self, ask: bool = True, assume_yes: bool = False):
        s, dev = self.s, self.dev
        h = self.load_history()
        if not h["params"]:
            raise RuntimeError(f"No iterations in {self.history_file}; run the optimisation first.")
        fp = self.folder / "final_sim_data.hdf5"
        if ask and not fp.exists() and not self.ask(iterations=False, final=True, assume_yes=assume_yes):
            self.log("Cancelled, nothing submitted.")
            return None
        b, rep = self.final_design(h)

        # fabrication figure
        self.plot_fabrication(b, rep, self.folder / "fabrication_check.png")

        sim = self.final_simulation(b)
        if fp.exists():
            data = td.SimulationData.from_hdf5(str(fp))
        else:
            t0 = time.time()
            data = web.run(sim, task_name=f"{s.name}_final", verbose=False)
            cost.record_timing(sim, time.time() - t0, s.name)
            data.to_hdf5(str(fp))

        eff = np.abs(data["eff"].amps.sel(direction="+", mode_index=0).values) ** 2 / dev.power_norm
        fig, ax = plt.subplots(2, 2, figsize=(9, 7), tight_layout=True)
        ax[0, 0].plot(h["fom"], "ro-", ms=4, label="FoM")
        ax[0, 0].plot(h["objective"], "ko-", ms=4, label="objective")
        for k in h["penalties"][0] if h["penalties"] else []:
            ax[0, 0].plot([p[k] for p in h["penalties"]], "o-", ms=4, label=k)
        ax[0, 0].set(xlabel="iteration", title="Learning curve")
        ax[0, 0].legend()
        sim.plot_eps(z=0, source_alpha=0, monitor_alpha=0, ax=ax[0, 1])
        ax[1, 0].plot(s.report_wavelengths, eff, "-k")
        ax[1, 0].set(xlabel="wavelength (um)", ylabel="efficiency", title="Coupling efficiency")
        data.plot_field("field_xy", "E", "abs^2", z=0, ax=ax[1, 1])
        fig.savefig(self.folder / "final_results.png")
        plt.close(fig)

        sim.to_gds_file(
            str(self.folder / "final_design.gds"),
            z=0,
            gds_precision=1e-4,
            gds_cell_name="MAIN",
            permittivity_threshold=(dev.eps_max + dev.eps_min) / 2,
            frequency=s.freq0,
        )
        i0 = int(np.argmin(np.abs(s.report_wavelengths - s.wavelength)))
        summary = {
            "efficiency_at_center": float(eff[i0]),
            "efficiency_max": float(eff.max()),
            "wavelength_at_max": float(s.report_wavelengths[np.argmax(eff)]),
            **self.dev.source_info,
            **{k: v for k, v in rep.items() if not isinstance(v, np.ndarray)},
        }
        if "objective_transmission" in summary:
            # relative to the beam entering the objective. The y-antisymmetric part
            # cannot couple to TE0 of the mirror-symmetric device, so it counts as lost.
            summary["efficiency_end_to_end_at_center"] = (
                summary["efficiency_at_center"]
                * summary["objective_transmission"]
                * summary["window_fraction"]
                * (1 - summary["discarded_by_y_symmetrisation"])
            )
        (self.folder / "summary.json").write_text(json.dumps(summary, indent=2))
        self.log(json.dumps(summary, indent=2))
        return summary
