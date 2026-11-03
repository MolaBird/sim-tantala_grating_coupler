"""Entry point. Edit / add a Study below, then

    python run.py check    <study>  # build everything locally, no cloud credits
    python run.py estimate <study>  # FlexCredit + time estimate only (uploads, does not run)
    python run.py optimize <study>  # estimate -> ask y/N -> run (or resume) + final report
    python run.py report   <study>  # estimate -> ask y/N -> final report only
    add --yes to skip the question

Results go to data-v2/<study>/.
"""

import sys

from tgc.config import Study

STUDIES = {
    # previous setup, with the new tantala data + connectivity / feature-size control
    "gaussian-frame": Study(name="gaussian-frame"),
    # no frame: connectivity is left entirely to the heat-method penalty
    "gaussian-noFrame": Study(name="gaussian-noFrame", masks=("arm",), init="random", beta_min=1, beta_max=30),
    # LMM40X-P01 reflective objective, 4.4 mm y-polarised input beam (lmm40x_focus.py).
    # The focused spot has wide rings: a 7 um half-width source plane holds 88 % of the
    # focused power (simulation box grows from 13.0 x 12.0 to 14.6 x 14.6 um).
    "lmm40x": Study(
        name="lmm40x",
        source="custom",
        beam_function="lmm40x",
        beam_kwargs=dict(focus_depth=0.0),  # um below the membrane top
        source_half_width=7.0,
    ),
    "lmm40x-fromRandom": Study(
        name="lmm40x-fromRandom",
        source="custom",
        beam_function="lmm40x",
        beam_kwargs=dict(focus_depth=0.0),  # um below the membrane top
        source_half_width=7.0,
        init="random",
        masks=("border",),
        min_feature_size=0.15,
        # v2: feature penalty ramps 0.05 -> 1.0 (quadratic), slower beta ramp.
        # v1 (constant weight 1.0, 15 iterations, FoM 0.224) is in data-v2/lmm40x-fromRandom_v1
        feature_penalty_start=0.05,
        feature_penalty_weight=1.0,
        feature_penalty_ramp_power=2.0,
        n_iter=50,
        beta_min=1,
        beta_max=30,
    ),
    # Fab penalty nearly off (0.02) until the FoM first reaches 30 %, then ramps to 1.0
    # (quadratic) over 25 iterations; ~25 more iterations at full weight to repair the
    # design. If 30 % is never reached, the ramp starts at iteration 44 anyway.
    "lmm40x-latePenalty": Study(
        name="lmm40x-latePenalty",
        source="custom",
        beam_function="lmm40x",
        beam_kwargs=dict(focus_depth=0.0),
        source_half_width=7.0,
        init="random",
        masks=("border",),
        min_feature_size=0.15,
        connectivity_weight=0.1,  # kept on throughout: cheap while connected, avoids islands
        feature_penalty_start=0.02,
        feature_penalty_trigger_fom=0.31,
        feature_penalty_ramp_iters=25,
        feature_penalty_ramp_power=2.0,
        feature_penalty_weight=1.0,
        n_iter=70,
        beta_min=1,
        beta_max=30,
    ),
    # Edge smoothing, restarting from the final lmm40x-fromRandom design (its last params,
    # so the filtered design is reproduced exactly). Curvature penalty removes the
    # conic-filter wiggle (boundary radius < 0.3 um); beta continues 30 -> 40.
    "lmm40x-smooth": Study(
        name="lmm40x-smooth",
        source="custom",
        beam_function="lmm40x",
        beam_kwargs=dict(focus_depth=0.0),
        source_half_width=7.0,
        init="study:lmm40x-fromRandom",
        masks=("border",),
        min_feature_size=0.15,
        feature_penalty_weight=1.0,
        connectivity_weight=0.1,
        curvature_weight=0.2,  # starting penalty 0.2 x 0.35 ~ 0.07 vs FoM ~0.32
        min_curvature_radius=0.2,  # scallops are R ~ 0.05-0.15 um
        n_iter=25,
        beta_min=30,
        beta_max=40,
        learning_rate=0.1,  # converged start: smaller steps
    ),
    # latePenalty + delayed beta + edge smoothing. Beta stays at 1 for 15 iterations, then
    # rises quadratically (still ~14 at iteration 55). The fab penalties (feature + curvature,
    # same ramp) wait for FoM >= 0.30 *and* the end of the beta hold, then ramp over 40
    # iterations; the last ~25 iterations run at full weight while beta climbs to 35.
    "lmm40x-latePenalty-smooth": Study(
        name="lmm40x-latePenalty-smooth",
        source="custom",
        beam_function="lmm40x",
        beam_kwargs=dict(focus_depth=0.0),
        source_half_width=7.0,
        init="study:lmm40x-latePenalty-smooth-v1",
        masks=("border",),
        min_feature_size=0.15,
        connectivity_weight=0.1,
        feature_penalty_start=0.02,
        # feature_penalty_weight=1.0,
        feature_penalty_weight=0.5,
        feature_penalty_trigger_fom=0.30,
        feature_penalty_ramp_iters=30,
        feature_penalty_ramp_power=0,
        curvature_weight=0.15,
        min_curvature_radius=0.25,
        n_iter=15,
        beta_min=25,
        beta_max=45,
        beta_hold_iters=0,
        beta_ramp_power=1.0,
    ),
}


def check(s: Study):
    """Offline sanity check: writes every figure that `optimize` writes before its first
    simulation (schedule, initial design, source) plus the masks and the simulation layout."""
    import matplotlib.pyplot as plt
    import numpy as np

    from tgc.materials import check_fit
    from tgc.optimize import Optimizer

    s.folder.mkdir(parents=True, exist_ok=True)
    print(f"tantala fit vs xlsx: max |dn| = {check_fit(s.tantala_medium_file):.2e}")
    opt = Optimizer(s)
    p = opt.initial_params()
    rho = opt.param.density(p, s.beta_schedule[0])
    print("design grid", (s.nx, s.ny), "| penalties at full weight",
          {k: round(float(v), 4) for k, v in opt.param.penalties(p, rho).items()})
    print("connectivity measure (~1-10 connected, >>10 islands):", round(float(opt.param.connectivity.measure(rho)), 3))
    from tgc import fabrication as fab

    b0 = (rho > 0.5) | opt.param.mask
    rep0 = {**fab.feature_size_report(b0, s), **opt.param.curvature.report(opt.param.filtered(p))}
    opt.plot_fabrication(b0, rep0, s.folder / "check_fabrication.png")  # of the initial design
    crv = rep0
    print(f"edges: {100 * crv['boundary_frac_below_min_radius']:.0f} % of boundary with radius < "
          f"{s.min_curvature_radius} um (curvature penalty {crv['curvature_penalty']:.3f}, weight {s.curvature_weight})")

    # schedule
    w = s.feature_weight_schedule
    idx = sorted({0, s.n_iter // 4, s.n_iter // 2, 3 * s.n_iter // 4, s.n_iter - 1})
    print(f"schedule: {s.n_iter} iterations, beta {s.beta_min} -> {s.beta_max}, learning rate {s.learning_rate}")
    print("  feature weight at iteration", {i: round(float(w[i]), 3) for i in idx},
          "(latest case: ramp waits for FoM >= %s)" % s.feature_penalty_trigger_fom
          if s.feature_penalty_trigger_fom is not None else "")
    print(f"  connectivity weight {s.connectivity_weight}, tol {s.connectivity_tol}, min feature {s.min_feature_size} um")

    sim = opt.dev.simulation(rho, unfold=True)
    sim.validate_pre_upload()
    print(f"simulation OK: {sim.num_cells / 1e6:.2f} M cells, {sim.num_time_steps} time steps, size {sim.size}")
    fig, ax = plt.subplots(1, 2, figsize=(9, 4), tight_layout=True)
    sim.plot_eps(z=0, ax=ax[0])
    sim.plot_eps(y=0, ax=ax[1])
    fig.savefig(s.folder / "check_sim.png")
    plt.close(fig)
    opt.plot_density(np.where(opt.param.mask, 1.0, 0.3), s.folder / "check_masks.png", "fixed solid (black)")
    opt.write_setup()  # optimization_schedule.png, source_profile.png, source_info.json
    opt.plot_initial(p)  # init_design.png, init_sim.png
    if s.source == "custom":
        print("source:", {k: round(v, 4) for k, v in opt.dev.source_info.items()})
    print("figures in", s.folder)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    yes = "--yes" in sys.argv
    cmd = args[0] if args else "check"
    name = args[1] if len(args) > 1 else "gaussian-frame"
    if name not in STUDIES:
        sys.exit(f"unknown study '{name}'. Defined: {', '.join(STUDIES)}")
    study = STUDIES[name]
    if cmd == "check":
        check(study)
    elif cmd in ("estimate", "optimize", "report"):
        from tgc.optimize import Optimizer

        opt = Optimizer(study)
        if cmd == "estimate":
            print(opt.estimate(iterations=True, final=True)[0])
        elif cmd == "optimize":
            if opt.run(ask=True, assume_yes=yes) is not None:
                opt.report(ask=False)  # already covered by the estimate above
        else:
            opt.report(ask=True, assume_yes=yes)
    else:
        sys.exit(f"unknown command '{cmd}'")
