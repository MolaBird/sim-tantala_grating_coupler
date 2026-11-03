"""Input beam.

Two options (``Study.source``):

* ``"gaussian"``: the analytic ``td.GaussianBeam`` used before.
* ``"custom"``: a ``td.CustomFieldSource`` built from an arbitrary wavefront.

Custom wavefront interface
--------------------------
Your wave-optics module provides a function (named in ``Study.beam_function``
as ``"module:function"``, or passed as a callable)::

    def field_on_plane(x, y, wavelength, **beam_kwargs):
        # x, y: 1D arrays (um), centred on the grating coupler
        # returns complex Ex, Ey of shape (len(x), len(y)) in the plane of
        # the membrane top surface (z = thickness / 2), beam travelling -z
        return Ex, Ey            # or (Ex, Ey, info_dict) to log extra numbers

Built in: ``beam_function="lmm40x"`` -> Thorlabs LMM40X-P01 objective focus
(``lmm40x_focus.py``, y polarisation, 4.4 mm input beam), ``"gaussian"``.

Phase convention is tidy3d's exp(-i w t), i.e. a plane wave is
exp(i (k.r - w t)). A beam converging to a focus a distance R *below* the
plane therefore has phase ``exp(-1j * k * (x**2 + y**2) / (2 * R))``.
If you only have intensity / phase of a scalar field use
``from_intensity_phase``. Only relative amplitude matters: the profile is
normalised here to 1 W.

What happens here:
1. sample the user field on a symmetric grid over the source plane,
2. (optional) symmetrise about y = 0 (the simulation uses a y-mirror),
3. angular-spectrum decomposition -> shift the field from the membrane top
   to the source plane, remove evanescent parts, compute Ez and the
   matching H = k x E / (w mu0) so the source is uni-directional,
4. normalise to 1 W downward flux and wrap into a ``CustomFieldSource``.
"""

from __future__ import annotations

import importlib
from typing import Callable

import numpy as np
import tidy3d as td

from .config import Study

# =====================================================================
# helpers for the user
# =====================================================================


def from_intensity_phase(intensity, phase, pol_angle=np.pi / 2):
    """Scalar (intensity, phase) -> (Ex, Ey) with linear polarisation angle from x."""
    u = np.sqrt(np.asarray(intensity)) * np.exp(1j * np.asarray(phase))
    return np.cos(pol_angle) * u, np.sin(pol_angle) * u


def gaussian_field(x, y, wavelength, waist_radius=2.621, focus_depth=0.0, pol_angle=np.pi / 2):
    """Reference beam: Gaussian whose waist is `focus_depth` below the plane (paraxial)."""
    k = 2 * np.pi / wavelength
    zr = np.pi * waist_radius**2 / wavelength
    q = -focus_depth - 1j * zr  # z measured along propagation (-z): plane is at -focus_depth
    X, Y = np.meshgrid(x, y, indexing="ij")
    u = (1 / q) * np.exp(1j * k * (X**2 + Y**2) / (2 * q))
    u = u / np.max(np.abs(u))
    return np.cos(pol_angle) * u, np.sin(pol_angle) * u


_LMM40X = None


def lmm40x_field(x, y, wavelength, focus_depth=0.0, beam_diameter=4.4e3, polarization=90.0, pupil_samples=301):
    """LMM40X-P01 reflective objective focus (lmm40x_focus.py), in simulation coordinates.

    focus_depth: distance (um) of the best focus *below* the membrane top surface.
    beam_diameter: 1/e^2 diameter of the collimated input beam (um), 4.4 mm.
    polarization: input polarisation angle from x (deg), 90 = y.

    lmm40x_focus works in metres with light along +z and arrays (ny, nx). The
    simulation beam travels -z, i.e. the objective frame rotated by 180 deg about
    x: (x, y, z)_sim = (x, -y, -z)_obj, so Ey_sim(x, y) = -Ey_obj(x, -y).
    """
    global _LMM40X
    import sys

    from .config import ROOT

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from lmm40x_focus import LMM40X

    if _LMM40X is None or _LMM40X[0] != pupil_samples:
        _LMM40X = (pupil_samples, LMM40X(pupil_samples=pupil_samples))
    obj = _LMM40X[1]
    um = 1e-6
    f = obj.focus(
        wavelength * um,
        beam_diameter * um,
        polarization=polarization,
        x=np.asarray(x) * um,
        y=-np.asarray(y) * um,  # y_obj = -y_sim
        z=-focus_depth * um,  # objective z: + = along propagation
    )
    Ex, Ey = f.Ex.T, -f.Ey.T  # (ny, nx) -> (nx, ny), rotate about x
    dA = (x[1] - x[0]) * (y[1] - y[0]) * um**2
    info = dict(
        objective_transmission=float(f.transmission),  # input power reaching the focus
        window_fraction=float(f.intensity.sum() * dA / f.transmission),  # of that, inside the source plane
    )
    return Ex, Ey, info


def _resolve(beam_function) -> Callable:
    if callable(beam_function):
        return beam_function
    if beam_function in ("", "gaussian"):
        return gaussian_field
    if beam_function == "lmm40x":
        return lmm40x_field
    mod, fn = beam_function.split(":")
    return getattr(importlib.import_module(mod), fn)


# =====================================================================
# E -> (E, H) on the source plane
# =====================================================================


def plane_wave_fields(x, y, Ex, Ey, wavelength, dz=0.0, n=1.0):
    """Angular spectrum of a -z travelling beam.

    Returns dict Ex, Ey, Ez, Hx, Hy, Hz on the same grid, with the field
    translated by `dz` along +z (i.e. *against* the propagation, dz > 0 moves
    the plane up). Evanescent components are discarded.
    """
    nx, ny = Ex.shape
    dx, dy = x[1] - x[0], y[1] - y[0]
    px, py = nx, ny  # zero padding to avoid wrap-around
    sh = (nx + 2 * px, ny + 2 * py)
    kx = 2 * np.pi * np.fft.fftfreq(sh[0], dx)
    ky = 2 * np.pi * np.fft.fftfreq(sh[1], dy)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    k = 2 * np.pi * n / wavelength
    kz2 = k**2 - KX**2 - KY**2
    prop = kz2 > 0
    kz = np.sqrt(np.where(prop, kz2, 1.0))  # propagation constant towards -z (> 0)

    def spec(a):
        a = np.pad(a, ((px, px), (py, py)))
        # account for the grid origin so that phases refer to physical x, y
        return np.fft.fft2(np.fft.ifftshift(a))

    # field ~ exp(i (kx x + ky y - kz z)); moving up by dz -> * exp(-i kz dz)
    shift = np.where(prop, np.exp(-1j * kz * dz), 0.0)
    Axs, Ays = spec(Ex) * shift, spec(Ey) * shift
    Azs = (KX * Axs + KY * Ays) / kz  # k . E = 0 with k = (kx, ky, -kz)
    omega_mu = 2 * np.pi * td.C_0 / wavelength * td.MU_0
    Hxs = (KY * Azs + kz * Ays) / omega_mu  # H = k x E / (w mu0)
    Hys = (-kz * Axs - KX * Azs) / omega_mu
    Hzs = (KX * Ays - KY * Axs) / omega_mu

    def back(a):
        a = np.fft.fftshift(np.fft.ifft2(a))
        return a[px : px + nx, py : py + ny]

    return {c: back(a) for c, a in zip(("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"), (Axs, Ays, Azs, Hxs, Hys, Hzs))}


def flux_z(fields, x, y) -> float:
    """0.5 Re int (E x H*)_z dA (tidy3d convention). Negative = downward."""
    Sz = 0.5 * np.real(fields["Ex"] * np.conj(fields["Hy"]) - fields["Ey"] * np.conj(fields["Hx"]))
    return float(np.sum(Sz) * (x[1] - x[0]) * (y[1] - y[0]))


# =====================================================================
# sources
# =====================================================================


def source_plane_size(s: Study) -> tuple:
    return s.source_plane_size


def custom_fields(s: Study, beam_function=None, resolution=None):
    """Sample, symmetrise, propagate and normalise the user beam -> (x, y, fields, info)."""
    fn = _resolve(beam_function if beam_function is not None else s.beam_function)
    Lx, Ly, _ = source_plane_size(s)
    dl = resolution or min(s.dr_grid_size, s.wavelength / 20)
    nxp, nyp = int(np.ceil(Lx / dl / 2)) * 2 + 1, int(np.ceil(Ly / dl / 2)) * 2 + 1
    x = np.linspace(-Lx / 2, Lx / 2, nxp)  # odd count -> contains 0, symmetric
    y = np.linspace(-Ly / 2, Ly / 2, nyp)
    out = fn(x, y, s.wavelength, **s.beam_kwargs)
    info = dict(out[2]) if len(out) > 2 else {}
    Ex, Ey = np.asarray(out[0], complex), np.asarray(out[1], complex)
    if Ex.shape != (nxp, nyp):
        raise ValueError(f"beam function returned {Ex.shape}, expected {(nxp, nyp)}")

    # y-mirror of the simulation (symmetry = -1 on y): Ey even, Ex odd in y
    Ey_s, Ex_s = 0.5 * (Ey + Ey[:, ::-1]), 0.5 * (Ex - Ex[:, ::-1])
    p_tot = np.sum(np.abs(Ex) ** 2 + np.abs(Ey) ** 2)
    asym = float(1 - np.sum(np.abs(Ex_s) ** 2 + np.abs(Ey_s) ** 2) / p_tot)
    info["discarded_by_y_symmetrisation"] = asym  # power fraction
    if asym > 0.05:
        td.log.warning(f"{asym:.1%} of the beam power is y-antisymmetric and is dropped (simulation uses a y-mirror).")
    Ex, Ey = Ex_s, Ey_s

    dz = s.beam_focus_offset if s.beam_backpropagate else 0.0
    f = plane_wave_fields(x, y, Ex, Ey, s.wavelength, dz=dz)
    P = flux_z(f, x, y)
    if P >= 0:
        raise RuntimeError("Computed flux is not downward; check the beam phase convention.")
    f = {c: a / np.sqrt(-P) for c, a in f.items()}
    return x, y, f, info


def make_source(s: Study, beam_function=None) -> tuple[td.Source, dict]:
    """Returns the source and a dict of beam numbers (empty for the Gaussian)."""
    pulse = td.GaussianPulse(freq0=s.freq0, fwidth=s.freq_width)
    center = (0.0, 0.0, s.source_z)
    size = source_plane_size(s)
    if s.source == "gaussian":
        return td.GaussianBeam(
            center=center,
            size=size,
            source_time=pulse,
            pol_angle=s.pol_angle,
            direction="-",
            num_freqs=7,
            waist_radius=s.beam_waist_radius,
        ), {}
    if s.source != "custom":
        raise ValueError(f"unknown source '{s.source}'")

    x, y, f, info = custom_fields(s, beam_function)
    coords = dict(x=x, y=y, z=[0.0], f=[s.freq0])
    data = {c: td.ScalarFieldDataArray(f[c][:, :, None, None], coords=coords) for c in ("Ex", "Ey", "Hx", "Hy")}
    return td.CustomFieldSource(
        center=center,
        size=size,
        source_time=pulse,
        field_dataset=td.FieldDataset(**data),
        name="custom_beam",
    ), info


def plot_source(s: Study, beam_function=None, fname=None):
    import matplotlib.pyplot as plt

    x, y, f, _ = custom_fields(s, beam_function)
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.2), tight_layout=True)
    ext = [x[0], x[-1], y[0], y[-1]]
    I = np.abs(f["Ex"]) ** 2 + np.abs(f["Ey"]) ** 2
    for a, img, title, cmap in (
        (ax[0], I, "|E|^2 at source plane", "magma"),
        (ax[1], np.angle(f["Ey"]) * (np.abs(f["Ey"]) > 0.05 * np.abs(f["Ey"]).max()), "arg Ey", "twilight"),
        (ax[2], np.abs(f["Ez"]) ** 2, "|Ez|^2", "magma"),
    ):
        im = a.imshow(img.T, origin="lower", extent=ext, cmap=cmap)
        a.set_title(title)
        a.set_xlabel("x (um)")
        fig.colorbar(im, ax=a)
    ax[0].set_ylabel("y (um)")
    if fname:
        fig.savefig(fname)
    return fig
