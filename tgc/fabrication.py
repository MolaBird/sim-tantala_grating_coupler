"""Fabrication constraints on the design density.

Pipeline (all differentiable with autograd):

    params --masks--> --conic filter + tanh projection--> --masks--> rho

* **Fixed-solid masks**: border frame, arm / waveguide interfaces, or any
  user .pkl mask (kept from the previous code).
* **Minimum feature size**: conic filter of radius `min_feature_size` +
  tanh projection, plus tidy3d's erosion/dilation penalty, which is zero only
  when opening/closing with that radius leaves the design unchanged.
* **Connectivity** (virtual-temperature / heat method, Li et al. 2016):
  solid generates heat, material conducts heat ~ rho^3, the arms and the
  waveguide are held at T = 0. Any tantala island that is not connected to an
  anchor can only lose heat through air (k_min) and gets very hot. The
  penalty is a p-norm of the temperature over solid, normalised to a fully
  solid design. The sparse solve has a hand-written adjoint.
* **Edge smoothness** (`CurvaturePenalty`): curvature of the boundary, taken
  as the 0.5 level set of the filtered (pre-projection) field. Boundary
  pieces with radius < `min_curvature_radius` are penalised; straight edges and
  gentle arcs cost nothing.
* **Hard guarantee at the end**: `remove_islands` drops every binary
  component not touching an anchor; `feature_size_report` measures what is
  left.
"""

from __future__ import annotations

import pickle

import autograd.numpy as anp
import numpy as np
import scipy.ndimage as ndi
import scipy.sparse as sps
import scipy.sparse.linalg as spla
from autograd.extend import defvjp, primitive
from tidy3d.plugins.autograd import make_conic_filter, make_erosion_dilation_penalty, make_filter_and_project

from .config import ROOT, Study

# =====================================================================
# masks
# =====================================================================


def _load_mask_pkl(path, nx, ny) -> np.ndarray:
    import cv2

    p = ROOT / path
    with open(p, "rb") as f:
        arr = np.asarray(pickle.load(f), dtype=float)
    return cv2.resize(arr, dsize=(ny, nx), interpolation=cv2.INTER_CUBIC) > 0.5


def fixed_solid_mask(s: Study) -> np.ndarray:
    """Boolean (nx, ny) array of pixels forced to be tantala."""
    xx, yy = s.xy_half
    Lx, Ly = s.dr_size_x, s.dr_size_y
    m = np.zeros((s.nx, s.ny), bool)
    for name in s.masks:
        if name == "border":
            m |= (np.abs(xx) > Lx / 2 - s.border_buffer) | (yy > Ly / 2 - s.border_buffer)
        elif name == "arm":
            y0 = s.arm_separation / 2
            m |= (xx < -Lx / 2 + s.arm_buffer) & (yy > y0) & (yy < y0 + s.arm_width)
        elif name == "waveguide":
            m |= (xx > Lx / 2 - s.wg_buffer_length) & (yy < s.wg_width * s.wg_buffer_width_frac / 2)
        else:  # path to a pkl array
            m |= _load_mask_pkl(name, s.nx, s.ny)
    return m


def anchor_cells(s: Study) -> np.ndarray:
    """Boolean (nx, ny): edge pixels touching the arms (left) or waveguide (right)."""
    _, yy = s.xy_half
    a = np.zeros((s.nx, s.ny), bool)
    y0 = s.arm_separation / 2
    a[0, :] = (yy[0] > y0) & (yy[0] < y0 + s.arm_width)
    a[-1, :] = yy[-1] < s.wg_width / 2
    return a


def apply_mask(params, mask: np.ndarray):
    m = mask.astype(float)
    return params * (1.0 - m) + m


# =====================================================================
# parametrisation
# =====================================================================


class Parametrization:
    """params (nx, ny in [0,1]) -> density rho (nx, ny in [0,1])."""

    def __init__(self, s: Study):
        self.s = s
        self.mask = fixed_solid_mask(s)
        self.filter_project = make_filter_and_project(
            s.min_feature_size, s.dr_grid_size, padding="reflect"
        )
        self.erode_dilate = make_erosion_dilation_penalty(s.min_feature_size, s.dr_grid_size)
        self.connectivity = HeatConnectivity(s)
        self.curvature = CurvaturePenalty(s)
        self.conic = make_conic_filter(s.min_feature_size, s.dr_grid_size, padding="reflect")

    def density(self, params, beta: float):
        rho = apply_mask(params, self.mask)
        rho = self.filter_project(rho, beta=beta)
        return apply_mask(rho, self.mask)

    def filtered(self, params):
        """Conic-filtered field before projection; its 0.5 level set is the boundary."""
        return apply_mask(self.conic(apply_mask(params, self.mask)), self.mask)

    def penalties(self, params, rho, feature_weight=None) -> dict:
        """Weighted penalties. `feature_weight` overrides s.feature_penalty_weight (ramp)."""
        s = self.s
        w = s.feature_penalty_weight if feature_weight is None else feature_weight
        out = {}
        if w:
            out["feature"] = w * self.erode_dilate(rho)
        if s.connectivity_weight:
            out["connectivity"] = s.connectivity_weight * self.connectivity.penalty(rho)
        if s.curvature_weight:
            # follows the same ramp as the feature penalty (factor 1 without a ramp)
            ramp = w / s.feature_penalty_weight if s.feature_penalty_weight else 1.0
            out["curvature"] = s.curvature_weight * ramp * self.curvature.penalty(self.filtered(params))
        return out


# =====================================================================
# edge smoothness: curvature of the design boundary
# =====================================================================


def _pad(a, r):
    """Mirror at y = 0 (symmetry plane), edge padding on the other three sides."""
    a = anp.concatenate((a[:, r - 1 :: -1], a), axis=1)
    a = anp.concatenate((a, anp.repeat(a[:, -1:], r, axis=1)), axis=1)
    return anp.concatenate((anp.repeat(a[:1], r, axis=0), a, anp.repeat(a[-1:], r, axis=0)), axis=0)


def _edge_pixels(b):
    """Pixels of a binary map with a 4-neighbour of the other phase."""
    p = np.pad(b, 1, mode="edge")
    c = p[1:-1, 1:-1]
    return (c != p[2:, 1:-1]) | (c != p[:-2, 1:-1]) | (c != p[1:-1, 2:]) | (c != p[1:-1, :-2])


class CurvaturePenalty:
    """Mean squared excess curvature along the design boundary.

    The boundary is the 0.5 level set of the *filtered* field
    phi = conic_filter(params) (the tanh projection only sharpens it: rho > 0.5
    <=> phi > 0.5). phi is already smooth on the filter scale, so it is used
    almost as is (0.75 px Gaussian against finite-difference noise); the
    conic-filter scallops (R ~ 0.05-0.15 um) are seen at full strength, and
    the gradient reaches the parameters directly instead of through the
    projection. kappa = div(grad phi / |grad phi|), within 4 % on disks of
    0.2-1 um. Each boundary point costs softplus(|kappa| * R_min - 1)^2 (~0 for
    radius > R_min); weighted by |grad phi| in a band around phi = 0.5, so the
    value is an average over boundary length.
    """

    sigma_px, band, sharpness = 0.75, 0.2, 5.0

    def __init__(self, s: Study):
        self.s, self.dl, self.r_min = s, s.dr_grid_size, s.min_curvature_radius
        r = int(np.ceil(3 * self.sigma_px))
        x = np.arange(-r, r + 1)
        k = np.exp(-(x**2) / (2 * self.sigma_px**2))
        self.kernel, self.r = k / k.sum(), r

    def _blur(self, a):
        nx, ny = a.shape
        b = _pad(a, self.r)
        b = sum(self.kernel[i] * b[i : i + nx, :] for i in range(2 * self.r + 1))
        return sum(self.kernel[j] * b[:, j : j + ny] for j in range(2 * self.r + 1))

    def curvature(self, phi):
        """(kappa, boundary weight) of the phi = 0.5 contour on the (nx, ny) grid."""
        dl = self.dl
        f = _pad(self._blur(phi), 1)
        c = f[1:-1, 1:-1]
        fx = (f[2:, 1:-1] - f[:-2, 1:-1]) / (2 * dl)
        fy = (f[1:-1, 2:] - f[1:-1, :-2]) / (2 * dl)
        fxx = (f[2:, 1:-1] - 2 * c + f[:-2, 1:-1]) / dl**2
        fyy = (f[1:-1, 2:] - 2 * c + f[1:-1, :-2]) / dl**2
        fxy = (f[2:, 2:] - f[2:, :-2] - f[:-2, 2:] + f[:-2, :-2]) / (4 * dl**2)
        g2 = fx**2 + fy**2 + 1e-8
        kappa = (fxx * fy**2 - 2 * fx * fy * fxy + fyy * fx**2) / g2**1.5
        w = anp.sqrt(g2) * anp.exp(-(((c - 0.5) / self.band) ** 2))
        return kappa, w

    def penalty(self, phi):
        kappa, w = self.curvature(phi)
        excess = anp.sqrt(kappa**2 + 1e-4) * self.r_min - 1.0
        cost = (anp.logaddexp(0.0, self.sharpness * excess) / self.sharpness) ** 2
        return anp.sum(w * cost) / (anp.sum(w) + 1e-12)

    def report(self, phi) -> dict:
        phi = np.asarray(getattr(phi, "_value", phi), float)
        kappa, w = self.curvature(phi)
        tight = np.abs(kappa) * self.r_min > 1
        return dict(
            curvature_penalty=float(self.penalty(phi)),
            boundary_frac_below_min_radius=float((w * tight).sum() / w.sum()),
            radius_map=np.where(_edge_pixels(phi > 0.5), 1 / np.maximum(np.abs(kappa), 1e-3), np.nan),
        )


# =====================================================================
# connectivity: heat method
# =====================================================================


def _assemble(kx, ky, ks):
    """5-point conductance matrix on an (nx, ny) grid.

    kx: (nx-1, ny) face conductances along x, ky: (nx, ny-1) along y,
    ks: (nx, ny) conductance to the T=0 sink. Other edges are adiabatic
    (y = 0 is the mirror plane, so Neumann is exact there).
    """
    nx, ny = ks.shape
    idx = np.arange(nx * ny).reshape(nx, ny)
    diag = ks.copy()
    diag[:-1, :] += kx
    diag[1:, :] += kx
    diag[:, :-1] += ky
    diag[:, 1:] += ky
    rows = [idx.ravel(), idx[:-1].ravel(), idx[1:].ravel(), idx[:, :-1].ravel(), idx[:, 1:].ravel()]
    cols = [idx.ravel(), idx[1:].ravel(), idx[:-1].ravel(), idx[:, 1:].ravel(), idx[:, :-1].ravel()]
    vals = [diag.ravel(), -kx.ravel(), -kx.ravel(), -ky.ravel(), -ky.ravel()]
    K = sps.csc_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(nx * ny,) * 2
    )
    return K


@primitive
def heat_solve(kx, ky, ks, q):
    """Solve K(kx, ky, ks) T = q, returns T with the shape of q."""
    K = _assemble(kx, ky, ks)
    return spla.spsolve(K, q.ravel()).reshape(q.shape)


def _vjp_maker(arg):
    def maker(T, kx, ky, ks, q):
        def vjp(g):
            K = _assemble(kx, ky, ks)  # symmetric -> adjoint uses the same matrix
            lam = spla.spsolve(K, np.asarray(g).ravel()).reshape(q.shape)
            if arg == "q":
                return lam
            if arg == "kx":
                return -(lam[1:] - lam[:-1]) * (T[1:] - T[:-1])
            if arg == "ky":
                return -(lam[:, 1:] - lam[:, :-1]) * (T[:, 1:] - T[:, :-1])
            return -lam * T  # ks

        return vjp

    return maker


defvjp(heat_solve, _vjp_maker("kx"), _vjp_maker("ky"), _vjp_maker("ks"), _vjp_maker("q"))


class HeatConnectivity:
    """Connectivity measure: ~1 for well connected designs, >> 1 with islands."""

    k_min = 1e-8  # conductivity of air (relative)
    simp = 3.0  # conductivity ~ rho^simp
    p = 8.0  # p-norm ~ soft maximum of the temperature over solid

    def __init__(self, s: Study):
        self.s = s
        self.anchor = anchor_cells(s).astype(float)
        self.T_ref = float(np.max(self.temperature(np.ones((s.nx, s.ny)))))

    def temperature(self, rho):
        k = self.k_min + (1.0 - self.k_min) * rho**self.simp
        kx = 2 * k[1:] * k[:-1] / (k[1:] + k[:-1])
        ky = 2 * k[:, 1:] * k[:, :-1] / (k[:, 1:] + k[:, :-1])
        ks = 2 * k * self.anchor
        return heat_solve(kx, ky, ks, rho)

    def measure(self, rho):
        """p-norm of T/T_ref over solid pixels (density weighted)."""
        t = self.temperature(rho) / self.T_ref
        return (anp.sum(rho * t**self.p) / anp.sum(rho)) ** (1.0 / self.p)

    def penalty(self, rho):
        """log-scaled, zero while the measure is below `connectivity_tol`."""
        g = self.measure(rho)
        return anp.log10(anp.maximum(g / self.s.connectivity_tol, 1.0))


# =====================================================================
# post-processing / checks on the binarised design
# =====================================================================

_FOUR = ndi.generate_binary_structure(2, 1)


def remove_islands(binary: np.ndarray, s: Study) -> tuple[np.ndarray, int]:
    """Keep only solid components (4-connected) touching an arm or the waveguide.

    Works on the y >= 0 half; the mirror plane is handled by symmetry.
    Returns the cleaned array and the number of removed components.
    """
    lab, n = ndi.label(binary, structure=_FOUR)
    keep = np.unique(lab[anchor_cells(s) & binary])
    keep = keep[keep > 0]
    out = np.isin(lab, keep)
    return out, int(n - len(keep))


def _disk(radius_px: float) -> np.ndarray:
    r = int(np.ceil(radius_px))
    yy, xx = np.mgrid[-r : r + 1, -r : r + 1]
    return xx**2 + yy**2 <= radius_px**2


def feature_size_report(binary: np.ndarray, s: Study) -> dict:
    """Pixels violating the minimum solid / void width (morphological open/close).

    Uses a disk of diameter `min_feature_size`. Reflect padding at y = 0
    (mirror plane), edge padding elsewhere.
    """
    se = _disk(s.min_feature_size / s.dr_grid_size / 2)
    pad = se.shape[0]
    b = np.pad(binary, ((pad, pad), (0, pad)), mode="edge")
    b = np.pad(b, ((0, 0), (pad, 0)), mode="reflect")
    opened = ndi.binary_opening(b, se)
    closed = ndi.binary_closing(b, se)
    crop = (slice(pad, -pad), slice(pad, -pad))
    solid_bad = (b & ~opened)[crop]
    void_bad = (closed & ~b)[crop]
    _, n = ndi.label(binary, structure=_FOUR)
    return dict(
        solid_violation_px=int(solid_bad.sum()),
        void_violation_px=int(void_bad.sum()),
        n_solid_components=int(n),
        solid_violation=solid_bad,
        void_violation=void_bad,
    )
