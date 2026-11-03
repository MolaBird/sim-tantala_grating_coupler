"""
lmm40x_focus - focal-plane field of a collimated, linearly polarised Gaussian
beam focused by the Thorlabs LMM40X-P01 reflective (Schwarzschild) objective.

Self-contained (numpy only; scipy optional for Gaussian fits).  The lens data
come from the Thorlabs Zemax file 265_LMM40X-P01.zmx and are embedded here.

Model
-----
* Exact real-ray trace through the convex secondary (R 6.175 mm) and the
  concave primary (R 16.18 mm), with all obscurations: secondary housing
  (blocks entrance-pupil radii < 1.22 mm), three curved spiders, 5.1 mm
  entrance pupil.  EFL 5.027 mm, NA 0.507, residual WFE 4.3 nm rms.
* Debye-Wolf vector diffraction integral at the best-focus plane, with the
  energy apodisation and the residual wavefront taken from the trace and the
  polarisation traced through both (ideal, retardance-free) mirrors.
* Normal incidence, beam centred on the axis, objective in the Zemax
  orientation (one spider vane along +y).  Tilt and objective rotation are
  ignored (they are negligible for small field angles).

Units: SI (metres, watts).  Field E in sqrt(W)/m, so that |E|^2 is in W/m^2
for the given input power (integral of |E|^2 over the focal plane equals the
transmitted power to within a few % - |E|^2 includes the axial component).

Example
-------
    from lmm40x_focus import LMM40X
    import numpy as np

    obj = LMM40X()                                    # traces once (~1 s)
    f = obj.focus(wavelength=2.923e-6, beam_diameter=4.4e-3,
                  polarization=90)                    # deg from x: y-pol
    f.intensity            # (ny, nx) W/m^2 for 1 W input
    f.phase()              # phase of the field along the input polarisation
    f.Ex, f.Ey, f.Ez       # complex components, (ny, nx)
    f.x, f.y               # 1-D coordinates, m (relative to the focus)
    f.fit_gaussian()       # 1/e^2 radii of the best-fit Gaussian

    # custom grid / defocus
    x = np.linspace(-10e-6, 10e-6, 301)
    f = obj.focus(2.923e-6, 4.4e-3, 0, x=x, y=x, z=0.5e-6)
"""
from dataclasses import dataclass, field

import numpy as np

__all__ = ["LMM40X", "FocalField"]

# ----------------------------------------------------------------------------
# Prescription (mm), from the Zemax file.  z along the axis, light enters +z.
# ----------------------------------------------------------------------------
_Z_PRI = 13.4                               # primary vertex
_Z_SEC = _Z_PRI + 9.97122                   # secondary vertex
_C_SEC = 1.619425104308223895e-01           # 1/mm
_C_PRI = 6.180355123205379803e-02           # 1/mm
_Z_S6 = _Z_SEC - 9.9712247255449995 + 11.271224725550001  # stop/spider plane
_ENPD = 5.1                                 # entrance pupil diameter
_R_SEC_CLEAR = 2.6
_R_PRI_IN, _R_PRI_OUT = 4.0, 10.75
_R_HOUSING = 2.5                            # blocks r < 2.5 at the stop plane
_R_STOP = 5.8753738441442476

# curved-spiders-LMM40X.uda (Zemax user aperture, units 0.01 mm)
_SPIDER_UDA = """
LIN 30 590
LIN -30 590
ARC -273.2 420 77.7 100
LIN 4.6 218.4
ARC -213.2 420 -77.7 100
BRK
LIN 496 -321
LIN 526 -269
ARC 500.3 26.6 77.7 100
LIN 186.8 -113.2
ARC 470.3 -25.4 -77.7 100
BRK
LIN -526 -269
LIN -496 -321
ARC -227.1 -446.6 77.7 100
LIN -191.4 -105.2
ARC -257.1 -394.6 -77.7 100
BRK
"""
_SPIDER_SCALE = 1e-2


def _spider_polygons():
    """UDA -> list of (N,2) polygons in mm.  ARC xc yc angle n sweeps about
    (xc, yc) from the current point; positive angle = clockwise."""
    polys, cur = [], []
    for line in _SPIDER_UDA.strip().splitlines():
        tok = line.split()
        if tok[0] == "LIN":
            cur.append((float(tok[1]), float(tok[2])))
        elif tok[0] == "ARC":
            xc, yc, ang, n = map(float, tok[1:5])
            x, y = cur[-1]
            r = np.hypot(x - xc, y - yc)
            a = np.arctan2(y - yc, x - xc) - np.deg2rad(ang) * np.arange(
                1, int(n) + 1) / n
            cur.extend(zip(xc + r * np.cos(a), yc + r * np.sin(a)))
        elif tok[0] == "BRK":
            polys.append(np.array(cur) * _SPIDER_SCALE)
            cur = []
    return polys


def _in_polygon(x, y, poly):
    inside = np.zeros(x.shape, bool)
    xp, yp = poly[:, 0], poly[:, 1]
    for x1, y1, x2, y2 in zip(xp, yp, np.roll(xp, 1), np.roll(yp, 1)):
        cond = (y1 > y) != (y2 > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            xi = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
        inside ^= cond & (x < xi)
    return inside


def _sphere_hit(p, d, z_v, c):
    R = 1.0 / c
    ctr = np.array([0.0, 0.0, z_v + R])
    oc = p - ctr
    b = np.einsum("ij,ij->i", oc, d)
    disc = np.sqrt(b * b - (np.einsum("ij,ij->i", oc, oc) - R * R))
    t1, t2 = -b - disc, -b + disc
    q1, q2 = p + t1[:, None] * d, p + t2[:, None] * d
    use1 = np.abs(q1[:, 2] - z_v) < np.abs(q2[:, 2] - z_v)
    t = np.where(use1, t1, t2)
    q = np.where(use1[:, None], q1, q2)
    n = (ctr - q) / R
    return q, n / np.linalg.norm(n, axis=1, keepdims=True), t


def _reflect(d, n):
    return d - 2 * np.einsum("ij,ij->i", d, n)[:, None] * n


def _trace(x0, y0, polys=None):
    """Rays parallel to +z through (x0, y0) [mm] at z = 0."""
    N = x0.size
    p = np.stack([x0, y0, np.zeros(N)], 1)
    d = np.tile([0.0, 0.0, 1.0], (N, 1))
    q1, n1, t1 = _sphere_hit(p, d, _Z_SEC, _C_SEC)
    d = _reflect(d, n1)
    q2, n2, t2 = _sphere_hit(q1, d, _Z_PRI, _C_PRI)
    d = _reflect(d, n2)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    t6 = (_Z_S6 - q2[:, 2]) / d[:, 2]
    p6 = q2[:, :2] + t6[:, None] * d[:, :2]
    r6 = np.hypot(p6[:, 0], p6[:, 1])
    r2 = np.hypot(q2[:, 0], q2[:, 1])
    ok = ((np.hypot(q1[:, 0], q1[:, 1]) <= _R_SEC_CLEAR)
          & (r2 >= _R_PRI_IN) & (r2 <= _R_PRI_OUT)
          & (r6 >= _R_HOUSING) & (r6 <= _R_STOP * (1 + 1e-9)))
    for poly in polys or []:
        ok &= ~_in_polygon(p6[:, 0], p6[:, 1], poly)
    return dict(s=d, P=q2, L=t1 + t2, ok=ok, n1=n1, n2=n2)


def _wavefront(tr, F):
    """Path of each ray's plane wave to the point F: L + s.(F - P)."""
    return tr["L"] + tr["s"] @ F - np.einsum("ij,ij->i", tr["s"], tr["P"])


def _best_focus_z():
    """Axial position minimising the rms wavefront of the open annulus."""
    h = 1e-4
    tr = _trace(np.array([h]), np.array([0.0]))
    s, P = tr["s"][0], tr["P"][0]
    z_par = P[2] - P[0] * s[2] / s[0]
    r = np.linspace(0, _ENPD / 2, 400)
    tr = _trace(r, 0 * r)
    ok = tr["ok"]
    best = (np.inf, z_par)
    for z in np.linspace(z_par - 0.01, z_par + 0.01, 401):
        W = _wavefront(tr, np.array([0.0, 0.0, z]))[ok]
        w = r[ok]
        W = W - np.sum(W * w) / np.sum(w)
        rms = np.sqrt(np.sum(W ** 2 * w) / np.sum(w))
        best = min(best, (rms, z))
    return best[1]


# ----------------------------------------------------------------------------
# public API
# ----------------------------------------------------------------------------
@dataclass
class FocalField:
    """Vector field near the focus.

    x, y : 1-D coordinates [m] relative to the focus;  z : defocus [m]
    E    : (3, ny, nx) complex field (Ex, Ey, Ez) [sqrt(W)/m]
    """
    x: np.ndarray
    y: np.ndarray
    z: float
    E: np.ndarray
    wavelength: float
    polarization: np.ndarray            # unit input Jones vector (px, py)
    transmission: float                 # fraction of input power focused
    input_power: float = 1.0
    _cache: dict = field(default_factory=dict, repr=False)

    @property
    def Ex(self):
        return self.E[0]

    @property
    def Ey(self):
        return self.E[1]

    @property
    def Ez(self):
        return self.E[2]

    @property
    def intensity(self):
        """|Ex|^2 + |Ey|^2 + |Ez|^2  [W/m^2]."""
        return np.sum(np.abs(self.E) ** 2, axis=0)

    @property
    def E_parallel(self):
        """Field component along the input polarisation (complex)."""
        px, py = self.polarization
        return px * self.E[0] + py * self.E[1]

    def phase(self, component="parallel", threshold=None):
        """Phase [rad] of 'parallel' (default), 'x', 'y' or 'z', referenced to
        the phase at the intensity peak.  threshold (e.g. 1e-4) masks pixels
        whose relative intensity in that component is below it."""
        c = {"parallel": self.E_parallel, "x": self.E[0], "y": self.E[1],
             "z": self.E[2]}[component]
        i0 = np.unravel_index(np.argmax(self.intensity), self.intensity.shape)
        ph = np.angle(c * np.exp(-1j * np.angle(c[i0])))
        if threshold is not None:
            a = np.abs(c) ** 2
            ph = np.ma.masked_where(a < threshold * a.max(), ph)
        return ph

    def fit_gaussian(self, method="intensity"):
        """Best-fit elliptical Gaussian, 1/e^2 radii (needs scipy).

        method='intensity': least squares to the total intensity (core shape).
        method='mode'     : TEM00 (flat phase, waist in this plane) maximising
                            power coupling with the parallel component;
                            also returns 'coupling', the fraction of the
                            focused (transmitted) power in that mode.  The
                            grid must contain the Gaussian (default is fine).
        Returns dict with wx, wy, x0, y0 [m] (and peak / coupling)."""
        X, Y = np.meshgrid(self.x, self.y)
        I = self.intensity
        w0 = 0.4 * self.wavelength
        if method == "intensity":
            from scipy.optimize import least_squares
            In = I / I.max()

            def res(p):
                return (p[0] * np.exp(-2 * ((X - p[1]) ** 2 / p[3] ** 2
                                            + (Y - p[2]) ** 2 / p[4] ** 2))
                        - In).ravel()

            p = least_squares(res, [1, 0, 0, w0, w0],
                              x_scale=[1, w0, w0, w0, w0]).x
            return dict(wx=abs(p[3]), wy=abs(p[4]), x0=p[1], y0=p[2],
                        peak=p[0] * I.max())
        if method == "mode":
            from scipy.optimize import minimize
            dA = (self.x[1] - self.x[0]) * (self.y[1] - self.y[0])
            P = self.transmission * self.input_power    # true focused power
            Ep = self.E_parallel

            lam = self.wavelength       # optimise in units of wavelength

            def neg(q):
                G = np.exp(-(X / lam - q[0]) ** 2 / np.exp(2 * q[2])
                           - (Y / lam - q[1]) ** 2 / np.exp(2 * q[3]))
                return -abs(np.sum(G * Ep) * dA) ** 2 / (np.sum(G * G) * dA * P)

            q0 = [0.01, 0.01, np.log(w0 / lam), np.log(w0 / lam)]
            r = minimize(neg, q0, method="Nelder-Mead",
                         options=dict(xatol=1e-7, fatol=1e-10, maxiter=4000))
            return dict(wx=np.exp(r.x[2]) * lam, wy=np.exp(r.x[3]) * lam,
                        x0=r.x[0] * lam, y0=r.x[1] * lam, coupling=-r.fun)
        raise ValueError("method must be 'intensity' or 'mode'")


class LMM40X:
    """Thorlabs LMM40X-P01 reflective objective, infinity corrected.

    Parameters
    ----------
    pupil_samples : int
        Samples across the 5.1 mm entrance pupil (301 is accurate to <0.1%
        for fields within ~±20 wavelengths; raise it for larger fields).
    spiders : bool
        Include the three curved spider vanes.
    """

    efl = 5.0273e-3                 # m
    na = 0.507
    entrance_pupil_diameter = _ENPD * 1e-3
    obscuration_radius = 1.2202e-3  # m, in the entrance pupil

    def __init__(self, pupil_samples=301, spiders=True):
        a = _ENPD / 2
        g = np.linspace(-a, a, pupil_samples)
        X, Y = np.meshgrid(g, g)
        m = np.hypot(X, Y) <= a
        x0, y0 = X[m], Y[m]
        tr = _trace(x0, y0, _spider_polygons() if spiders else None)
        ok = tr["ok"]
        F = np.array([0.0, 0.0, _best_focus_z()])
        W = _wavefront(tr, F)[ok]
        s = tr["s"][ok]
        # sqrt(dOmega/dA) from a finite-difference Jacobian of the mapping
        x0, y0, h = x0[ok], y0[ok], 1e-4
        st = lambda dx, dy: _trace(x0 + dx, y0 + dy)["s"]
        sxp, sxm, syp, sym = st(h, 0), st(-h, 0), st(0, h), st(0, -h)
        J = np.abs((sxp[:, 0] - sxm[:, 0]) * (syp[:, 1] - sym[:, 1])
                   - (syp[:, 0] - sym[:, 0]) * (sxp[:, 1] - sxm[:, 1])) / (
            4 * h * h) / s[:, 2]
        # store in SI
        self._x0, self._y0 = x0 * 1e-3, y0 * 1e-3
        self._s = s
        self._W = (W - W.mean()) * 1e-3
        self._apod = np.sqrt(J) * 1e3
        self._dA = (g[1] - g[0]) ** 2 * 1e-6
        self._n1, self._n2 = tr["n1"][ok], tr["n2"][ok]
        self.focus_position = F[2] * 1e-3   # from the rear (S1) plane, m

    # ------------------------------------------------------------------
    def transmission(self, beam_diameter):
        """Fraction of a centred Gaussian beam (1/e^2 diameter, m) that
        reaches the focus."""
        w = beam_diameter / 2
        a2 = np.exp(-2 * (self._x0 ** 2 + self._y0 ** 2) / w ** 2)
        return np.sum(a2) * self._dA / (np.pi * w ** 2 / 2)

    def _output_polarization(self, pol):
        E = np.tile([pol[0], pol[1], 0.0], (self._s.shape[0], 1))
        for n in (self._n1, self._n2):
            E = 2 * np.einsum("ij,ij->i", n, E)[:, None] * n - E
        return E

    def focus(self, wavelength, beam_diameter, polarization=0.0, x=None,
              y=None, z=0.0, power=1.0, half_width=None, n=201,
              chunk=20000):
        """Field near the focus.

        wavelength    : m
        beam_diameter : 1/e^2 intensity diameter of the collimated input, m
        polarization  : linear polarisation angle from +x in degrees
                        (0 = x, 90 = y), or a Jones vector (px, py)
        x, y          : 1-D coordinates [m] relative to the focus; default
                        an n x n grid of half width `half_width`
                        (default 8 wavelengths)
        z             : defocus [m] (+ = away from the objective)
        power         : input beam power, W
        """
        if np.ndim(polarization) == 0:
            t = np.deg2rad(polarization)
            pol = np.array([np.cos(t), np.sin(t)])
        else:
            pol = np.asarray(polarization, float)
            pol = pol / np.linalg.norm(pol)
        if x is None or y is None:
            hw = 8 * wavelength if half_width is None else half_width
            g = np.linspace(-hw, hw, n)
            x = g if x is None else x
            y = g if y is None else y
        x, y = np.atleast_1d(x).astype(float), np.atleast_1d(y).astype(float)

        k = 2 * np.pi / wavelength
        w = beam_diameter / 2
        A0 = np.sqrt(2 * power / (np.pi * w ** 2))
        amp = A0 * np.exp(-(self._x0 ** 2 + self._y0 ** 2) / w ** 2)
        c = (amp * self._apod * self._dA
             * np.exp(1j * k * (self._W + self._s[:, 2] * z)))
        vec = self._output_polarization(pol)
        E = np.zeros((3, y.size, x.size), complex)
        for i0 in range(0, c.size, chunk):
            sl = slice(i0, i0 + chunk)
            ex = np.exp(1j * k * np.outer(x, self._s[sl, 0]))
            ey = np.exp(1j * k * np.outer(y, self._s[sl, 1]))
            for m in range(3):
                E[m] += (ey * (c[sl] * vec[sl, m])) @ ex.T
        E *= -1j / wavelength
        return FocalField(x=x, y=y, z=z, E=E, wavelength=wavelength,
                          polarization=pol,
                          transmission=self.transmission(beam_diameter),
                          input_power=power)


if __name__ == "__main__":
    obj = LMM40X()
    f = obj.focus(2.923e-6, 4.4e-3, polarization=90)
    dA = (f.x[1] - f.x[0]) * (f.y[1] - f.y[0])
    g = f.fit_gaussian()
    gm = f.fit_gaussian("mode")
    print(f"focus at {obj.focus_position*1e3:.4f} mm from S1")
    print(f"transmission {f.transmission:.4f}, focal-plane power "
          f"{f.intensity.sum()*dA:.4f} W (1 W in)")
    print(f"peak {f.intensity.max():.4e} W/m^2")
    print(f"LSQ Gaussian  wx={g['wx']*1e6:.3f} um  wy={g['wy']*1e6:.3f} um")
    print(f"mode-matched  wx={gm['wx']*1e6:.3f} um  wy={gm['wy']*1e6:.3f} um"
          f"  coupling={gm['coupling']:.3f}")
