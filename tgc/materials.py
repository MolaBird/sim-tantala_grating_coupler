"""Tantala material.

* Fixed structures (arms, waveguide) use the dispersive PoleResidue fitted to
  `Ta2O5_Index_of_refraction.xlsx` (stored in `Tantala_tidy3d_Medium.pkl`).
* The design region is a CustomMedium whose permittivity is interpolated
  between air and eps_max = n(lambda0)^2 of the same fit (autograd needs a
  non-dispersive custom medium). Over the source band the fit varies by
  < 0.2 %, so this is a good approximation.
"""

from __future__ import annotations

import pickle
import re
import zipfile
from functools import lru_cache

import numpy as np
import tidy3d as td

from .config import ROOT


@lru_cache
def tantala(medium_file: str = "Tantala_tidy3d_Medium.pkl") -> td.PoleResidue:
    with open(ROOT / medium_file, "rb") as f:
        return pickle.load(f)


def n_tantala(wavelength: float, medium_file: str = "Tantala_tidy3d_Medium.pkl") -> float:
    n, _ = tantala(medium_file).nk_model(td.C_0 / wavelength)
    return float(n)


def read_index_xlsx(path=ROOT / "Ta2O5_Index_of_refraction.xlsx") -> np.ndarray:
    """Return [[wavelength_um, n], ...] from the xlsx (no openpyxl needed)."""
    xml = zipfile.ZipFile(path).read("xl/worksheets/sheet1.xml").decode()
    rows = re.findall(r'<c r="A\d+"><v>([^<]+)</v></c><c r="B\d+"><v>([^<]+)</v>', xml)
    data = np.array(rows, float)
    data[:, 0] /= 1000.0  # nm -> um
    return data


def check_fit(medium_file: str = "Tantala_tidy3d_Medium.pkl") -> float:
    """Max |n_fit - n_data| over the tabulated range."""
    data = read_index_xlsx()
    n_fit = np.array([n_tantala(wl, medium_file) for wl in data[:, 0]])
    return float(np.max(np.abs(n_fit - data[:, 1])))
