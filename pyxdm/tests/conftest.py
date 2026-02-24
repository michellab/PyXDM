from pathlib import Path

import numpy as np
import pytest

from .. import XDMSession


@pytest.fixture(scope="module")
def session() -> XDMSession:
    """Fully initialised XDMSession for water."""
    DATA_DIR = Path(__file__).parent / "data"
    MOLDEN_FILE = DATA_DIR / "water.orca.molden.input"
    s = XDMSession(str(MOLDEN_FILE))
    s.load_molecule()
    s.setup_grid()
    s.setup_calculator()
    return s


@pytest.fixture(scope="session")
def water_ref_mbis():
    """MBIS reference values for water from postg."""
    return {
        "atoms": ["O", "H", "H"],
        "coordinates": np.array(
            [
                [29.8423891, 29.1739636, 30.1820667],
                [29.2303350, 30.8776384, 29.9591468],
                [29.5019369, 28.5230590, 28.4334475],
            ]
        ),
        "distances": np.array(
            [
                [0.0, 1.823955, 1.896643],
                [1.823955, 0.0, 2.818789],
                [1.896643, 2.818789, 0.0],
            ]
        ),
        "M1_sq": np.array([6.4291878706, 0.71388001856, 0.77681399532]),
        "M2_sq": np.array([55.613596185, 3.0621180895, 3.5204186535]),
        "M3_sq": np.array([546.96781273, 23.420604202, 28.596484963]),
        "volume": np.array([27.77429239, 1.6595269, 1.8437556174]),
        "volume_free": np.array([22.954305454, 8.3489695346, 8.3489695346]),
        "atomic_pols": np.array([6.5486204764, 0.89442435842, 0.99371690524]),
        "charges": np.array([-0.79875051794, 0.4046151335, 0.39401645935]),
        "population_alpha": np.array([5.000059]),
        "population_beta": np.array([5.000059]),
        "population_total": np.array([10.000119]),
        "c6": np.array([21.05115567, 2.578600571, 2.832048538, 0.3192558388, 0.3510110849, 0.3859665997]),
        "c8": np.array([546.2885021, 50.04899698, 55.99827667, 4.108249511, 4.644545228, 5.247449294]),
        "c10": np.array([13779.45636, 1009.789031, 1156.672599, 66.56662258, 77.53266991, 90.1266692]),
        "r_crit": np.array([5.058204, 4.448619, 4.495673, 3.804169, 3.859485, 3.913542]),
        "E_disp": -5.276362287585,
        "E_scf": -76.42073582869,
        "E_total": -81.69709811627,
    }
