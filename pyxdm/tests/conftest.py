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
def water_ref_hirshfeld():
    """Hirshfeld reference values for water from postg."""
    return {
        "atoms": ["O", "H", "H"],
        "M1_sq": np.array([5.0923733396e00, 1.3460974275e00, 1.3642669290e00]),
        "M2_sq": np.array([3.7106386300e01, 1.0354718636e01, 1.0359074964e01]),
        "M3_sq": np.array([2.9495030904e02, 1.2952508632e02, 1.2940187218e02]),
        "volume": np.array([2.0826176950e01, 4.9243621684e00, 4.9182151641e00]),
        "volume_free": np.array([2.2954305454e01, 8.3489695346e00, 8.3489695346e00]),
        "alpha": np.array([4.9103943641e00, 2.6540512680e00, 2.6507382573e00]),
        "charges": np.array([-3.0937961120e-01, 1.5185106963e-01, 1.5740961649e-01]),
        "C6": np.array([1.250278067e01, 4.438950293e00, 4.477156066e00, 1.786305792e00, 1.797150063e00, 1.808157271e00]),
        "C8": np.array([2.733104852e02, 9.973693925e01, 9.992879673e01, 4.122293119e01, 4.120562156e01, 4.118879448e01]),
        "C10": np.array([5.684772971e03, 2.413471824e03, 2.408360865e03, 1.131476973e03, 1.127652173e03, 1.123874401e03]),
        "E_disp": -5.063647335380e-04,
        "E_scf": -7.642073582869e01,
        "E_total": -7.642124219342e01,
    }


@pytest.fixture(scope="session")
def water_ref_mbis():
    """MBIS reference values for water from postg."""
    return {
        "atoms": ["O", "H", "H"],
        "M1_sq": np.array([6.4291878706, 0.71388001856, 0.77681399532]),
        "M2_sq": np.array([55.613596185, 3.0621180895, 3.5204186535]),
        "M3_sq": np.array([546.96781273, 23.420604202, 28.596484963]),
        "volume": np.array([27.77429239, 1.6595269, 1.8437556174]),
        "volume_free": np.array([22.954305454, 8.3489695346, 8.3489695346]),
        "alpha": np.array([6.5486204764, 0.89442435842, 0.99371690524]),
        "charges": np.array([-0.79875051794, 0.4046151335, 0.39401645935]),
        "C6": np.array([21.05115567, 2.578600571, 2.832048538, 0.3192558388, 0.3510110849, 0.3859665997]),
        "C8": np.array([546.2885021, 50.04899698, 55.99827667, 4.108249511, 4.644545228, 5.247449294]),
        "C10": np.array([13779.45636, 1009.789031, 1156.672599, 66.56662258, 77.53266991, 90.1266692]),
        "E_disp": -5.276362287585,
        "E_scf": -76.42073582869,
        "E_total": -81.69709811627,
    }
