"""Tests for XDM calculations on a water molecule."""

from pathlib import Path

import numpy as np
import pytest

from pyxdm import XDMCalculator, XDMSession

DATA_DIR = Path(__file__).parent / "data"
MOLDEN_FILE = DATA_DIR / "water.orca.molden.input"
PROATOMDB_FILE = DATA_DIR / "proatomdb.h5"

XDM_ORDERS = [1, 2, 3]


@pytest.fixture(scope="module")
def mbis_session(session: XDMSession) -> XDMSession:
    """Session with the MBIS partition already computed."""
    session.setup_partition_schemes(["mbis"], proatomdb=PROATOMDB_FILE)
    return session


class TestXDMMoments:
    def test_xdm_moments_shape(self, mbis_session: XDMSession) -> None:
        results = mbis_session.calculator.calculate_xdm_moments(
            partition_obj=mbis_session.partitions["mbis"],
            grid=mbis_session.grid,
            order=XDM_ORDERS,
        )
        xdm = results["mbis"]["xdm_results"]
        for n in XDM_ORDERS:
            assert xdm[f"<M{n}^2>"].shape == (mbis_session.mol.natom,)

    def test_xdm_moments(self, mbis_session: XDMSession, water_ref_mbis) -> None:
        """Moments should match postg mbis reference values within 1 %."""
        results = mbis_session.calculator.calculate_xdm_moments(
            partition_obj=mbis_session.partitions["mbis"],
            grid=mbis_session.grid,
            order=XDM_ORDERS,
        )
        xdm = results["mbis"]["xdm_results"]
        for key, ref_key in (("<M1^2>", "M1_sq"), ("<M2^2>", "M2_sq"), ("<M3^2>", "M3_sq")):
            np.testing.assert_allclose(xdm[key], water_ref_mbis[ref_key], rtol=1e-2, err_msg=f"{key} deviates from postg reference")

    def test_radial_moments(self, mbis_session: XDMSession, water_ref_mbis) -> None:
        """Radial moments should match postg mbis reference values within 1 %."""
        results = mbis_session.calculator.calculate_radial_moments(
            partition_obj=mbis_session.partitions["mbis"],
            order=XDM_ORDERS,
        )
        radial = results["mbis"]["radial_moments"]["<r^3>"]
        np.testing.assert_allclose(radial, water_ref_mbis["volume"], rtol=1e-2, err_msg="Radial moment <r^3> deviates from postg reference")


# ---------------------------------------------------------------------------
# Radial moment tests
# ---------------------------------------------------------------------------


class TestRadialMoments:
    def test_radial_moments_shape(self, mbis_session: XDMSession) -> None:
        results = mbis_session.calculator.calculate_radial_moments(
            partition_obj=mbis_session.partitions["mbis"],
            order=[1, 2, 3],
        )
        radial = results["mbis"]["radial_moments"]
        for o in [1, 2, 3]:
            assert radial[f"<r^{o}>"].shape == (mbis_session.mol.natom,)

    def test_radial_moments_positive(self, mbis_session: XDMSession) -> None:
        results = mbis_session.calculator.calculate_radial_moments(
            partition_obj=mbis_session.partitions["mbis"],
            order=3,
        )
        assert np.all(results["mbis"]["radial_moments"]["<r^3>"] > 0)


# ---------------------------------------------------------------------------
# Geometry factor tests
# ---------------------------------------------------------------------------


class TestGeometryFactor:
    def test_geom_factor_identity(self) -> None:
        """For an isotropic tensor the geometric factor should be 1."""
        tensor = np.eye(3)
        f = XDMCalculator.geom_factor(tensor)
        assert pytest.approx(f, rel=1e-6) == 1.0

    def test_geom_factor_range(self) -> None:
        """f_geom must lie in (0, 1]."""
        rng = np.random.default_rng(42)
        A = rng.random((3, 3))
        tensor = A @ A.T  # symmetric positive-semidefinite
        f = XDMCalculator.geom_factor(tensor)
        assert 0.0 < f <= 1.0 + 1e-9
