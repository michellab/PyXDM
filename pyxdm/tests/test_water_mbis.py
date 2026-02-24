"""Tests for XDM calculations on a water molecule."""

from pathlib import Path

import numpy as np
import pytest

from .. import XDMCalculator, XDMSession

DATA_DIR = Path(__file__).parent / "data"
MOLDEN_FILE = DATA_DIR / "water.orca.molden.input"

XDM_ORDERS = [1, 2, 3]


@pytest.fixture(scope="module")
def mbis_session(session: XDMSession) -> XDMSession:
    """Session with the MBIS partition already computed."""
    session.setup_partition_schemes(["mbis"])
    return session


class TestCoordinates:
    def test_coordinates(self, mbis_session: XDMSession, water_ref_mbis) -> None:
        """Test coordinates against postg mbis reference values."""
        coordinates = mbis_session.mol.coordinates
        np.testing.assert_allclose(coordinates, water_ref_mbis["coordinates"], rtol=1e-2, err_msg="Coordinates deviate from postg reference")


class TestDistances:
    def test_distances(self, mbis_session: XDMSession, water_ref_mbis) -> None:
        from ..core import compute_distances

        distances = compute_distances(mbis_session.mol.coordinates)
        print(distances)
        np.testing.assert_allclose(distances, water_ref_mbis["distances"], rtol=1e-2, err_msg="Distances deviate from postg reference")


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
        """Test XDM moments against postg mbis reference values."""
        results = mbis_session.calculator.calculate_xdm_moments(
            partition_obj=mbis_session.partitions["mbis"],
            grid=mbis_session.grid,
            order=XDM_ORDERS,
        )
        xdm = results["mbis"]["xdm_results"]
        for key, ref_key in (("<M1^2>", "M1_sq"), ("<M2^2>", "M2_sq"), ("<M3^2>", "M3_sq")):
            np.testing.assert_allclose(xdm[key], water_ref_mbis[ref_key], rtol=1e-2, err_msg=f"{key} deviates from postg reference")


class TestDispersionCoefficients:
    @pytest.fixture(scope="module")
    def dispersion_coefficients_postg(self, mbis_session: XDMSession, water_ref_mbis) -> None:
        """Calculate dispersion coefficients using the atomic polarizabilities and XDM moments provided by postg."""
        atomic_pols = water_ref_mbis["atomic_pols"]
        m1 = water_ref_mbis["M1_sq"]
        m2 = water_ref_mbis["M2_sq"]
        m3 = water_ref_mbis["M3_sq"]
        return mbis_session.calculator.calculate_dispersion_coefficients(
            partition_obj=mbis_session.partitions["mbis"],
            atomic_pols=atomic_pols,
            m1=m1,
            m2=m2,
            m3=m3,
        )

    def test_c6(self, mbis_session: XDMSession, water_ref_mbis, dispersion_coefficients_postg) -> None:
        """Test C6 coefficients against postg mbis reference values."""
        c6 = dispersion_coefficients_postg["mbis"]["c6"]
        c6 = c6[np.triu_indices_from(c6)]
        np.testing.assert_allclose(c6, water_ref_mbis["c6"], rtol=1e-2, err_msg="C6 coefficients deviate from postg reference")

    def test_c8(self, mbis_session: XDMSession, water_ref_mbis, dispersion_coefficients_postg) -> None:
        """Test C8 coefficients against postg mbis reference values."""
        c8 = dispersion_coefficients_postg["mbis"]["c8"]
        c8 = c8[np.triu_indices_from(c8)]
        np.testing.assert_allclose(c8, water_ref_mbis["c8"], rtol=1e-2, err_msg="C8 coefficients deviate from postg reference")

    def test_c10(self, mbis_session: XDMSession, water_ref_mbis, dispersion_coefficients_postg) -> None:
        """Test C10 coefficients against postg mbis reference values."""
        c10 = dispersion_coefficients_postg["mbis"]["c10"]
        c10 = c10[np.triu_indices_from(c10)]
        np.testing.assert_allclose(c10, water_ref_mbis["c10"], rtol=1e-2, err_msg="C10 coefficients deviate from postg reference")


class TestDispersionEnergy:
    @pytest.fixture(scope="module")
    def dispersion_coefficients_postg(self, mbis_session: XDMSession, water_ref_mbis) -> None:
        """Calculate dispersion coefficients using the atomic polarizabilities and XDM moments provided by postg."""
        C6 = np.zeros((3, 3))
        C8 = np.zeros((3, 3))
        C10 = np.zeros((3, 3))
        C6[np.triu_indices_from(C6)] = water_ref_mbis["c6"]
        C8[np.triu_indices_from(C8)] = water_ref_mbis["c8"]
        C10[np.triu_indices_from(C10)] = water_ref_mbis["c10"]
        return {"mbis": {"c6": C6, "c8": C8, "c10": C10}}

    def test_dispersion_energy(self, mbis_session: XDMSession, water_ref_mbis, dispersion_coefficients_postg) -> None:
        """Test dispersion energy against postg mbis reference values."""
        results = mbis_session.calculator.calculate_dispersion_energy(
            partition_obj=mbis_session.partitions["mbis"],
            coordinates=mbis_session.mol.coordinates,
            dispersion_coefficients=dispersion_coefficients_postg,
            order=XDM_ORDERS,
        )
        np.testing.assert_allclose(
            results["mbis"]["E_disp"], water_ref_mbis["E_disp"], rtol=1e-2, err_msg="Dispersion energy deviates from postg reference"
        )


class TestRadialMoments:
    def test_radial_moments(self, mbis_session: XDMSession, water_ref_mbis) -> None:
        """Test radial moments against postg mbis reference values."""
        results = mbis_session.calculator.calculate_radial_moments(
            partition_obj=mbis_session.partitions["mbis"],
            order=XDM_ORDERS,
        )
        radial = results["mbis"]["radial_moments"]["<r^3>"]
        np.testing.assert_allclose(radial, water_ref_mbis["volume"], rtol=1e-2, err_msg="Radial moment <r^3> deviates from postg reference")


class TestGeometryFactor:
    def test_geom_factor_identity(self) -> None:
        """Test geometry factor against identity tensor. For an isotropic tensor the geometric factor should be 1."""
        tensor = np.eye(3)
        f = XDMCalculator.geom_factor(tensor)
        assert pytest.approx(f, rel=1e-6) == 1.0

    def test_geom_factor_range(self) -> None:
        """Test geometry factor against random tensor. f_geom must lie in (0, 1]."""
        rng = np.random.default_rng(42)
        A = rng.random((3, 3))
        tensor = A @ A.T  # symmetric positive-semidefinite
        f = XDMCalculator.geom_factor(tensor)
        assert 0.0 < f <= 1.0 + 1e-9


class TestCharges:
    def test_charges(self, mbis_session: XDMSession, water_ref_mbis) -> None:
        """Test charges against postg mbis reference values."""
        results = mbis_session.partition_schemes["mbis"].get_charges(mbis_session.mol, mbis_session.calculator.dm_full)
        charges = results["mbis"]["charges"]
        np.testing.assert_allclose(charges, water_ref_mbis["charges"], rtol=1e-2, err_msg="Charges deviate from postg reference")


class TestPopulations:
    def test_populations(self, mbis_session: XDMSession, water_ref_mbis) -> None:
        """Test populations against postg mbis reference values."""
        # Alpha-spin populations
        results_alpha = mbis_session.partition_schemes["mbis"].get_populations(mbis_session.mol, mbis_session.calculator.dm_alpha)
        populations_alpha = sum(results_alpha["mbis"]["populations"])
        np.testing.assert_allclose(
            populations_alpha, water_ref_mbis["population_alpha"], rtol=1e-2, err_msg="Populations deviate from postg reference"
        )
        # Beta-spin populations
        results_beta = mbis_session.partition_schemes["mbis"].get_populations(mbis_session.mol, mbis_session.calculator.dm_beta)
        populations_beta = sum(results_beta["mbis"]["populations"])
        np.testing.assert_allclose(
            populations_beta, water_ref_mbis["population_beta"], rtol=1e-2, err_msg="Populations deviate from postg reference"
        )
        # Total-spin populations
        results_total = mbis_session.partition_schemes["mbis"].get_populations(mbis_session.mol, mbis_session.calculator.dm_full)
        populations_total = sum(results_total["mbis"]["populations"])
        np.testing.assert_allclose(
            populations_total, water_ref_mbis["population_total"], rtol=1e-2, err_msg="Populations deviate from postg reference"
        )
