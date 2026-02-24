"""Unit tests for pyxdm.utils.io."""

import h5py
import numpy as np
import pytest

from ..utils.io import dump_to_h5, write_h5_output

XDM_ORDERS = [1, 2, 3]


class TestDumpToH5:
    def test_flat_dict_numpy(self, tmp_path):
        out = tmp_path / "test.h5"
        data = {"charges": np.array([1.0, -1.0]), "populations": np.array([7.0, 1.0])}
        dump_to_h5(str(out), data)

        with h5py.File(out, "r") as f:
            np.testing.assert_allclose(f["charges"][:], data["charges"])
            np.testing.assert_allclose(f["populations"][:], data["populations"])

    def test_scalar_values(self, tmp_path):
        out = tmp_path / "test.h5"
        data = {"an_int": 42, "a_float": 3.14}
        dump_to_h5(str(out), data)

        with h5py.File(out, "r") as f:
            assert f["an_int"][()] == 42
            assert abs(f["a_float"][()] - 3.14) < 1e-10

    def test_string_value(self, tmp_path):
        out = tmp_path / "test.h5"
        data = {"label": "water"}
        dump_to_h5(str(out), data)

        with h5py.File(out, "r") as f:
            stored = f["label"][()]
            assert stored in (b"water", "water")

    def test_nested_dict(self, tmp_path):
        out = tmp_path / "test.h5"
        data = {"outer": {"inner": np.array([1.0, 2.0, 3.0])}}
        dump_to_h5(str(out), data)

        with h5py.File(out, "r") as f:
            np.testing.assert_allclose(f["outer/inner"][:], [1.0, 2.0, 3.0])

    def test_overwrite_existing_key(self, tmp_path):
        out = tmp_path / "test.h5"
        dump_to_h5(str(out), {"arr": np.array([1.0, 2.0])})
        dump_to_h5(str(out), {"arr": np.array([9.0, 8.0])})

        with h5py.File(out, "r") as f:
            np.testing.assert_allclose(f["arr"][:], [9.0, 8.0])


@pytest.fixture(scope="module")
def mbis_session(session):
    """Session with MBIS partitioning already set up (reuses conftest session)."""
    session.setup_partition_schemes(["mbis"])
    return session


@pytest.fixture(scope="module")
def real_mbis_results(mbis_session):
    """
    Full dict of XDM results with charges, populations, and moments deep-merged
    under the same scheme key.

    Note: get_charges / get_populations / calculate_xdm_moments all return
    {"mbis": {...}}, so naive dict.update() overwrites earlier values.
    We merge explicitly to preserve all fields.
    """
    scheme = "mbis"
    partition_obj = mbis_session.partitions[scheme]
    partitioning_scheme = mbis_session.partition_schemes[scheme]

    charges = partitioning_scheme.get_charges(mbis_session.mol, mbis_session.calculator.dm_full)
    populations = partitioning_scheme.get_populations(mbis_session.mol, mbis_session.calculator.dm_full)
    xdm = mbis_session.calculator.calculate_xdm_moments(
        partition_obj=partition_obj,
        grid=mbis_session.grid,
        order=XDM_ORDERS,
    )

    # Deep-merge all results under the same inner scheme key
    scheme_results = {"mbis": {}}
    scheme_results["mbis"].update(charges[scheme])
    scheme_results["mbis"].update(populations[scheme])
    scheme_results["mbis"].update(xdm[scheme])

    return {"mbis": scheme_results}


class TestWriteH5Output:
    def test_top_level_groups_present(self, tmp_path, mbis_session, real_mbis_results):
        out = tmp_path / "water.h5"
        write_h5_output(str(out), mbis_session, real_mbis_results, write_horton=False)

        with h5py.File(out, "r") as f:
            assert "metadata" in f
            assert "molecule" in f
            assert "mbis" in f

    def test_metadata_attributes(self, tmp_path, mbis_session, real_mbis_results):
        out = tmp_path / "water.h5"
        write_h5_output(str(out), mbis_session, real_mbis_results, write_horton=False)

        with h5py.File(out, "r") as f:
            assert "pyxdm_version" in f["metadata"].attrs
            assert "wavefunction_file" in f["metadata"].attrs

    def test_molecule_atomic_numbers(self, tmp_path, mbis_session, real_mbis_results):
        out = tmp_path / "water.h5"
        write_h5_output(str(out), mbis_session, real_mbis_results, write_horton=False)

        with h5py.File(out, "r") as f:
            np.testing.assert_array_equal(f["molecule/atomic_numbers"][:], mbis_session.mol.numbers)

    def test_molecule_coordinates(self, tmp_path, mbis_session, real_mbis_results):
        out = tmp_path / "water.h5"
        write_h5_output(str(out), mbis_session, real_mbis_results, write_horton=False)

        with h5py.File(out, "r") as f:
            np.testing.assert_allclose(f["molecule/coordinates"][:], mbis_session.mol.coordinates)

    def test_molecule_atomic_symbols(self, tmp_path, mbis_session, real_mbis_results):
        out = tmp_path / "water.h5"
        write_h5_output(str(out), mbis_session, real_mbis_results, write_horton=False)

        with h5py.File(out, "r") as f:
            symbols = [s.decode() for s in f["molecule/atomic_symbols"][:]]
            assert symbols == ["O", "H", "H"]

    def test_roundtrip_charges(self, tmp_path, mbis_session, real_mbis_results):
        """Charges written to H5 must read back bit-for-bit identical."""
        out = tmp_path / "water.h5"
        write_h5_output(str(out), mbis_session, real_mbis_results, write_horton=False)

        original = np.array(real_mbis_results["mbis"]["mbis"]["charges"])
        with h5py.File(out, "r") as f:
            stored = f["mbis/xdm/mbis/charges"][:]
        np.testing.assert_allclose(stored, original, rtol=1e-10)

    def test_roundtrip_xdm_moments(self, tmp_path, mbis_session, real_mbis_results):
        """XDM moments written to H5 must read back bit-for-bit identical."""
        out = tmp_path / "water.h5"
        write_h5_output(str(out), mbis_session, real_mbis_results, write_horton=False)

        xdm_src = real_mbis_results["mbis"]["mbis"]["xdm_results"]
        with h5py.File(out, "r") as f:
            for key in ("<M1^2>", "<M2^2>", "<M3^2>"):
                stored = f[f"mbis/xdm/mbis/xdm_results/{key}"][:]
                np.testing.assert_allclose(stored, xdm_src[key], rtol=1e-10)

    def test_roundtrip_charges_vs_reference(self, tmp_path, mbis_session, real_mbis_results, water_ref_mbis):
        """Charges read from H5 must agree with postg reference to 1 %."""
        out = tmp_path / "water.h5"
        write_h5_output(str(out), mbis_session, real_mbis_results, write_horton=False)

        with h5py.File(out, "r") as f:
            stored = f["mbis/xdm/mbis/charges"][:]
        np.testing.assert_allclose(stored, water_ref_mbis["charges"], rtol=1e-2)

    def test_roundtrip_moments_vs_reference(self, tmp_path, mbis_session, real_mbis_results, water_ref_mbis):
        """XDM moments read from H5 must agree with postg reference to 1 %."""
        out = tmp_path / "water.h5"
        write_h5_output(str(out), mbis_session, real_mbis_results, write_horton=False)

        ref_map = {"<M1^2>": "M1_sq", "<M2^2>": "M2_sq", "<M3^2>": "M3_sq"}
        with h5py.File(out, "r") as f:
            for key, ref_key in ref_map.items():
                stored = f[f"mbis/xdm/mbis/xdm_results/{key}"][:]
                np.testing.assert_allclose(stored, water_ref_mbis[ref_key], rtol=1e-2)
