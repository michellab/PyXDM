"""Tests for partitioning module."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from ..partitioning.partitioning import (
    BeckePartitioning,
    HirshfeldIPartitioning,
    HirshfeldPartitioning,
    IterativeStockholderPartitioning,
    MBISPartitioning,
    PartitioningScheme,
    PartitioningSchemeFactory,
)


class TestPartitioningSchemeFactory:
    def test_available_schemes_returns_list(self):
        schemes = PartitioningSchemeFactory.available_schemes()
        assert isinstance(schemes, list)
        assert len(schemes) > 0

    def test_all_expected_schemes_available(self):
        schemes = PartitioningSchemeFactory.available_schemes()
        assert "mbis" in schemes
        assert "becke" in schemes
        assert "hirshfeld" in schemes
        assert "hirshfeld-i" in schemes
        assert "iterative-stockholder" in schemes

    def test_create_mbis_returns_correct_type(self):
        scheme = PartitioningSchemeFactory.create_scheme("mbis")
        assert isinstance(scheme, MBISPartitioning)

    def test_create_becke_returns_correct_type(self):
        scheme = PartitioningSchemeFactory.create_scheme("becke")
        assert isinstance(scheme, BeckePartitioning)

    def test_create_hirshfeld_returns_correct_type(self):
        scheme = PartitioningSchemeFactory.create_scheme("hirshfeld")
        assert isinstance(scheme, HirshfeldPartitioning)

    def test_create_hirshfeld_i_returns_correct_type(self):
        scheme = PartitioningSchemeFactory.create_scheme("hirshfeld-i")
        assert isinstance(scheme, HirshfeldIPartitioning)

    def test_create_iterative_stockholder_returns_correct_type(self):
        scheme = PartitioningSchemeFactory.create_scheme("iterative-stockholder")
        assert isinstance(scheme, IterativeStockholderPartitioning)

    def test_unknown_scheme_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown partitioning scheme"):
            PartitioningSchemeFactory.create_scheme("nonexistent-scheme")

    def test_unknown_scheme_error_mentions_available(self):
        with pytest.raises(ValueError, match="Available"):
            PartitioningSchemeFactory.create_scheme("bad-scheme")

    # kwarg filtering -----------------------------------------------------------

    def test_becke_ignores_extra_kwargs(self):
        """Becke scheme silently ignores unknown kwargs (they are filtered)."""
        scheme = PartitioningSchemeFactory.create_scheme("becke", maxiter=999)
        assert isinstance(scheme, BeckePartitioning)

    def test_mbis_accepts_maxiter_and_threshold(self):
        scheme = PartitioningSchemeFactory.create_scheme("mbis", maxiter=100, threshold=1e-4)
        assert scheme.maxiter == 100
        assert scheme.threshold == 1e-4

    def test_hirshfeld_i_accepts_maxiter_threshold_proatomdb(self):
        scheme = PartitioningSchemeFactory.create_scheme(
            "hirshfeld-i",
            maxiter=200,
            threshold=1e-5,
            proatom_db="/fake/path",
        )
        assert scheme.maxiter == 200
        assert scheme.threshold == 1e-5
        assert scheme.proatom_db == "/fake/path"

    def test_iterative_stockholder_filters_proatomdb(self):
        """iterative-stockholder doesn't accept proatom_db — it should be filtered."""
        scheme = PartitioningSchemeFactory.create_scheme(
            "iterative-stockholder",
            maxiter=50,
            proatom_db="/should/be/ignored",
        )
        assert isinstance(scheme, IterativeStockholderPartitioning)
        assert scheme.maxiter == 50
        assert not hasattr(scheme, "proatom_db")

    def test_returned_object_is_partitioning_scheme_instance(self):
        for name in PartitioningSchemeFactory.available_schemes():
            scheme = PartitioningSchemeFactory.create_scheme(name)
            assert isinstance(scheme, PartitioningScheme)


class TestPartitioningSchemeBaseClass:
    def test_get_partition_object_initially_none(self):
        scheme = MBISPartitioning()
        assert scheme.get_partition_object() is None

    def test_get_charges_raises_before_compute(self):
        scheme = MBISPartitioning()
        with pytest.raises(ValueError, match="Partition object not computed"):
            scheme.get_charges(MagicMock(), MagicMock())

    def test_get_populations_raises_before_compute(self):
        scheme = BeckePartitioning()
        with pytest.raises(ValueError, match="Partition object not computed"):
            scheme.get_populations(MagicMock(), MagicMock())


class TestSchemeDefaults:
    def test_mbis_defaults(self):
        scheme = MBISPartitioning()
        assert scheme.maxiter == 500
        assert scheme.threshold == 1e-6

    def test_hirshfeld_i_defaults(self):
        scheme = HirshfeldIPartitioning()
        assert scheme.maxiter == 500
        assert scheme.threshold == 1e-6
        assert scheme.proatom_db is None

    def test_iterative_stockholder_defaults(self):
        scheme = IterativeStockholderPartitioning()
        assert scheme.maxiter == 500
        assert scheme.threshold == 1e-6

    def test_hirshfeld_proatom_db_default_none(self):
        scheme = HirshfeldPartitioning()
        assert scheme.proatom_db is None

    def test_hirshfeld_proatom_db_set(self):
        scheme = HirshfeldPartitioning(proatom_db="/a/b/c")
        assert scheme.proatom_db == "/a/b/c"


class TestProatomDBRequired:
    def test_hirshfeld_requires_proatom_db(self):
        scheme = HirshfeldPartitioning()
        with pytest.raises(ValueError, match="proatom_db"):
            scheme.compute_weights(MagicMock(), MagicMock())

    def test_hirshfeld_i_requires_proatom_db(self):
        scheme = HirshfeldIPartitioning()
        with pytest.raises(ValueError, match="proatom_db"):
            scheme.compute_weights(MagicMock(), MagicMock())


def _make_mock_mol(natom=2, numbers=None):
    mol = MagicMock()
    mol.natom = natom
    mol.numbers = numbers if numbers is not None else [8, 1]
    return mol


def _make_mock_partition_obj(natom=2, name="mbis"):
    part = MagicMock()
    part.name = name
    for i in range(natom):
        subgrid = MagicMock()
        subgrid.integrate.return_value = float(i + 1)
        part.get_grid.return_value = subgrid
        part.cache.load.return_value = np.ones(10)
    return part


class TestGetChargesWithMock:
    def test_get_charges_returns_dict(self):
        scheme = MBISPartitioning()
        scheme._partition_obj = _make_mock_partition_obj(natom=2, name="mbis")

        mol = _make_mock_mol(natom=2, numbers=[8, 1])
        mol.obasis.compute_grid_density_dm.return_value = np.ones(10)

        result = scheme.get_charges(mol, MagicMock())
        assert isinstance(result, dict)
        assert "mbis" in result
        assert "charges" in result["mbis"]

    def test_get_charges_length_matches_natom(self):
        scheme = MBISPartitioning()
        natom = 3
        scheme._partition_obj = _make_mock_partition_obj(natom=natom, name="mbis")

        mol = _make_mock_mol(natom=natom, numbers=[8, 1, 1])
        mol.obasis.compute_grid_density_dm.return_value = np.ones(10)

        result = scheme.get_charges(mol, MagicMock())
        assert len(result["mbis"]["charges"]) == natom


class TestGetPopulationsWithMock:
    def test_get_populations_returns_dict(self):
        scheme = BeckePartitioning()
        scheme._partition_obj = _make_mock_partition_obj(natom=2, name="becke")

        mol = _make_mock_mol(natom=2)
        mol.obasis.compute_grid_density_dm.return_value = np.ones(10)

        result = scheme.get_populations(mol, MagicMock())
        assert isinstance(result, dict)
        assert "becke" in result
        assert "populations" in result["becke"]

    def test_get_populations_values_are_floats(self):
        scheme = BeckePartitioning()
        scheme._partition_obj = _make_mock_partition_obj(natom=2, name="becke")

        mol = _make_mock_mol(natom=2)
        mol.obasis.compute_grid_density_dm.return_value = np.ones(10)

        result = scheme.get_populations(mol, MagicMock())
        for pop in result["becke"]["populations"]:
            assert isinstance(pop, float)
