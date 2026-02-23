"""Tests for XDMSession setup."""

from .. import XDMCalculator, XDMSession


class TestXDMSessionSetup:
    def test_molecule_loaded(self, session: XDMSession) -> None:
        assert session.mol is not None

    def test_correct_n_atoms(self, session: XDMSession) -> None:
        assert session.mol.natom == 3

    def test_grid_initialised(self, session: XDMSession) -> None:
        assert session.grid is not None

    def test_calculator_initialised(self, session: XDMSession) -> None:
        assert isinstance(session.calculator, XDMCalculator)

    def test_atomic_numbers(self, session: XDMSession) -> None:
        assert session.mol.pseudo_numbers.tolist() == [8, 1, 1]
        assert session.mol.numbers.tolist() == [8, 1, 1]
