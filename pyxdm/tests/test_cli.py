"""Tests for the CLI entry point (cli.py)."""

import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from ..cli import main

DATA_DIR = Path(__file__).parent / "data"
MOLDEN_FILE = str(DATA_DIR / "water.orca.molden.input")


def run_cli(monkeypatch, args: list[str]) -> None:
    """Patch sys.argv and invoke main()."""
    monkeypatch.setattr(sys, "argv", ["pyxdm"] + args)
    main()


class TestCLI:
    def test_runs_without_error(self, monkeypatch, tmp_path):
        out = tmp_path / "water.h5"
        run_cli(monkeypatch, [MOLDEN_FILE, "--scheme", "mbis", "-o", str(out)])

    def test_output_file_created(self, monkeypatch, tmp_path):
        out = tmp_path / "water.h5"
        run_cli(monkeypatch, [MOLDEN_FILE, "--scheme", "mbis", "-o", str(out)])
        assert out.exists()

    def test_output_has_expected_groups(self, monkeypatch, tmp_path):
        out = tmp_path / "water.h5"
        run_cli(monkeypatch, [MOLDEN_FILE, "--scheme", "mbis", "-o", str(out)])
        with h5py.File(out, "r") as f:
            assert "metadata" in f
            assert "molecule" in f
            assert "mbis" in f

    def test_molecule_metadata(self, monkeypatch, tmp_path):
        out = tmp_path / "water.h5"
        run_cli(monkeypatch, [MOLDEN_FILE, "--scheme", "mbis", "-o", str(out)])
        with h5py.File(out, "r") as f:
            symbols = [s.decode() for s in f["molecule/atomic_symbols"][:]]
            assert symbols == ["O", "H", "H"]
            assert f["molecule"].attrs["natom"] == 3

    def test_xdm_moments_vs_reference(self, monkeypatch, tmp_path, water_ref_mbis):
        out = tmp_path / "water.h5"
        run_cli(monkeypatch, [MOLDEN_FILE, "--scheme", "mbis", "-o", str(out)])
        ref_map = {"<M1^2>": "M1_sq", "<M2^2>": "M2_sq", "<M3^2>": "M3_sq"}
        with h5py.File(out, "r") as f:
            for h5_key, ref_key in ref_map.items():
                stored = f[f"mbis/xdm/mbis/xdm_results/{h5_key}"][:]
                np.testing.assert_allclose(
                    stored,
                    water_ref_mbis[ref_key],
                    rtol=1e-2,
                    err_msg=f"{h5_key} from CLI output doesn't match postg reference",
                )

    def test_invalid_scheme_exits(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", [MOLDEN_FILE, "--scheme", "invalid_scheme"])
        with pytest.raises(SystemExit):
            main()
