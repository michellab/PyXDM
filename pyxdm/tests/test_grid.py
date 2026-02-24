"""Tests for grid utilities (load_mesh and CustomGrid)."""

from pathlib import Path

import numpy as np
import pytest

from ..grids.grid import CustomGrid, load_mesh


def _write_mesh(tmp_path: Path, lines: list[str]) -> Path:
    """Write *lines* to a temporary mesh file and return its path."""
    mesh_file = tmp_path / "test.mesh"
    mesh_file.write_text("\n".join(lines))
    return mesh_file


class TestLoadMesh:
    def test_basic_points_and_weights(self, tmp_path):
        """Three valid data lines are parsed correctly."""
        mesh = _write_mesh(
            tmp_path,
            [
                "1.0 2.0 3.0 0.5",
                "4.0 5.0 6.0 0.25",
                "7.0 8.0 9.0 0.125",
            ],
        )
        points, weights = load_mesh(str(mesh))

        assert points.shape == (3, 3)
        assert weights.shape == (3,)
        np.testing.assert_allclose(points[0], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(weights, [0.5, 0.25, 0.125])

    def test_comment_lines_skipped(self, tmp_path):
        """Lines starting with '#' are ignored."""
        mesh = _write_mesh(
            tmp_path,
            [
                "# This is a comment",
                "1.0 2.0 3.0 1.0",
                "# another comment",
                "4.0 5.0 6.0 2.0",
            ],
        )
        points, weights = load_mesh(str(mesh))
        assert points.shape == (2, 3)
        assert weights.shape == (2,)

    def test_empty_lines_skipped(self, tmp_path):
        """Blank lines are ignored."""
        mesh = _write_mesh(
            tmp_path,
            [
                "",
                "1.0 2.0 3.0 1.0",
                "   ",
                "4.0 5.0 6.0 2.0",
            ],
        )
        points, _ = load_mesh(str(mesh))
        assert points.shape == (2, 3)

    def test_inline_comments_stripped(self, tmp_path):
        """Text after '#' on a data line is stripped before parsing."""
        mesh = _write_mesh(
            tmp_path,
            ["1.0 2.0 3.0 0.5  # inline comment"],
        )
        points, weights = load_mesh(str(mesh))
        np.testing.assert_allclose(points[0], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(weights[0], 0.5)

    def test_non_float_lines_skipped(self, tmp_path):
        """Lines that cannot be parsed as floats are skipped silently."""
        mesh = _write_mesh(
            tmp_path,
            [
                "a b c d",
                "1.0 2.0 3.0 1.0",
            ],
        )
        points, weights = load_mesh(str(mesh))
        assert points.shape == (1, 3)

    def test_lines_with_fewer_than_4_columns_skipped(self, tmp_path):
        """Lines with fewer than 4 columns are ignored."""
        mesh = _write_mesh(
            tmp_path,
            [
                "1.0 2.0 3.0",  # only 3 columns → skipped
                "1.0 2.0 3.0 1.0",
            ],
        )
        points, _ = load_mesh(str(mesh))
        assert points.shape == (1, 3)

    def test_empty_file_returns_empty_arrays(self, tmp_path):
        """A file with no valid data returns arrays with zero rows."""
        mesh = _write_mesh(tmp_path, ["# only comments"])
        points, weights = load_mesh(str(mesh))
        assert points.shape == (0, 3)
        assert weights.shape == (0,)

    def test_extra_columns_ignored(self, tmp_path):
        """Only the first four columns (x, y, z, w) are used."""
        mesh = _write_mesh(
            tmp_path,
            ["1.0 2.0 3.0 0.5 99.9 100.0"],
        )
        points, weights = load_mesh(str(mesh))
        assert points.shape == (1, 3)
        np.testing.assert_allclose(weights[0], 0.5)


class TestCustomGrid:
    @pytest.fixture
    def simple_grid(self):
        points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        weights = np.array([0.5, 0.3, 0.2])
        return CustomGrid(points, weights)

    def test_attributes_stored(self, simple_grid):
        assert simple_grid.points.shape == (3, 3)
        assert simple_grid.weights.shape == (3,)
        assert simple_grid.n == 3

    def test_integrate_constant_function(self, simple_grid):
        """Integrating a constant of 1 should equal the sum of weights."""
        ones = np.ones(3)
        result = simple_grid.integrate(ones)
        assert isinstance(result, float)
        np.testing.assert_allclose(result, 1.0)

    def test_integrate_zero_function(self, simple_grid):
        result = simple_grid.integrate(np.zeros(3))
        np.testing.assert_allclose(result, 0.0)

    def test_integrate_weighted(self, simple_grid):
        """Test that integration applies weights correctly."""
        values = np.array([2.0, 2.0, 2.0])
        result = simple_grid.integrate(values)
        np.testing.assert_allclose(result, 2.0)

    def test_single_point_grid(self):
        points = np.array([[1.0, 2.0, 3.0]])
        weights = np.array([0.7])
        grid = CustomGrid(points, weights)
        assert grid.n == 1
        np.testing.assert_allclose(grid.integrate(np.array([3.0])), 2.1)

    def test_load_mesh_into_custom_grid_roundtrip(self, tmp_path):
        """load_mesh output feeds directly into CustomGrid."""
        mesh = _write_mesh(
            tmp_path,
            ["0.0 0.0 0.0 1.0", "1.0 0.0 0.0 2.0"],
        )
        points, weights = load_mesh(str(mesh))
        grid = CustomGrid(points, weights)
        assert grid.n == 2
        np.testing.assert_allclose(grid.integrate(np.ones(2)), 3.0)
