"""Unit tests for pyxdm.utils.formatting."""

import logging

import numpy as np

from ..utils.formatting import format_scientific, get_atomic_symbol, log_boxed_title, log_table


class TestGetAtomicSymbol:
    def test_known_elements(self):
        assert get_atomic_symbol(1) == "H"
        assert get_atomic_symbol(6) == "C"
        assert get_atomic_symbol(7) == "N"
        assert get_atomic_symbol(8) == "O"
        assert get_atomic_symbol(9) == "F"
        assert get_atomic_symbol(15) == "P"
        assert get_atomic_symbol(16) == "S"
        assert get_atomic_symbol(17) == "Cl"

    def test_unknown_element_fallback(self):
        assert get_atomic_symbol(99) == "Z99"
        assert get_atomic_symbol(118) == "Z118"


class TestFormatScientific:
    def test_nonzero_value(self):
        result = format_scientific(1.23456789e-3)
        assert "E" in result
        assert "1.234568" in result

    def test_negative_value(self):
        result = format_scientific(-3.14159e5)
        assert "E" in result
        assert "-" in result

    def test_near_zero_returns_zero(self):
        result = format_scientific(0.0)
        assert "0.0" in result

    def test_below_threshold_returns_zero(self):
        result = format_scientific(1e-11)
        assert "0.0" in result

    def test_custom_precision(self):
        result = format_scientific(1.23456789, precision=3)
        assert "1.235E" in result


class TestLogTable:
    def test_basic_table_logged(self, caplog):
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        with caplog.at_level(logging.INFO):
            log_table(
                logging.getLogger("test"),
                columns=["A", "B"],
                rows=["row1", "row2"],
                data=data,
            )
        full_output = "\n".join(caplog.messages)
        assert "A" in full_output
        assert "B" in full_output
        assert "row1" in full_output
        assert "row2" in full_output

    def test_table_with_title(self, caplog):
        data = np.array([[1.0]])
        with caplog.at_level(logging.INFO):
            log_table(
                logging.getLogger("test"),
                columns=["X"],
                rows=["r"],
                data=data,
                title="My Title",
            )
        full_output = "\n".join(caplog.messages)
        assert "My Title" in full_output

    def test_table_without_scientific_notation(self, caplog):
        data = np.array([[42.0]])
        with caplog.at_level(logging.INFO):
            log_table(
                logging.getLogger("test"),
                columns=["Val"],
                rows=["r"],
                data=data,
                scientific_notation=False,
            )
        full_output = "\n".join(caplog.messages)
        assert "42.0" in full_output


class TestLogBoxedTitle:
    def test_title_appears_in_output(self, caplog):
        with caplog.at_level(logging.INFO):
            log_boxed_title("Hello World", logger=logging.getLogger("test"))
        full_output = "\n".join(caplog.messages)
        assert "Hello World" in full_output

    def test_border_characters_present(self, caplog):
        with caplog.at_level(logging.INFO):
            log_boxed_title("Test", logger=logging.getLogger("test"))
        full_output = "\n".join(caplog.messages)
        assert "+" in full_output
        assert "-" in full_output
        assert "|" in full_output

    def test_long_title_expands_box(self, caplog):
        long_title = "A" * 60
        with caplog.at_level(logging.INFO):
            log_boxed_title(long_title, width=50, logger=logging.getLogger("test"))
        full_output = "\n".join(caplog.messages)
        assert long_title in full_output
