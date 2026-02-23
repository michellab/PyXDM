"""Formatting utilities for XDM output."""

import logging

import horton as ht

logger = logging.getLogger(__name__)
# Constants for formatting
ATOMIC_SYMBOLS = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 15: "P", 16: "S", 17: "Cl"}


def get_atomic_symbol(atomic_number: int) -> str:
    """
    Get atomic symbol from atomic number.

    Parameters
    ----------
    atomic_number : int
        Atomic number

    Returns
    -------
    str
        Atomic symbol or fallback format
    """
    return ATOMIC_SYMBOLS.get(atomic_number, f"Z{atomic_number}")


def format_scientific(value: float, precision: int = 6) -> str:
    """
    Format a number in scientific notation like POSTG.

    Parameters
    ----------
    value : float
        Value to format
    precision : int, optional
        Number of decimal places, default 6

    Returns
    -------
    str
        Formatted scientific notation string
    """
    if abs(value) < 1e-10:
        return f"{'0.0':>13}"
    else:
        return f"{value:.{precision}E}"


def log_mol_info(mol: ht.IOData) -> None:
    """
    Log information about the loaded molecule.

    Parameters
    ----------
    mol : ht.IOData
        Loaded molecule object
    """
    logger.info("Loaded molecule information:")
    logger.info(f"No. of atoms: {mol.natom}")
    logger.info(f"Atomic numbers: {mol.numbers}")
    logger.info(f"Pseudo atomic numbers: {mol.pseudo_numbers}")
    log_table(
        logger,
        columns=["x", "y", "z"],
        rows=[get_atomic_symbol(num) for num in mol.numbers],
        data=mol.coordinates,
        title="Coordinates [bohr]",
    )


def log_table(logger, columns, rows, data, level="info", title=None, scientific_notation=True):
    """
    Log a clean, aligned table with optional title.

    Title is printed above the table with an underline.
    """

    def log(msg):
        getattr(logger, level)(msg)

    log("")

    n_rows = len(rows)
    n_cols = len(columns)

    formatted_data = []
    for r_idx in range(n_rows):
        row_vals = []
        for c_idx in range(n_cols):
            val = data[r_idx][c_idx]
            val_str = format_scientific(val) if scientific_notation else str(val)
            row_vals.append(val_str)
        formatted_data.append(row_vals)

    col_widths = [max(len(str(columns[c])), *(len(formatted_data[r][c]) for r in range(n_rows))) for c in range(n_cols)]
    row_label_width = max(len(str(r)) for r in rows) if rows else 0

    if title:
        log(title)
        # log("-" * max(len(title), row_label_width + 2 + sum(col_widths) + 2 * n_cols))

    header = " " * row_label_width + "  " + "  ".join(str(columns[i]).rjust(col_widths[i]) for i in range(n_cols))
    log(header)
    # log("-" * len(header))

    for r_idx, row_label in enumerate(rows):
        row_values = [formatted_data[r_idx][c].rjust(col_widths[c]) for c in range(n_cols)]
        log(str(row_label).ljust(row_label_width) + "  " + "  ".join(row_values))

    log("")


def log_boxed_title(title: str, width: int = 50, logger=None, level="info"):
    """
    Print or log a title inside a box of given width.

    Parameters
    ----------
    title : str
        The text to display inside the box.
    width : int
        Total width of the box (including borders).
    logger : logging.Logger, optional
        If provided, logs the box instead of printing.
    level : str
        Logging level if logger is provided ("info", "warning", etc.).
    """

    def log(msg):
        getattr(logger, level)(msg)

    # Ensure width is enough for borders
    width = max(width, len(title) + 4)
    border = "+" + "-" * (width - 2) + "+"
    log(border)
    log("|" + title.center(width - 2) + "|")
    log(border)


def log_charges_populations(session, partition_obj, logger, level="info"):
    """
    Compute and log atomic charges and populations in a table.

    Parameters
    ----------
    session : object
        Session object containing molecule and calculator.
    partition_obj : object
        Object providing atomic grids and cached weights.
    logger : logging.Logger
        Logger instance to log table.
    level : str
        Logging level ("info", "warning", etc.).
    """
    import numpy as np

    charges = []
    populations = []

    for i in range(session.mol.natom):
        subgrid = partition_obj.get_grid(i)
        weights_i = partition_obj.cache.load("at_weights", i)
        rho_subgrid = session.mol.obasis.compute_grid_density_dm(session.calculator.dm_full, subgrid.points)
        population = subgrid.integrate(weights_i * rho_subgrid)
        charge = session.mol.numbers[i] - population

        populations.append(population)
        charges.append(charge)

    populations.append(np.sum(populations))
    charges.append(np.sum(charges))

    log_table(
        logger=logger,
        columns=["Charge [e]", "Population [e]"],
        rows=[get_atomic_symbol(num) for num in session.mol.numbers] + ["Σ_atoms"],
        data=np.column_stack((charges, populations)),
        level=level,
        scientific_notation=True,
    )
