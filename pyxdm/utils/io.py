"""Input/output utilities for PyXDM."""

import logging
from typing import Any, Dict

import h5py
import numpy as np

from .. import __version__
from .formatting import get_atomic_symbol

logger = logging.getLogger(__name__)


def dump_to_h5(grp: str, data: Any) -> None:
    """
    Dump a dictionary or Horton object to an HDF5 file.

    Notes
    -----
    Adapted from: https://github.com/theochem/horton/blob/master/horton/io/internal.py#L31-L63

    grp
        A HDF5 group or a filename of a new HDF5 file.

    data
        The object to be written. This can be a dictionary of objects or
        an instance of a HORTON class that has a ``to_hdf5`` method. The
        dictionary my contain numpy arrays
    """
    if isinstance(grp, str):
        with h5py.File(grp, "w") as f:
            dump_to_h5(f, data)
    elif isinstance(data, dict):
        for key, value in data.items():
            # Simply overwrite old data
            if key in grp:
                del grp[key]
            if isinstance(value, int) or isinstance(value, float) or isinstance(value, np.ndarray) or isinstance(value, str):
                grp[key] = value
            else:
                subgrp = grp.require_group(key)
                dump_to_h5(subgrp, value)
    else:
        # clear the group if anything was present
        for key in list(grp.keys()):
            del grp[key]
        for key in list(grp.attrs.keys()):
            del grp.attrs[key]
        data.to_hdf5(grp)
        # The following is needed to create object of the right type when
        # reading from the checkpoint:
        grp.attrs["class"] = data.__class__.__name__


def write_h5_output(filename: str, session, xdm_results: Dict[str, Any], write_horton: bool = True) -> None:
    """
    Write calculated data to HDF5 file.

    Parameters
    ----------
    filename : str
        Output HDF5 filename
    session : XDMSession
        XDM session containing molecule and calculation data
    xdm_results : dict
        Dictionary containing all calculated results by scheme
    write_horton
        Whether to attempt to write the Horton partitioning results.
    """

    def get_horton_results(part, keys):
        results = {}
        for key in keys:
            if isinstance(key, str):
                results[key] = part[key]
            elif isinstance(key, tuple):
                if len(key) > 2:
                    # Skip lines such as ('isolated', -1, 0, 132516082542464) for hirshfeld-i
                    continue
                assert len(key) >= 2
                index = key[1]
                assert isinstance(index, int)
                assert index >= 0
                assert index < part.natom
                atom_results = results.setdefault("atom_%05i" % index, {})
                atom_results[key[0]] = part[key]
        return results

    logger.info(f"Writing results to {filename}")

    with h5py.File(filename, "w") as f:
        # Write metadata
        metadata = f.create_group("metadata")
        metadata.attrs["pyxdm_version"] = __version__
        metadata.attrs["wavefunction_file"] = str(session.wfn_file)

        # Write molecular information
        molecule = f.create_group("molecule")
        molecule.create_dataset("atomic_numbers", data=session.mol.numbers)
        molecule.create_dataset("coordinates", data=session.mol.coordinates)

        # Calculate nelec from populations (sum of all atomic populations)
        nelec = None
        if xdm_results:
            first_scheme_results = next(iter(xdm_results.values()))
            if "populations" in first_scheme_results:
                nelec = int(round(np.sum(first_scheme_results["populations"])))

        if nelec is None:
            nelec = np.nan

        molecule.create_dataset("nelec", data=nelec)
        molecule.attrs["natom"] = session.mol.natom

        # Write atomic symbols as string dataset
        symbols_str = [get_atomic_symbol(num).encode("utf-8") for num in session.mol.numbers]
        molecule.create_dataset("atomic_symbols", data=symbols_str)

        # Write results for each partitioning scheme
        for scheme_name, results_data in xdm_results.items():
            scheme_group = f.create_group(f"{scheme_name}")

            # Write PyXDM-calculated results
            xdm_group = scheme_group.create_group("xdm")
            dump_to_h5(xdm_group, results_data)

            # Write Horton data
            if write_horton:
                horton_group = scheme_group.create_group("horton")
                horton_data = get_horton_results(session.partitions[scheme_name], session.partitions[scheme_name].cache.keys())
                dump_to_h5(horton_group, horton_data)
