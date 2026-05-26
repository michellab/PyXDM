"""XDM calculation session management."""

import logging
import sys
from pathlib import Path
from typing import Any, Optional

from ..core import XDMCalculator
from ..grids import CustomGrid, load_mesh
from ..partitioning import PartitioningSchemeFactory

logger = logging.getLogger(__name__)

try:
    import horton as ht
except ImportError as e:
    logger.error("horton package is required for XDM calculations.")
    sys.exit(1)


class XDMSession:
    """
    Session manager for XDM calculations.

    This class manages the overall workflow of XDM calculations, including
    molecule loading, grid setup, and coordination between different components.

    Attributes
    ----------
    wfn_file : Path
        Path to the wavefunction file
    mol : object or None
        Loaded molecule object
    grid : object or None
        Computational grid object
    calculator : XDMCalculator or None
        XDM calculator instance
    partitions : dict or None
        Dictionary of partition objects
    """

    def __init__(self, wfn_file: str) -> None:
        """
        Initialize XDM calculation session.

        Parameters
        ----------
        wfn_file : str
            Path to the wavefunction file

        Returns
        -------
        None
        """
        self.wfn_file = Path(wfn_file)
        self.mol = None
        self.grid = None
        self.calculator = None
        self.partitions: dict[str, object] = {}
        self.partition_schemes: dict[str, object] = {}

    def load_molecule(self) -> None:
        """
        Load molecule from wavefunction file.

        Returns
        -------
        None
        """
        try:
            self.mol = ht.IOData.from_file(str(self.wfn_file))
        except Exception as e:
            logger.error(f"Error loading wavefunction file: {e}")
            sys.exit(1)

    def setup_grid(self, mesh_file: Optional[str] = None, grid_definition: Any = "ultrafine") -> None:
        """
        Setup computational grid.

        Parameters
        ----------
        mesh_file : str, optional
            Path to custom mesh file, if None uses default Becke grid
        grid_definition : Any, optional
            Definition for the Becke grid (default is 'ultrafine').
            See Horton documentation for valid options:
            https://theochem.github.io/horton/2.0.1/lib/mod_horton_grid_atgrid.html?highlight=ultrafine#horton.grid.atgrid.AtomicGridSpec

        Returns
        -------
        None
        """
        if not self.mol:
            raise RuntimeError("Molecule must be loaded before setting up grid")

        if mesh_file:
            try:
                mesh_points, mesh_weights = load_mesh(mesh_file)
                self.grid = CustomGrid(mesh_points, mesh_weights)
                logger.info(f"Loaded custom mesh with {len(mesh_points)} points")
            except Exception as e:
                logger.error(f"Error loading mesh file: {e}")
                sys.exit(1)
        else:
            agspec = ht.AtomicGridSpec(grid_definition)
            self.grid = ht.BeckeMolGrid(
                self.mol.coordinates,
                self.mol.numbers,
                self.mol.pseudo_numbers,
                agspec=agspec,
                mode="keep",
                random_rotate=False,
            )
            logger.info(f"Using Becke grid with {len(self.grid.points)} points")

    def setup_calculator(self) -> None:
        """
        Setup XDM calculator.

        Returns
        -------
        None
        """
        if not self.mol or not self.grid:
            raise RuntimeError("Molecule and grid must be set up before creating calculator")

        self.calculator = XDMCalculator(self.mol)
        logger.debug("XDM calculator initialized")

    def setup_partition_schemes(self, schemes, proatomdb: Optional[str] = None) -> dict:
        """
        Setup partitioning schemes for the session.

        Parameters
        ----------
        schemes : list of str or dict
            List of partitioning scheme names to use, or a dictionary mapping
            scheme names to their configuration options.
            
            If dict, keys are scheme names and values are dicts of kwargs:
            - mbis: lmax (int, default=3), maxiter (int, default=500), threshold (float, default=1e-6)
            - becke: lmax (int, default=3), k (int, default=3)
            - hirshfeld: lmax (int, default=3)
            - hirshfeld-i: lmax (int, default=3), maxiter (int, default=500), threshold (float, default=1e-6)
            - iterative-stockholder: lmax (int, default=3), maxiter (int, default=500), threshold (float, default=1e-6)
            
            Example:
                schemes = {
                    "mbis": {"lmax": 4, "maxiter": 500, "threshold": 1e-6},
                    "hirshfeld-i": {"lmax": 3, "maxiter": 1000, "threshold": 1e-5},
                    "becke": {"lmax": 3, "k": 3}
                }
                
        proatomdb : str, optional
            Path to proatom database for Hirshfeld-based schemes (hirshfeld, hirshfeld-i)

        Returns
        -------
        dict
            Dictionary of partition objects.
        """
        self.partitions = {}
        self.partition_schemes = {}

        if isinstance(schemes, dict):
            scheme_configs = schemes
        else:
            scheme_configs = {scheme: {} for scheme in schemes}

        for scheme, config in scheme_configs.items():
            try:
                scheme_kwargs = config.copy() if config else {}
                if proatomdb:
                    scheme_kwargs["proatom_db"] = proatomdb

                partitioning = PartitioningSchemeFactory.create_scheme(scheme, **scheme_kwargs)

                partitioning.compute_weights(self.mol, self.grid)

                partition_obj = partitioning.get_partition_object()
                if partition_obj is not None:
                    self.partitions[scheme] = partition_obj
                    self.partition_schemes[scheme] = partitioning
                else:
                    logger.warning(f"No partition object available for {scheme}")

            except Exception as e:
                logger.warning(f"Failed to compute {scheme} partition: {e}")
                continue

        return self.partitions
