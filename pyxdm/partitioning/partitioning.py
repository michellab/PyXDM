"""Atomic partitioning schemes for XDM calculations with comprehensive type hints."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional, Union

from ..grids import CustomGrid

if TYPE_CHECKING:
    pass


class PartitioningScheme(ABC):
    """
    Abstract base class for atomic partitioning schemes.

    This class defines the interface for all partitioning schemes used in XDM
    calculations. Subclasses must implement the compute_weights method to
    perform the actual partitioning calculation.

    Attributes
    ----------
    _partition_obj : Optional[Any]
        Internal partition object storing computed weights and metadata
    """

    def __init__(self) -> None:
        """
        Initialize the partitioning scheme.

        Returns
        -------
        None
        """
        self._partition_obj: Optional[Any] = None

    @abstractmethod
    def compute_weights(self, mol: Any, grid: Union[CustomGrid, Any]) -> None:
        """
        Compute partition object for the given molecule and grid.

        This method creates the partition object and performs the partitioning calculation.
        Weights are later loaded directly from the partition object's cache during
        grid projection.

        Parameters
        ----------
        mol : Any
            Molecule object containing basis set and geometry information
        grid : Union[CustomGrid, Any]
            Integration grid for partitioning calculations

        Returns
        -------
        None
        """
        pass

    def get_partition_object(self) -> Optional[Any]:
        """
        Get the underlying partition object for grid projection.

        Returns
        -------
        Optional[Any]
            The computed partition object if available, None otherwise
        """
        return self._partition_obj

    def get_charges(self, mol: Any, dm_full: Any) -> list[float]:
        """
        Compute atomic charges from the partition object.

        Parameters
        ----------
        mol : Any
            Molecule object containing atomic numbers and basis information
        dm_full : Any
            Full density matrix

        Returns
        -------
        list[float]
            List of atomic charges in electrons
        """
        if self._partition_obj is None:
            raise ValueError("Partition object not computed. Run compute_weights first.")

        charges = []
        for i in range(mol.natom):
            subgrid = self._partition_obj.get_grid(i)
            weights_i = self._partition_obj.cache.load("at_weights", i)
            rho_subgrid = mol.obasis.compute_grid_density_dm(dm_full, subgrid.points)
            population = subgrid.integrate(weights_i * rho_subgrid)
            charge = mol.numbers[i] - population
            charges.append(float(charge))

        return {self._partition_obj.name: {"charges": charges}}

    def get_populations(self, mol: Any, dm_full: Any) -> list[float]:
        """
        Compute atomic populations from the partition object.

        Parameters
        ----------
        mol : Any
            Molecule object containing atomic numbers and basis information
        dm_full : Any
            Full density matrix

        Returns
        -------
        list[float]
            List of atomic populations in electrons
        """
        if self._partition_obj is None:
            raise ValueError("Partition object not computed. Run compute_weights first.")

        populations = []
        for i in range(mol.natom):
            subgrid = self._partition_obj.get_grid(i)
            weights_i = self._partition_obj.cache.load("at_weights", i)
            rho_subgrid = mol.obasis.compute_grid_density_dm(dm_full, subgrid.points)
            population = subgrid.integrate(weights_i * rho_subgrid)
            populations.append(float(population))

        return {self._partition_obj.name: {"populations": populations}}


class BeckePartitioning(PartitioningScheme):
    """
    Becke partitioning scheme.

    Uses Becke's fuzzy atom approach with smooth cutoff functions
    to partition molecular electron density into atomic contributions.
    """

    NAME: str = "becke"

    def __init__(self) -> None:
        """
        Initialize Becke partitioning.

        Returns
        -------
        None
        """
        super().__init__()

    def compute_weights(self, mol: Any, grid: Union[CustomGrid, Any]) -> None:
        """
        Create Becke partition object for grid projection.

        This method creates a Horton Becke partition object that computes
        atomic weights using Becke's fuzzy atom partitioning scheme.

        Parameters
        ----------
        mol : Any
            Molecule object containing coordinates and atomic numbers
        grid : Union[CustomGrid, Any]
            Integration grid used for partitioning calculations

        Returns
        -------
        None
        """
        from horton.part import BeckeWPart

        dm_full = mol.get_dm_full()
        rho_total = mol.obasis.compute_grid_density_dm(dm_full, grid.points)

        becke = BeckeWPart(
            mol.coordinates,
            mol.numbers,
            mol.pseudo_numbers,
            grid,
            rho_total,
            local=False,
        )
        becke.do_all()

        self._partition_obj = becke
        self._partition_obj.name = self.NAME


class HirshfeldPartitioning(PartitioningScheme):
    """
    Hirshfeld partitioning scheme.

    Uses reference atomic densities to define partitioning weights
    based on the ratio of atomic to molecular density contributions.
    """

    NAME: str = "hirshfeld"

    def __init__(self, proatom_db: Optional[str] = None) -> None:
        """
        Initialize Hirshfeld partitioning.

        Parameters
        ----------
        proatom_db : Optional[str], default=None
            Path to pro-atom database. If None, uses default database.

        Returns
        -------
        None
        """
        super().__init__()
        self.proatom_db = proatom_db

    def compute_weights(self, mol: Any, grid: Union[CustomGrid, Any]) -> None:
        """
        Create Hirshfeld partition object for grid projection.

        This method creates a Horton Hirshfeld partition object that computes
        atomic weights using reference atomic densities.

        Parameters
        ----------
        mol : Any
            Molecule object containing coordinates, atomic numbers, and density matrix
        grid : Union[CustomGrid, Any]
            Integration grid used for partitioning calculations

        Returns
        -------
        None
        """
        from horton.part import HirshfeldWPart
        from horton.part.proatomdb import ProAtomDB

        if self.proatom_db is None:
            raise ValueError("Hirshfeld partitioning requires proatom_db path. Use --proatomdb argument.")

        # Load proatom database
        proatomdb = ProAtomDB.from_file(self.proatom_db)

        dm_full = mol.get_dm_full()
        rho_total = mol.obasis.compute_grid_density_dm(dm_full, grid.points)

        hirshfeld = HirshfeldWPart(
            mol.coordinates,
            mol.numbers,
            mol.pseudo_numbers,
            grid,
            rho_total,
            proatomdb,
            local=False,
        )
        hirshfeld.do_all()

        # Store partition object for grid projection
        self._partition_obj = hirshfeld
        self._partition_obj.name = self.NAME


class HirshfeldIPartitioning(PartitioningScheme):
    """
    Hirshfeld-I (Iterative) partitioning scheme.

    Iterative version of Hirshfeld partitioning that self-consistently
    updates reference atomic densities based on computed atomic charges.
    """

    NAME: str = "hirshfeld-i"

    def __init__(
        self,
        proatom_db: Optional[str] = None,
        maxiter: int = 500,
        threshold: float = 1e-6,
    ) -> None:
        """
        Initialize Hirshfeld-I partitioning.

        Parameters
        ----------
        proatom_db : Optional[str], default=None
            Path to pro-atom database. If None, uses default database.
        maxiter : int, default=500
            Maximum number of iterations for convergence
        threshold : float, default=1e-6
            Convergence threshold for iterative process

        Returns
        -------
        None
        """
        super().__init__()
        self.proatom_db = proatom_db
        self.maxiter = maxiter
        self.threshold = threshold

    def compute_weights(self, mol: Any, grid: Union[CustomGrid, Any]) -> None:
        """
        Create Hirshfeld-I partition object for grid projection.

        This method creates a Horton Hirshfeld-I partition object that computes
        atomic weights using iteratively refined reference atomic densities.

        Parameters
        ----------
        mol : Any
            Molecule object containing coordinates, atomic numbers, and density matrix
        grid : Union[CustomGrid, Any]
            Integration grid used for partitioning calculations
        """
        from horton.part import HirshfeldIWPart
        from horton.part.proatomdb import ProAtomDB

        if self.proatom_db is None:
            raise ValueError("Hirshfeld-I partitioning requires proatom_db path. Use --proatomdb argument.")

        # Load proatom database
        proatomdb = ProAtomDB.from_file(self.proatom_db)

        dm_full = mol.get_dm_full()
        rho_total = mol.obasis.compute_grid_density_dm(dm_full, grid.points)

        hirshfeld_i = HirshfeldIWPart(
            mol.coordinates,
            mol.numbers,
            mol.pseudo_numbers,
            grid,
            rho_total,
            proatomdb,
            local=False,
            maxiter=self.maxiter,
            threshold=self.threshold,
        )
        hirshfeld_i.do_all()

        # Store partition object for grid projection
        self._partition_obj = hirshfeld_i
        self._partition_obj.name = self.NAME


class IterativeStockholderPartitioning(PartitioningScheme):
    """
    Iterative Stockholder (IS) partitioning scheme.

    The Iterative Stockholder partitioning scheme is an iterative extension
    of Hirshfeld partitioning that aims to reduce the dependence on pro-atoms.
    It iteratively updates the pro-atom densities using the current atomic
    populations until convergence.

    This is a local partitioning scheme that creates atomic subgrids for
    improved computational efficiency.

    Attributes
    ----------
    maxiter : int
        Maximum number of iterations for convergence
    threshold : float
        Convergence threshold for density changes
    """

    NAME: str = "iterative-stockholder"

    def __init__(
        self,
        maxiter: int = 500,
        threshold: float = 1e-6,
    ) -> None:
        """
        Initialize Iterative Stockholder partitioning.

        Parameters
        ----------
        maxiter : int, default=500
            Maximum number of iterations for self-consistent procedure
        threshold : float, default=1e-6
            Convergence threshold for density changes between iterations
        """
        super().__init__()
        self.maxiter = maxiter
        self.threshold = threshold

    def compute_weights(self, mol: Any, grid: Union[CustomGrid, Any]) -> None:
        """
        Compute Iterative Stockholder atomic weights.

        This method creates an IterativeStockholderWPart object and performs
        the iterative partitioning calculation on the provided grid.

        Parameters
        ----------
        mol : Any
            Molecule object containing basis set and geometry information
        grid : Union[CustomGrid, Any]
            Integration grid for partitioning calculations
        """
        from horton.part import IterativeStockholderWPart

        dm_full = mol.get_dm_full()
        rho_total = mol.obasis.compute_grid_density_dm(dm_full, grid.points)

        iterstock = IterativeStockholderWPart(
            mol.coordinates,
            mol.numbers,
            mol.pseudo_numbers,
            grid,
            rho_total,
            maxiter=self.maxiter,
            threshold=self.threshold,
        )
        iterstock.do_all()
        iterstock.update_at_weights()

        # Store partition object for grid projection
        self._partition_obj = iterstock
        self._partition_obj.name = self.NAME


class MBISPartitioning(PartitioningScheme):
    """
    MBIS (Minimal Basis Iterative Stockholder) partitioning.

    MBIS uses iterative refinement to achieve self-consistent partitioning
    based on minimal basis set approximations of atomic densities.
    """

    NAME: str = "mbis"

    def __init__(self, maxiter: int = 500, threshold: float = 1e-6) -> None:
        """Initialize MBIS partitioning.

        Parameters
        ----------
        maxiter : int, default=500
            Maximum number of iterations for convergence
        threshold : float, default=1e-6
            Convergence threshold for iterative process
        """
        super().__init__()
        self.maxiter = maxiter
        self.threshold = threshold

    def compute_weights(self, mol: Any, grid: Union[CustomGrid, Any]) -> None:
        """Create MBIS partition object for grid projection.

        This method creates a Horton MBIS partition object that computes
        atomic weights using the Minimal Basis Iterative Stockholder approach.
        The partition object is stored for later use in grid projection.

        Parameters
        ----------
        mol : Any
            Molecule object containing coordinates, atomic numbers, and density matrix
        grid : Union[CustomGrid, Any]
            Target integration grid (not used directly by MBIS but kept for interface consistency)
        """
        from horton.part import MBISWPart

        dm_full = mol.get_dm_full()
        rho_total = mol.obasis.compute_grid_density_dm(dm_full, grid.points)
        mbis = MBISWPart(
            mol.coordinates,
            mol.numbers,
            mol.pseudo_numbers,
            grid,
            rho_total,
            maxiter=self.maxiter,
            threshold=self.threshold,
        )
        mbis.do_all()
        mbis.update_at_weights()

        # Store partition object for grid projection
        self._partition_obj = mbis
        self._partition_obj.name = self.NAME


class PartitioningSchemeFactory:
    """
    Factory for creating partitioning schemes.

    This factory provides a unified interface for creating different
    partitioning schemes used in XDM calculations.
    """

    _schemes = {
        MBISPartitioning.NAME: MBISPartitioning,
        BeckePartitioning.NAME: BeckePartitioning,
        HirshfeldPartitioning.NAME: HirshfeldPartitioning,
        HirshfeldIPartitioning.NAME: HirshfeldIPartitioning,
        IterativeStockholderPartitioning.NAME: IterativeStockholderPartitioning,
    }

    @classmethod
    def create_scheme(cls, scheme_name: str, **kwargs: Any) -> "PartitioningScheme":
        """
        Create a partitioning scheme.

        Parameters
        ----------
        scheme_name : str
            Name of the partitioning scheme to create.
            Available schemes: 'mbis', 'becke', 'hirshfeld', 'hirshfeld-i',
            'iterative-stockholder'
        **kwargs
            Additional keyword arguments passed to the scheme constructor.
            Different schemes accept different parameters:
            - mbis: maxiter, threshold, agspec
            - becke: (no parameters)
            - hirshfeld: proatom_db
            - hirshfeld-i: proatom_db, maxiter, threshold
            - iterstock/iterative-stockholder/is: maxiter, threshold

        Returns
        -------
        PartitioningScheme
            Instance of the requested partitioning scheme

        Raises
        ------
        ValueError
            If scheme_name is not recognized
        """
        if scheme_name not in cls._schemes:
            available = ", ".join(cls._schemes.keys())
            raise ValueError(f"Unknown partitioning scheme '{scheme_name}'. Available: {available}")

        # Filter kwargs based on scheme requirements
        if scheme_name in [BeckePartitioning.NAME]:
            # Becke doesn't accept any special parameters
            filtered_kwargs = {}
        elif scheme_name in [HirshfeldPartitioning.NAME]:
            # Hirshfeld only accepts 'proatom_db'
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in ["proatom_db"]}
        elif scheme_name in [HirshfeldIPartitioning.NAME]:
            # Hirshfeld-I accepts proatom_db, maxiter, threshold
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in ["proatom_db", "maxiter", "threshold"]}
        elif scheme_name in [IterativeStockholderPartitioning.NAME, "iterstock", "is"]:
            # Iterative Stockholder accepts maxiter, threshold
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in ["maxiter", "threshold"]}
        else:
            # MBIS accepts maxiter, threshold, agspec
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in ["maxiter", "threshold"]}

        scheme_class = cls._schemes[scheme_name]
        result = scheme_class(**filtered_kwargs)
        if not isinstance(result, PartitioningScheme):
            raise TypeError("Returned object is not a PartitioningScheme")
        return result

    @classmethod
    def available_schemes(cls) -> list[str]:
        """
        Return list of available partitioning schemes.

        Returns
        -------
        List[str]
            List of available scheme names that can be used with create_scheme()
        """
        return list(cls._schemes.keys())
