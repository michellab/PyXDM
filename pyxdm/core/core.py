"""Core XDM calculator for computing multipole moments and related properties."""

import logging
from typing import Any, List, Optional, Union

import numpy as np

try:
    from .exchange_hole_cpp import compute_b_sigma
except ImportError:
    from .exchange_hole import compute_b_sigma

logger = logging.getLogger(__name__)


class XDMCalculator:
    """Core calculator for XDM multipole moments."""

    _MOLECULAR_GRID_SCHEMES = ["hirshfeld", "hirshfeld-i", "becke"]

    def __init__(self, mol: Any) -> None:
        self.mol = mol
        self.n_atoms: int = mol.natom

        dm_full = mol.get_dm_full()
        dm_spin = mol.get_dm_spin()

        if dm_spin is None:
            self.dm_alpha = dm_full * 0.5
            self.dm_beta = dm_full * 0.5
            self._closed_shell = True
        else:
            self.dm_alpha = 0.5 * (dm_full + dm_spin)
            self.dm_beta = 0.5 * (dm_full - dm_spin)
            self._closed_shell = False

        self.dm_full = dm_full
        self.dm_spin = dm_spin

        logger.debug(f"Initialized XDMCalculator for molecule with {self.n_atoms} atoms.")
        logger.info(f"Assuming {'closed-shell' if self._closed_shell else 'open-shell'} configuration.")

    def _compute_density_properties(self, grid_points: np.ndarray, anisotropic: bool = True) -> dict:
        """Compute density and derived properties on grid points."""
        if self._closed_shell:
            rho_alpha = rho_beta = self.mol.obasis.compute_grid_density_dm(self.dm_alpha, grid_points)
            tau_alpha = tau_beta = 2.0 * self.mol.obasis.compute_grid_kinetic_dm(self.dm_alpha, grid_points)
            nabla_alpha = nabla_beta = self.mol.obasis.compute_grid_gradient_dm(self.dm_alpha, grid_points)
            hessian_alpha = hessian_beta = self.mol.obasis.compute_grid_hessian_dm(self.dm_alpha, grid_points)
            laplacian_alpha = laplacian_beta = hessian_alpha[:, 0] + hessian_alpha[:, 3] + hessian_alpha[:, 5]
        else:
            rho_alpha = self.mol.obasis.compute_grid_density_dm(self.dm_alpha, grid_points)
            rho_beta = self.mol.obasis.compute_grid_density_dm(self.dm_beta, grid_points)
            tau_alpha = 2.0 * self.mol.obasis.compute_grid_kinetic_dm(self.dm_alpha, grid_points)
            tau_beta = 2.0 * self.mol.obasis.compute_grid_kinetic_dm(self.dm_beta, grid_points)
            nabla_alpha = self.mol.obasis.compute_grid_gradient_dm(self.dm_alpha, grid_points)
            nabla_beta = self.mol.obasis.compute_grid_gradient_dm(self.dm_beta, grid_points)
            hessian_alpha = self.mol.obasis.compute_grid_hessian_dm(self.dm_alpha, grid_points)
            hessian_beta = self.mol.obasis.compute_grid_hessian_dm(self.dm_beta, grid_points)
            laplacian_alpha = hessian_alpha[:, 0] + hessian_alpha[:, 3] + hessian_alpha[:, 5]
            laplacian_beta = hessian_beta[:, 0] + hessian_beta[:, 3] + hessian_beta[:, 5]

        nabla_alpha_mag2 = np.sum(nabla_alpha**2, axis=1)
        nabla_beta_mag2 = np.sum(nabla_beta**2, axis=1)

        d_alpha = tau_alpha - 0.25 * nabla_alpha_mag2 / rho_alpha
        d_beta = tau_beta - 0.25 * nabla_beta_mag2 / rho_beta

        Q_alpha = (laplacian_alpha - 2.0 * d_alpha) / 6.0
        Q_beta = (laplacian_beta - 2.0 * d_beta) / 6.0

        b_alpha = compute_b_sigma(rho_alpha, Q_alpha)
        b_beta = compute_b_sigma(rho_beta, Q_beta)

        props = {"rho_alpha": rho_alpha, "rho_beta": rho_beta, "b_alpha": b_alpha, "b_beta": b_beta}

        if anisotropic:
            epsilon = 1e-12
            nabla_alpha_norm = np.linalg.norm(nabla_alpha, axis=1)[:, None]
            nabla_beta_norm = np.linalg.norm(nabla_beta, axis=1)[:, None]

            props["u_alpha"] = nabla_alpha / (nabla_alpha_norm + epsilon)
            props["u_beta"] = nabla_beta / (nabla_beta_norm + epsilon)

        return props

    def _compute_moment_for_atom(
        self, atom_idx: int, grid: Any, weights_i: np.ndarray, order: int, density_props: dict, anisotropic: bool = False
    ) -> np.ndarray:
        """Compute multipole moment for a single atom."""
        r_vec = grid.points - self.mol.coordinates[atom_idx]
        r_i = np.linalg.norm(r_vec, axis=1)

        if not anisotropic:
            logger.debug(f"Calculating isotropic moment for atom {atom_idx} with order {order}")
            b_alpha_capped = np.minimum(density_props["b_alpha"], r_i)
            b_beta_capped = np.minimum(density_props["b_beta"], r_i)

            term_alpha_iso = r_i**order - (r_i - b_alpha_capped) ** order
            term_beta_iso = r_i**order - (r_i - b_beta_capped) ** order

            integrand_alpha = density_props["rho_alpha"] * term_alpha_iso**2
            integrand_beta = density_props["rho_beta"] * term_beta_iso**2

            lambda_alpha = grid.integrate(integrand_alpha * weights_i)
            lambda_beta = grid.integrate(integrand_beta * weights_i)
            lambda_iso = lambda_alpha + lambda_beta
            return lambda_iso, None
        else:
            logger.debug(f"Calculating anisotropic moment for atom {atom_idx} with order {order}")
            b_alpha_capped = np.minimum(density_props["b_alpha"], r_i)
            b_beta_capped = np.minimum(density_props["b_beta"], r_i)

            moment_mag_alpha = r_i**order - (r_i - b_alpha_capped) ** order
            moment_mag_beta = r_i**order - (r_i - b_beta_capped) ** order

            term_alpha_vec = moment_mag_alpha[:, None] * density_props["u_alpha"]
            term_beta_vec = moment_mag_beta[:, None] * density_props["u_beta"]

            outer_alpha = np.einsum("ni,nj->nij", term_alpha_vec, term_alpha_vec)
            outer_beta = np.einsum("ni,nj->nij", term_beta_vec, term_beta_vec)

            integrand_alpha = density_props["rho_alpha"][:, None, None] * outer_alpha
            integrand_beta = density_props["rho_beta"][:, None, None] * outer_beta

            lambda_alpha_tensor = np.sum(grid.weights[:, None, None] * weights_i[:, None, None] * integrand_alpha, axis=0)
            lambda_beta_tensor = np.sum(grid.weights[:, None, None] * weights_i[:, None, None] * integrand_beta, axis=0)

            lambda_tensor = lambda_alpha_tensor + lambda_beta_tensor
            lambda_iso = np.trace(lambda_tensor)

            return lambda_iso, lambda_tensor

    def calculate_xdm_moments(self, partition_obj: Any, grid: Any, order: Union[List[int], int], anisotropic: bool = False) -> dict:
        """Calculate XDM multipole moments for all atoms."""
        logger.debug("Calculating XDM multipole moments for all atoms...")
        order = order if isinstance(order, list) else [order]
        xdm_results = {f"<M{n}^2>": np.zeros(self.n_atoms) for n in order}
        xdm_results_tensor = {f"<M{n}^2>_tensor": np.full((self.n_atoms, *(3, 3)), np.nan) for n in order}

        recompute_density_props = partition_obj.name not in self._MOLECULAR_GRID_SCHEMES
        if not recompute_density_props:
            grid = partition_obj.get_grid(0)
            density_props = self._compute_density_properties(grid.points, anisotropic=anisotropic)

        for atom_idx in range(self.n_atoms):
            if recompute_density_props:
                grid = partition_obj.get_grid(atom_idx)
                density_props = self._compute_density_properties(grid.points, anisotropic=anisotropic)

            weights_i = partition_obj.cache.load("at_weights", atom_idx)
            for o in order:
                iso, tensor = self._compute_moment_for_atom(atom_idx, grid, weights_i, o, density_props, anisotropic=anisotropic)
                xdm_results[f"<M{o}^2>"][atom_idx] = iso
                xdm_results_tensor[f"<M{o}^2>_tensor"][atom_idx] = tensor

        return {partition_obj.name: {"xdm_results": xdm_results, "xdm_results_tensor": xdm_results_tensor}}

    def calculate_radial_moments(self, partition_obj: Any, order: Union[List[int], int] = 3) -> dict:
        """Calculate radial moments <r^order> for all atoms."""
        logger.debug("Calculating radial moments for all atoms...")
        order = order if isinstance(order, list) else [order]
        moments_dict = {f"<r^{o}>": np.zeros(self.mol.natom) for o in order}

        for i in range(self.mol.natom):
            subgrid = partition_obj.get_grid(i)
            weights_i = partition_obj.cache.load("at_weights", i)
            rho_subgrid = self.mol.obasis.compute_grid_density_dm(self.dm_full, subgrid.points)
            r = np.linalg.norm(subgrid.points - self.mol.coordinates[i], axis=1)

            for o in order:
                moments_dict[f"<r^{o}>"][i] = subgrid.integrate(r**o * weights_i * rho_subgrid)

        return {partition_obj.name: {"radial_moments": moments_dict}}

    @staticmethod
    def geom_factor(tensor: np.ndarray) -> float:
        """
        Compute the geometric factor f_geom for a tensor based on its eigenvalues.

        Parameters
        ----------
        eigenvalues : np.ndarray
            Eigenvalues of the tensor (length 3).

        Returns
        -------
        float
            Geometric scaling factor f_geom.
        """
        eigenvalues = np.linalg.eigvalsh(tensor)
        assert all(eigenvalues >= 0), "Eigenvalues must be non-negative for geometric factor calculation"
        mean_val = np.mean(eigenvalues)
        geom_mean = np.prod(eigenvalues) ** (1 / 3)
        f_geom = (geom_mean**2) / (mean_val**2)
        return f_geom

    @staticmethod
    def calculate_c6(m1_i, m1_j, alpha_i, alpha_j):
        """Calculate the C6 interaction between two atoms."""
        return (m1_i * m1_j * alpha_i * alpha_j) / (m1_i * alpha_j + m1_j * alpha_i)

    @staticmethod
    def calculate_c8(m1_i, m1_j, m2_i, m2_j, alpha_i, alpha_j):
        """Calculate the C8 interaction between two atoms."""
        return (3.0 / 2.0) * (alpha_i * alpha_j * (m1_i * m2_j + m2_i * m1_j)) / (m1_i * alpha_j + m1_j * alpha_i)

    @staticmethod
    def calculate_c10(m1_i, m1_j, m2_i, m2_j, m3_i, m3_j, alpha_i, alpha_j):
        """Calculate the C10 interaction between two atoms."""
        oct_dip = 2 * alpha_i * alpha_j * (m1_i * m3_j + m3_i * m1_j) / (m1_i * alpha_j + m1_j * alpha_i)
        quad_quad = (21.0 / 5.0) * (alpha_i * alpha_j * m2_i * m2_j) / (m1_i * alpha_j + m1_j * alpha_i)
        return oct_dip + quad_quad

    def calculate_dispersion_coefficients(
        self, partition_obj: Any, atomic_pols: np.ndarray, m1: np.ndarray, m2: np.ndarray, m3: np.ndarray
    ) -> dict:
        """Calculate dispersion coefficients for all atoms."""
        c6 = np.zeros((atomic_pols.shape[0], atomic_pols.shape[0]))
        c8 = np.zeros((atomic_pols.shape[0], atomic_pols.shape[0]))
        c10 = np.zeros((atomic_pols.shape[0], atomic_pols.shape[0]))
        for i in range(atomic_pols.shape[0]):
            for j in range(i, atomic_pols.shape[0]):
                c6[i, j] = self.calculate_c6(m1[i], m1[j], atomic_pols[i], atomic_pols[j])
                c8[i, j] = self.calculate_c8(m1[i], m1[j], m2[i], m2[j], atomic_pols[i], atomic_pols[j])
                c10[i, j] = self.calculate_c10(m1[i], m1[j], m2[i], m2[j], m3[i], m3[j], atomic_pols[i], atomic_pols[j])
        return {partition_obj.name: {"c6": c6, "c8": c8, "c10": c10}}

    def calculate_dispersion_energy(
        self,
        partition_obj: Any,
        coordinates: np.ndarray,
        dispersion_coefficients: dict,
        r_crit: Optional[np.ndarray] = None,
        order: Union[List[int], int] = [1, 2, 3],
    ) -> dict:
        """Calculate dispersion energy for all atoms."""
        from . import compute_distances

        order = order if isinstance(order, list) else [order]
        c6 = dispersion_coefficients[partition_obj.name]["c6"] if 1 in order else 0
        c8 = dispersion_coefficients[partition_obj.name]["c8"] if 2 in order else 0
        c10 = dispersion_coefficients[partition_obj.name]["c10"] if 3 in order else 0

        # Compute distances and
        r = compute_distances(coordinates)
        rcrit = r_crit if r_crit is not None else np.zeros_like(r)

        # Compute dispersion energy using only the upper triangle without the diagonal
        E6 = np.triu(c6 / (r**6 + rcrit**6), k=1).sum()
        E8 = np.triu(c8 / (r**8 + rcrit**8), k=1).sum()
        E10 = np.triu(c10 / (r**10 + rcrit**10), k=1).sum()
        E_disp = -np.sum(E6 + E8 + E10)
        return {partition_obj.name: {"E_disp": E_disp}}
