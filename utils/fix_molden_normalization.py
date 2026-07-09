"""
Fix molden files with incorrect orbital normalization.

orca_2mkl seems to have a bug where certain molecules get written with
incorrect basis set normalization AND orbital coefficient scaling.
This causes HORTON to fail with "Could not correct the data read from..." errors.

This script applies ORCA basis fix then renormalizes MO coefficients.
"""

import sys
from pathlib import Path

import numpy as np

try:
    import horton.io.molden as molden_module
    from horton import IOData
except ImportError:
    print("Error: HORTON not installed")
    sys.exit(1)


def fix_molden_file(input_file, output_file=None):
    """
    Load molden file, apply ORCA basis fix, renormalize orbitals, and save.

    Parameters
    ----------
    input_file : str
        Path to input molden file
    output_file : str, optional
        Path to output molden file. Defaults to fixed_input_file

    Returns
    -------
    str
        Path to the output file
    """
    if output_file is None:
        output_file = "fixed_" + input_file

    print(f"Loading {input_file}...")

    # Temporarily patch HORTON to not raise errors on bad normalization
    original_fix = molden_module._fix_molden_from_buggy_codes

    def patched_fix(result, filename):
        try:
            original_fix(result, filename)
        except IOError:
            print("  [Note: HORTON's automatic fixes failed, proceeding to manual fix...]")
            # Apply ORCA basis fix manually
            from horton import GOBasis

            obasis = result["obasis"]
            orca_con_coeffs = molden_module._get_fixed_con_coeffs(obasis, "orca")
            if orca_con_coeffs is not None:
                orca_obasis = GOBasis(obasis.centers, obasis.shell_map, obasis.nprims, obasis.shell_types, obasis.alphas, orca_con_coeffs)
                result["obasis"] = orca_obasis
                print("  Applied ORCA basis set fix")

    molden_module._fix_molden_from_buggy_codes = patched_fix

    try:
        result = molden_module.load_molden(input_file)
    finally:
        molden_module._fix_molden_from_buggy_codes = original_fix

    obasis = result["obasis"]
    orb_alpha = result["orb_alpha"]
    orb_beta = result.get("orb_beta")

    print(f"  {obasis.nbasis} basis functions, {orb_alpha.nfn} alpha orbitals")

    # Compute overlap matrix
    olp = obasis.compute_overlap()

    # Renormalize alpha orbitals
    print("  Renormalizing alpha orbitals...")
    max_norm = 0.0
    for i in range(orb_alpha.nfn):
        c = orb_alpha._coeffs[:, i]
        norm_sq = np.dot(c, np.dot(olp, c))
        norm = np.sqrt(norm_sq)
        max_norm = max(max_norm, norm)
        orb_alpha._coeffs[:, i] /= norm

    print(f"    Max initial norm: {max_norm:.6f} (should be ~1.0 for correct files)")

    # Renormalize beta orbitals if present
    if orb_beta is not None:
        print("  Renormalizing beta orbitals...")
        for i in range(orb_beta.nfn):
            c = orb_beta._coeffs[:, i]
            norm_sq = np.dot(c, np.dot(olp, c))
            norm = np.sqrt(norm_sq)
            orb_beta._coeffs[:, i] /= norm

    # Verify correction
    print("  Verifying normalization...")
    max_error = 0.0
    for i in range(orb_alpha.nfn):
        c = orb_alpha._coeffs[:, i]
        norm = np.dot(c, np.dot(olp, c))
        error = abs(norm - 1.0)
        max_error = max(max_error, error)

    print(f"    Max error: {max_error:.2e} (target: < 1e-4)")

    # Save corrected molden file
    print(f"\nSaving to {output_file}...")

    kwargs = {
        "coordinates": result["coordinates"],
        "numbers": result["numbers"],
        "obasis": result["obasis"],
        "orb_alpha": orb_alpha,
    }

    if orb_beta is not None:
        kwargs["orb_beta"] = orb_beta

    iodata = IOData(**kwargs)

    # Add optional fields
    if "energy" in result:
        iodata.energy = result["energy"]
    if "permutation" in result:
        iodata.permutation = result["permutation"]

    iodata.to_file(output_file)

    print("Done. Use the fixed file:")
    print(f"  pyxdm {output_file} --scheme mbis")

    return output_file


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fix_molden_normalization.py <input> [output]")
        print("\nExample:")
        print("  python fix_molden_normalization.py orca.molden.input")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    fix_molden_file(input_file, output_file)
