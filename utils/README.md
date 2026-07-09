# Utilities

## fix_molden_normalization.py

`orca_2mkl` seems to have a bug where certain molecules get written with incorrect orbital normalization. 

When you hit an error like:

```
Error loading wavefunction file: Could not correct the data read from orca.molden.input. The molden or mkl file you are trying to load contains errors. Please report this problem to Toon.Verstraelen@UGent.be, so he can fix it.
```

Use this script to fix the normalization:

```bash
python utils/fix_molden_normalization.py orca.molden.input
```

This creates `fixed_orca.molden.input` with properly normalized orbitals that PyXDM can process.

### Technical details

HORTON tries 4 automatic fixes (ORCA-specific, PSI4, Turbomole, general renormalization), but all fail for this bug. The issue requires both fixing the basis set contractions (ORCA fix) AND renormalizing the MO coefficients. HORTON only attempts basis fixes but checks the final orbital normalization, which still fails. This script applies HORTON's ORCA basis fix first, then renormalizes each orbital coefficient against the corrected overlap matrix.
