# Hydrogen_bond_network_OH_SiO2

Hydrogen Bond and Tilt Angle Analysis for Functionalized OH Groups on SiO2 Surface

---

## Table of Contents

- [Theory](#theory)  
- [Installation](#installation)  
- [Global Variables](#global-variables)  
- [How to Cite](#how-to-cite)  

---

## Theory

### Modifications in this version

1. **Global cutoff variables** are defined so that the hydrogen‐bond cutoff (`HB_CUTOFF`) and the O1–H cutoff (`OH_CUTOFF`) can be adjusted in one place.  
2. For each MD frame, the hydrogen bond (O2 acceptor) is **recalculated** for each OH group so that the number of valid hydrogen bonds is updated every frame.  
3. **Progress reporting** is added to show how many frames have been analyzed.

This script:

- **Reads** `XDATCAR` (with non‑orthogonal cells and periodic boundaries).  
- **Adjusts z coordinates** (mapping z in \[0.9,1) to negative values, shifting upward, and recentering so that the slab center is 0.5).  
- In the **first frame**, identifies OH groups (each H bonded to an oxygen within `OH_CUTOFF`).  
- For **every frame**:  
  - Calculates the O1–H bond length and tilt angle (for every OH group).  
  - Recalculates the hydrogen bond acceptor (O2) for each OH group and, if found, validates the hydrogen bond if H–O2 distance < `HB_CUTOFF` and the O1–H–O2 angle is between 120° and 180°.  
- **Writes** per-frame data and overall averages to output files.

---

## Installation
   - Python 3.x (≥ 3.6)  
   - No external Python dependencies beyond the standard library.

## Global Variables

```bash
OH_CUTOFF=1.2   # O1–H bond cutoff distance (Å)
HB_CUTOFF=3.0   # Hydrogen‐bond cutoff distance (Å)
```


## How to Cite

If you use this code in your work, please cite it using the following BibTeX entry:

```bash
@misc{strugovshchikov2025hydrogen,
  author       = {Strugovshchikov, E. and Mandrolko, V.},
  title        = {{Hydrogen\_bond\_network\_OH\_SiO2}},
  year         = {2025},
  howpublished = {GitHub repository},
  url          = {https://github.com/your-username/Hydrogen_bond_network_OH_SiO2},
  note         = {Accessed: May 20, 2025},
}
```

