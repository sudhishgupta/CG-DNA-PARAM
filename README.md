# 🧬 **CG-DNA-PARAM – Coarse-Grained DNA Parameter Calculator**

This repository provides a **Python3 implementation** of helical parameter calculation for coarse-grained (CG) DNA, based on the original implementation of cgheliparam

It supports:

- **Single-frame structures** (`.pdb`, `.gro`, etc.)
- **Topology + trajectory** (`.pdb/.gro` + `.xtc/.trr`)

and computes:

---

## ✔ **Base-Pair Parameters**
Shear, Stretch, Stagger, Buckle, Propeller, Opening

## ✔ **Step Parameters**
Shift, Slide, Rise, Tilt, Roll, Twist

## ✔ **Local Helical Parameters**
X-disp, Y-disp, Helical Rise, Inclination, Tip, Helical Twist

---

## 📦 1. Requirements

Install dependencies:
```
pip install numpy MDAnalysis tqdm
```

If MDAnalysis fails to install with pip:
```
conda install -c conda-forge mdanalysis
```

## 🚩 2. Usage Guidelines

When analyzing a trajectory, make sure to remove all other atoms(solvent, ions, ligands, etc.) other than the DNA atoms.
You can use the following command for the same, if you are using GROMACS

```
gmx trjconv -f <traj.xtc> -s <md.tpr> -o <onlydna_traj.xtc>
```

Make sure to choose DNA group for output trajectory.


Once you have the trajectory file you can run the pipeline using the following command

```
python cg_analysis_pipeline.py --top_pdb <.pdb/.gro> --traj_file <.xtc> --out_folder <output_folder path/name>
```

```
options:
  -h, --help            show this help message and exit
  --top_pdb TOP_PDB     Topology/structure file (PDB or GRO)
  --traj_file TRAJ_FILE
                        Trajectory file (XTC/TRR) or 'None' for single-frame mode
  --out_folder OUT_FOLDER
                        Output folder to store results
```


### 📚 Citation

If you use **CG-DNA-PARAM** or this repository in your research, please cite:

> **Faustino, I., & Marrink, S. J. (2017).  
> cgHeliParm: analysis of dsDNA helical parameters for coarse-grained MARTINI molecular dynamics simulations.  
> *Bioinformatics*. btx444.**  
> https://doi.org/10.1093/bioinformatics/btx444

Please include this citation in any publications, presentations, or derived work.


Output is stored as a .npz file in the output folder with keys {basepair_parameters, step_parameters, local_parameters}
Each key corresponds to value which is a numpy array with dimensions (n_frames, n_residues, params)



