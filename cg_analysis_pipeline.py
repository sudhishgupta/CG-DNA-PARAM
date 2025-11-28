import warnings
warnings.filterwarnings('ignore')
import numpy as np
import sys
import os
import glob
import pickle
import pandas as pd
# from cgheliparm_py3 import *
from cgheliparam_core import *
import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Run cgheliparam_core")

    parser.add_argument(
        "--top_pdb",
        type=str,
        default=None,
        help="Topology/structure file (PDB or GRO)"
    )

    parser.add_argument(
        "--traj_file",
        type=str,
        default=None,
        help="Trajectory file (XTC/TRR) or 'None' for single-frame mode"
    )

    parser.add_argument(
        "--out_folder",
        type=str,
        default=None,
        help="Output folder to store results"
    )

    return parser.parse_args()

def main():

    args = get_args()

    top_pdb = args.top_pdb
    traj_file = args.traj_file
    out_folder = args.out_folder

    if out_folder is not None:
        
        # top_pdb = sys.argv[1]
        # traj_file = sys.argv[2]
        # out_folder = sys.argv[3]

        #pdb_frames = glob.glob(os.path.join(pdb_folder,'*.pdb'))

        basepair_data = {}
        step_data = {}
        local_data = {}

        print('='*20)
        print("COARSE-GRAIN-DNA-PARAMETERS-ANALYSIS")
        print("Usage : python cg_analysis_")
        print('='*20)
        try:
            if traj_file is not None:
                bp, step, local = run_cgheliparam(top_pdb, trajectory=traj_file, library_path='./data/', verbose=True)
            else:
                bp, step, local = run_cgheliparam(top_pdb, trajectory=None, library_path='./data/')
        except Exception as e:
            print(f'Failed Due to Exception  :: {e}')

        
        os.makedirs(out_folder, exist_ok=True)

        outfile = os.path.join(out_folder,'cg_helix_parameters')

        np.savez_compressed(f"{outfile}.npz",
                                basepair_parameters=bp,
                                step_parameters=step,
                                local_parameters=local)
        
        print('='*20)
        print(f"Saved results to {outfile}.npz")
        print('='*20)
    
    else:

        print('Output folder not specified, please specify output_folder as --out_folder <path/name>')


if __name__ == '__main__':
    main()
        


