#########################################################
#                        std Lib                        #
#########################################################
import os, sys, subprocess

#########################################################
#                      Dependencies                     #
#########################################################
from genseq.train_dir.train import main

#########################################################
#                      Own modules                      #
#########################################################

def run_own_eaDCA(infile_path : str, outdir: str, DCA_model : str = "eaDCA", bias : bool = False):
    main(infile_path, outdir, DCA_model, bias)

def run_own_energies(dca_seq_path : str, param_dca_path : str, outdir : str):
    pass

def run_eaDCA(infile_path : str, outdir: str, DCA_model_ : str = "eaDCA", bias : bool = False):
    """ Run the desired DCA model """
    if bias :
        subprocess.run(
            ["adabmDCA", "train" ,"-m", DCA_model_, "-d", infile_path, "-o", outdir, "--alphabet", "rna", "--gap_bias_flag", str(bias)]
        )
    else:
        subprocess.run(
            ["adabmDCA", "train" ,"-m", DCA_model_, "-d", infile_path, "-o", outdir, "--alphabet", "rna"]
        )
    return 0

def compute_DCA_energies(dca_seq_path : str, param_dca_path : str, outdir : str):
    """ Compute the DCA energy for all generated sequences """
    subprocess.run(
        ["adabmDCA", "energies" ,"-d", dca_seq_path, "-p", param_dca_path, "-o", outdir, "--alphabet", "rna"]
    )
    return 0