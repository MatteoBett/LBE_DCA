#########################################################
#                        std Lib                        #
#########################################################
import os, sys, argparse, shutil

#########################################################
#                      Dependencies                     #
#########################################################
import torch

#########################################################
#                      Own modules                      #
#########################################################
from genseq.tools import dataset, energies
import genseq.display.plot as display
import genseq.old
import genseq.old.generator
import genseq.train_dir.train as push


if __name__ == "__main__":
    family_dir = r'/home/mbettiati/LBE_MatteoBettiati/tests/Artificial_rybo_gen_jl/own_dca/data/input_test'
    outdir = r'/home/mbettiati/LBE_MatteoBettiati/tests/Artificial_rybo_gen_jl/own_dca/output'
    model_type = "eaDCA"
    run_generation = True
    run_energy = True
    bias = False

    for family_file, infile_path in dataset.family_stream(family_dir=family_dir):
        if bias:
            family_outdir = os.path.join(outdir, "sequences", family_file.split('.')[0], "biased")
        else:
            family_outdir = os.path.join(outdir, "sequences", family_file.split('.')[0], "non_biased")
        os.makedirs(family_outdir, exist_ok=True)
        
        if run_generation:
            if bias is True:
                push.main(infile_path=infile_path, outdir=family_outdir, DCA_model_=model_type, bias=bias)
            else:
                push.main(infile_path=infile_path, outdir=family_outdir, DCA_model_=model_type, bias=bias)

        chain_file = os.path.join(family_outdir, "chains.fasta")
        params_dca = os.path.join(family_outdir, "params.dat")
        weights_dca = os.path.join(family_outdir, "weights.dat")
        
        if run_energy:
            genseq.old.generator.compute_DCA_energies(dca_seq_path=chain_file, param_dca_path=params_dca, outdir=family_outdir)

        base_outdir = "/".join(family_outdir.split("/")[:-1])
        display.plot_energy_vs_gaps(base_outdir=base_outdir, path_save=os.path.join(outdir, "figures", f"{family_file.split('.')[0]}.png"))
