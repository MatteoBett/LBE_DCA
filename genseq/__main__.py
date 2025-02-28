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
import genseq.secondary_structure.make_struct as make_SS
import genseq.display.plot_ss as plot_ss
import genseq.train_dir.train as push


if __name__ == "__main__":
    family_dir = r'/home/mbettiati/LBE_MatteoBettiati/tests/Artificial_rybo_gen_jl/own_dca/data/input_test'
    outdir = r'/home/mbettiati/LBE_MatteoBettiati/tests/Artificial_rybo_gen_jl/own_dca/output'
    model_type = "eaDCA"
    run_generation = False
    run_energy = False
    bias = True 
    indel = False
    gaps_fraction = 0.15
    nchains = 2721

    for family_file, infile_path in dataset.family_stream(family_dir=family_dir):
        if bias:
            family_outdir = os.path.join(outdir, "sequences", family_file.split('.')[0], "biased")
        else:
            family_outdir = os.path.join(outdir, "sequences", family_file.split('.')[0], "non_biased")
        os.makedirs(family_outdir, exist_ok=True)

        chain_file = os.path.join(family_outdir, "chains.fasta")
        params_dca = os.path.join(family_outdir, "params.dat")
        weights_dca = os.path.join(family_outdir, "weights.dat")
        energy_chains = os.path.join(family_outdir, "chains_energies.fasta")
        
        if run_generation:
            if bias is True:
                push.main(infile_path=infile_path, 
                          outdir=family_outdir, 
                          DCA_model_=model_type, 
                          bias=bias, 
                          nchains=nchains, 
                          gaps_fraction=gaps_fraction,
                          target_pearson=0.83)
            else:
                push.main(infile_path=infile_path, 
                          outdir=family_outdir, 
                          DCA_model_=model_type, 
                          bias=bias, 
                          nchains=nchains, 
                          gaps_fraction=gaps_fraction,
                          target_pearson=0.83)

        
        if run_energy:
            energies.main(dca_seq_path=chain_file, param_dca_path=params_dca, outdir=family_outdir)

        base_outdir = "/".join(family_outdir.split("/")[:-1])

        if os.path.exists(energy_chains):      
            plot_ss.homology_vs_gaps(chains_file=chain_file, infile_path=infile_path, bias=bias, indel=indel)
        else:
            print("Sequences energy is not available. Switching to normal chains. Energy will not be plotted")
            plot_ss.homology_vs_gaps(chains_file=energy_chains, infile_path=infile_path, bias=bias, indel=indel)
