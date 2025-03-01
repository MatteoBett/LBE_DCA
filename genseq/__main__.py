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
import genseq.secondary_structure.make_struct as make_SS
import genseq.display.plot_ss as plot_ss
import genseq.train_dir.train as push

"""
Semaine prochaine : Alphabet custom avec ajout d'un caractère dédié aux délétions
Just have to change the alphabet in the input : no need to change anything else.
    - Character for "deletions" : *
    - Alphabet to use: *-ACGU 
    - The token's dictionnary is therefore: 
        {'*':0, '-':1, 'A':2, 'C':3, 'G':4, 'U':5}
        and reverse:
        {0:'*', 1:'-', 2:'A', 3:'C', 4:'G', 5:'U}
"""

if __name__ == "__main__":
    family_dir = r'/home/mbettiati/LBE_MatteoBettiati/tests/Artificial_rybo_gen_jl/own_dca/data/input_test'
    outdir = r'/home/mbettiati/LBE_MatteoBettiati/tests/Artificial_rybo_gen_jl/own_dca/output'
    fig_dir = r'/home/mbettiati/LBE_MatteoBettiati/tests/Artificial_rybo_gen_jl/own_dca/output/figures'
    model_type = "eaDCA"
    run_generation = True
    run_energy = True
    bias = False 
    indel = False
    do_one = False

    gaps_fraction = 0.15
    nchains = 1613

    for family_file, infile_path in dataset.family_stream(family_dir=family_dir):
        if indel:
            alphabet = '*-ACGU'
        else:
            alphabet = 'rna'
            
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
                          target_pearson=0.83,
                          alphabet=alphabet)
            else:
                push.main(infile_path=infile_path, 
                          outdir=family_outdir, 
                          DCA_model_=model_type, 
                          bias=bias, 
                          nchains=nchains, 
                          gaps_fraction=gaps_fraction,
                          target_pearson=0.83,
                          alphabet=alphabet)

        
        if run_energy:
            energies.main(dca_seq_path=chain_file, param_dca_path=params_dca, outdir=family_outdir)

        base_outdir = "/".join(family_outdir.split("/")[:-1])

        biased_seqs = os.path.join(outdir, "sequences", family_file.split('.')[0], "biased", "chains_energies.fasta")
        unbiased_seqs = os.path.join(outdir, "sequences", family_file.split('.')[0], "non_biased", "chains_energies.fasta")
        
        plot_ss.homology_vs_gaps(chains_file_ref=unbiased_seqs, infile_path=infile_path, chains_file_bias=biased_seqs,indel=indel, fig_dir=fig_dir)

        if do_one:
            break