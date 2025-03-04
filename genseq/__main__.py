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
import genseq.display.plot as plot
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
    bias = True 
    double_bias = False
    indel = True
    do_one = True
    plotting = True

    nchains = 2720
    target_pearson = 0.94

    if double_bias:
        assert not bias, "Simple and Double bias cannot be considered at the same time!"

    if indel:
        alphabet = '*-AUCG'
        out = 'indel'
        family_dir = os.path.join(family_dir, "indel")
        gaps_fraction = (0, 0.03)
    elif double_bias:
        alphabet = '*-AUCG'
        out = 'double'
        family_dir = os.path.join(family_dir, "double")
        gaps_fraction = (0.15, 0.03)
    else:
        alphabet = 'rna'
        out = 'raw'
        family_dir = os.path.join(family_dir, 'raw')
        gaps_fraction = (0.15, 0)

    plot.get_summary(family_dir)

    for family_file, infile_path in dataset.family_stream(family_dir=family_dir):
        if bias:
            family_outdir = os.path.join(outdir, "sequences", out, family_file.split('.')[0], "biased")
        else:
            family_outdir = os.path.join(outdir, "sequences", out, family_file.split('.')[0], "non_biased")
        
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
                          target_pearson=target_pearson,
                          alphabet=alphabet)
            else:
                push.main(infile_path=infile_path, 
                          outdir=family_outdir, 
                          DCA_model_=model_type, 
                          bias=bias, 
                          nchains=nchains, 
                          gaps_fraction=gaps_fraction,
                          target_pearson=target_pearson,
                          alphabet=alphabet)

        if run_energy:
            energies.main(dca_seq_path=chain_file, param_dca_path=params_dca, outdir=family_outdir, alphabet=alphabet)

        base_outdir = "/".join(family_outdir.split("/")[:-1])

        biased_seqs = os.path.join(outdir, "sequences", out, family_file.split('.')[0], "biased", "chains_energies.fasta")
        unbiased_seqs = os.path.join(outdir, "sequences", out, family_file.split('.')[0], "non_biased", "chains_energies.fasta")

        biased_params = os.path.join(outdir, "sequences", out, family_file.split('.')[0], "biased", "params.dat")
        unbiased_params = os.path.join(outdir, "sequences", out, family_file.split('.')[0], "non_biased", "params.dat")
        
        if plotting:
            fig_dir = os.path.join(fig_dir, family_file.split('.')[0])
            os.makedirs(fig_dir, exist_ok=True)
            plot_ss.homology_vs_gaps(chains_file_ref=unbiased_seqs, 
                                     infile_path=infile_path, 
                                     chains_file_bias=biased_seqs,
                                     indel=indel, 
                                     fig_dir=fig_dir,
                                     params_path_unbiased=unbiased_params,
                                     params_path_biased=biased_params,
                                     alphabet=alphabet)

        if do_one:
            break