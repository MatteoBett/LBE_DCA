import os, sys, argparse

import BW_imp.loader as loader
import BW_imp.models as models

if __name__ == "__main__":
    all_families = r'/home/mbettiati/LBE_MatteoBettiati/tests/Artificial_rybo_gen_jl/own_dca/data/input_test/raw'
    outpath = r'/home/mbettiati/LBE_MatteoBettiati/tests/Artificial_rybo_gen_jl/own_dca/data/input_test/indel'
    strategy = "mean"

    model = models.distributor(strategy=strategy)
    for family_file, infile_path in loader.family_stream(all_families):
        data = loader.load_sequences(infile_path)
        alt_msa = models.get_mean_gaps_per_position(data)

        family_outdir = os.path.join(outpath, family_file.split(".")[0])
        os.makedirs(family_outdir, exist_ok=True)
        fam_outpath = os.path.join(family_outdir, f"{family_file}.fasta")
        models.send_result(alt_msa, outpath=fam_outpath)
        