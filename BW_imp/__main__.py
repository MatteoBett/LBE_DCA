import os, sys, argparse

import BW_imp.loader as loader
import BW_imp.models as models

if __name__ == "__main__":
    all_families = r'C:\Subpbiotech_cours\BT5\Stage\DCA\LBE_DCA\data\families_GII'
    outpath = r'C:\Subpbiotech_cours\BT5\Stage\DCA\LBE_DCA\output_BW_imp\converted_fam'
    strategy = "mean"

    model = models.distributor(strategy=strategy)
    for family_file, infile_path in loader.family_stream(all_families):
        data = loader.load_sequences(infile_path)
        alt_msa = models.get_mean_gaps_per_position(data)

        fam_outpath = os.path.join(outpath, f"{family_file}.fasta")
        models.send_result(alt_msa, outpath=fam_outpath)
        