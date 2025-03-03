import os, re
from typing import List

from Bio import SeqIO
import torch
import numpy as np

def encode_seq(seq : SeqIO.SeqRecord, dico : dict[str, int] = {"-":0, "A":1, "C":2, "G":3, "U":4}):
    """ Numerically encode the sequence """
    return [dico[i] for i in seq]

def family_stream(family_dir : str):
    """ Yield the output of load_msa function for each family directory """
    for family_file in os.listdir(family_dir):
        yield family_file, os.path.join(family_dir, family_file, f"{family_file}.fasta")

def load_sequences(path_seq : str) -> List[List[str]]:
    """ 
    Load MSA in the form of a tensor matrix of shape L*N with:
        - L the length of the sequences
        - N the number of sequences 
    """
    return np.matrix([list(record.seq) for record in SeqIO.parse(path_seq, 'fasta')]).T.tolist()



