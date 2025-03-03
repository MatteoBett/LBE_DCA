import os, re
from typing import List
from collections import Counter

from Bio import SeqIO
import torch
import numpy as np

def get_mean_gaps_per_position(msa : List[List[str]]) -> List[str]:
    """
    Takes as input a matrix msa of size L*N and returns a single vector of
    size L where each index element correspond to either an insertion or a
    deletion.
    """
    freq = []
    for msa_col in msa:
        gaps = msa_col.count('-')
        freq.append(gaps/len(msa_col))

    new_msa = []
    threshold = np.mean(freq)
    for index, msa_col in enumerate(msa):
        if freq[index] < threshold:
            new_msa.append(list(re.sub("-", "*", "".join(msa_col))))
        else:
            new_msa.append(msa_col)

    return np.matrix(new_msa).T.tolist()

def send_result(new_msa : List[List[str]], outpath : str) -> None:
    with open(outpath, 'w') as outfasta:
        for index, seqlist in enumerate(new_msa):
            outfasta.write(f">sequence {index}\n{''.join(seqlist)}\n")



    
def viterbi_main():
    pass

def get_initial_states(msa : List[str]) -> torch.Tensor:
    """ 
    Determine the initial vector of probabilities of size q*S with S the number of 
    states in the model. Here |S| = 3, so that S = {'M', 'I', 'D'} with:
        - M a match : less than 10% of gaps at a position i
        - I an insertion : more than 50% of gaps at a position i
        - D a deletion : between 10 and 50% of gaps at a position

    This is susceptible to evolve depending on how much the bias should then be applied
    strongly on deletion or insertions for DCA.

    Determines 
    """
    counting = Counter()
    transition_matrix = torch.zeros((5, 5))
    for seq in msa:
        counting += Counter(seq)

    


def distributor(strategy : str):
    if strategy == "mean":
        return get_mean_gaps_per_position
    elif strategy == "viterbi":
        return viterbi_main

