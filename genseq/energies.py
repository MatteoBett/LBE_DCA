import os
from pathlib import Path

import torch

from genseq.fasta import get_tokens
from genseq.io import load_params, import_from_fasta
from genseq.fasta import encode_sequence
from genseq.functional import one_hot
from genseq.statmech import compute_energy

def main(dca_seq_path : str, 
         param_dca_path : str, 
         outdir : str,
         alphabet: str = 'rna'):
    
    print("\n" + "".join(["*"] * 10) + f" Computing DCA energies " + "".join(["*"] * 10) + "\n")
    
    
    # Set the device
    device = 'cuda'
    dtype = torch.float32
    
    # Check if the data file exists
    if not Path(dca_seq_path).exists():
        raise FileNotFoundError(f"Data file {dca_seq_path} not found.")
    
    # Check if the parameters file exists
    if not Path(param_dca_path).exists():
        raise FileNotFoundError(f"Parameters file {param_dca_path} not found.")
    
    # import data
    tokens = get_tokens(alphabet)
    names, sequences = import_from_fasta(dca_seq_path)
    data = encode_sequence(sequences, tokens)
    data = torch.tensor(data, device=device, dtype=torch.int32)
    
    # import parameters and compute DCA energies
    print(f"Loading parameters from {param_dca_path}...")
    params = load_params(param_dca_path, tokens=tokens, device=device, dtype=dtype)
    q = params["bias"].shape[1]
    data = one_hot(data, num_classes=q).to(dtype)
    print(f"Computing DCA energies...")
    energies = compute_energy(data, params).cpu().numpy()
    
    # Save results in a file
    print("Saving results...")
    fname_out = outdir / Path(dca_seq_path.split('.')[0] + "_energies.fasta")
    with open(fname_out, "w") as f:
        for n, s, e in zip(names, sequences, energies):
            f.write(f">{n} | DCAenergy: {e:.3f}\n")
            f.write(f"{s}\n")
    
    print(f"Process completed. Output saved in {fname_out}")
    
    