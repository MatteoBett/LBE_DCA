from pathlib import Path
import numpy as np

import torch

from genseq.dataset import DatasetDCA
from genseq.fasta import get_tokens
from genseq.stats import get_freq_single_point, get_freq_two_points
from genseq.utils import init_chains, init_parameters
from genseq.sampling import get_sampler
from genseq.functional import one_hot
import genseq.model.eaDCA as model
from genseq.checkpoint import LinearCheckpoint


def main(infile_path : str, 
         outdir: str, 
         DCA_model_ : str = "eaDCA", 
         bias : bool = False,
         MCMC_sampler : str = "gibbs",
         alphabet : str = 'rna',
         nchains : int = 10000):

    print("\n" + "".join(["*"] * 10) + f" Training {DCA_model_} model " + "".join(["*"] * 10) + "\n")
    # Set the device
    device = 'cuda'
    dtype = torch.float32

    template = "{0:<30} {1:<50}"
    print(template.format("Input MSA:", infile_path))
    print(template.format("Output folder:", outdir))
    print(template.format("Alphabet:", 'RNA (-AUCG)'))
    print(template.format("Sampler:", MCMC_sampler))
    print("\n")
    
    # Check if the data file exist
    if not Path(infile_path).exists():
        raise FileNotFoundError(f"Data file {infile_path} not found.")
    
    # Create the folder where to save the model
    folder = Path(outdir)
    folder.mkdir(parents=True, exist_ok=True)

    file_paths = {
    "params" : folder / Path(f"params.dat"),
    "chains" : folder / Path(f"chains.fasta")
    }

    # Import dataset
    print("Importing dataset...")
    dataset = DatasetDCA(
        path_data=infile_path,
        alphabet=alphabet,
        device=device,
        dtype=dtype,
    )
    
    tokens = get_tokens(alphabet)
    
    # Save the weights if not already provided
    path_weights = folder / "weights.dat"    
    np.savetxt(path_weights, dataset.weights.cpu().numpy())
    print(f"Weights saved in {path_weights}")
        
    # Set the random seed
    torch.manual_seed(42)
        
    # Shuffle the dataset
    dataset.shuffle()
    
    # Compute statistics of the data
    L = dataset.get_num_residues()
    q = dataset.get_num_states()
    
    pseudocount = 1. / dataset.get_effective_size()
    print(f"Pseudocount automatically set to {pseudocount}.")

    data_oh = one_hot(dataset.data, num_classes=q).to(dtype)
    fi_target = get_freq_single_point(data=data_oh, weights=dataset.weights, pseudo_count=pseudocount)
    fij_target = get_freq_two_points(data=data_oh, weights=dataset.weights, pseudo_count=pseudocount) 
    
    params = init_parameters(fi=fi_target)
    mask = torch.zeros(size=(L, q, L, q), device=device, dtype=torch.bool)
    
    print(f"Number of chains set to {nchains}.")
    chains = init_chains(num_chains=nchains, L=L, q=q, fi=fi_target, device=device, dtype=dtype)
    log_weights = torch.zeros(size=(nchains,), device=device, dtype=dtype)
        
    # Select the sampling function
    sampler = get_sampler(MCMC_sampler)
    print("\n")

    checkpoint = LinearCheckpoint(file_paths=file_paths,
                                  tokens=tokens)

    model.fit(
        sampler=sampler,
        fij_target=fij_target,
        fi_target=fi_target,
        params=params,
        mask=mask,
        chains=chains,
        log_weights=log_weights,
        tokens=tokens,
        target_pearson=0.95,
        pseudo_count=pseudocount,
        nsweeps=10,
        nepochs=500,
        lr=0.05,
        factivate=0.001,
        gsteps=10,
        gap_bias_flag=bias,
        drate=None,
        target_density=None,
        checkpoint=checkpoint
    )

