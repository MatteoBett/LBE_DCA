from pathlib import Path
import numpy as np
import importlib
from typing import Tuple
import re

import torch

from genseq.tools.dataset import DatasetDCA
from genseq.tools.fasta import get_tokens
from genseq.tools.stats import get_freq_single_point, get_freq_two_points
from genseq.tools.utils import init_chains, init_parameters
from genseq.train_dir.sampling import get_sampler
from genseq.tools.functional import one_hot
import genseq.model.eaDCA as model
from genseq.tools.checkpoint import get_checkpoint

def get_target_gap_distribution(frac_target : float, 
                                data_distrib : torch.Tensor, 
                                distrib : torch.Tensor, 
                                passed : list = []) -> torch.Tensor:
    """ Determines the target frequency distribution of gaps given an overall mean """
    if frac_target <= 0:
        return distrib

    else:
        for index, val in enumerate(data_distrib):
            target_val = val*(frac_target/data_distrib.mean())
            if target_val > 1 and index not in passed:
                target_val = 1
                passed.append(index)
            
            new_val = distrib[index] + target_val
            if new_val > 1:
                passed.append(index)
                distrib[index] = 1
            else:
                distrib[index] = new_val

        unused_frac = frac_target-distrib.mean()
        return get_target_gap_distribution(frac_target=unused_frac,
                                           data_distrib=data_distrib,
                                           distrib=distrib,
                                           passed=passed)
    
def load_dataset(
        infile_path : str,
        device : str,
        clustering_th : float,
        folder : str, 
        alphabet = 'rna',
        dtype : torch.dtype = torch.float32,
        reset : bool = False):
    
    if reset:
        infile_path = re.sub("indel", "raw", infile_path)

    dataset = DatasetDCA(
        path_data=infile_path,
        alphabet=alphabet,
        device=device,
        dtype=dtype,
        clustering_th=clustering_th
    )

    tokens = get_tokens(alphabet=alphabet)
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

    return tokens, L, q, pseudocount, data_oh, fi_target, fij_target

def main(
        infile_path : str, 
        outdir: str, 
        DCA_model_ : str = "eaDCA", 
        bias : bool = False,
        MCMC_sampler : str = "gibbs",
        alphabet : str = 'rna',
        nchains : int = 10000,
        target_pearson : float=0.95,
        nsweeps : int =10,
        nepochs : int =500,
        lr : float =0.05,
        factivate : float =0.001,
        gsteps : int =10,
        drate : float =0.01,
        target_density : float=0.02,
        gaps_fraction : Tuple[float] = (0.0, 0.0),
        cluster_th : float = 0.92):
    
    gap_bias, del_bias = gaps_fraction
    args = locals()

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
        "log" : folder / Path(f"adabmDCA.log"),
        "params" : folder / Path(f"params.dat"),
        "chains" : folder / Path(f"chains.fasta")
    }

    # Import dataset
    print("Importing dataset...")
    tokens, L, q, pseudocount, data_oh, fi_target, fij_target = load_dataset(infile_path=infile_path,
                                                                            device=device,
                                                                            clustering_th=cluster_th,
                                                                            folder=folder,
                                                                            alphabet=alphabet,
                                                                            reset=False)                                                                        

    #dataset.get_indels_info()
    DCA_model = importlib.import_module(f"genseq.model.{DCA_model_}")
    
    if gap_bias > 0.0 and del_bias > 0.0 : #double bias
        fi_target_gap_distribution = fi_target[:, 1].cpu()
        fi_target_del_distribution = fi_target[:, 0].cpu()
        
        target_gap_distribution = torch.zeros((len(fi_target_gap_distribution), 1), dtype=torch.float32).cpu()
        target_del_distribution = torch.zeros((len(fi_target_del_distribution), 1), dtype=torch.float32).cpu()

        target_gap_distribution = get_target_gap_distribution(frac_target=gap_bias, 
                                                            data_distrib=fi_target_gap_distribution, 
                                                            distrib=target_gap_distribution)
        target_del_distribution = get_target_gap_distribution(frac_target=del_bias, 
                                                            data_distrib=fi_target_del_distribution, 
                                                            distrib=target_del_distribution)
        tokens, L, q, pseudocount, data_oh, fi_target, fij_target = load_dataset(infile_path=infile_path,
                                                                            device=device,
                                                                            clustering_th=cluster_th,
                                                                            folder=folder,
                                                                            alphabet='rna',
                                                                            reset=True)
        target_gap_distribution += target_del_distribution
                                                                        
    elif gap_bias > 0.0 and del_bias == 0.0 : #gap bias
        fi_target_gap_distribution = fi_target[:, 0].cpu()
        target_gap_distribution = torch.zeros((len(fi_target_gap_distribution), 1), dtype=torch.float32).cpu()

        target_gap_distribution = get_target_gap_distribution(frac_target=gap_bias, 
                                                            data_distrib=fi_target_gap_distribution, 
                                                            distrib=target_gap_distribution)

    elif gap_bias == 0.0 and del_bias > 0.0 : #indel bias
        fi_target_del_distribution = fi_target[:, 0].cpu()
        target_del_distribution = torch.zeros((len(fi_target_del_distribution), 1), dtype=torch.float32).cpu()

        target_gap_distribution = get_target_gap_distribution(frac_target=del_bias, 
                                                            data_distrib=fi_target_del_distribution, 
                                                            distrib=target_del_distribution)

    print("Targetted average gap frequency", target_gap_distribution.mean())

    params = init_parameters(fi=fi_target)
    if DCA_model_ in ["bmDCA", "edDCA"]:
        mask = torch.ones(size=(L, q, L, q), dtype=torch.bool, device=device)
        mask[torch.arange(L), :, torch.arange(L), :] = 0
        
    else:
        mask = torch.zeros(size=(L, q, L, q), device=device, dtype=torch.bool)
    

    print(f"Number of chains set to {nchains}.")
    chains = init_chains(num_chains=nchains, L=L, q=q, fi=fi_target, device=device, dtype=dtype)
    log_weights = torch.zeros(size=(nchains,), device=device, dtype=dtype)
    
    pi_train = get_freq_single_point(chains)
    print("pi_train gap freq: ", pi_train[:, 0].mean())
    # Select the sampling function
    sampler = get_sampler(MCMC_sampler)
    
    print("\n")
    

    checkpoint = get_checkpoint('linear')(
        file_paths=file_paths,
        tokens=tokens,
        args = args,
        params=params,
        chains=chains,
    )

    DCA_model.fit(
        sampler=sampler,
        fij_target=fij_target,
        fi_target=fi_target,
        params=params,
        mask=mask,
        chains=chains,
        log_weights=log_weights,
        tokens=tokens,
        target_pearson=target_pearson,
        pseudo_count=pseudocount,
        nsweeps=10,
        nepochs=500,
        lr=0.05,
        factivate=0.001,
        gsteps=10,
        gap_bias_flag=bias,
        drate=0.01,
        target_density=0.02,
        checkpoint=checkpoint,
        target_gap_distribution = target_gap_distribution
    )
    
    
if __name__ == "__main__":
    main()
