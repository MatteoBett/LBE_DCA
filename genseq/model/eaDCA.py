import time
from tqdm import tqdm
from typing import Callable, Tuple

import torch
import numpy as np
import matplotlib.pyplot as plt

from genseq.tools.stats import get_freq_single_point, get_freq_two_points, get_correlation_two_points
from genseq.train_dir.training import train_graph
from genseq.tools.utils import get_mask_save
from genseq.train_dir.graph import activate_graph, compute_density
from genseq.tools.statmech import compute_log_likelihood, compute_entropy, _compute_ess
from genseq.tools.checkpoint import Checkpoint

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

"""
    gaps_distribution = data[:, 0]
    target_gaps_distribution = []
    unused_frac = 0
    for index, val in gaps_distribution:
        target_val = gaps_distribution*(frac/gaps_distribution.mean())
        if target_val > 1:
            unused_frac += abs(1-target_val)*gaps_distribution.mean()/(frac*gaps_distribution)
            target_val = 1
"""

def fit(
    sampler: Callable,
    fi_target: torch.Tensor,
    fij_target: torch.Tensor,
    params: dict,
    mask: torch.Tensor,
    chains: torch.Tensor,
    log_weights: torch.Tensor,
    target_pearson: float,
    nsweeps: int,
    nepochs: int,
    pseudo_count: float,
    lr: float,
    factivate: float,
    gsteps: int,
    gap_bias_flag: bool,
    checkpoint: Checkpoint | None = None,
    gaps_fraction : Tuple[float, float] = (0.0, 0.0),
    *args, **kwargs,
) -> None:
    """
    Fits an eaDCA model on the training data and saves the results in a file.

    Args:
        sampler (Callable): Sampling function to be used.
        fi_target (torch.Tensor): Single-point frequencies of the data.
        fij_target (torch.Tensor): Two-point frequencies of the data.
        params (dict): Initialization of the model's parameters.
        mask (torch.Tensor): Initialization of the coupling matrix's mask.
        chains (torch.Tensor): Initialization of the Markov chains.
        log_weights (torch.Tensor): Log-weights of the chains. Used to estimate the log-likelihood.
        target_pearson (float): Pearson correlation coefficient on the two-points statistics to be reached.
        nsweeps (int): Number of Monte Carlo steps to update the state of the model.
        nepochs (int): Maximum number of epochs to be performed. Defaults to 50000.
        pseudo_count (float): Pseudo count for the single and two points statistics. Acts as a regularization.
        lr (float): Learning rate.
        factivate (float): Fraction of inactive couplings to activate at each step.
        gsteps (int): Number of gradient updates to be performed on a given graph.
        checkpoint (Checkpoint | None): Checkpoint class to be used to save the model. Defaults to None.
    """
    # Check the input sizes
    if fi_target.dim() != 2:
        raise ValueError("fi_target must be a 2D tensor")
    if fij_target.dim() != 4:
        raise ValueError("fij_target must be a 4D tensor")
    if chains.dim() != 3:
        raise ValueError("chains must be a 3D tensor")
    
    gap_bias, del_bias = gaps_fraction
    device = fi_target.device
    dtype = fi_target.dtype
    checkpoint.checkpt_interval = 10 # Save the model every 10 graph updates
    checkpoint.max_epochs = nepochs
    
    fi_target_gap_distribution = fi_target[:, 0].cpu()
    target_gap_distribution = torch.zeros((len(fi_target_gap_distribution), 1), dtype=torch.float32).cpu()

    target_gap_distribution = get_target_gap_distribution(frac_target=gaps_fraction[0], 
                                                          data_distrib=fi_target_gap_distribution, 
                                                          distrib=target_gap_distribution)
    
    target_del_distribution = get_target_gap_distribution(frac_target=gaps_fraction[1], 
                                                          data_distrib=fi_target_gap_distribution, 
                                                          distrib=target_gap_distribution)
    
    print("Targetted average gap frequency",target_gap_distribution.mean())
    print("Targetted average gap frequency",target_del_distribution.mean())

    graph_upd = 0
    density = compute_density(mask) * 100
    L, q = fi_target.shape
        
    # Mask for saving only the upper-diagonal matrix
    mask_save = get_mask_save(L, q, device=device)
    
    # log_weights used for the online computing of the log-likelihood
    logZ = (torch.logsumexp(log_weights, dim=0) - torch.log(torch.tensor(len(chains), device=device, dtype=dtype))).item()
    
    # Compute the single-point and two-points frequencies of the simulated data
    pi = get_freq_single_point(data=chains)
    pij = get_freq_two_points(data=chains)
    pearson = max(0, float(get_correlation_two_points(fij=fij_target, pij=pij, fi=fi_target, pi=pi)[0]))
    
    #print("pi gap freq eaDCA: ", pi[:, 0].mean())
    
    # Number of active couplings
    nactive = mask.sum()
    
    # Training loop
    time_start = time.time()
    log_likelihood = compute_log_likelihood(fi=fi_target, fij=fij_target, params=params, logZ=logZ)

    entropy = compute_entropy(chains=chains, params=params, logZ=logZ)
        
    pbar = tqdm(initial=max(0, float(pearson)), total=target_pearson, colour="red", dynamic_ncols=True, ascii="-#",
                bar_format="{desc}: {percentage:.2f}%[{bar}] Pearson: {n:.3f}/{total_fmt} [{elapsed}]")
    pbar.set_description(f"Graph updates: {graph_upd} - Gap avg freq: {pi[:,0].mean():.3f} - New active couplings: {0} - LL: {log_likelihood:.3f}")
    
    while pearson < target_pearson:
        # Old number of active couplings
        nactive_old = nactive
        
        # Compute the two-points frequencies of the simulated data with pseudo-count
        pij_Dkl = get_freq_two_points(data=chains, weights=None, pseudo_count=pseudo_count)

        # Update the graph
        nactivate = int(((L**2 * q**2) - mask.sum().item()) * factivate)
        mask = activate_graph(
            mask=mask,
            fij=fij_target,
            pij=pij_Dkl,
            nactivate=nactivate,
        )
        
        # New number of active couplings
        nactive = mask.sum()


        # Bring the model at convergence on the graph
        chains, params, log_weights = train_graph(
            sampler=sampler,
            chains=chains,
            mask=mask,
            fi=fi_target,
            fij=fij_target,
            params=params,
            nsweeps=nsweeps,
            lr=lr,
            max_epochs=gsteps,
            target_pearson=target_pearson,
            log_weights=log_weights,
            check_slope=False,
            checkpoint=None,
            progress_bar=False,
            gap_bias_flag=gap_bias_flag,
            gaps_target=target_gap_distribution
        )

        graph_upd += 1
        
        # Compute the single-point and two-points frequencies of the simulated data
        pi = get_freq_single_point(data=chains)
        pij = get_freq_two_points(data=chains)
        
        # Compute statistics of the training
        pearson, slope = get_correlation_two_points(fij=fij_target, pij=pij, fi=fi_target, pi=pi)
        density = compute_density(mask) * 100
        logZ = (torch.logsumexp(log_weights, dim=0) - torch.log(torch.tensor(len(chains), device=device, dtype=dtype))).item()
        log_likelihood = compute_log_likelihood(fi=fi_target, fij=fij_target, params=params, logZ=logZ)
        pbar.set_description(f"Graph updates: {graph_upd} - Gap avg freq: {pi[:,0].mean():.3f} - New active couplings: {int(nactive - nactive_old)} - DCA LL: {log_likelihood:.3f}")

        # Save the model if a checkpoint is reached
        if checkpoint.check(graph_upd, params, chains):
            entropy = compute_entropy(chains=chains, params=params, logZ=logZ)
            ess = _compute_ess(log_weights)
            checkpoint.log(
                {
                    "Epochs": graph_upd,
                    "Pearson": pearson,
                    "Slope": slope,
                    "LL_train": log_likelihood,
                    "ESS": ess,
                    "Entropy": entropy,
                    "Density": density,
                    "Time": time.time() - time_start,
                    "Gaps_freq": pi[:, 0].mean()
                }
            )
            
            checkpoint.save(
                params=params,
                mask=torch.logical_and(mask, mask_save),
                chains=chains,
                log_weights=log_weights,
                )
        pbar.n = min(max(0, float(pearson)), target_pearson)

    entropy = compute_entropy(chains=chains, params=params, logZ=logZ)
    ess = _compute_ess(log_weights)
    checkpoint.log(
        {
            "Epochs": graph_upd,
            "Pearson": pearson,
            "Slope": slope,
            "LL_train": log_likelihood,
            "ESS": ess,
            "Entropy": entropy,
            "Density": density,
            "Time": time.time() - time_start,
            "Gaps_freq": pi[:, 0].mean()
        }
    )
                
    checkpoint.save(
        params=params,
        mask=torch.logical_and(mask, mask_save),
        chains=chains,
        log_weights=log_weights,
        )
    print(f"Completed, model parameters saved in {checkpoint.file_paths['params']}")
    pbar.close()