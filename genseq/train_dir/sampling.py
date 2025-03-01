from typing import Dict, Callable, Tuple

import torch
from torch.nn.functional import one_hot as one_hot_torch

from genseq.tools.functional import one_hot

@torch.jit.script
def compute_gap_gradient(target_dist : torch.Tensor,
                         dist_sample : torch.Tensor,
                         params : Dict[str, torch.Tensor],
                         device : str = 'cuda'
                         ) -> Dict[str, torch.Tensor]:
    """
    Computes the gradient of the bias applied to the gaps frequency and adjust it 
    toward a target distribution of gaps corresponding to a mean frequency of gaps in the sequence.
    """ 
    target_dist = target_dist.to(device=device)
    loss = target_dist - dist_sample
    new_bias = params["gaps_lr"] * loss # positive result
    #print("loss: ",loss.shape, "target: ", target_dist.shape, "dist: ",dist_sample.shape, "new val", new_bias[:, 0].shape, "param", params["bias"][:, 0].shape)
    params["gaps_bias"][:, 0] += new_bias[:, 0]
    return params

@torch.jit.script
def _gibbs_sweep(
    chains: torch.Tensor,
    residue_idxs: torch.Tensor,
    params: Dict[str, torch.Tensor],
    beta: float,
    target_gaps_dist : torch.Tensor,
    gap_bias_flag: bool = False
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Performs a Gibbs sweep over the chains.

    Args:
        chains (torch.Tensor): One-hot encoded sequences.
        residue_idxs (torch.Tensor): List of residue indices in random order.
        params (Dict[str, torch.Tensor]): Parameters of the model.
        beta (float): Inverse temperature.

    Returns:
        torch.Tensor: Updated chains.
    """
    N, L, q = chains.shape
    alloc_bias = torch.zeros((L, 1), device='cuda')
    for i in residue_idxs:
        # Select the couplings attached to the residue we are considering (i) and flatten along the other residues ({j})
        couplings_residue = params["coupling_matrix"][i].view(q, L * q)
        # Update the chains
        logit_residue = params["bias"][i].unsqueeze(0) + chains.reshape(N, L * q) @ couplings_residue.T # (N, q)
        make_proba = torch.softmax(beta*logit_residue, -1)

        sampled = torch.multinomial(make_proba, 1)
        chains[:, i, :] = one_hot(sampled, num_classes=q).to(logit_residue.dtype).squeeze(1)

        alloc_bias[i] = make_proba[:, 0].mean()
    
    if gap_bias_flag:
        params = compute_gap_gradient(
            target_dist=target_gaps_dist,
            dist_sample=alloc_bias,
            params=params
        )
        
    return chains, params


def gibbs_sampling(
    chains: torch.Tensor,
    params: Dict[str, torch.Tensor],
    nsweeps: int,
    gaps_target : torch.Tensor,
    beta: float = 1.0,
    gap_bias_flag: bool = False,
) -> torch.Tensor:
    """Gibbs sampling.
    
    Args:
        chains (torch.Tensor): Initial chains.
        params (Dict[str, torch.Tensor]): Parameters of the model.
        nsweeps (int): Number of sweeps.
        beta (float, optional): Inverse temperature. Defaults to 1.0.
    
    Returns:
        torch.Tensor: Updated chains.
    """
    L = params["bias"].shape[0]
    
    for t in torch.arange(nsweeps):
        # Random permutation of the residues
        residue_idxs = torch.randperm(L)
        chains, params = _gibbs_sweep(chains, residue_idxs, params, beta, gaps_target, gap_bias_flag)
        
    return chains, params


def _get_deltaE(
        idx: int,
        chain: torch.Tensor,
        residue_old: torch.Tensor,
        residue_new: torch.Tensor,
        params: Dict[str, torch.Tensor],
        L: int,
        q: int,
    ) -> float:
    
        coupling_residue = chain.view(-1, L * q) @ params["coupling_matrix"][:, :, idx, :].view(L * q, q) # (N, q)
        E_old = - residue_old @ params["bias"][idx] - torch.vmap(torch.dot, in_dims=(0, 0))(coupling_residue, residue_old)
        E_new = - residue_new @ params["bias"][idx] - torch.vmap(torch.dot, in_dims=(0, 0))(coupling_residue, residue_new)
        
        return E_new - E_old
    

def _metropolis_sweep(
    chains: torch.Tensor,
    params: Dict[str, torch.Tensor],
    beta: float,
) -> torch.Tensor:
    """Performs a Metropolis sweep over the chains.

    Args:
        chains (torch.Tensor): One-hot encoded sequences.
        params (Dict[str, torch.Tensor]): Parameters of the model.
        beta (float): Inverse temperature.

    Returns:
        torch.Tensor: Updated chains.
    """
    
    N, L, q = chains.shape
    residue_idxs = torch.randperm(L)
    for i in residue_idxs:
        res_old = chains[:, i, :]
        res_new = one_hot_torch(torch.randint(0, q, (N,), device=chains.device), num_classes=q).float()
        delta_E = _get_deltaE(i, chains, res_old, res_new, params, L, q)
        accept_prob = torch.exp(- beta * delta_E).unsqueeze(-1)
        chains[:, i, :] = torch.where(accept_prob > torch.rand((N, 1), device=chains.device, dtype=chains.dtype), res_new, res_old)

    return chains
    

def metropolis(
    chains: torch.Tensor,
    params: Dict[str, torch.Tensor],
    nsweeps: int,
    beta: float = 1.0,
) -> torch.Tensor:
    """Metropolis sampling.

    Args:
        chains (torch.Tensor): One-hot encoded sequences.
        params (Dict[str, torch.Tensor]): Parameters of the model.
        nsweeps (int): Number of sweeps to be performed.
        beta (float, optional): Inverse temperature. Defaults to 1.0.

    Returns:
        torch.Tensor: Updated chains.
    """

    for _ in range(nsweeps):
        chains = _metropolis_sweep(chains, params, beta)

    return chains


def get_sampler(sampling_method: str) -> Callable:
    """Returns the sampling function corresponding to the chosen method.

    Args:
        sampling_method (str): String indicating the sampling method. Choose between 'metropolis' and 'gibbs'.

    Raises:
        KeyError: Unknown sampling method.

    Returns:
        Callable: Sampling function.
    """
    if sampling_method == "gibbs":
        return gibbs_sampling
    elif sampling_method == "metropolis":
        return metropolis
    else:
        raise KeyError("Unknown sampling method. Choose between 'metropolis' and 'gibbs'.")