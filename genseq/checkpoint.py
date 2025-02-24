from typing import Dict, Any
from abc import ABC, abstractmethod
import torch
import h5py
import wandb

from adabmDCA.io import save_chains, save_params
from adabmDCA.statmech import _get_acceptance_rate


class Checkpoint(ABC):
    """Helper class to save the model's parameters and chains at regular intervals during training and to log the
    progress of the training.
    """
    def __init__(
        self,
        file_paths: dict,
        tokens: str,
    ):
        """Initializes the Checkpoint class.

        Args:
            file_paths (dict): Dictionary containing the paths of the files to be saved.
            tokens (str): Alphabet to be used for encoding the sequences.
            args (dict): Dictionary containing the arguments of the training.
            params (Dict[str, torch.Tensor] | None, optional): Parameters of the model. Defaults to None.
            chains (Dict[str, torch.Tensor] | None, optional): Chains. Defaults to None.
            use_wandb (bool, optional): Whether to use Weights & Biases for logging. Defaults to False.
        """           
        self.file_paths = file_paths
        self.tokens = tokens
        
        
    @abstractmethod 
    def save(
        self,
        params: Dict[str, torch.Tensor],
        mask: torch.Tensor,
        chains: Dict[str, torch.Tensor],
        log_weights: torch.Tensor,
    ) -> None:
        """Saves the chains and the parameters of the model.

        Args:
            params (Dict[str, torch.Tensor]): Parameters of the model.
            mask (torch.Tensor): Mask of the model's coupling matrix representing the interaction graph
            chains (Dict[str, torch.Tensor]): Chains.
            log_weights (torch.Tensor): Log of the chain weights. Used for AIS.
        """
        pass
    
    
class LinearCheckpoint(Checkpoint):
    def __init__(
        self,
        file_paths: dict,
        tokens: str,
        checkpt_interval: int = 50,
    ):
        super().__init__(
            file_paths=file_paths,
            tokens=tokens,
        )
        self.checkpt_interval = checkpt_interval

    def save(
        self,
        params: Dict[str, torch.Tensor],
        mask: torch.Tensor,
        chains: Dict[str, torch.Tensor],
        log_weights: torch.Tensor,
    ) -> None:
        """Saves the chains and the parameters of the model.

        Args:
            params (Dict[str, torch.Tensor]): Parameters of the model.
            mask (torch.Tensor): Mask of the model's coupling matrix representing the interaction graph
            chains (Dict[str, torch.Tensor]): Chains.
            log_weights (torch.Tensor): Log of the chain weights. Used for AIS.
        """           
        save_params(fname=self.file_paths["params"], params=params, mask=mask, tokens=self.tokens)
        save_chains(fname=self.file_paths["chains"], chains=chains.argmax(dim=-1), tokens=self.tokens, log_weights=log_weights)
        
            
            
class AcceptanceCheckpoint(Checkpoint):
    def __init__(
        self,
        file_paths: Dict,
        tokens: str,
        args: Dict,
        params: Dict[str, torch.Tensor] | None = None,
        chains: Dict[str, torch.Tensor] | None = None,
        target_acc_rate: float = 0.5,
        use_wandb: bool = False,
        **kwargs,
    ):
        super().__init__(
            file_paths=file_paths,
            tokens=tokens,
            args=args,
            params=params,
            chains=chains,
            use_wandb=use_wandb,
        )
        self.target_acc_rate = target_acc_rate
        self.num_saved_models = 0
        # Create a .h5 archive for storing the history of the parameters
        self.file_paths["params_history"] = self.file_paths["params"].with_suffix(".h5")
        with h5py.File(self.file_paths["params_history"], "w") as f:
            f["alphabet"] = self.tokens
            f.create_group("update_{0}".format(self.logs["Epochs"]))
            for key, value in params.items():
                f["update_{0}".format(self.logs["Epochs"])].create_dataset(key, data=value.cpu().numpy())
        
    def check(
        self,
        updates: int,
        curr_params: Dict[str, torch.Tensor],
        curr_chains: Dict[str, torch.Tensor],
        *args,
        **kwargs,
    ) -> bool:
        """Checks if a checkpoint has been reached by computing the acceptance rate of swapping the 
        configurations of the present model and the last saved model.
        
        Args:
            updates (int): Number of gradient updates performed.
            curr_params (Dict[str, torch.Tensor]): Current parameters of the model.
            curr_chains (Dict[str, torch.Tensor]): Current chains of the model.

        Returns:
            bool: Whether a checkpoint has been reached.
        """
        acc_rate = _get_acceptance_rate(
            prev_params=self.params,
            curr_params=curr_params,
            prev_chains=self.chains,
            curr_chains=curr_chains,
        )
        return (acc_rate < self.target_acc_rate) or (updates == self.max_epochs)
    
    
    def save(
        self,
        params: Dict[str, torch.Tensor],
        mask: torch.Tensor,
        chains: Dict[str, torch.Tensor],
        log_weights: torch.Tensor,
        *args,
        **kwargs,
    ) -> None:
        """Saves the chains and the parameters of the model and appends the current parameters to the
        file containing the parameters history.

        Args:
            params (Dict[str, torch.Tensor]): Parameters of the model.
            mask (torch.Tensor): Mask of the model's coupling matrix representing the interaction graph.
            chains (Dict[str, torch.Tensor]): Chains.
            log_weights (torch.Tensor): Log of the chain weights. Used for AIS.
        """
        if self.wandb:
            wandb.log(self.logs)
            
        # Store the current parameters and chains
        self.params = {key: value.clone() for key, value in params.items()}
        self.chains = chains.clone()
        # Append the current parameters to the history
        with h5py.File(self.file_paths["params_history"], "a") as f:
            f.create_group("update_{0}".format(self.logs["Epochs"]))
            for key, value in params.items():
                f["update_{0}".format(self.logs["Epochs"])].create_dataset(key, data=value.cpu().numpy())
        # Save the current parameters and chains
        save_params(fname=self.file_paths["params"], params=params, mask=mask, tokens=self.tokens)
        save_chains(fname=self.file_paths["chains"], chains=chains.argmax(dim=-1), tokens=self.tokens, log_weights=log_weights)
        # Update the log file
        out_string = " ".join([f"{value:<10.3f}" if isinstance(value, float) else f"{value:<10}" for value in self.logs.values()])
        with open(self.file_paths["log"], "a") as f:
            f.write(out_string + "\n")
        
            
def get_checkpoint(chpt: str) -> Checkpoint:
    if chpt == "linear":
        return LinearCheckpoint
    elif chpt == "acceptance":
        return AcceptanceCheckpoint
    else:
        raise ValueError(f"Checkpoint type {chpt} not recognized.")        