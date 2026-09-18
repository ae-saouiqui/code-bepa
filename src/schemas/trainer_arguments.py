
import torch
from typing import NamedTuple

class CodeBEPATrainingArgs(NamedTuple):
    model : torch.nn.Module
    optimizer : torch.optim.Optimizer
    train_dataloader : torch.utils.data.DataLoader
    val_dataloader : torch.utils.data.DataLoader
    alignement_weight : float
    device : str
    accumulation_step : int


