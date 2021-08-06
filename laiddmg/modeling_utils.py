
from abc import abstractmethod

import torch
import torch.nn as nn

from . import logging


logger = logging.get_logger(__name__)


class BaseModel(nn.Module):

  @property
  def device(self) -> torch.device:
    try:
      return next(self.parameters()).device
    except StopIteration:
      return 0

  @property
  def dtype(self) -> torch.dtype:
    try:
      return next(self.parameters()).dtype
    except StopIteration:
      return 0

  def num_parameters(self, only_trainable: bool = False, exclude_embeddings: bool = False) -> int:

    def parameter_filter(x):
      return (x.requires_grad or not only_trainable) and not (
        isinstance(x, nn.Embedding) and exclude_embeddings
      )

    params = filter(parameter_filter, self.parameters()) if only_trainable else self.parameters()
    return sum(p.numel() for p in params)

  @abstractmethod
  def generate(self, **kwargs):
    pass
