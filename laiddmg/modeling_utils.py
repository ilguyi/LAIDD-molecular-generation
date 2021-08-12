import os
from abc import abstractmethod
from typing import List, Union, Any

import torch
import torch.nn as nn

from .tokenization_utils import Tokenizer

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

  @classmethod
  def from_pretrained(
    cls,
    config: Any,
    ckpt_path: Union[str, os.PathLike],
    output_info: bool = False
  ):
    model = cls(config)
    state_dict = torch.load(ckpt_path, map_location='cpu')

    epoch = state_dict['epoch']
    global_step = state_dict['global_step']
    model.load_state_dict(state_dict['model_state_dict'])
    logger.info('All keys matched successfully and success to load')

    if output_info:
      return model, epoch, global_step
    else:
      return model

  def postprocessing(
    self,
    generated_sequences: torch.LongTensor,
    tokenizer: Tokenizer,
  ) -> List[List[int]]:
    pad_index = tokenizer.convert_token_to_id(tokenizer.pad_token)
    end_index = tokenizer.convert_token_to_id(tokenizer.end_token)
    new_sequences = []
    for sequence in generated_sequences:
      new_seq = []
      for token in sequence:
        if token.item() not in [pad_index, end_index]:
          new_seq.append(token.item())
        else:
          break
      new_sequences.append(new_seq)

    return new_sequences
