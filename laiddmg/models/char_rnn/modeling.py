from typing import Tuple

import torch
import torch.nn as nn
# import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils

from .configuration import CharRNNConfig
from ...modeling_utils import BaseModel
from ... import logging

logger = logging.get_logger(__name__)


class CharRNNModel(BaseModel):

  def __init__(self, config: CharRNNConfig):
    super(CharRNNModel, self).__init__()
    self.config = config
    self.vocab_size = config.vocab_size
    self.embedding_dim = config.embedding_dim
    self.hidden_dim = config.hidden_dim
    self.num_layers = config.num_layers
    self.dropout = config.dropout
    self.padding_value = config.padding_value
    self.output_dim = self.vocab_size

    self.embeddings = nn.Embedding(self.vocab_size, self.embedding_dim,
                                   padding_idx=self.padding_value)
    self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim,
                        self.num_layers,
                        batch_first=True,
                        dropout=self.dropout)
    self.fc = nn.Linear(self.hidden_dim, self.output_dim)

  def forward(
    self,
    input_ids: torch.Tensor,  # (batch_size, seq_len)
    lengths: torch.Tensor,  # (batch_size,)
    hiddens: Tuple[torch.Tensor] = None,  # (num_layers, batch_size, hidden_dim)
    **kwargs,
  ) -> Tuple[torch.Tensor, Tuple[torch.Tensor]]:
    x = self.embeddings(input_ids)  # x: (batch_size, seq_len, embedding_dim)
    x = rnn_utils.pack_padded_sequence(
      x,
      lengths.cpu(),
      batch_first=True,
      enforce_sorted=False,
    )
    x, hiddens = self.lstm(x, hiddens)
    # hiddens: (h, c); (num_layers, batch_size, hidden_dim), respectively
    x, _ = rnn_utils.pad_packed_sequence(
      x,
      batch_first=True,
    )  # x: (batch_size, seq_len, hidden_dim)
    outputs = self.fc(x)  # outputs: (batch_size, seq_len, vocab_size)

    return outputs, hiddens
