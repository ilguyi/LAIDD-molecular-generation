from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils

from .configuration import CharRNNConfig
from ...tokenization_utils import Tokenizer
from ...modeling_utils import BaseModel
from ... import logging_utils

logger = logging_utils.get_logger(__name__)


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

  def reset_states(self, batch_size: int):
    h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(self.device)
    c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(self.device)

    return (h0, c0)

  @torch.no_grad()
  def generate(
    self,
    tokenizer: Tokenizer = None,
    max_length: int = 128,
    num_return_sequences: int = 1,
    skip_special_tokens: bool = False,
    **kwargs,
  ) -> Union[List[List[int]], List[List[str]]]:

    initial_inputs = torch.full((num_return_sequences, 1),
                                tokenizer.convert_token_to_id(tokenizer.start_token),
                                dtype=torch.long,
                                device=self.device)
    generated_sequences = initial_inputs
    input_ids = initial_inputs  # input_ids: [batch_size, 1]
    hiddens = self.reset_states(num_return_sequences)

    for i in range(max_length + 1):
      x = self.embeddings(input_ids)  # x: [batch_size, 1, embedding_dim]
      x, hiddens = self.lstm(x, hiddens)  # x: [batch_size, 1, hidden_dim]
      logits = self.fc(x)  # logits: [batch_size, 1, vocab_size]
      next_token_logits = logits.squeeze(1)  # next_token_logits: [batch_size, vocab_size]

      probabilities = F.softmax(next_token_logits, dim=-1)  # probabilities: [batch_size, vocab_size]
      next_tokens = torch.multinomial(probabilities, num_samples=1)
      # next_tokens: [batch_size, 1]

      input_ids = next_tokens
      generated_sequences = torch.cat([generated_sequences, next_tokens], dim=1)
      # generated_sequences: [batch_size, max_length]

    generated_sequences = self.postprocessing(generated_sequences, tokenizer)

    generated_SMILES = []
    for sequence in generated_sequences:
      generated_SMILES.append(tokenizer.decode(sequence, skip_special_tokens))

    return generated_SMILES
