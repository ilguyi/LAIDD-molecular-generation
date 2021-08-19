from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils

from .configuration import VAEConfig
from ...tokenization_utils import Tokenizer
from ...modeling_utils import BaseModel
from ... import logging_utils

logger = logging_utils.get_logger(__name__)


class Encoder(nn.Module):

  def __init__(self, config: VAEConfig, embeddings: nn.Module = None):
    super(Encoder, self).__init__()
    self.vocab_size = config.vocab_size
    self.embedding_dim = config.embedding_dim
    self.encoder_hidden_dim = config.encoder_hidden_dim
    self.encoder_num_layers = config.encoder_num_layers
    self.encoder_dropout = config.encoder_dropout
    self.latent_dim = config.latent_dim
    self.padding_value = config.padding_value

    if embeddings is not None:
      self.embeddings = embeddings
    else:
      self.embeddings = nn.Embedding(self.vocab_size, self.embedding_dim,
                                     padding_idx=self.padding_value)

    self.gru = nn.GRU(self.embedding_dim,
                      self.encoder_hidden_dim,
                      self.encoder_num_layers,
                      batch_first=True,
                      dropout=self.encoder_dropout)
    self.fc = nn.Linear(self.encoder_hidden_dim, self.latent_dim * 2)

  def forward(
    self,
    input_ids: torch.Tensor,  # (batch_size, seq_len)
    lengths: torch.Tensor,  # (batch_size,)
    **kwargs,
  ) -> Tuple[torch.Tensor]:
    x = self.embeddings(input_ids)  # x: (batch_size, seq_len, embedding_dim)
    x = rnn_utils.pack_padded_sequence(
      x,
      lengths.cpu(),
      batch_first=True,
      enforce_sorted=False,
    )
    _, hiddens = self.gru(x, None)  # hiddens: (num_layers, batch_size, encoder_hidden_dim)

    hiddens = hiddens[-1]  # hiddens: (batch_size, encoder_hidden_dim) for last layer

    z_mu, z_logvar = torch.split(self.fc(hiddens), self.latent_dim, dim=-1)
    # z_mu, z_logvar: (batch_size, latent_dim)

    return z_mu, z_logvar


class Decoder(nn.Module):

  def __init__(self, config: VAEConfig, embeddings: nn.Module = None):
    super(Decoder, self).__init__()
    self.vocab_size = config.vocab_size
    self.embedding_dim = config.embedding_dim
    self.latent_dim = config.latent_dim
    self.decoder_hidden_dim = config.decoder_hidden_dim
    self.decoder_num_layers = config.decoder_num_layers
    self.decoder_dropout = config.decoder_dropout
    self.input_dim = self.embedding_dim + self.latent_dim
    self.output_dim = config.vocab_size
    self.padding_value = config.padding_value

    if embeddings is not None:
      self.embeddings = embeddings
    else:
      self.embeddings = nn.Embedding(self.vocab_size, self.embedding_dim,
                                     padding_idx=self.padding_value)

    self.gru = nn.GRU(self.input_dim,
                      self.decoder_hidden_dim,
                      self.decoder_num_layers,
                      batch_first=True,
                      dropout=self.decoder_dropout)
    self.z2hidden = nn.Linear(self.latent_dim, self.decoder_hidden_dim)
    self.fc = nn.Linear(self.decoder_hidden_dim, self.output_dim)

  def forward(
    self,
    input_ids: torch.Tensor,  # (batch_size, seq_len)
    lengths: torch.Tensor,  # (batch_size,)
    z: torch.Tensor,  # (batch_size, latent_dim)
    **kwargs,
  ) -> Tuple[torch.Tensor]:
    x = self.embeddings(input_ids)  # x: (batch_size, seq_len, embedding_dim)
    hiddens = self.z2hidden(z)  # hiddens: (batch_size, decoder_hidden_dim)
    hiddens = hiddens.unsqueeze(0).repeat(self.decoder_num_layers, 1, 1)
    # hiddens: (num_layers, batch_size, decoder_hidden_dim)

    z_ = z.unsqueeze(1).repeat(1, x.shape[1], 1)  # z: (batch_size, seq_len, latent_dim)
    x = torch.cat((x, z_), dim=-1)  # x: (batch_size, seq_len, embedding_dim + latent_dim)

    x = rnn_utils.pack_padded_sequence(
      x,
      lengths.cpu(),
      batch_first=True,
      enforce_sorted=False
    )
    x, _ = self.gru(x, hiddens)
    x, _ = rnn_utils.pad_packed_sequence(
      x,
      batch_first=True,
    )  # x: (batch_size, seq_len, hidden_dim)
    outputs = self.fc(x)  # outputs: (batch_size, seq_len, vocab_size)

    return outputs


class VAEModel(BaseModel):

  def __init__(self, config: VAEConfig):
    super(VAEModel, self).__init__()
    self.config = config
    self.vocab_size = config.vocab_size
    self.embedding_dim = config.embedding_dim
    self.latent_dim = config.latent_dim
    self.padding_value = config.padding_value

    self.embeddings = nn.Embedding(self.vocab_size, self.embedding_dim,
                                   padding_idx=self.padding_value)

    self.encoder = Encoder(self.config, self.embeddings)
    self.decoder = Decoder(self.config, self.embeddings)

  def reparameterize(self, mean, logvar):
    epsilon = torch.rand_like(mean)
    z = epsilon * torch.exp(logvar * .5) + mean  # mean, logvar, z: (batch_size, latent_dim)

    return z

  def forward(
    self,
    input_ids: torch.Tensor,  # (batch_size, seq_len)
    lengths: torch.Tensor,  # (batch_size,)
    **kwargs,
  ) -> Tuple[torch.Tensor]:
    z_mu, z_logvar = self.encoder(input_ids, lengths)
    z = self.reparameterize(z_mu, z_logvar)  # z: (batch_size, latent_dim)
    y = self.decoder(input_ids, lengths, z)  # y: (batch_size, seq_len, vocab_size)

    return y, z_mu, z_logvar

  def sample_gaussian_dist(self, batch_size: int):
    return torch.randn(batch_size, self.latent_dim).to(self.device)

  @torch.no_grad()
  def generate(
    self,
    tokenizer: Tokenizer = None,
    max_length: int = 128,
    num_return_sequences: int = 1,
    skip_special_tokens: bool = False,
    **kwargs,
  ) -> Union[List[List[int]], List[List[str]]]:

    z = kwargs.pop('z', None)
    z = z if z is not None else self.sample_gaussian_dist(num_return_sequences)
    assert z.shape == (num_return_sequences, self.latent_dim)  # z: [batch_size, latent_dim]
    z_ = z.unsqueeze(1)  # z_: [batch_size, 1, latent_dim]

    initial_inputs = torch.full((num_return_sequences, 1),
                                tokenizer.convert_token_to_id(tokenizer.start_token),
                                dtype=torch.long,
                                device=self.device)
    generated_sequences = initial_inputs
    input_ids = initial_inputs  # input_ids: [batch_size, 1]

    # z -> initial hiddens
    hiddens = self.decoder.z2hidden(z)  # hiddens: [batch_size, hidden_dim]
    hiddens = hiddens.unsqueeze(0).repeat(self.config.decoder_num_layers, 1, 1)
    # hiddens: [decoder_num_layers, batch_size, hidden_dim]

    for i in range(max_length + 1):
      x = self.decoder.embeddings(input_ids)  # x: [batch_size, 1, embedding_dim]
      x = torch.cat((x, z_), dim=-1)  # x: [batch_size, 1, embedding_dim + latent_dim]
      x, hiddens = self.decoder.gru(x, hiddens)  # x: [batch_size, 1, hidden_dim]
      logits = self.decoder.fc(x)  # logits: [batch_size, 1, vocab_size]
      next_token_logits = logits.squeeze(1)  # next_token_logits: [batch_size, vocab_size]

      probabilities = F.softmax(next_token_logits, dim=-1)  # probabilities: [batch_size, vocab_size]
      next_tokens = torch.multinomial(probabilities, num_samples=1)
      # next_tokens: [batch_size, 1]

      input_ids = next_tokens
      generated_sequences = torch.cat([generated_sequences, next_tokens], dim=1)

    generated_sequences = self.postprocessing(generated_sequences, tokenizer)

    generated_SMILES = []
    for sequence in generated_sequences:
      generated_SMILES.append(tokenizer.decode(sequence, skip_special_tokens))

    return generated_SMILES
