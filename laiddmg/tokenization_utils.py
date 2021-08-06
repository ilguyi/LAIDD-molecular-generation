from collections import OrderedDict
from typing import List, Dict, Union, Optional

import numpy as np

import torch
import torch.nn.utils.rnn as rnn_utils

from . import logging


logger = logging.get_logger(__name__)

SPEICIAL_TOKENS = OrderedDict([
  ('pad_token', '<PAD>'),
  ('start_token', '<BOS>'),
  ('end_token', '<EOS>'),
  ('unknown_token', '<UNK>'),
])

MOSES_VOCAB = OrderedDict([
  (SPEICIAL_TOKENS['pad_token'], 0),
  (SPEICIAL_TOKENS['start_token'], 1),
  (SPEICIAL_TOKENS['end_token'], 2),
  (SPEICIAL_TOKENS['unknown_token'], 3),
  ('#', 4),
  ('(', 5),
  (')', 6),
  ('-', 7),
  ('1', 8),
  ('2', 9),
  ('3', 10),
  ('4', 11),
  ('5', 12),
  ('6', 13),
  ('=', 14),
  ('B', 15),
  ('C', 16),
  ('F', 17),
  ('H', 18),
  ('N', 19),
  ('O', 20),
  ('S', 21),
  ('[', 22),
  (']', 23),
  ('c', 24),
  ('l', 25),
  ('n', 26),
  ('o', 27),
  ('r', 28),
  ('s', 29),
])


class Tokenizer:

  def __init__(self, vocab_type: str = 'moses'):
    self.vocab_type = vocab_type
    self.vocab = MOSES_VOCAB
    self.ids_to_tokens = OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])

  @property
  def vocab_size(self) -> int:
    return len(self.vocab)

  def __len__(self) -> int:
    return len(self.vocab)

  @property
  def start_token(self) -> str:
    return SPEICIAL_TOKENS['start_token']

  @property
  def end_token(self) -> str:
    return SPEICIAL_TOKENS['end_token']

  @property
  def pad_token(self) -> str:
    return SPEICIAL_TOKENS['pad_token']

  @property
  def padding_value(self) -> int:
    return self.vocab[self.pad_token]

  def __call__(
    self,
    text: Union[str, List[str]],
    add_special_tokens: str = 'both',  # one of [`start`, `end`, `both`, `none`]
    max_length: Optional[int] = None,
    return_tensors: Optional[bool] = True,
  ) -> Dict:
    assert isinstance(text, str) or isinstance(text, (list, tuple)) or isinstance(text, np.ndarray), (
        'input must of type `str` (single example), `List[str]` (batch example)'
    )

    assert add_special_tokens in ['start', 'end', 'both', 'none']
    if isinstance(text, str):
      return self.encode(text, add_special_tokens, max_length, return_tensors)
    else:
      return self.batch_encode(text, add_special_tokens, max_length, return_tensors)

  def encode(
    self,
    text: str,
    add_special_tokens: str = 'both',
    max_length: Optional[int] = None,
    return_tensors: Optional[bool] = True,
  ) -> List[int]:
    tokens = self.tokenize(text)

    if add_special_tokens == 'both':
      special_tokens_len = 2
    elif add_special_tokens == 'none':
      special_tokens_len = 0
    else:
      special_tokens_len = 1
    total_len = len(tokens) + special_tokens_len

    if max_length is not None:
      num_tokens_to_remove = total_len - max_length
    else:
      num_tokens_to_remove = 0

    tokens = self.truncate_sequences(tokens, num_tokens_to_remove)
    tokens = self.add_special_tokens(tokens, add_special_tokens)
    tokens = self.convert_tokens_to_ids(tokens)

    if return_tensors:
      return {'input_ids': torch.LongTensor([tokens]),
              'lengths': torch.LongTensor([len(tokens)])}
    else:
      return tokens

  def truncate_sequences(
    self,
    tokens: List[str],
    num_tokens_to_remove: int = 0,
  ) -> List[str]:
    if num_tokens_to_remove <= 0:
      return tokens
    else:
      return tokens[:-num_tokens_to_remove]

  def tokenize(self, text: str) -> List[str]:
    return [token for token in text]

  def add_special_tokens(self, tokens: List[str], mode: str) -> List[str]:
    if mode == 'both':
      return self.add_end_token(self.add_start_token(tokens))
    elif mode == 'start':
      return self.add_start_token(tokens)
    elif mode == 'end':
      return self.add_end_token(tokens)
    else:
      return tokens

  def add_start_token(self, tokens: List[str]) -> List[str]:
    return [self.start_token] + tokens

  def add_end_token(self, tokens: List[str]) -> List[str]:
    return tokens + [self.end_token]

  def convert_token_to_id(self, token: str) -> int:
    try:
      return self.vocab[token]
    except BaseException:
      return self.vocab[SPEICIAL_TOKENS['unknown_token']]

  def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
    return [self.convert_token_to_id(token) for token in tokens]

  def batch_encode(
    self,
    text: Union[List[str], np.ndarray],
    add_special_tokens: str = 'both',
    max_length: Optional[int] = None,
    return_tensors: Optional[bool] = True,
  ) -> List[List[int]]:
    if return_tensors:
      encoded = [self.encode(t,
                             add_special_tokens=add_special_tokens,
                             max_length=max_length,
                             return_tensors=False)
                 for t in text]
      inputs = []
      lengths = []
      for e in encoded:
        inputs.append(torch.LongTensor(e))
        lengths.append(len(e))

      inputs = rnn_utils.pad_sequence(inputs,
                                      batch_first=True,
                                      padding_value=self.padding_value)
      lengths = torch.LongTensor(lengths)

      return {'input_ids': inputs,
              'lengths': lengths}
    else:
      return [self.encode(t,
                          add_special_tokens=add_special_tokens,
                          max_length=max_length,
                          return_tensors=return_tensors)
              for t in text]

  def decode(
    self,
    token_ids: Union[List[int], np.ndarray, torch.Tensor],
    skip_special_tokens: bool = False
  ) -> str:
    assert isinstance(token_ids, (list, tuple)) or isinstance(token_ids, np.ndarray) or isinstance(token_ids, torch.Tensor)
    if not isinstance(token_ids, (list, tuple)):
      assert len(token_ids.shape) == 1, 'Available only 1D array for decoding'
      if isinstance(token_ids, torch.Tensor):
        assert token_ids.dtype in [torch.int, torch.long]
        token_ids = token_ids.tolist()

    if skip_special_tokens:
      SPEICIAL_TOKENS_VALUE = SPEICIAL_TOKENS.values()
      decoded = [self.ids_to_tokens[index] for index in token_ids if not self.ids_to_tokens[index] in SPEICIAL_TOKENS_VALUE]
      return ''.join(decoded)
    else:
      decoded = [self.ids_to_tokens[index] for index in token_ids]
      return ''.join(decoded)
