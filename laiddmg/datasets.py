import os
from typing import List, Dict, Union, Callable

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

from .tokenization_utils import Tokenizer


from . import logging_utils


logger = logging_utils.get_logger(__name__)


def get_rawdataset(split: str = 'train',
                   path: Union[str, os.PathLike] = '../datasets/moses') -> np.ndarray:

  assert split in ['train', 'test']

  smiles_path = os.path.join(path, f'{split}.csv.gz')

  logger.info(f'read {smiles_path} file')
  smiles = pd.read_csv(smiles_path)['smiles'].values

  logger.info(f'number of {split} dataset: {len(smiles)}')

  return smiles


class SMILESDataset(Dataset):

  def __init__(self,
               smiles: Union[List[str], np.ndarray],
               tokenizer: Tokenizer,
               transfrom: Callable = None):

    self.smiles = np.asarray(smiles)
    assert len(self.smiles.shape) == 1, 'dataset must be `1-D` array'
    self.tokenizer = tokenizer
    self.transfrom = transfrom

  def __len__(self) -> int:
    return len(self.smiles)

  def __getitem__(self, index: int) -> Dict[str, Union[List[str], List[int], int]]:
    smiles = self.smiles[index]
    input_id = self.tokenizer(smiles, return_tensors=False)

    sample = {
        'index': index,
        'smiles': smiles,
        'input_id': input_id[:-1],
        'target': input_id[1:],
        'length': len(input_id[:-1])
    }

    if self.transfrom:
      sample = self.transfrom(sample)

    return sample


class ToTensor:

  def __init__(self):
    pass

  def __call__(self, sample: Dict) -> Dict:
    inp = sample['input_id']
    tar = sample['target']
    length = sample['length']
    sample['input_id'] = torch.LongTensor(inp)
    sample['target'] = torch.LongTensor(tar)
    sample['length'] = torch.LongTensor([length])

    return sample


def get_dataset(smiles: Union[List[str], np.ndarray] = None,
                tokenizer: Tokenizer = None) -> SMILESDataset:
  if smiles is not None:
    if tokenizer is not None:
      return SMILESDataset(smiles,
                           tokenizer,
                           transfrom=ToTensor())
    else:
      raise ValueError('tokenizer must be needed.')
  else:
    raise ValueError('smiles must be needed.')
