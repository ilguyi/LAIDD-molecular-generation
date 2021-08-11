from argparse import ArgumentParser
from typing import List, Dict, Union

import torch
import torch.nn as nn
import torch.optim as optim

from ...trainer import Trainer
from ... import logging


logger = logging.get_logger(__name__)


def train_parser(parser: ArgumentParser) -> ArgumentParser:
  if parser is None:
    parser = ArgumentParser()

  char_rnn_parser = parser.add_parser('char_rnn')

  char_rnn_parser.add_argument('--num_train_epochs',
                               default=50,
                               type=int,
                               help='number of epochs for training')
  char_rnn_parser.add_argument('--train_batch_size',
                               default=64,
                               type=int,
                               help='batch size per device for training')
  char_rnn_parser.add_argument('--lr',
                               default=1e-3,
                               type=float,
                               help='learning rate for training')
  char_rnn_parser.add_argument('--step_size',
                               default=10,
                               type=int,
                               help='period of learning rate decay (decay unit: epoch)')
  char_rnn_parser.add_argument('--gamma',
                               default=0.5,
                               type=float,
                               help='multiplicative factor of learning rate decay')

  return char_rnn_parser


class CharRNNTrainer(Trainer):

  def __init__(self, **kwargs):
    super(CharRNNTrainer, self).__init__(**kwargs)
    pass

  def _pad_sequence(self,
                    data: List[torch.Tensor],
                    padding_value: int = 0) -> torch.Tensor:
    return torch.nn.utils.rnn.pad_sequence(data,
                                           batch_first=True,
                                           padding_value=padding_value)

  def _collate_fn(self,
                  batch: List[Dict[str, Union[torch.Tensor, str, int]]],
                  **kwargs) -> Dict[str, Union[torch.Tensor, List[str], List[int]]]:

    indexes = [item['index'] for item in batch]
    smiles = [item['smiles'] for item in batch]
    input_ids = [item['input_id'] for item in batch]
    targets = [item['target'] for item in batch]
    lengths = [item['length'] for item in batch]

    padding_value = self.tokenizer.padding_value
    input_ids = self._pad_sequence(input_ids, padding_value)
    targets = self._pad_sequence(targets, padding_value)
    lengths = torch.LongTensor(lengths)

    return {'input_ids': input_ids,
            'targets': targets,
            'lengths': lengths,
            'smiles': smiles,
            'indexes': indexes}

  def _train_step(
    self,
    data: Dict[str, Union[torch.Tensor, List[str], List[int]]],
    loss_fn: 'nn.modules.loss',
    optimizer: 'optim'
  ) -> float:
    optimizer.zero_grad()

    data = self._prepare_inputs(data)
    outputs, _ = self.model(**data)

    loss = loss_fn(outputs.view(-1, outputs.shape[-1]),
                   data['targets'].view(-1))

    loss.backward()
    optimizer.step()
    self.global_step += 1

    return loss.item()

  def _train_epoch(
    self,
    epoch: int,
    loss_fn: 'nn.modules.loss',
    optimizer: 'optim',
    lr_scheduler: 'optim.lr_scheduler',
  ):
    self.model.train()

    for i, data in enumerate(self.train_dataloader):
      loss = self._train_step(data, loss_fn, optimizer)

      logger.info(
        f'{epoch} Epochs | {i + 1}/{self.args.num_training_steps_per_epoch} | loss: {loss:.4g} | '
        f'lr: {lr_scheduler.get_last_lr()[0]:.4g}'
      )

  def train(self):

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer,
                                             self.args.step_size,
                                             self.args.gamma)

    for epoch in range(1, self.args.num_train_epochs + 1):
      logger.info(f'Start training: {epoch} Epoch')

      self._train_epoch(epoch, loss_fn, optimizer, lr_scheduler)
      self.save_model(epoch)

    logger.info('Training done!!')
