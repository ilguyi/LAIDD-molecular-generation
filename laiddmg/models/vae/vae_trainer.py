from argparse import ArgumentParser
from typing import List, Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim

from ...trainer import Trainer
from ...utils import AnnealingSchedules
from ... import logging_utils


logger = logging_utils.get_logger(__name__)


def train_parser(parser: ArgumentParser) -> ArgumentParser:
  if parser is None:
    parser = ArgumentParser()

  vae_parser = parser.add_parser('vae')

  vae_parser.add_argument('--num_train_epochs',
                          default=50,
                          type=int,
                          help='number of epochs for training')
  vae_parser.add_argument('--train_batch_size',
                          default=64,
                          type=int,
                          help='batch size per device for training')
  vae_parser.add_argument('--lr',
                          default=3e-4,
                          type=float,
                          help='learning rate for training')

  return vae_parser


def generate_parser(parser: ArgumentParser) -> ArgumentParser:
  if parser is None:
    parser = ArgumentParser()

  vae_parser = parser.add_parser('vae')

  vae_parser.add_argument('--batch_size_for_generation',
                          default=128,
                          type=int,
                          help='batch size for generation')

  return vae_parser


class VAETrainer(Trainer):

  def __init__(self, **kwargs):
    super(VAETrainer, self).__init__(**kwargs)
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
  ) -> Tuple[float]:
    optimizer.zero_grad()

    data = self._prepare_inputs(data)
    outputs, z_mu, z_logvar = self.model(**data)

    reconstruction_loss = loss_fn(
      outputs.view(-1, outputs.shape[-1]),
      data['targets'].view(-1)
    )

    kl_loss = .5 * (torch.exp(z_logvar) + z_mu**2 - 1. - z_logvar).sum(1).mean()

    kl_annealing_weight = self.kl_annealing(self.global_step)

    total_loss = reconstruction_loss + kl_annealing_weight * kl_loss

    total_loss.backward()
    optimizer.step()
    self.global_step += 1

    return total_loss.item(), reconstruction_loss.item(), kl_loss.item()

  def _train_epoch(
    self,
    epoch: int,
    loss_fn: 'nn.modules.loss',
    optimizer: 'optim',
  ):
    self.model.train()

    for i, data in enumerate(self.train_dataloader):
      total_loss, reconstruction_loss, kl_loss = self._train_step(data, loss_fn, optimizer)

      logger.info(
        f'{epoch} Epochs | {i + 1}/{self.args.num_training_steps_per_epoch} | reconst_loss: {reconstruction_loss:.4g} '
        f'kl_loss: {kl_loss:.4g}, total_loss: {total_loss:.4g}, kl_annealing: {self.kl_annealing(self.global_step -1):.4g} | '
      )

  def train(self):

    reconstruction_loss_fn = nn.CrossEntropyLoss(ignore_index=self.tokenizer.padding_value)
    optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)

    # setup kl annealing weight
    self.kl_annealing = AnnealingSchedules(
      method='cycle_linear',
      update_unit='epoch',
      num_training_steps=self.args.num_training_steps,
      num_training_steps_per_epoch=self.args.num_training_steps_per_epoch,
      # start_weight=0.0,
      # stop_weight=0.05,
      # n_cycle=1,
      # ratio=1.0,
    )

    for epoch in range(1, self.args.num_train_epochs + 1):
      logger.info(f'Start training: {epoch} Epoch')

      self._train_epoch(epoch, reconstruction_loss_fn, optimizer)
      self.save_model(epoch)

    logger.info('Training done!!')
