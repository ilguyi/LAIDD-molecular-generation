from argparse import ArgumentParser

# import torch
# import torch.nn as nn
# import torch.optim as optim

from ...trainer import Trainer
from ... import logging


logger = logging.get_logger(__name__)


def train_parser(parser: ArgumentParser) -> ArgumentParser:
  if parser is None:
    parser = ArgumentParser()

  vae_parser = parser.add_parser('vae')

  vae_parser.add_argument('--num_train_epochs',
                          default=100,
                          type=int,
                          help='number of epochs for training')
  vae_parser.add_argument('--train_batch_size',
                          default=64,
                          type=int,
                          help='batch size per device for training')

  logger.info('vae_parser')

  return vae_parser


class VAETrainer(Trainer):

  def __init__(self, **kwargs):
    super(VAETrainer, self).__init__(**kwargs)

    pass
