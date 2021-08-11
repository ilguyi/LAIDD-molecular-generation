import argparse

import torch

from argparse import ArgumentParser

from .models.char_rnn.char_rnn_trainer import train_parser as char_rnn_parser
from .models.vae.vae_trainer import train_parser as vae_parser

from .models.char_rnn.char_rnn_trainer import generate_parser as char_rnn_g_parser
from .models.vae.vae_trainer import generate_parser as vae_g_parser

from . import logging


logger = logging.get_logger(__name__)


def add_common_args(parser: ArgumentParser) -> None:
  parser = parser.add_argument_group('common')

  parser.add_argument('--seed',
                      type=int,
                      default=219,
                      help='seed number')
  parser.add_argument('--output_dir',
                      default=None,
                      type=str,
                      help='directory where to save checkpoint')


def add_train_args(parser: ArgumentParser) -> None:
  add_common_args(parser)

  parser = parser.add_argument_group('train')

  parser.add_argument('--dataset_path',
                      default='../datasets',
                      type=str,
                      help='dataset path where train, test datasets are')
  parser.add_argument('--log_steps',
                      default=10,
                      type=int,
                      help='number of steps before two (tensorboard) logs write.')


def get_train_parser() -> ArgumentParser:
  parser = ArgumentParser('Molecular generation train tool',
                          usage='laiddmg-train <model> [<args>]')
  subparser = parser.add_subparsers()

  # get parser of all models
  add_train_args(char_rnn_parser(subparser))
  add_train_args(vae_parser(subparser))

  return parser


def add_generate_args(parser: ArgumentParser) -> None:
  add_common_args(parser)

  parser = parser.add_argument_group('generate')

  parser.add_argument('--checkpoint_dir',
                      type=str,
                      required=True,
                      help='directory where to load checkpoint')
  parser.add_argument('--weights_name',
                      default=None,
                      type=str,
                      help='checkpoint file name to load weights')
  parser.add_argument('--num_generation',
                      default=10000,
                      type=int,
                      help='the number of generated SMILES')
  parser.add_argument('--max_length',
                      default=128,
                      type=int,
                      help='the maximum length of the sequence to be generated')


def get_generate_parser() -> ArgumentParser:
  parser = ArgumentParser('Molecular generation generate tool',
                          usage='laiddmg-generate <model> [<args>]')
  subparser = parser.add_subparsers()

  # get parser of all models
  add_generate_args(char_rnn_g_parser(subparser))
  add_generate_args(vae_g_parser(subparser))

  return parser


def setup_devices(args) -> torch.device:
  logger.info('PyTorch: setting up devices')
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  logger.info(f'set torch.device: {device}')

  if device.type == 'cuda':
    torch.cuda.set_device(device)

  return device


def get_train_args() -> argparse.Namespace:
  parser = get_train_parser()
  args = parser.parse_args()

  args.device = setup_devices(args)

  return args


def get_generate_args() -> argparse.Namespace:
  parser = get_generate_parser()
  args = parser.parse_args()

  args.device = setup_devices(args)

  return args
