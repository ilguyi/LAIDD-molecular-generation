import argparse

from argparse import ArgumentParser

from .models.char_rnn.trainer import train_parser as char_rnn_parser
from .models.vae.trainer import train_parser as vae_parser

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

  parser.add_argument('--config_path',
                      type=str,
                      required=True,
                      help='directory name where to load config file path')
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


def get_train_args() -> argparse.Namespace:
  parser = get_train_parser()
  args = parser.parse_args()

  return args