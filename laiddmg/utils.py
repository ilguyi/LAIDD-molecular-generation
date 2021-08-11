import argparse
import copy
import datetime
import json
import os
import random
import numpy as np
from typing import Any

import torch

from . import logging


OUTPUT_DIR = 'outputs'
TRAINING_ARGS = 'training_ags.json'
CONFIG_NAME = 'config.json'

logger = logging.get_logger(__name__)


def set_output_dir(model_type: str, args: argparse.Namespace) -> argparse.Namespace:
  if args.output_dir is not None:
    output_dir = os.path.join(OUTPUT_DIR, model_type, args.output_dir)
  else:
    time_stamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
    output_dir = os.path.join(OUTPUT_DIR, model_type, time_stamp)

  logger.info(f'output_dir: {output_dir}')
  args.output_dir = output_dir

  return args


def set_output_dir_for_generation(model_type: str, args: argparse.Namespace) -> argparse.Namespace:
  if args.output_dir is not None:
    output_dir = os.path.join(args.checkpoint_dir, args.output_dir)
  else:
    output_dir = os.path.join(args.checkpoint_dir, 'generate')

  logger.info(f'output_dir: {output_dir}')
  args.output_dir = output_dir

  return args


def measure_duration_time(duration_time: datetime.timedelta):
  days = duration_time.days
  seconds = duration_time.seconds
  hours, remainder = divmod(seconds, 3600)
  minutes, seconds = divmod(remainder, 60)
  print(f'total duration time: {days}days {hours}hours {minutes}minutes {seconds}seconds')


def set_seed(seed: int = 219):
  logger.info(f'Set seed number {seed}')
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)


def args_to_json_file(args: argparse.Namespace):
  args_dict = copy.deepcopy(vars(args))
  if args_dict['device'] == torch.device('cuda:0'):
    args_dict['device'] = 'cuda:0'
  else:
    args_dict['device'] = 'cpu'

  args_json_path = os.path.join(args.output_dir, TRAINING_ARGS)
  logger.info(f'write training args to `{args_json_path}`')
  with open(args_json_path, 'w', encoding='utf-8') as f:
    f.write(json.dumps(args_dict, indent=2, sort_keys=True))


def config_to_json_file(config: Any, output_dir: str = None):
  config_dict = copy.deepcopy(vars(config))

  config_json_path = os.path.join(output_dir, CONFIG_NAME)
  logger.info(f'write model config to `{config_json_path}`')
  with open(config_json_path, 'w', encoding='utf-8') as f:
    f.write(json.dumps(config_dict, indent=2, sort_keys=True))


def args_and_config_to_json_files(
  args: argparse.Namespace,
  config: Any,
):
  args_to_json_file(args)
  config_to_json_file(config, args.output_dir)


# This code (`Annealingschedules` class) that is borrowed from `https://github.com/haofuml/cyclical_annealing`
# is modified by Il Gu Yi
class AnnealingSchedules:

  def __init__(self,
               method: str = 'cycle_linear',
               update_unit: str = 'epoch',  # ('step' or 'epoch')
               num_training_steps: int = None,
               num_training_steps_per_epoch: int = None,
               **kwargs):
    self.method = method
    assert update_unit in ['step', 'epoch']
    self.update_unit = update_unit
    self.num_training_steps = num_training_steps
    self.num_training_steps_per_epoch = num_training_steps_per_epoch
    self.kwargs = kwargs

    self._calculate_annealing_schedule(**self.kwargs)

  def _get_annealing_value(self, w: float) -> float:
    if self.method == 'cycle_linear':
      return w
    elif self.method == 'cycle_sigmoid':
      return 1.0 / (1.0 + np.exp(- (w * 12. - 6.)))
    elif self.method == 'cycle_cosine':
      return .5 - .5 * np.cos(w * np.pi)

  def _calculate_annealing_schedule(
    self,
    start_weight: float = 0.0,
    stop_weight: float = 1.0,
    n_cycle: int = 1,
    ratio: float = 1.0,
  ):
    self.L = np.ones(self.num_training_steps) * stop_weight
    period = self.num_training_steps / n_cycle
    weight_step = (stop_weight - start_weight) / (period * ratio)  # linear schedule

    for c in range(n_cycle):
      w, i = start_weight, 0
      while w <= stop_weight and (int(i + c * period) < self.num_training_steps):
        self.L[int(i + c * period)] = self._get_annealing_value(w)
        w += weight_step
        i += 1

    if self.update_unit == 'epoch':
      for global_step, w in enumerate(self.L):
        quotient = global_step // self.num_training_steps_per_epoch
        self.L[global_step] = self.L[quotient * self.num_training_steps_per_epoch]

  def __call__(self, global_step: int):
    assert global_step < self.num_training_steps
    return self.L[global_step]

  def get_annealing_schedule(self):
    return self.L
