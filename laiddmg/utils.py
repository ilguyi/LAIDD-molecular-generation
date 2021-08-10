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
