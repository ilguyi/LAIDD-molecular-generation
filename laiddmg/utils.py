import argparse
import datetime
import os

from . import logging


OUTPUT_DIR = 'outputs'
TRAINING_ARGS = 'training_ags.json'

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
