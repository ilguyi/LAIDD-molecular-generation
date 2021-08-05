#!/usr/bin/env/python

from datetime import datetime

from laiddmg import (
  get_train_args,
  measure_duration_time,
)

from . import logging


def main():

  start_time = datetime.now()
  # get training args
  args = get_train_args()
  logger = logging.get_logger(__name__)
  logger.info('logger start')
  logger.info(args)

  end_time = datetime.now()
  measure_duration_time(end_time - start_time)


if __name__ == '__main__':
  main()
