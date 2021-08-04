#!/usr/bin/env/python

from datetime import datetime

from laiddmg import (
  get_logger,
  measure_duration_time,
)


def main():

  start_time = datetime.now()

  logger = get_logger(__name__)
  logger.info('logger start')

  end_time = datetime.now()
  measure_duration_time(end_time - start_time)


if __name__ == '__main__':
  main()
