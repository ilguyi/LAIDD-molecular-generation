import datetime


def measure_duration_time(duration_time: datetime.timedelta):
  days = duration_time.days
  seconds = duration_time.seconds
  hours, remainder = divmod(seconds, 3600)
  minutes, seconds = divmod(remainder, 60)
  print(f'total duration time: {days}days {hours}hours {minutes}minutes {seconds}seconds')
