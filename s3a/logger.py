import logging

from utilitys.fns import AppLogger

def getAppLogger(name='s3a'):
  return AppLogger.getAppLogger(name)

# Populate initially
getAppLogger().setLevel(logging.INFO)