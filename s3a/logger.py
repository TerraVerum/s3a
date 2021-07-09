import logging

from utilitys.fns import AppLogger

def getAppLogger(name='s3a'):
  # TODO: Get nesting working right with dialog registration
  oldCls = logging.getLoggerClass()
  try:
    logging.setLoggerClass(AppLogger)
    logger = logging.getLogger('s3a')
  finally:
    logging.setLoggerClass(oldCls)
  return logger

# Populate initially
getAppLogger().setLevel(logging.INFO)