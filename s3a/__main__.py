import sys
from typing import Optional

import fire
from pyqtgraph.Qt import QtCore

from . import appInst, PRJ_SINGLETON
from .views.s3agui import S3A
from utilitys.fns import makeExceptionsShowDialogs


def main(loadLastState=True, help=False, **profileArgs):
  """
  Calling code for the S3A application.

  :param loadLastState: When the app is closed, all settings are saved. If this is *True*,
    these settings are restored on startup. If *False*, they aren't.

  :param help: Show help documentation

  :key image: Optional initial image to be annotated
  :key annotations: Optional initial annotation file loaded.
  :key `param editor name`: Name of the parameter editor within S3A with a loadable state.
    This can be e.g. `colorscheme`, `shortcuts`, etc.
  """
  # Handle here for faster bootup
  win = S3A(guiMode=not help, loadLastState=loadLastState, **profileArgs)
  if help:
    return win
  makeExceptionsShowDialogs(win)
  QtCore.QTimer.singleShot(0, win.showMaximized)
  sys.exit(appInst.exec_())

if __name__ == '__main__':
    fire.Fire(main)