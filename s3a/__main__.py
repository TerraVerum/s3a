import sys

import fire
from pyqtgraph.Qt import QtCore

from s3a.constants import PRJ_ENUMS
from . import appInst
from .views.s3agui import S3A
from utilitys.fns import makeExceptionsShowDialogs


def main(loadLastState=True, version=False, **profileArgs):
  """
  Calling code for the S3A application.

  :param loadLastState: When the app is closed, all settings are saved. If this is *True*,
    these settings are restored on startup. If *False*, they aren't.

  :param version: Show version information

  :key image: Optional initial image to be annotated
  :key annotations: Optional initial annotation file loaded.
  :key `param editor name`: Name of the parameter editor within S3A with a loadable state.
    This can be e.g. `colorscheme`, `shortcuts`, etc.
  """
  # Handle here for faster bootup
  if version:
    from .__version__ import __version__
    print(__version__)
    return
  win = S3A(log=PRJ_ENUMS.LOG_GUI, loadLastState=loadLastState, **profileArgs)
  QtCore.QTimer.singleShot(0, win.showMaximized)
  sys.exit(appInst.exec_())

if __name__ == '__main__':
    fire.Fire(main)