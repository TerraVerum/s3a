import sys
from typing import Optional

import fire
from pyqtgraph.Qt import QtCore

from . import appInst, PRJ_SINGLETON
from .views.s3agui import S3A
from utilitys.fns import makeExceptionsShowDialogs


def main(guiMode=True, loadLastState=None, **profileArgs) -> Optional[S3A]:
  """
  Calling code for the S3A application.

  :param guiMode: Whether to run in the Qt event loop or not. If false, the user can inject
    interactions into the app using the returned :class:`S3A` object. Otherwise (default),
    the GUI application is shown and the Qt event loop is executed.

  :param tableCfg: YAML configuration file for table fields and classes. Should contain
    one or more of the following specifications:
      * table-fields: Fields to include in the component table
      * classes: Allowed class for a component

  :param loadLastState: When the app is closed, all settings are saved. If this is *True*,
    these settings are restored on startup. If *False*, they aren't. If *None*, the
    user is prompted for whether the settings should be loaded.

  :key image: Optional initial image to be annotated
  :key annotations: Optional initial annotation file loaded.
  :key `param editor name`: Name of the parameter editor within S3A with a loadable state.
    This can be e.g. `colorscheme`, `shortcuts`, etc.
  """
  # Handle here for faster bootup
  tableCfg = profileArgs.pop('tablecfg', None)
  if tableCfg is not None:
    PRJ_SINGLETON.tableData.loadCfg(tableCfg)
  win = S3A(guiMode=guiMode, loadLastState=loadLastState, **profileArgs)
  if guiMode:
    makeExceptionsShowDialogs(win)
    QtCore.QTimer.singleShot(0, win.showMaximized)
    sys.exit(appInst.exec_())
  else:
    return win

if __name__ == '__main__':
    fire.Fire(main)