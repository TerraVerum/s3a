import sys
from typing import Optional
from .structures import FilePath

import fire
from pyqtgraph.Qt import QtCore

from . import appInst, FR_SINGLETON
from .views.s3agui import S3A


def main(gui=True, tableCfg: FilePath=None, loadLastState=True, **profileArgs) -> Optional[S3A]:
  """
  Calling code for the S3A application.

  :param gui: Whether to run in the Qt event loop or not. If false, the user can inject
    interactions into the app using the returned :class:`S3A` object. Otherwise (default),
    the GUI application is shown and the Qt event loop is executed.

  :param tableCfg: YAML configuration file for table fields and classes. Should contain
    one or more of the following specifications:
      * opt-tbl-fields: Fields to include in the component table
      * classes: Allowed class for a component

  :key author: Required if no default author exists for the application.
    The default author is updated every time an author name is given.
  :key Image: Optional initial image to be annotated
  :key Annotations: Optional initial annotation file loaded.
  """
  # Parameter editors are named in title-case, so ensure this is how keys are formatted
  if tableCfg is not None:
    FR_SINGLETON.tableData.loadCfg(tableCfg)
  profileArgs = {k.replace(' ', '').lower(): v for k, v in profileArgs.items()}
  win = S3A(exceptionsAsDialogs=gui, loadLastState=loadLastState, **profileArgs)
  if gui:
    QtCore.QTimer.singleShot(0, win.showMaximized)
    sys.exit(appInst.exec_())
  else:
    return win

fire.Fire(main)