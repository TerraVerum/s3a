import sys
from typing import Optional

import fire
from pyqtgraph.Qt import QtCore

from . import appInst
from .cdefapp import FRCdefApp


def main(gui=True, **profileArgs) -> Optional[FRCdefApp]:
  """
  Calling code for the CDEF application.

  :param gui: Whether to run in the Qt event loop or not. If false, the user can inject
    interactions into the app using the returned :class:`FRCdefApp` object. Otherwise (default),
    the GUI application is shown and the Qt event loop is executed.
  :key author: Required if no default author exists for the application.
    The default author is updated every time an author name is given.
  :key Image: Optional initial image to be annotated
  :key Annotations: Optional initial annotation file loaded.
  """
  # Parameter editors are named in title-case, so ensure this is how keys are formatted
  profileArgs = {k.title(): v for k, v in profileArgs.items()}
  win = FRCdefApp(**profileArgs)
  if gui:
    QtCore.QTimer.singleShot(0, win.showMaximized)
    sys.exit(appInst.exec())
  else:
    return win

fire.Fire(main)