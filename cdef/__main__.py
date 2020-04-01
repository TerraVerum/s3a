from pyqtgraph.Qt import QtCore

import sys
from typing import Optional

import fire

from . import appInst
from .MainWindow import MainWindow


def main(img: str = None, ann: str = None, author: str = None, gui=True) -> Optional[MainWindow]:
  """
  Calling code for the CDEF application.

  :param img: Optional initial image to be annotated
  :param ann: Optional initial annotation file loaded.
  :param author: Required if no default author exists for the application.
    The default author is updated every time an author name is given.
  :param gui: Whether to run in the Qt event loop or not. If false, the user can inject
    interactions into the app using the returned :class:`MainWindow` object. Otherwise (default),
    the GUI application is shown and the Qt event loop is executed.
  """
  win = MainWindow(img, ann, author)
  if gui:
    QtCore.QTimer.singleShot(0, win.showMaximized)
    sys.exit(appInst.exec())
  else:
    return win

fire.Fire(main)