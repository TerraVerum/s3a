from __future__ import annotations

import sys

import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets

__all__ = ['appInst', 'FRCdefApp']

# Makes sure that when the folder is run as a module, the app exists in the outermost
# scope of the application
appInst = QtWidgets.QApplication.instance()
if appInst is None:
  appInst = QtWidgets.QApplication(sys.argv)
# Now that the app was created with sys args, populate pg instance
pg.mkQApp()

# Import here to resolve resolution order
import cdef.projectvars
import cdef.structures

import cdef.frgraphics.parameditors
import cdef.interfaceimpls

from cdef.cdefapp import FRCdefApp