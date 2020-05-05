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

from cdef.frgraphics.parameditors import FR_SINGLETON
import cdef.interfaceimpls as impls
for key, val in vars(impls).items():
  if 'Processor' in key:
    FR_SINGLETON.algParamMgr.addProcessCtor(val)

from cdef.cdefapp import FRCdefApp