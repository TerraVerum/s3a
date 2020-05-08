from __future__ import annotations

import inspect
import sys

import numpy as np

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
from cdef.interfaceimpls import FRTopLevelProcessors
## TODO: A little sad to set default function this way
# Reverse sort so 'region grow' is first
for name, func in reversed(inspect.getmembers(FRTopLevelProcessors, inspect.isfunction)):
  FR_SINGLETON.algParamMgr.addProcessCtor(func)

from cdef.cdefapp import FRCdefApp