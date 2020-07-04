from __future__ import annotations

import inspect
import sys

import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets

__all__ = ['appInst', 'S3A', 'FR_SINGLETON']

pg.setConfigOptions(imageAxisOrder='row-major')


# Makes sure that when the folder is run as a module, the app exists in the outermost
# scope of the application
appInst = QtWidgets.QApplication.instance()
if appInst is None:
  appInst = QtWidgets.QApplication(sys.argv)
# Now that the app was created with sys args, populate pg instance
pg.mkQApp()
# Allow selectable text in message boxes
appInst.setStyleSheet("QMessageBox { messagebox-text-interaction-flags: 5; }")

import s3a.frgraphics.graphicsutils as gutils
appInst.installEventFilter(
  gutils.QAwesomeTooltipEventFilter(appInst))

# Import here to resolve resolution order
import s3a.projectvars
import s3a.structures

from s3a.frgraphics.parameditors import FR_SINGLETON
from s3a.processingimpls import FRTopLevelProcessors
for name, func in inspect.getmembers(FRTopLevelProcessors, inspect.isfunction):
  FR_SINGLETON.algParamMgr.addProcessCtor(func)

from s3a.s3app import S3A