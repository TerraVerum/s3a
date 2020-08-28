from __future__ import annotations

import inspect
import sys
import os

import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets

__all__ = ['appInst', 'FR_SINGLETON', 'S3A', 'FRParamEditor', 'FRParam', 'REQD_TBL_FIELDS',
           'FRComplexVertices', 'FRVertices', 'FRComponentIO', 'FR_CONSTS']

pg.setConfigOptions(imageAxisOrder='row-major')


# Makes sure that when the folder is run as a module, the app exists in the outermost
# scope of the application
customPlatform = os.environ.get('S3A_PLATFORM')
appInst = QtWidgets.QApplication.instance()
if appInst is None:
  args = list(sys.argv)
  if customPlatform is not None:
    args += ['-platform', customPlatform]
  appInst = QtWidgets.QApplication(args)
# Now that the app was created with sys args, populate pg instance
pg.mkQApp()
# Allow selectable text in message boxes
appInst.setStyleSheet("QMessageBox { messagebox-text-interaction-flags: 5; }")

from . import graphicsutils as gutils

# appInst.installEventFilter(
#   gutils.QAwesomeTooltipEventFilter(appInst))

# Import here to resolve resolution order
import s3a.constants
import s3a.structures

from s3a.parameditors import FR_SINGLETON, FRParamEditor
from s3a.models.tablemodel import FRComponentIO
from s3a.processing.algorithms import FRTopLevelProcessors
from s3a.structures import FRVertices, FRComplexVertices, FRParam
from s3a.constants import REQD_TBL_FIELDS, FR_CONSTS

for name, func in inspect.getmembers(FRTopLevelProcessors, inspect.isfunction):
  FR_SINGLETON.imgProcClctn.addProcessCtor(func)

from s3a.plugins import FRTableVertsPlugin
FR_SINGLETON.addPlugin(FRTableVertsPlugin)

# Minimal means no GUI is needed. Things work faster when they don't have to be
# shown through the comp display filter
if customPlatform is not None:
  from .models.s3abase import S3ABase as S3A
else:
  from .views.s3agui import S3A