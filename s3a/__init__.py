from __future__ import annotations

import os
import sys

import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets

__all__ = ['appInst', 'S3A', 'REQD_TBL_FIELDS',
           'ComplexXYVertices', 'XYVertices', 'PRJ_CONSTS', 'ComponentIO',
           'ProjectData', '__version__', 'PRJ_ENUMS']

pg.setConfigOptions(imageAxisOrder='row-major')

from .__version__ import __version__

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


# Import here to resolve resolution order
import s3a.constants
import s3a.structures

from ._io import ComponentIO, defaultIo
from s3a.structures import XYVertices, ComplexXYVertices
from s3a.constants import REQD_TBL_FIELDS, PRJ_CONSTS, CFG_DIR, PRJ_ENUMS

from .views.s3agui import S3A

from s3a.plugins.misc import RandomToolsPlugin
from s3a.plugins.file import ProjectData
