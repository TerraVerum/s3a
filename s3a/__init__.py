from __future__ import annotations

import os
from pkg_resources import parse_version
import sys

import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets

__all__ = ['appInst', 'S3A', 'REQD_TBL_FIELDS',
           'ComplexXYVertices', 'XYVertices', 'PRJ_CONSTS', 'ComponentIO',
           'ProjectData', '__version__', 'PRJ_ENUMS', 'TableData']

pg.setConfigOptions(imageAxisOrder='row-major')

# OpenCV can be satisfied by many different libraries, so check for it early with a helpful error message
try:
    import cv2
    assert parse_version(cv2.__version__) >= parse_version('4.1.2.30')

except (ImportError, AssertionError):
    raise ImportError(
      'S3A requires OpenCV (cv2) >= 4.1.2.30. This can come from "opencv-python-headless" (preferred),'
      ' "opencv-python", "opencv-contrib-python", etc.'
    )

from .__version__ import __version__

# Makes sure that when the folder is run as a module, the app exists in the outermost
# scope of the application
customPlatform = os.environ.get('S3A_PLATFORM')
args = list(sys.argv)
if customPlatform is not None:
  args += ['-platform', customPlatform]
sys.argv = args
appInst = pg.mkQApp()
# Allow selectable text in message boxes
appInst.setStyleSheet("QMessageBox { messagebox-text-interaction-flags: 5; }")

from . import graphicsutils as gutils


# Import here to resolve resolution order
import s3a.constants
import s3a.structures

from .compio import ComponentIO, defaultIo
from s3a.structures import XYVertices, ComplexXYVertices
from s3a.constants import REQD_TBL_FIELDS, PRJ_CONSTS, CFG_DIR, PRJ_ENUMS

from .views.s3agui import S3A

from s3a.plugins.misc import RandomToolsPlugin
from s3a.plugins.file import ProjectData
from s3a.parameditors.table import TableData
