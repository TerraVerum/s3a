from __future__ import annotations

import os
import sys

import pyqtgraph as pg
from pkg_resources import parse_version

__all__ = [
    "appInst",
    "mkQApp",
    "S3A",
    "REQD_TBL_FIELDS",
    "ComplexXYVertices",
    "XYVertices",
    "PRJ_CONSTS",
    "ComponentIO",
    "ProjectData",
    "__version__",
    "PRJ_ENUMS",
    "TableData",
    "defaultIo",
    "SharedAppSettings",
]

pg.setConfigOptions(imageAxisOrder="row-major")

# OpenCV can be satisfied by many libraries, so check for it early with a
# helpful error message
try:
    import cv2

    assert parse_version(cv2.__version__) >= parse_version("4.1.2.30")

except (ImportError, AssertionError):
    raise ImportError(
        "S3A requires OpenCV (cv2) >= 4.1.2.30. This can come from "
        "`opencv-python-headless` (preferred), `opencv-python`, `opencv-contrib-python`, "
        "etc."
    )

from .__version__ import __version__
from . import graphicsutils as gutils
from .constants import REQD_TBL_FIELDS, PRJ_CONSTS, CFG_DIR, PRJ_ENUMS
from .structures import ComplexXYVertices, XYVertices
from .compio import ComponentIO, defaultIo
from .views.s3agui import S3A
from .shared import SharedAppSettings

from .plugins.misc import RandomToolsPlugin
from .plugins.file import ProjectData
from .parameditors.table import TableData

appInst = None


def mkQApp(*args):
    global appInst
    if appInst is not None:
        return appInst
    if pg.QAPP is not None:
        appInst = pg.QAPP
        return appInst
    # else
    args = list(sys.argv) + list(args)
    # Makes sure that when the folder is run as a module, the app exists in the outermost
    # scope of the application
    customPlatform = os.environ.get("S3A_PLATFORM")
    if customPlatform is not None:
        args += ["-platform", customPlatform]
    oldArgs = sys.argv
    sys.argv = args
    # No option to pass custom args to mkQApp, so temporarily override sys.argv
    appInst = pg.mkQApp()
    sys.argv = oldArgs
    # Allow selectable text in message boxes
    appInst.setStyleSheet("QMessageBox { messagebox-text-interaction-flags: 5; }")
    return appInst
