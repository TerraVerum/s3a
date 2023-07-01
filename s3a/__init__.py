from __future__ import annotations

import os
import sys

import pyqtgraph as pg
from packaging.version import Version

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
    "ToolsPlugin",
    "defaultIo",
    "SharedAppSettings",
]

pg.setConfigOptions(imageAxisOrder="row-major")

# OpenCV can be satisfied by many libraries, so check for it early with a
# helpful error message
try:
    import cv2

    assert Version(cv2.__version__) >= Version("4.1.2.30")

except (ImportError, AssertionError):
    raise ImportError(
        "S3A requires OpenCV (cv2) >= 4.1.2.30. This can come from "
        "`opencv-python-headless` (preferred), `opencv-python`, `opencv-contrib-python`, "
        "etc."
    )

from . import graphicsutils as gutils
from .compio import ComponentIO, defaultIo
from .constants import CONFIG_DIR, PRJ_CONSTS, PRJ_ENUMS, REQD_TBL_FIELDS
from .plugins.file import ProjectData
from .plugins.tools import ToolsPlugin
from .shared import SharedAppSettings
from .structures import ComplexXYVertices, XYVertices
from .tabledata import TableData
from .views.s3agui import S3A

appInst = None
__version__ = "0.7.0"


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
