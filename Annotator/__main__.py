#!/usr/bin/env python

import sys
from os import path
from pyqtgraph.Qt import QtWidgets

from . import Annotator, appInst
from .constants import BASE_DIR

startImgFpath = path.join(BASE_DIR, '../Images/fast.tif')
if len(sys.argv) > 1:
  startImgFpath = path.join(BASE_DIR, sys.argv[1])
win = Annotator(startImgFpath)
# p = run('profileFunc(win.estBoundsBtnClicked, 1)')
win.showMaximized()
sys.exit(appInst.exec())