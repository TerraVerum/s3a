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
try:
  sys.exit(appInst.exec())
except Exception as ex:
  QtWidgets.QMessageBox().information(win, 'Error', f'An error occurred:\n{str(ex)}')