#!/usr/bin/env python

import sys
from os import path

from . import appInst
from .MainWindow import MainWindow
from cdef.projectvars.constants import BASE_DIR

startImgFpath = path.join(BASE_DIR, '../Images/circuitBoard.png')
if len(sys.argv) > 1:
  startImgFpath = path.join(BASE_DIR, sys.argv[1])
win = MainWindow(startImgFpath)
# p = run('profileFunc(win.estBoundsBtnClicked, 1)')
win.showMaximized()
sys.exit(appInst.exec())