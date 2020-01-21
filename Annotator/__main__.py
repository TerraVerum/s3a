#!/usr/bin/env python
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui
from cProfile import run

from .constants import BASE_DIR
from . import Annotator, appInst

from os import path
import sys

startImgFpath = path.join(BASE_DIR, '../Images/fast.tif')
if len(sys.argv) > 1:
  startImgFpath = path.join(BASE_DIR, sys.argv[1])
win = Annotator(startImgFpath)
# p = run('profileFunc(win.estBoundsBtnClicked, 1)')
win.showMaximized()
appInst.exec()
