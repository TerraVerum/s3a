#!/usr/bin/env python
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui
from cProfile import run

from .MainWindow import MainWindow
from .constants import BASE_DIR

from os import path
import sys

def profileFunc(func, numTimes, *funcArgs, **funcKwargs):
  for _ in range(numTimes):
    func(*funcArgs, **funcKwargs)
app = pg.mkQApp()
args = sys.argv
runProfile = len(args) > 1
startImgFpath = path.join(BASE_DIR, './Images/fast.tif')
win = MainWindow(startImgFpath)
if runProfile:
  p = run('profileFunc(win.estBoundsBtnClicked, 1)')
else:
  win.show()
  ret = app.exec()

