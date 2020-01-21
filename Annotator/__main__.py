#!/usr/bin/env python
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui
from cProfile import run

from .constants import BASE_DIR
from . import Annotator, appInst

from os import path
import sys

def profileFunc(func, numTimes, *funcArgs, **funcKwargs):
  for _ in range(numTimes):
    func(*funcArgs, **funcKwargs)
args = sys.argv
runProfile = len(args) > 1
startImgFpath = path.join(BASE_DIR, '../Images/fast.tif')
win = Annotator(startImgFpath)
if runProfile:
  p = run('profileFunc(win.estBoundsBtnClicked, 1)')
else:
  win.show()
  ret = appInst.exec()

