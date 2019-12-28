import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets, QtGui
from typing import Union

class TformHelper:
  def __init__(self, tformObj: Union[QtGui.QTransform,type(None)] = None):
    self.matValList = []
    for ii in range(1,4):
      for jj in range(1,4):
        initialVal = getattr(tformObj, f'm{ii}{jj}', lambda: None)()
        setattr(self, f'm{ii}{jj}', initialVal)
  def getTransform(self) -> QtGui.QTransform:
    matEls = [getattr(self, f'm{ii}{jj}') for ii in range(1,4) for jj in range(1,4)]
    return QtGui.QTransform(*matEls)

def flipHorizontal(gItem: QtWidgets.QGraphicsItem):
  origTf = gItem.transform()
  newTf = origTf.scale(1,-1)
  gItem.setTransform(newTf)