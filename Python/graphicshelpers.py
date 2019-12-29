import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets, QtGui
Signal = QtCore.pyqtSignal
QCursor = QtGui.QCursor

from contextlib import contextmanager
from functools import wraps

from processing import segmentComp

from typing import Union

import numpy as np

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

def applyWaitCursor(func):
  @wraps(func)
  def wrapWithWaitCursor(*args, **kwargs):
    try:
      pg.QAPP.setOverrideCursor(QCursor(QtCore.Qt.WaitCursor))
      return func(*args, **kwargs)
    finally:
      pg.QAPP.restoreOverrideCursor()
  return wrapWithWaitCursor

class ABTextItem(pg.TextItem):
  sigClicked = Signal()

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.origCursor = self.cursor()
    self.hoverCursor = QtCore.Qt.PointingHandCursor
    self.setAnchor((0.5,0.5))
    self.setAcceptHoverEvents(True)

  def hoverEnterEvent(self, ev):
    self.setCursor(self.hoverCursor)

  def hoverLeaveEvent(self, ev):
    #self.setCursor(self.origCursor)
    self.unsetCursor()

  def mousePressEvent(self, ev):
    self.sigClicked.emit()

class ImageROISuite(pg.PlotWidget):
  # Import here to resolve cyclic dependence
  from component import Component

  compImgItem: pg.ImageItem
  regionImgItem: pg.PlotDataItem

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.setAspectLocked(True)

    self.compImgItem = pg.ImageItem()
    self.addItem(self.compImgItem)

    self.region = pg.PolyLineROI([], closed=True)
    self.addItem(self.region)

  def setImage(self, image=None, autoLevels=None):
    return self.compImgItem.setImage(image, autoLevels)

  def update(self, mainImg: np.array, newComp:Component,
             margin: int, segThresh: float):
    offset = self._updateImg_getOffset(mainImg, newComp, margin, segThresh)
    self._updateRegion(newComp, offset)

  def _updateImg_getOffset(self, mainImg, newComp, margin, segThresh) -> np.array:
    bbox = np.vstack((newComp.vertices.min(0),
          newComp.vertices.max(0)))
    # Account for margins
    for ii in range(2):
      bbox[0,ii] = np.maximum(0, bbox[0,ii]-margin)
      bbox[1,ii] = np.minimum(mainImg.shape[1-ii], bbox[1,ii]+margin)

    newCompImg = mainImg[bbox[0,1]:bbox[1,1], bbox[0,0]:bbox[1,0],:]
    segImg = segmentComp(newCompImg, segThresh)
    self.setImage(segImg)
    # Return offset so polygons can be swapped between this
    # component image and the main image
    return bbox[0,:]

  def _updateRegion(self, newComp: Component, offset: np.array):
    # Subtract offset from vertices so they are in reference to
    # (0,0) on the component image
    newVerts = newComp.vertices - offset
    self.region.setPoints([vert for vert in newVerts], closed=True)