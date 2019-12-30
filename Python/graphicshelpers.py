import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets, QtGui
Signal = QtCore.pyqtSignal
QCursor = QtGui.QCursor

from contextlib import contextmanager
from functools import wraps

from processing import segmentComp

from typing import Union

import numpy as np
import cv2 as cv

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

class ClickableImageItem(pg.ImageItem):
  sigClicked = Signal(object)

  def mouseClickEvent(self, ev):
    if ev.button() == QtCore.Qt.LeftButton:
      self.sigClicked.emit(ev)
      return

class SaveablePolyROI(pg.PolyLineROI):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    # Force new menu options
    self.getMenu()

  def getMenu(self, *args, **kwargs):
    '''
    Adds context menu option to add current ROI area to existing region
    '''
    if self.menu is None:
      menu = super().getMenu()
      addAct = QtGui.QAction("Add to Region", menu)
      menu.addAction(addAct)
      self.addAct = addAct
      self.menu = menu
    return self.menu

class FocusedComp(pg.PlotWidget):
  # Import here to resolve cyclic dependence
  from component import Component

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.setAspectLocked(True)

    self.compImgItem = ClickableImageItem()
    self.addItem(self.compImgItem)

    self.region = pg.ImageItem()
    self.addItem(self.region)

    self.interactor = SaveablePolyROI([], pen=(6,9), closed=False, removable=True)
    self.addItem(self.interactor)

    # Remove points when user asks to delete polygon
    self.interactor.sigRemoveRequested.connect(self.interactor.clearPoints)
    # Expand current mask by region on request
    self.interactor.addAct.triggered.connect(self._addRoiToRegion)

    self.compImgItem.sigClicked.connect(self.compImageClicked)

  def compImageClicked(self, ev: QtWidgets.QGraphicsSceneMouseEvent):
    # Capture clicks for ROI if component is present
    if self.compImgItem.image is None:
      return
    newVert = ev.pos()
    # Account for moved ROI
    newVert.setX(newVert.x() - self.interactor.x())
    newVert.setY(newVert.y() - self.interactor.y())
    # If enough points exist and new point is 'close' to origin,
    # consider the ROI complete
    handles = self.interactor.handles
    lastPos = handles[-1]['pos'] if len(handles) > 2 else None
    if lastPos is not None and abs(lastPos.x() - newVert.x()) < 5 \
                           and abs(lastPos.y() - newVert.y()) < 5:
      self.interactor.addAct.triggered.emit()
    else:
      # Add point as normal
      prevVerts = [handle['pos'] for handle in handles]
      self.interactor.setPoints(prevVerts + [newVert])

  def setImage(self, image=None, autoLevels=None):
    self.interactor.clearPoints()
    return self.compImgItem.setImage(image, autoLevels)

  def update(self, mainImg: np.array, newComp:Component,
             margin: int, segThresh: float):
    # --------
    # Update background image
    # --------
    bbox = np.vstack((newComp.vertices.min(0),
          newComp.vertices.max(0)))
    # Account for margins
    for ii in range(2):
      bbox[0,ii] = np.maximum(0, bbox[0,ii]-margin)
      bbox[1,ii] = np.minimum(mainImg.shape[1-ii], bbox[1,ii]+margin)

    newCompImg = mainImg[bbox[0,1]:bbox[1,1], bbox[0,0]:bbox[1,0],:]
    segImg = segmentComp(newCompImg, segThresh)
    self.setImage(segImg)

    # --------
    # Update image making up the region
    # --------
    offset = bbox[0,:]
    vertices = newComp.vertices - offset
    region = np.zeros(segImg.shape, dtype='uint8')
    cv.fillPoly(region, [vertices], (0,255,0))
    alpha = region[:,:,1].copy()//4
    alpha[vertices[:,1], vertices[:,0]] = 255
    self.region.setImage(np.concatenate((region, alpha[:,:,None]), axis=2))

  def _addRoiToRegion(self):
    roiArea = self.interactor.getArrayRegion(self.region.image, self.compImgItem)
    breakp = 1
    breakp2 = 2