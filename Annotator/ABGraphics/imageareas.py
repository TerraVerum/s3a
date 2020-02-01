from typing import Union

import numpy as np
import pyqtgraph as pg
from PIL import Image
from pandas import DataFrame as df
from pyqtgraph.Qt import QtCore, QtGui
from pyqtgraph import Point

from .clickables import ClickableImageItem
from .regions import VertexRegion, SaveablePolyROI
from ..constants import TEMPLATE_COMP as TC
from ..processing import segmentComp, getVertsFromBwComps, growSeedpoint
from Annotator.generalutils import getClippedBbox
from ..tablemodel import makeCompDf

Signal = QtCore.pyqtSignal
QCursor = QtGui.QCursor


class ABViewBox(pg.ViewBox):
  sigSelectionCreated = Signal(object)
  sigComponentCreated = Signal(object)

  def mouseDragEvent(self, ev, axis=None):
    """
    Most of the desired functionality for drawing a selection rectangle on the main image
    already exists within the default viewbox. However, pyqtgraph behavior is to zoom on
    the selected region once the drag is done. We don't want that -- instead, we want the
    components within the selected rectangle to be selected within the table. This requires
    overloading only a small portion of
    :func:`ViewBox.mouseDragEvent()<pyqtgraph.ViewBox.mouseDragEvent>`.
    """
    # TODO: Make this more robust, since it is a temporary measure at the moment
    callSuperMethod = True
    modifiers = ev.modifiers()
    if modifiers != QtCore.Qt.NoModifier:
      self.state['mouseMode'] = pg.ViewBox.RectMode
      if ev.isFinish():
        callSuperMethod = False
        bounds = self.getSelectionBounds(ev)
        if modifiers == QtCore.Qt.ShiftModifier:
          self.sigSelectionCreated.emit(bounds)
        elif modifiers == QtCore.Qt.ControlModifier:
          self.sigComponentCreated.emit(bounds)
    else:
      self.state['mouseMode'] = pg.ViewBox.PanMode
    if callSuperMethod:
      super().mouseDragEvent(ev, axis)

  def getSelectionBounds(self, ev):
    pos = ev.pos()
    self.rbScaleBox.hide()
    ax = QtCore.QRectF(Point(ev.buttonDownPos(ev.button())), Point(pos))
    selectionBounds = self.childGroup.mapRectFromParent(ax)
    return selectionBounds.getCoords()


class MainImageArea(pg.PlotWidget):
  def __init__(self, parent=None, background='default', imgSrc=None, **kargs):
    super().__init__(parent, background, viewBox=ABViewBox(), **kargs)

    self.allowNewComps = True

    self.setAspectLocked(True)
    self.viewbox: ABViewBox = self.getViewBox()
    self.viewbox.invertY()
    self.sigSelectionCreated = self.viewbox.sigSelectionCreated
    self.sigComponentCreated = self.viewbox.sigComponentCreated

    # -----
    # Image Item
    # -----
    self.imgItem = ClickableImageItem()
    # Ensure image is behind plots
    self.imgItem.setZValue(-100)
    self.setImage(imgSrc)
    self.addItem(self.imgItem)

  def keyPressEvent(self, ev: QtGui.QKeyEvent):
    if ev.key() == QtCore.Qt.Key_Escape:
      # Simulate empty bounding box to deselect points
      self.sigSelectionCreated.emit((-1,-1,-1,-1))

  @property
  def image(self):
    return self.imgItem.image

  @property
  def clickable(self):
    return self.compImgItem.clickable

  @clickable.setter
  def clickable(self, newVal):
    self.compImgItem.clickable = bool(newVal)

  def setImage(self, imgSrc: Union[str, np.ndarray]=None):
    """
    Allows the user to change the main image either from a filepath or array data
    """
    if isinstance(imgSrc, str):
      imgSrc = np.array(Image.open(imgSrc))

    self.imgItem.setImage(imgSrc)

class FocusedComp(pg.PlotWidget):
  sigEnterPressed = Signal()

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    # Whether drawn items should be added or removed from current component
    self.inAddMode = True
    # Type of region to add once the user clicks. See radio buttons on the
    # image annotator UI
    self.drawType = 'seedpoint'

    self.setAspectLocked(True)
    self.getViewBox().invertY()

    self.compSer = makeCompDf().squeeze()
    self.deletedPrevComponent = False

    self.bbox = np.zeros((2,2), dtype='int32')

    # Items directly updated from the gui
    self.seedThresh = 0.

    self.compImgItem = ClickableImageItem()
    self.compImgItem.requireCtrlKey = False
    self.addItem(self.compImgItem)

    self.region = VertexRegion()
    self.addItem(self.region)

    self.interactor = SaveablePolyROI([], pen=(6,9), closed=False, removable=True)
    self.addItem(self.interactor)

    # Remove points when user asks to delete polygon
    self.interactor.sigRemoveRequested.connect(self.interactor.clearPoints)
    # Expand current mask by region on request
    self.interactor.finishPolyAct.triggered.connect(self._addRoiToRegion)

    self.compImgItem.sigClicked.connect(self.compImageClicked)

  @property
  def clickable(self):
    return self.compImgItem.clickable

  @clickable.setter
  def clickable(self, newVal):
    self.compImgItem.clickable = bool(newVal)

  def keyPressEvent(self, ev: QtGui.QKeyEvent):
    pressedKey = ev.key()
    if pressedKey == QtCore.Qt.Key_Enter or pressedKey == QtCore.Qt.Key_Return or \
        pressedKey == QtCore.Qt.Key_1:
      self.sigEnterPressed.emit()
      ev.accept()
    super().keyPressEvent(ev)

  def mouseMoveEvent(self, ev: QtGui.QKeyEvent):
    if ev.modifiers() == QtCore.Qt.ControlModifier \
       and ev.buttons() == QtCore.Qt.LeftButton:
      # Simulate click in that location
      posRelToImg = self.compImgItem.mapFromScene(ev.pos())
      xyCoord = np.round(np.array([[posRelToImg.x(), posRelToImg.y()]], dtype='int'))
      self.compImageClicked(xyCoord)
      ev.accept()
    else:










      super().mouseMoveEvent(ev)

  def compImageClicked(self, newVert: np.ndarray):
    # Capture clicks only if component is present and user allows it
    # TODO: Expand to include ROI, superpixel, etc.
    # Change vertex from x-y to row-col
    newVert = np.fliplr(newVert)
    newArea = growSeedpoint(self.compImgItem.image, newVert, self.seedThresh)
    curRegionMask = self.region.embedMaskInImg(newArea.shape)
    if self.inAddMode:
      newArea |= curRegionMask
    else:
      newArea = ~newArea & curRegionMask
    # TODO: handle case of multiple regions existing after click. For now, just use
    # the largest
    vertsPerComp = getVertsFromBwComps(newArea)
    vertsToUse = np.empty((0,2), dtype='int')
    for verts in vertsPerComp:
      if verts.shape[0] > vertsToUse.shape[0]:
        vertsToUse = verts
    self.updateRegion(vertsToUse, [0,0])

  def addRoiVertex(self, newVert: QtCore.QPointF):
    # Account for moved ROI
    newVert.setX(newVert.x() - self.interactor.x())
    newVert.setY(newVert.y() - self.interactor.y())
    # If enough points exist and new point is 'close' to origin,
    # consider the ROI complete
    handles = self.interactor.handles
    lastPos = handles[-1]['pos'] if len(handles) > 2 else None
    if lastPos is not None and abs(lastPos.x() - newVert.x()) < 5 \
                           and abs(lastPos.y() - newVert.y()) < 5:
      self.interactor.finishPolyAct.triggered.emit()
    else:
      # Add point as normal
      prevVerts = [handle['pos'] for handle in handles]
      self.interactor.setPoints(prevVerts + [newVert])

  def setImage(self, image=None, autoLevels=None):
    self.interactor.clearPoints()
    return self.compImgItem.setImage(image, autoLevels)

  def updateAll(self, mainImg: np.array, newComp:df,
             margin: int, segThresh: float):
    newVerts = newComp[TC.VERTICES].squeeze()
    # If the previous component had no vertices, signal its removal
    if len(self.compSer[TC.VERTICES].squeeze()) == 0 and not self.deletedPrevComponent:
      self.deletedPrevComponent = True
    # Since values INSIDE the dataframe are reset instead of modified, there is no
    # need to go through the trouble of deep copying
    self.compSer = newComp.copy(deep=False)
    self.deletedPrevComponent = False
    self.updateBbox(mainImg.shape, newVerts, margin)
    self.updateCompImg(mainImg, segThresh)
    self.updateRegion(newVerts)
    self.autoRange()
    return self.deletedPrevComponent

  def updateBbox(self, mainImgShape, newVerts: np.ndarray, margin: int):
    # Ignore NAN entries during computation
    bbox = np.vstack([np.nanmin(newVerts, 0),
          np.nanmax(newVerts, 0)])
    # Account for margins
    self.bbox = getClippedBbox(mainImgShape, bbox, margin)

  def updateCompImg(self, mainImg, segThresh, bbox=None):
    if bbox is None:
      bbox = self.bbox
    # Account for nan entries
    newCompImg = mainImg[bbox[0,1]:bbox[1,1],
                         bbox[0,0]:bbox[1,0],
                         :]
    segImg = segmentComp(newCompImg, segThresh)
    self.setImage(segImg)

  def updateRegion(self, newVerts, offset=None):
    # Component vertices are nan-separated regions
    if offset is None:
      offset = self.bbox[0,:]
    if newVerts is None:
      newVerts = np.ones((0,2), dtype=int)
    # 0-center new vertices relative to FocusedComp image
    # Make a copy of each list first so we aren't modifying the
    # original data
    centeredVerts = newVerts.copy()
    centeredVerts -= offset
    self.region.updateVertices(centeredVerts)

  def saveNewVerts(self):
    # Add in offset from main image to VertexRegion vertices
    self.compSer.loc[TC.VERTICES] = self.region.verts + self.bbox[0,:]

  def _addRoiToRegion(self):
    imgMask = self.interactor.getImgMask(self.compImgItem)
    newRegion = np.bitwise_or(imgMask, self.region.image)
    newVerts = getVertsFromBwComps(newRegion)
    # TODO: Handle case of poly not intersecting existing region
    newVerts = newVerts[0]
    self.updateRegion(newVerts, [0,0])
    # Now that the ROI was added to the region, remove it
    self.interactor.clearPoints()