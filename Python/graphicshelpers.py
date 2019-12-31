import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets, QtGui
Signal = QtCore.pyqtSignal
QCursor = QtGui.QCursor

from functools import wraps

from processing import segmentComp, getVertsFromBwComps, growSeedpoint
from skimage.morphology import closing

import numpy as np
import cv2 as cv

# Must import everything to avoid cyclic dependency
from component import Component

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

  def getImgMask(self, imgItem: pg.ImageItem):
    imgMask = np.zeros(imgItem.image.shape[0:2], dtype='bool')
    roiSlices,_ = self.getArraySlice(imgMask, imgItem)
    # TODO: Clip regions that extend beyond image dimensions
    roiSz = [curslice.stop - curslice.start for curslice in roiSlices]
    # renderShapeMask takes width, height args. roiSlices has row/col sizes,
    # so switch this order when passing to renderShapeMask
    roiSz = roiSz[::-1]
    roiMask = self.renderShapeMask(*roiSz).astype('uint8')
    # Also, the return value for renderShapeMask is given in col-major form.
    # Transpose this, since all other data is in row-major.
    roiMask = roiMask.T
    imgMask[roiSlices[0], roiSlices[1]] = roiMask
    return imgMask

class FocusedComp(pg.PlotWidget):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.setAspectLocked(True)

    self.comp = Component()

    self.bbox = np.zeros((2,2), dtype='int32')

    # Items directly updated from the gui
    self.seedThresh = 0.

    self.compImgItem = ClickableImageItem()
    self.addItem(self.compImgItem)

    self.region = pg.ImageItem()
    # Default LUT is green with alpha at interior
    self.setRegionLUT(np.array([[0,0,0,0],
                                [0,255,0,70],
                                [0,255,0,255]]))
    self.addItem(self.region)

    self.interactor = SaveablePolyROI([], pen=(6,9), closed=False, removable=True)
    self.addItem(self.interactor)

    # Remove points when user asks to delete polygon
    self.interactor.sigRemoveRequested.connect(self.interactor.clearPoints)
    # Expand current mask by region on request
    self.interactor.addAct.triggered.connect(self._addRoiToRegion)

    self.compImgItem.sigClicked.connect(self.compImageClicked)

  def setRegionLUT(self, lutArr:np.array):
    # Define LUT that properly colors vertices and interior of region
    cmap = pg.ColorMap([0,1,2], lutArr)
    self.region.setLookupTable(cmap.getLookupTable(0,2,nPts=3,alpha=True))

  def compImageClicked(self, ev: QtWidgets.QGraphicsSceneMouseEvent):
    # Capture clicks only if component is present
    if self.compImgItem.image is None:
      return
    # TODO: Expand to include ROI, superpixel, etc.
    # y -> row, x -> col
    newVert = np.round(np.array([[ev.pos().y(), ev.pos().x()]], dtype='int32'))
    newArea = growSeedpoint(self.compImgItem.image, newVert, self.seedThresh).astype('uint8')
    newArea |= self.region.image
    newArea = closing(newArea, np.ones((5,5)))
    # TODO: handle case of multiple regions existing after click. For now, just use
    # the largest
    vertsPerComp = getVertsFromBwComps(newArea)
    vertsToUse = np.array([])
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
      self.interactor.addAct.triggered.emit()
    else:
      # Add point as normal
      prevVerts = [handle['pos'] for handle in handles]
      self.interactor.setPoints(prevVerts + [newVert])

  def setImage(self, image=None, autoLevels=None):
    self.interactor.clearPoints()
    return self.compImgItem.setImage(image, autoLevels)

  def updateAll(self, mainImg: np.array, newComp:Component,
             margin: int, segThresh: float):

    self.comp = newComp
    self.updateBbox(mainImg.shape, newComp, margin)
    self.updateCompImg(mainImg, segThresh)
    self.updateRegion(newComp.vertices)

  def updateBbox(self, mainImgShape, newComp, margin):
    bbox = np.vstack((newComp.vertices.min(0),
          newComp.vertices.max(0)))
    # Account for margins
    for ii in range(2):
      bbox[0,ii] = np.maximum(0, bbox[0,ii]-margin)
      bbox[1,ii] = np.minimum(mainImgShape[1-ii], bbox[1,ii]+margin)
    self.bbox = bbox

  def updateCompImg(self, mainImg, segThresh, bbox=None):
    if bbox is None:
      bbox = self.bbox
    newCompImg = mainImg[self.bbox[0,1]:self.bbox[1,1],
                         self.bbox[0,0]:self.bbox[1,0],
                         :]
    segImg = segmentComp(newCompImg, segThresh)
    self.setImage(segImg)

  def updateRegion(self, newVerts, offset=None):
    if offset is None:
      offset = self.bbox[0,:]
    newImgShape = self.compImgItem.image.shape
    regionData = np.zeros(newImgShape[0:2], dtype='uint8')
    # No need to look for polygons if vertices are empty
    if newVerts.shape[0] > 0:
      vertices = newVerts - offset
      cv.fillPoly(regionData, [vertices], 1)
      # Make vertices full brightness
      regionData[vertices[:,1], vertices[:,0]] = 2
    self.region.setImage(regionData)

  def _addRoiToRegion(self):
    imgMask = self.interactor.getImgMask(self.compImgItem)
    newRegion = np.bitwise_or(imgMask, self.region.image)
    newVerts = getVertsFromBwComps(newRegion)
    # TODO: Handle case of poly not intersecting existing region
    newVerts = newVerts[0]
    self._updateRegion(newVerts, [0,0])
    # Now that the ROI was added to the region, remove it
    self.interactor.clearPoints()

