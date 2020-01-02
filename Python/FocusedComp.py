import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets, QtGui
Signal = QtCore.pyqtSignal
QCursor = QtGui.QCursor

from processing import segmentComp, getVertsFromBwComps, growSeedpoint
from skimage.morphology import closing

import numpy as np

from ABGraphics.clickables import ClickableImageItem
from ABGraphics.regions import VertexRegion, SaveablePolyROI
from component import *
from SchemeEditor import SchemeEditor
from constants import SchemeValues as SV

class FocusedComp(pg.PlotWidget):
  scheme = SchemeEditor()

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.setAspectLocked(True)

    self.comp = Component()

    self.bbox = np.zeros((2,2), dtype='int32')

    # Items directly updated from the gui
    self.seedThresh = 0.

    self.compImgItem = ClickableImageItem()
    self.addItem(self.compImgItem)

    self.region = VertexRegion()
    self.addItem(self.region)

    self.interactor = SaveablePolyROI([], pen=(6,9), closed=False, removable=True)
    self.addItem(self.interactor)

    # Remove points when user asks to delete polygon
    self.interactor.sigRemoveRequested.connect(self.interactor.clearPoints)
    # Expand current mask by region on request
    self.interactor.addAct.triggered.connect(self._addRoiToRegion)

    self.compImgItem.sigClicked.connect(self.compImageClicked)

  def compImageClicked(self, ev: QtWidgets.QGraphicsSceneMouseEvent):
    # Capture clicks only if component is present
    if self.compImgItem.image is None:
      return
    # TODO: Expand to include ROI, superpixel, etc.
    # y -> row, x -> col
    newVert = np.round(np.array([[ev.pos().y(), ev.pos().x()]], dtype='int32'))
    newArea = growSeedpoint(self.compImgItem.image, newVert, self.seedThresh).astype('uint8')
    newArea |= self.region.embedMaskInImg(newArea.shape)
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
    compCenteredVertices = newVerts - offset
    self.region.updateVertices(compCenteredVertices)

  def _addRoiToRegion(self):
    imgMask = self.interactor.getImgMask(self.compImgItem)
    newRegion = np.bitwise_or(imgMask, self.region.image)
    newVerts = getVertsFromBwComps(newRegion)
    # TODO: Handle case of poly not intersecting existing region
    newVerts = newVerts[0]
    self.updateRegion(newVerts, [0,0])
    # Now that the ROI was added to the region, remove it
    self.interactor.clearPoints()

  @staticmethod
  def setScheme(scheme: SchemeEditor):
    # Pass scheme to VertexRegion
    VertexRegion.setScheme(scheme)