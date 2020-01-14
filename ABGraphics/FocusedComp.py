import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets, QtGui
Signal = QtCore.pyqtSignal
QCursor = QtGui.QCursor

from processing import segmentComp, getVertsFromBwComps, growSeedpoint, getClippedBbox
from skimage.morphology import closing, opening

import numpy as np

from ABGraphics.clickables import ClickableImageItem
from ABGraphics.regions import VertexRegion, SaveablePolyROI
from component import *
from ABGraphics.parameditors import SchemeEditor

class FocusedComp(pg.PlotWidget):
  scheme = SchemeEditor()

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

    # Whether drawn items should be added or removed from current component
    self.inAddMode = True
    # Type of region to add once the user clicks. See radio buttons on the
    # image annotator UI
    self.drawType = 'seedpoint'

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
    self.interactor.finishPolyAct.triggered.connect(self._addRoiToRegion)

    self.compImgItem.sigClicked.connect(self.compImageClicked)

  def setClickable(self, isClickable):
    self.compImgItem.clickable = isClickable

  def compImageClicked(self, newVert: np.ndarray):
    # Capture clicks only if component is present and user allows it
    # TODO: Expand to include ROI, superpixel, etc.
    # Change vertex from x-y to row-col
    newVert = np.fliplr(newVert)
    newArea = growSeedpoint(self.compImgItem.image, newVert, self.seedThresh)
    curRegionMask = self.region.embedMaskInImg(newArea.shape)
    if self.inAddMode:
      newArea |= curRegionMask
      newArea = closing(newArea, np.ones((5,5)))
    else:
      newArea = ~newArea & curRegionMask
      newArea = opening(newArea, np.ones((5,5)))
    newArea = closing(newArea, np.ones((5,5)))
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

  def updateAll(self, mainImg: np.array, newComp:Component,
             margin: int, segThresh: float):
    deletePrevComponent = False
    # If the previous component had no vertices, signal its removal
    if len(self.comp.vertices) == 0:
      deletePrevComponent = True
    self.comp = newComp
    self.updateBbox(mainImg.shape, newComp, margin)
    self.updateCompImg(mainImg, segThresh)
    self.updateRegion(newComp.vertices)
    return deletePrevComponent

  def updateBbox(self, mainImgShape, newComp: Component, margin):
    # Ignore NAN entries during computation
    bbox = np.vstack([np.nanmin(newComp.vertices, 0),
          np.nanmax(newComp.vertices, 0)])
    # Account for margins
    self.bbox = getClippedBbox(mainImgShape, bbox, margin)

  def updateCompImg(self, mainImg, segThresh, bbox=None):
    if bbox is None:
      bbox = self.bbox
    newCompImg = mainImg[self.bbox[0,1]:self.bbox[1,1],
                         self.bbox[0,0]:self.bbox[1,0],
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
    self.comp.vertices = self.region.verts + self.bbox[0,:]

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