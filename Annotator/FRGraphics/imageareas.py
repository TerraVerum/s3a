from typing import Union, Optional

import numpy as np
import pyqtgraph as pg
from PIL import Image
from pandas import DataFrame as df
from pyqtgraph.Qt import QtCore, QtGui
from skimage import morphology

from .clickables import DraggableViewBox
from .parameditors import FR_SINGLETON
from ..constants import FR_CONSTS, FR_ENUMS
from ..params import FRImageProcessor
from ..processing import growBoundarySeeds
from .clickables import ClickableImageItem
from .regions import FRVertexRegion, FRShape
from ..constants import TEMPLATE_COMP as TC
from ..processing import segmentComp, getVertsFromBwComps, growSeedpoint
from Annotator.generalutils import getClippedBbox, nanConcatList, ObjUndoBuffer
from ..tablemodel import makeCompDf

Signal = QtCore.pyqtSignal
QCursor = QtGui.QCursor

@FR_SINGLETON.registerClass(FR_CONSTS.CLS_REGION_BUF)
class RegionVertsUndoBuffer(ObjUndoBuffer):
  @FR_SINGLETON.generalProps.registerProp(FR_CONSTS.PROP_UNDO_BUF_SZ)
  def maxBufferLen(self): pass

  @FR_SINGLETON.generalProps.registerProp(FR_CONSTS.PROP_STEPS_BW_SAVE)
  def stepsBetweenBufSave(self): pass

  _bwPrevRegion: Optional[np.ndarray] = None

  def __init__(self):
    super().__init__(self.maxBufferLen, self.stepsBetweenBufSave)

  def update(self, newVerts, newBwArea=None, overridingUpdateCondtn=None):
    if self._bwPrevRegion is None:
      # Ensure update takes place
      self._bwPrevRegion = ~newBwArea
    curPrevAreaDiff = np.bitwise_xor(newBwArea, self._bwPrevRegion).sum() / \
                      newBwArea.size
    # Force cancel update if region hasn't changed
    if curPrevAreaDiff == 0:
      overridingUpdateCondtn = False
    # Don't forget to update bwArea after checking diff
    self._bwPrevRegion = newBwArea
    super().update(newVerts, curPrevAreaDiff > 0.05, overridingUpdateCondtn)

class FREditableImg(pg.PlotWidget):
  def __init__(self, parent=None, background='default', processor: FRImageProcessor=None, **kargs):
    super().__init__(parent, background, **kargs)
    self.setAspectLocked(True)
    self.getViewBox().invertY()
    self.setMouseEnabled(True)

    if processor is None:
      processor = FRImageProcessor()
    self.processor = processor

    # -----
    # DRAWING OPTIONS
    # -----
    self.drawShapeOpt = FR_CONSTS.DRAW_SHAPE_PAINT
    self.drawActionOpt = FR_CONSTS.DRAW_ACT_PAN
    self.drawShape = FRShape(self)

    # -----
    # IMAGE
    # -----
    self.image = pg.ImageItem()
    self.image.setZValue(-100)
    self.addItem(self.image)

  def mousePressEvent(self, ev: QtGui.QMouseEvent):
    if ev.buttons() == QtCore.Qt.LeftButton:
      self.drawShape.buildRoi(self.image, ev)

    super().mousePressEvent(ev)

  def mouseMoveEvent(self, ev: QtGui.QMouseEvent):
    """
    Mouse move behavior is contingent on which shape is currently selected
    """
    # Nothing to do if panning or no button is pressed
    if ev.buttons() == QtCore.Qt.LeftButton:
      self.drawShape.buildRoi(self.image, ev)
    super().mouseMoveEvent(ev)

  def mouseReleaseEvent(self, ev: QtGui.QMouseEvent):
    """
    Perform a processing method depending on what the current draw action is
    """
    isRoiDone = self.drawShape.buildRoi(self.image, ev)

    if isRoiDone:
      newVerts = self.processor.vertsFromPoints()
    super().mouseReleaseEvent(ev)

@FR_SINGLETON.registerClass(FR_CONSTS.CLS_MAIN_IMG_AREA)
class MainImageArea(pg.PlotWidget):
  sigComponentCreated = Signal(object)
  # Hooked up during __init__
  sigSelectionBoundsMade = Signal(object)

  @FR_SINGLETON.generalProps.registerProp(FR_CONSTS.PROP_NEW_COMP_SZ)
  def newCompSz(self): pass
  @FR_SINGLETON.generalProps.registerProp(FR_CONSTS.PROP_MIN_COMP_SZ)
  def minCompSz(self): pass
  @FR_SINGLETON.generalProps.registerProp(FR_CONSTS.PROP_MAIN_IMG_SEED_THRESH)
  def mainImgSeedThresh(self): pass


  def __init__(self, parent=None, background='default', imgSrc=None, **kargs):
    super().__init__(parent, background, viewBox=DraggableViewBox(), **kargs)

    self.allowNewComps = True

    self.setAspectLocked(True)
    self.viewbox: DraggableViewBox = self.getViewBox()
    self.viewbox.invertY()
    self.viewbox.sigCreationBoundsMade.connect(self.createCompFromBounds)
    self.sigSelectionBoundsMade = self.viewbox.sigSelectionBoundsMade

    # -----
    # Image Item
    # -----
    self.imgItem = ClickableImageItem()
    self.imgItem.sigClicked.connect(self.createCompAtClick)
    # Ensure image is behind plots
    self.imgItem.setZValue(-100)
    self.setImage(imgSrc)
    self.addItem(self.imgItem)

  # Spoof selection of empty area on escape to deselect components
  @FR_SINGLETON.shortcuts.registerMethod(FR_CONSTS.SHC_DESEL_ALL_BOUNDARIES)
  def deselectAllBoundaries(self):
    self.sigSelectionBoundsMade.emit((-1,-1,-1,-1))

  def createCompFromBounds(self, bounds:tuple):
    # TODO: Make this code more robust
    img_np = self.image
    compCoords = np.reshape(bounds, (2, 2)).astype(int)
    compCoords = getClippedBbox(img_np.shape, compCoords, 0).flatten()
    croppedImg = self.image[compCoords[1]:compCoords[3], compCoords[0]:compCoords[2], :]
    if croppedImg.size == 0: return
    # Performance for using all bounds is prohibitive for large components
    # TODO: Find a better method of determining whether to use all bounds
    if np.prod(croppedImg.shape[0:2]) > 25e3:
      shouldUseAllBounds = False
    else:
      shouldUseAllBounds = True
    newRegion = growBoundarySeeds(croppedImg, self.mainImgSeedThresh, self.minCompSz, useAllBounds=shouldUseAllBounds)
    newRegion = morphology.opening(newRegion, morphology.square(3))

    newVerts = getVertsFromBwComps(newRegion)
    # Remember to account for the vertex offset
    if len(newVerts) == 0: return
    # TODO: Determine more robust solution for separated vertices. For now use largest component
    elif len(newVerts) > 1:
      lens = np.array([len(v) for v in newVerts])
      newVerts = np.array(newVerts)[lens == lens.max()]
    newVerts[0] += compCoords[0:2].reshape(1, 2)
    newComp = makeCompDf(1)
    newComp[TC.VERTICES] = newVerts
    # newComp[TC.VERTICES] = [newVerts]
    self.sigComponentCreated.emit(newComp)
    return newComp

  def createCompAtClick(self, xyCoord: tuple):
    """
    Forms a box with a center at the clicked location, and passes the box
    edges as vertices for a new component.
    """
    vertBox = np.vstack((xyCoord, xyCoord))
    vertBox = getClippedBbox(self.image.shape, vertBox, self.newCompSz)
    miniImg = self.image[
      vertBox[0,1]:vertBox[1,1], vertBox[0,0]:vertBox[1,0],:
    ]
    # Account for mini img offset and x-y -> row-col
    xyCoord = xyCoord[::-1]
    xyCoord -= vertBox[0,:]
    bwCompMask = growSeedpoint(miniImg, xyCoord, self.mainImgSeedThresh, self.minCompSz)

    compVerts = getVertsFromBwComps(bwCompMask)
    if len(compVerts) == 0:
      return
    # Turn list-of-lists into plottable, nan-separated vertices
    # Reverse for row-col -> x-y
    compVerts = nanConcatList(compVerts)
    # Make sure the vertices are translated to the appropriate coordinates
    compVerts += vertBox[0,:]
    newComp = makeCompDf()
    newComp[TC.VERTICES] = [compVerts]
    self.sigComponentCreated.emit(newComp)
    return newComp

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

@FR_SINGLETON.registerClass(FR_CONSTS.CLS_FOCUSED_IMG_AREA)
class FocusedComp(FREditableImg):
  @FR_SINGLETON.generalProps.registerProp(FR_CONSTS.PROP_FOCUSED_SEED_THRESH)
  def seedThresh(self): pass
  @FR_SINGLETON.generalProps.registerProp(FR_CONSTS.PROP_MARGIN)
  def compCropMargin(self): pass
  @FR_SINGLETON.generalProps.registerProp(FR_CONSTS.PROP_SEG_THRESH)
  def segThresh(self): pass

  def __init__(self, parent=None):
    super().__init__(parent)
    self.inAddMode = True
    self.region = FRVertexRegion()
    self.addItem(self.region)

    self.compSer = makeCompDf().squeeze()

    self.bbox = np.zeros((2,2), dtype='int32')
    self.regionBuffer = RegionVertsUndoBuffer()

  def updateAll(self, mainImg: np.array, newComp:df):
    newVerts = newComp[TC.VERTICES].squeeze()
    # Since values INSIDE the dataframe are reset instead of modified, there is no
    # need to go through the trouble of deep copying
    self.compSer = newComp.copy(deep=False)

    # Reset the undo buffer
    self.regionBuffer = RegionVertsUndoBuffer()

    # Propagate all resultant changes
    self.updateBbox(mainImg.shape, newVerts)
    self.updateCompImg(mainImg)
    self.updateRegionFromVerts(newVerts)
    self.autoRange()

  def updateBbox(self, mainImgShape, newVerts: np.ndarray):
    # Ignore NAN entries during computation
    bbox = np.vstack([np.nanmin(newVerts, 0),
          np.nanmax(newVerts, 0)])
    # Account for margins
    self.bbox = getClippedBbox(mainImgShape, bbox, self.compCropMargin)

  def updateCompImg(self, mainImg, bbox=None):
    if bbox is None:
      bbox = self.bbox
    # Account for nan entries
    newCompImg = mainImg[bbox[0,1]:bbox[1,1],
                         bbox[0,0]:bbox[1,0],
                         :]
    segImg = segmentComp(newCompImg, self.segThresh)
    self.image.setImage(segImg)

  def updateRegionFromVerts(self, newVerts, offset=None):
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

  def updateRegionFromBwMask(self, bwImg: np.ndarray):
    vertsPerComp = getVertsFromBwComps(bwImg)
    vertsToUse = np.empty((0, 2), dtype='int')
    for verts in vertsPerComp:
      # TODO: handle case of multiple regions existing in bwImg. For now, just use
      #   the largest
      if verts.shape[0] > vertsToUse.shape[0]:
        vertsToUse = verts
    # TODO: Find a computationally good way to check for large changes every time a
    #  region is modified. The current solution can only work from mask-based updates
    #  unless an expensive cv.floodFill is used
    self.regionBuffer.update(vertsToUse, bwImg)
    self.updateRegionFromVerts(vertsToUse, [0, 0])

  @FR_SINGLETON.shortcuts.registerMethod(FR_CONSTS.SHC_UNDO_MOD_REGION, [FR_ENUMS.BUFFER_UNDO])
  @FR_SINGLETON.shortcuts.registerMethod(FR_CONSTS.SHC_REDO_MOD_REGION, [FR_ENUMS.BUFFER_REDO])
  def undoRedoRegionChange(self, undoOrRedo: str):
    # Ignore requests when no region present
    if self.compImgItem.image is None:
      return
    if undoOrRedo == FR_ENUMS.BUFFER_UNDO:
      self.updateRegionFromVerts(self.regionBuffer.undo_getObj(), [0, 0])
    elif undoOrRedo == FR_ENUMS.BUFFER_REDO:
      self.updateRegionFromVerts(self.regionBuffer.redo_getObj(), [0, 0])

  def saveNewVerts(self):
    # Add in offset from main image to FRVertexRegion vertices
    self.compSer.loc[TC.VERTICES] = self.region.verts + self.bbox[0,:]



# @FR_SINGLETON.registerClass(FR_CONSTS.CLS_FOCUSED_IMG_AREA)
# class FocusedComp(pg.PlotWidget):
#   sigModeChanged = Signal(bool)
#
#   @FR_SINGLETON.generalProps.registerProp(FR_CONSTS.PROP_FOCUSED_SEED_THRESH)
#   def seedThresh(self): pass
#   @FR_SINGLETON.generalProps.registerProp(FR_CONSTS.PROP_MARGIN)
#   def compCropMargin(self): pass
#   @FR_SINGLETON.generalProps.registerProp(FR_CONSTS.PROP_SEG_THRESH)
#   def segThresh(self): pass
#
#
#   def __init__(self, *args, **kwargs):
#     super().__init__(*args, **kwargs)
#     # Whether drawn items should be added or removed from current component
#     self.inAddMode = True
#     # Type of region to add once the user clicks. See radio buttons on the
#     # image annotator UI
#     self.drawType = 'seedpoint'
#
#     # For limiting number of mouse move-caused click simulations
#     self.lastXyVertFromMouseMove = np.array([0,0])
#
#     self.setAspectLocked(True)
#     self.getViewBox().invertY()
#
#     self.compSer = makeCompDf().squeeze()
#
#     self.bbox = np.zeros((2,2), dtype='int32')
#
#     self.compImgItem = ClickableImageItem()
#     self.compImgItem.requireCtrlKey = False
#     self.addItem(self.compImgItem)
#
#     self.region = FRVertexRegion()
#     self.addItem(self.region)
#
#     self.regionBuffer = RegionVertsUndoBuffer()
#
#     # self.interactor = _FRGeneralROI([], pen=(6, 9), removable=True)
#     # self.addItem(self.interactor)
#
#     # Remove points when user asks to delete polygon
#     # self.interactor.sigRemoveRequested.connect(self.interactor.clearPoints)
#     # Expand current mask by region on request
#     # self.interactor.finishPolyAct.triggered.connect(self._addRoiToRegion)
#
#     self.compImgItem.sigClicked.connect(self.compImageClicked)
#
#   @property
#   def clickable(self):
#     return self.compImgItem.clickable
#
#   @clickable.setter
#   def clickable(self, newVal):
#     self.compImgItem.clickable = bool(newVal)
#
#   def mouseMoveEvent(self, ev: QtGui.QKeyEvent):
#     if ev.modifiers() == QtCore.Qt.ControlModifier \
#        and ev.buttons() == QtCore.Qt.LeftButton:
#       # Simulate click in that location
#       posRelToImg = self.compImgItem.mapFromScene(ev.pos())
#       # Form of rate-limiting -- only simulate click if the next pixel is at least one away
#       # from the previous pixel location
#       xyCoord = np.round(np.array([[posRelToImg.x(), posRelToImg.y()]], dtype='int'))
#       if np.abs(xyCoord - self.lastXyVertFromMouseMove).sum() >= 1 \
#          and self.compImgItem.image is not None \
#          and np.all(xyCoord[:,::-1] < self.compImgItem.image.shape[:2]) \
#          and np.all(xyCoord >= 0):
#         # Above line also makes sure the click was within the image
#         self.compImageClicked(xyCoord)
#         ev.accept()
#         self.lastXyVertFromMouseMove = xyCoord
#     else:
#       super().mouseMoveEvent(ev)
#
#   def compImageClicked(self, newVert: np.ndarray):
#     # Capture clicks only if component is present and user allows it
#     # TODO: Expand to include ROI, superpixel, etc.
#     # Change vertex from x-y to row-col
#     newVert = np.fliplr(newVert)
#     newArea = growSeedpoint(self.compImgItem.image, newVert, self.seedThresh)
#     imgSize = newArea.shape
#     curRegionMask = self.region.embedMaskInImg(newArea.shape)
#     if self.inAddMode:
#       newArea |= curRegionMask
#     else:
#       newArea = ~newArea & curRegionMask
#
#     self.updateRegionFromBwMask(newArea)
#
#   def addRoiVertex(self, newVert: QtCore.QPointF):
#     # Account for moved ROI
#     newVert.setX(newVert.x() - self.interactor.x())
#     newVert.setY(newVert.y() - self.interactor.y())
#     # If enough points exist and new point is 'close' to origin,
#     # consider the ROI complete
#     handles = self.interactor.handles
#     lastPos = handles[-1]['pos'] if len(handles) > 2 else None
#     if lastPos is not None and abs(lastPos.x() - newVert.x()) < 5 \
#                            and abs(lastPos.y() - newVert.y()) < 5:
#       self.interactor.finishPolyAct.triggered.emit()
#     else:
#       # Add point as normal
#       prevVerts = [handle['pos'] for handle in handles]
#       self.interactor.setPoints(prevVerts + [newVert])
#
#   def setImage(self, image=None, autoLevels=None):
#     return self.compImgItem.setImage(image, autoLevels)
#
#   def updateAll(self, mainImg: np.array, newComp:df):
#     newVerts = newComp[TC.VERTICES].squeeze()
#     # Since values INSIDE the dataframe are reset instead of modified, there is no
#     # need to go through the trouble of deep copying
#     self.compSer = newComp.copy(deep=False)
#
#     # Reset the undo buffer
#     self.regionBuffer = RegionVertsUndoBuffer()
#
#     # Propagate all resultant changes
#     self.updateBbox(mainImg.shape, newVerts)
#     self.updateCompImg(mainImg)
#     self.updateRegionFromVerts(newVerts)
#     self.autoRange()
#
#   def updateBbox(self, mainImgShape, newVerts: np.ndarray):
#     # Ignore NAN entries during computation
#     bbox = np.vstack([np.nanmin(newVerts, 0),
#           np.nanmax(newVerts, 0)])
#     # Account for margins
#     self.bbox = getClippedBbox(mainImgShape, bbox, self.compCropMargin)
#
#   def updateCompImg(self, mainImg, bbox=None):
#     if bbox is None:
#       bbox = self.bbox
#     # Account for nan entries
#     newCompImg = mainImg[bbox[0,1]:bbox[1,1],
#                          bbox[0,0]:bbox[1,0],
#                          :]
#     segImg = segmentComp(newCompImg, self.segThresh)
#     self.setImage(segImg)
#
#   def updateRegionFromVerts(self, newVerts, offset=None):
#     # Component vertices are nan-separated regions
#     if offset is None:
#       offset = self.bbox[0,:]
#     if newVerts is None:
#       newVerts = np.ones((0,2), dtype=int)
#     # 0-center new vertices relative to FocusedComp image
#     # Make a copy of each list first so we aren't modifying the
#     # original data
#     centeredVerts = newVerts.copy()
#     centeredVerts -= offset
#     self.region.updateVertices(centeredVerts)
#
#   def updateRegionFromBwMask(self, bwImg: np.ndarray):
#     vertsPerComp = getVertsFromBwComps(bwImg)
#     vertsToUse = np.empty((0, 2), dtype='int')
#     for verts in vertsPerComp:
#       # TODO: handle case of multiple regions existing in bwImg. For now, just use
#       #   the largest
#       if verts.shape[0] > vertsToUse.shape[0]:
#         vertsToUse = verts
#     # TODO: Find a computationally good way to check for large changes every time a
#     #  region is modified. The current solution can only work from mask-based updates
#     #  unless an expensive cv.floodFill is used
#     self.regionBuffer.update(vertsToUse, bwImg)
#     self.updateRegionFromVerts(vertsToUse, [0, 0])
#
#   @FR_SINGLETON.shortcuts.registerMethod(FR_CONSTS.SHC_UNDO_MOD_REGION, [FR_ENUMS.BUFFER_UNDO])
#   @FR_SINGLETON.shortcuts.registerMethod(FR_CONSTS.SHC_REDO_MOD_REGION, [FR_ENUMS.BUFFER_REDO])
#   def undoRedoRegionChange(self, undoOrRedo: str):
#     # Ignore requests when no region present
#     if self.compImgItem.image is None:
#       return
#     if undoOrRedo == FR_ENUMS.BUFFER_UNDO:
#       self.updateRegionFromVerts(self.regionBuffer.undo_getObj(), [0, 0])
#     elif undoOrRedo == FR_ENUMS.BUFFER_REDO:
#       self.updateRegionFromVerts(self.regionBuffer.redo_getObj(), [0, 0])
#
#   def saveNewVerts(self):
#     # Add in offset from main image to FRVertexRegion vertices
#     self.compSer.loc[TC.VERTICES] = self.region.verts + self.bbox[0,:]
#
#   def _addRoiToRegion(self):
#     imgMask = self.interactor.getImgMask(self.compImgItem)
#     newRegion = np.bitwise_or(imgMask, self.region.image)
#     # TODO: Handle case of poly not intersecting existing region
#     self.updateRegionFromBwMask(newRegion)
#     # Now that the ROI was added to the region, remove it
#     self.interactor.clearPoints()