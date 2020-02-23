from typing import Union, Optional, Tuple

import numpy as np
import pyqtgraph as pg
from PIL import Image
from pandas import DataFrame as df
from pyqtgraph import Point
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from skimage import morphology

from Annotator.FRGraphics.clickables import RightPanViewBox
from Annotator.FRGraphics.rois import FRExtendedROI
from Annotator.interfaceimpls import RegionGrow
from Annotator.params import FRVertices
from .clickables import ClickableImageItem
from .clickables import DraggableViewBox
from .drawopts import FRDrawOpts
from .parameditors import FR_SINGLETON
from .regions import FRVertexRegion, FRShapeCollection
from ..constants import TEMPLATE_COMP as TC, FR_CONSTS, FR_ENUMS
from ..generalutils import getClippedBbox, nanConcatList, ObjUndoBuffer
from ..params import FRImageProcessor
from ..params import FRParam
from ..processing import segmentComp, getVertsFromBwComps, growSeedpoint, growBoundarySeeds
from ..tablemodel import makeCompDf

Signal = QtCore.pyqtSignal
Slot = QtCore.pyqtSlot
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
  def __init__(self, parent=None, processor: FRImageProcessor=None,
               allowableShapes: Tuple[FRParam,...]=None, allowableActions: Tuple[FRParam,...]=None,
               **kargs):
    super().__init__(parent, viewBox=RightPanViewBox(), **kargs)
    self.setAspectLocked(True)
    self.getViewBox().invertY()
    self.setMouseEnabled(True)

    # -----
    # IMAGE
    # -----
    self.imgItem = pg.ImageItem()
    self.imgItem.setZValue(-100)
    self.addItem(self.imgItem)


    if processor is None:
      processor = RegionGrow()
    processor.image = self.imgItem.image
    self.processor = processor

    # -----
    # DRAWING OPTIONS
    # -----
    self.drawAction: FRParam = FR_CONSTS.DRAW_ACT_PAN
    self.shapeCollection = FRShapeCollection(parent=self, allowableShapes=allowableShapes)

    # Make sure panning is allowed before creating draw widget
    if FR_CONSTS.DRAW_ACT_PAN not in allowableActions:
      allowableActions += (FR_CONSTS.DRAW_ACT_PAN,)
    self.drawOptsWidget = FRDrawOpts(parent, allowableShapes, allowableActions)
    btnGroups = [self.drawOptsWidget.shapeBtnGroup, self.drawOptsWidget.actionBtnGroup]
    for group in btnGroups:
      group.buttonToggled.connect(self._handleBtnToggle)
    self.drawOptsWidget.selectOpt(self.drawAction)
    self.drawOptsWidget.selectOpt(self.shapeCollection.curShape)

  @Slot(QtWidgets.QAbstractButton, bool)
  def _handleBtnToggle(self, btn: QtWidgets.QPushButton, isChecked: bool):
    """
    This function will be called for each button check and corresponding UNCHECK
    of the previously selected button. So, only listen for the changes of the
    checked button, since this is the new action/shape
    :param btn: the toggled QPushButton
    :param isChecked: Whether the button was checked or unchecked
    :return: None
    """
    if not isChecked: return
    if btn in self.drawOptsWidget.actionBtnParamMap:
      self.drawAction = self.drawOptsWidget.actionBtnParamMap[btn]
    else:
      # Shape toggle
      self.shapeCollection.curShape = self.drawOptsWidget.shapeBtnParamMap[btn]

  def mousePressEvent(self, ev: QtGui.QMouseEvent):
    if ev.buttons() == QtCore.Qt.LeftButton \
        and self.drawAction != FR_CONSTS.DRAW_ACT_PAN:
      self.shapeCollection.buildRoi(self.imgItem, ev)

    super().mousePressEvent(ev)

  def mouseMoveEvent(self, ev: QtGui.QMouseEvent):
    """
    Mouse move behavior is contingent on which shape is currently selected,
    unless we are panning
    """
    if self.drawAction != FR_CONSTS.DRAW_ACT_PAN:
      self.shapeCollection.buildRoi(self.imgItem, ev)
    super().mouseMoveEvent(ev)

  def mouseReleaseEvent(self, ev: QtGui.QMouseEvent):
    """
    Perform a processing method depending on what the current draw action is

    :return: Whether the mouse release completes the current ROI
    """
    # if ev.buttons() == QtCore.Qt.LeftButton:
    self.shapeCollection.buildRoi(self.imgItem, ev)

    super().mouseReleaseEvent(ev)

  def clearCurDrawShape(self):
    self.shapeCollection.clearAllRois()

@FR_SINGLETON.registerClass(FR_CONSTS.CLS_MAIN_IMG_AREA)
class MainImageArea(FREditableImg):
  sigComponentCreated = Signal(object)
  # Hooked up during __init__
  sigSelectionBoundsMade = Signal(object)

  @FR_SINGLETON.generalProps.registerProp(FR_CONSTS.PROP_NEW_COMP_SZ)
  def newCompSz(self): pass
  @FR_SINGLETON.generalProps.registerProp(FR_CONSTS.PROP_MIN_COMP_SZ)
  def minCompSz(self): pass
  @FR_SINGLETON.generalProps.registerProp(FR_CONSTS.PROP_MAIN_IMG_SEED_THRESH)
  def mainImgSeedThresh(self): pass


  def __init__(self, parent=None, imgSrc=None, **kargs):
    allowedShapes = (FR_CONSTS.DRAW_SHAPE_RECT, FR_CONSTS.DRAW_SHAPE_POLY)
    allowedActions = (FR_CONSTS.DRAW_ACT_SELECT,FR_CONSTS.DRAW_ACT_ADD)
    super().__init__(parent, allowableShapes=allowedShapes,
                     allowableActions=allowedActions, **kargs)

    # self.viewbox.sigCreationBoundsMade.connect(self.createCompFromBounds)
    # self.sigSelectionBoundsMade = self.viewbox.sigSelectionBoundsMade
    self.shapeCollection.sigShapeFinished.connect(self._handleShapeFinished)
    # -----
    # Image Item
    # -----
    self.setImage(imgSrc)

  def _handleShapeFinished(self, roi: FRExtendedROI):
    if self.drawAction == FR_CONSTS.DRAW_ACT_SELECT \
        and roi.connected:
      # Selection
      self.sigSelectionBoundsMade.emit(self.shapeCollection.shapeVerts)
    elif self.drawAction != FR_CONSTS.DRAW_ACT_PAN:
      # Component modification subject to processor
      prevComp = np.zeros(self.image.shape[0:2], dtype=bool)
      fgBgVerts = [None, None]
      if self.drawAction == FR_CONSTS.DRAW_ACT_ADD:
        # Assume user selected foreground
        fgBgVerts[0] = self.shapeCollection.shapeVerts
      elif self.drawAction == FR_CONSTS.DRAW_ACT_REM:
        fgBgVerts[1] = self.shapeCollection.shapeVerts
      newVerts = self.processor.localCompEstimate(prevComp, *fgBgVerts)
      if len(newVerts) == 0: return
        # TODO: Determine more robust solution for separated vertices. For now use largest component
      elif len(newVerts) > 1:
        lens = np.array([len(v) for v in newVerts])
        newVerts = np.array(newVerts)[lens == lens.max()]
      newComp = makeCompDf(1)
      newComp[TC.VERTICES] = newVerts
      # newComp[TC.VERTICES] = [newVerts]
      self.sigComponentCreated.emit(newComp)
      return newComp


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

  def setImage(self, imgSrc: Union[str, np.ndarray]=None):
    """
    Allows the user to change the main image either from a filepath or array data
    """
    if isinstance(imgSrc, str):
      imgSrc = np.array(Image.open(imgSrc))

    self.imgItem.setImage(imgSrc)
    self.processor.image = imgSrc

  @FR_SINGLETON.shortcuts.registerMethod(FR_CONSTS.SHC_CLEAR_SHAPE_MAIN)
  def clearCurDrawShape(self):
    super().clearCurDrawShape()

@FR_SINGLETON.registerClass(FR_CONSTS.CLS_FOCUSED_IMG_AREA)
class FocusedImg(FREditableImg):
  @FR_SINGLETON.generalProps.registerProp(FR_CONSTS.PROP_FOCUSED_SEED_THRESH)
  def seedThresh(self): pass
  @FR_SINGLETON.generalProps.registerProp(FR_CONSTS.PROP_MARGIN)
  def compCropMargin(self): pass
  @FR_SINGLETON.generalProps.registerProp(FR_CONSTS.PROP_SEG_THRESH)
  def segThresh(self): pass

  def __init__(self, parent=None, processor: FRImageProcessor = None, **kargs):
    allowableShapes = (
      FR_CONSTS.DRAW_SHAPE_RECT, FR_CONSTS.DRAW_SHAPE_POLY, FR_CONSTS.DRAW_SHAPE_PAINT
    )
    allowableActions = (
      FR_CONSTS.DRAW_ACT_ADD, FR_CONSTS.DRAW_ACT_REM
    )
    super().__init__(parent, processor, allowableShapes, allowableActions, **kargs)
    self.region = FRVertexRegion()
    self.addItem(self.region)

    self.compSer = makeCompDf().squeeze()

    self.bbox = np.zeros((2, 2), dtype='int32')
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
    self.updateRegionFromVerts(FRVertices(newVerts))
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
    self.imgItem.setImage(segImg)

  def updateRegionFromVerts(self, newVerts: FRVertices, offset=None):
    # Component vertices are nan-separated regions
    if offset is None:
      offset = self.bbox[0,:]
    if newVerts is None:
      newVerts = FRVertices(dtype=int)
    # 0-center new vertices relative to FocusedImg image
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

  def clearCurDrawShape(self):
    super().clearCurDrawShape()

  @FR_SINGLETON.shortcuts.registerMethod(FR_CONSTS.SHC_CLEAR_SHAPE_FOC)
  def saveNewVerts(self):
    # Add in offset from main image to FRVertexRegion vertices
    self.compSer.loc[TC.VERTICES] = self.region.verts + self.bbox[0,:]