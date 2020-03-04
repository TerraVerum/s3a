from typing import Union, Tuple, List, Optional

import numpy as np
import pyqtgraph as pg
from PIL import Image
from pandas import DataFrame as df
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from Annotator.FRGraphics.regions import FRVertexDefinedImg
from Annotator.params import FRComplexVertices
from Annotator.processing import getVertsFromBwComps
from .clickables import RightPanViewBox
from .drawopts import FRDrawOpts
from .parameditors import FR_SINGLETON
from .regions import FRShapeCollection
# Required to trigger property registration
from .rois import FRExtendedROI
from ..constants import TEMPLATE_COMP as TC, FR_CONSTS, FR_ENUMS
from ..generalutils import getClippedBbox, ObjUndoBuffer
from ..params import FRParam
from ..params import FRVertices
from ..processing import segmentComp
from ..tablemodel import makeCompDf

Signal = QtCore.pyqtSignal
Slot = QtCore.pyqtSlot
QCursor = QtGui.QCursor

@FR_SINGLETON.registerClass(FR_CONSTS.CLS_REGION_BUF)
class FRRegionVertsUndoBuffer(ObjUndoBuffer):
  @FR_SINGLETON.generalProps.registerProp(FR_CONSTS.PROP_UNDO_BUF_SZ)
  def maxBufferLen(self): pass
  @FR_SINGLETON.generalProps.registerProp(FR_CONSTS.PROP_STEPS_BW_SAVE)
  def stepsBetweenBufSave(self): pass

  def __init__(self):
    """
    Save buffer for region modification.
    """
    super().__init__(self.maxBufferLen, self.stepsBetweenBufSave)

class FREditableImg(pg.PlotWidget):
  def __init__(self, parent=None, allowableShapes: Tuple[FRParam,...]=None,
               allowableActions: Tuple[FRParam,...]=None, **kargs):
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


    self.procCollection = FR_SINGLETON.algParamMgr.createProcessorForClass(self)
    self.procCollection.image = self.imgItem.image

    # -----
    # DRAWING OPTIONS
    # -----
    self.drawAction: FRParam = FR_CONSTS.DRAW_ACT_PAN
    self.shapeCollection = FRShapeCollection(parent=self, allowableShapes=allowableShapes)
    self.shapeCollection.sigShapeFinished.connect(self.handleShapeFinished)

    # Make sure panning is allowed before creating draw widget
    if FR_CONSTS.DRAW_ACT_PAN not in allowableActions:
      allowableActions += (FR_CONSTS.DRAW_ACT_PAN,)
    self.drawOptsWidget = FRDrawOpts(parent, allowableShapes, allowableActions)
    btnGroups = [self.drawOptsWidget.shapeBtnGroup, self.drawOptsWidget.actionBtnGroup]
    for group in btnGroups:
      group.buttonToggled.connect(self._handleBtnToggle)
    self.drawOptsWidget.selectOpt(self.drawAction)
    self.drawOptsWidget.selectOpt(self.shapeCollection.curShape)

  def handleShapeFinished(self, roi: FRExtendedROI, fgVerts: FRVertices=None, bgVerts: FRVertices=None, prevComp=None) \
      -> Optional[np.ndarray]:
    """
    Overloaded in child classes to process new regions
    """

  def switchBtnMode(self, newMode: FRParam):
    for curDict in [self.drawOptsWidget.actionBtnParamMap.inv,
                    self.drawOptsWidget.shapeBtnParamMap.inv]:
      if newMode in curDict:
        curDict[newMode].setChecked(True)
        return
    # If this is reached, a param was passed in that doesn't correspond to a valid button
    # TODO: return soemthing else?

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
    if self.drawAction != FR_CONSTS.DRAW_ACT_PAN:
      self.shapeCollection.buildRoi(self.imgItem, ev)

    super().mouseReleaseEvent(ev)

  def clearCurDrawShape(self):
    self.shapeCollection.clearAllRois()

@FR_SINGLETON.registerClass(FR_CONSTS.CLS_MAIN_IMG_AREA)
class FRMainImage(FREditableImg):
  sigComponentCreated = Signal(object)
  # Hooked up during __init__
  sigSelectionBoundsMade = Signal(object)

  def __init__(self, parent=None, imgSrc=None, **kargs):
    allowedShapes = (FR_CONSTS.DRAW_SHAPE_RECT, FR_CONSTS.DRAW_SHAPE_POLY)
    allowedActions = (FR_CONSTS.DRAW_ACT_SELECT,FR_CONSTS.DRAW_ACT_ADD)
    super().__init__(parent, allowableShapes=allowedShapes,
                     allowableActions=allowedActions, **kargs)

    # -----
    # Image Item
    # -----
    self.setImage(imgSrc)

    self.switchBtnMode(FR_CONSTS.DRAW_ACT_ADD)

  def handleShapeFinished(self, roi: FRExtendedROI, fgVerts: FRVertices=None, bgVerts: FRVertices=None,
                          prevComp=None) -> Optional[np.ndarray]:
    if self.drawAction == FR_CONSTS.DRAW_ACT_SELECT \
        and roi.connected:
      # Selection
      self.sigSelectionBoundsMade.emit(self.shapeCollection.shapeVerts)
    elif self.drawAction != FR_CONSTS.DRAW_ACT_PAN:
      # Component modification subject to processor
      prevComp = np.zeros(self.image.shape[0:2], dtype=bool)
      # For now assume a single point indicates foreground where multiple indicate
      # background selection
      verts = self.shapeCollection.shapeVerts.astype(int)

      newCompMask = self.procCollection.curProcessor.localCompEstimate(prevComp, verts, None)
      newVerts = getVertsFromBwComps(newCompMask)
      if len(newVerts.stack()) == 0:
        return
      # TODO: Determine more robust solution for separated vertices. For now use largest component
      newComp = makeCompDf(1)
      newComp[TC.VERTICES] = [newVerts.copy()]
      self.sigComponentCreated.emit(newComp)

  @FR_SINGLETON.shortcuts.registerMethod(FR_CONSTS.SHC_DRAW_FG, [FR_CONSTS.DRAW_ACT_ADD])
  @FR_SINGLETON.shortcuts.registerMethod(FR_CONSTS.SHC_DRAW_PAN, [FR_CONSTS.DRAW_ACT_PAN])
  @FR_SINGLETON.shortcuts.registerMethod(FR_CONSTS.SHC_DRAW_SELECT, [FR_CONSTS.DRAW_ACT_SELECT])
  @FR_SINGLETON.shortcuts.registerMethod(FR_CONSTS.SHC_DRAW_RECT, [FR_CONSTS.DRAW_SHAPE_RECT])
  @FR_SINGLETON.shortcuts.registerMethod(FR_CONSTS.SHC_DRAW_POLY, [FR_CONSTS.DRAW_SHAPE_POLY])
  def switchBtnMode(self, newMode: FRParam):
    super().switchBtnMode(newMode)

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
    self.procCollection.image = imgSrc

  @FR_SINGLETON.shortcuts.registerMethod(FR_CONSTS.SHC_CLEAR_SHAPE_MAIN)
  def clearCurDrawShape(self):
    super().clearCurDrawShape()

@FR_SINGLETON.registerClass(FR_CONSTS.CLS_FOCUSED_IMG_AREA)
class FRFocusedImage(FREditableImg):
  @FR_SINGLETON.generalProps.registerProp(FR_CONSTS.PROP_MARGIN)
  def compCropMargin(self): pass
  @FR_SINGLETON.generalProps.registerProp(FR_CONSTS.PROP_SEG_THRESH)
  def segThresh(self): pass

  def __init__(self, parent=None, **kargs):
    allowableShapes = (
      FR_CONSTS.DRAW_SHAPE_RECT, FR_CONSTS.DRAW_SHAPE_POLY, FR_CONSTS.DRAW_SHAPE_PAINT
    )
    allowableActions = (
      FR_CONSTS.DRAW_ACT_ADD, FR_CONSTS.DRAW_ACT_REM
    )
    super().__init__(parent, allowableShapes, allowableActions, **kargs)
    self.region = FRVertexDefinedImg()
    self.addItem(self.region)

    self.compSer = makeCompDf().squeeze()
    # Image representation of component boundaries
    self.compMask = np.zeros((1,1), bool)

    self.bbox = np.zeros((2, 2), dtype='int32')
    self.regionBuffer = FRRegionVertsUndoBuffer()

    self.switchBtnMode(FR_CONSTS.DRAW_ACT_ADD)
    self.switchBtnMode(FR_CONSTS.DRAW_SHAPE_PAINT)

  @FR_SINGLETON.shortcuts.registerMethod(FR_CONSTS.SHC_DRAW_BG, [FR_CONSTS.DRAW_ACT_REM])
  @FR_SINGLETON.shortcuts.registerMethod(FR_CONSTS.SHC_DRAW_FG, [FR_CONSTS.DRAW_ACT_ADD])
  @FR_SINGLETON.shortcuts.registerMethod(FR_CONSTS.SHC_DRAW_PAN, [FR_CONSTS.DRAW_ACT_PAN])
  @FR_SINGLETON.shortcuts.registerMethod(FR_CONSTS.SHC_DRAW_RECT, [FR_CONSTS.DRAW_SHAPE_RECT])
  @FR_SINGLETON.shortcuts.registerMethod(FR_CONSTS.SHC_DRAW_POLY, [FR_CONSTS.DRAW_SHAPE_POLY])
  @FR_SINGLETON.shortcuts.registerMethod(FR_CONSTS.SHC_DRAW_PAINT, [FR_CONSTS.DRAW_SHAPE_PAINT])
  def switchBtnMode(self, newMode: FRParam):
    super().switchBtnMode(newMode)

  @FR_SINGLETON.shortcuts.registerMethod(FR_CONSTS.SHC_CLEAR_SHAPE_FOC)
  def clearCurDrawShape(self):
    super().clearCurDrawShape()

  def resetImage(self):
    self.updateAll(None)

  def handleShapeFinished(self, roi: FRExtendedROI, fgVerts: FRVertices=None, bgVerts: FRVertices=None,
                          prevComp=None) -> Optional[np.ndarray]:
    if self.drawAction == FR_CONSTS.DRAW_ACT_PAN:
      return

    # Component modification subject to processor
    # For now assume a single point indicates foreground where multiple indicate
    # background selection
    verts = self.shapeCollection.shapeVerts.astype(int)
    fgBgVerts = [None, None]
    if self.drawAction == FR_CONSTS.DRAW_ACT_ADD:
      fgBgVerts[0] = verts
    elif self.drawAction == FR_CONSTS.DRAW_ACT_REM:
      fgBgVerts[1] = verts
    # Check for flood fill

    newMask = self.procCollection.curProcessor.localCompEstimate(
      self.compMask, *fgBgVerts)
    if not np.all(newMask == self.compMask):
      self.compMask = newMask
      self.region.updateFromMask(self.compMask)
      self.regionBuffer.update((self.compMask, (0,0)))

  def updateAll(self, mainImg: Optional[np.array], newComp:Optional[df]=None):
    if mainImg is None:
      mainImg = np.zeros((1,1,3))
      self.imgItem.setImage(mainImg)
      self.region.updateFromVertices(FRComplexVertices())
      return
    newVerts: FRComplexVertices = newComp[TC.VERTICES]
    # Since values INSIDE the dataframe are reset instead of modified, there is no
    # need to go through the trouble of deep copying
    self.compSer = newComp.copy(deep=False)

    # Reset the undo buffer
    self.regionBuffer = FRRegionVertsUndoBuffer()

    # Propagate all resultant changes
    self.updateBbox(mainImg.shape, newVerts)
    self.updateCompImg(mainImg)
    self.updateRegionFromVerts(newVerts)
    self.autoRange()

  def updateBbox(self, mainImgShape, newVerts: FRComplexVertices):
    concatVerts = newVerts.stack()
    # Ignore NAN entries during computation
    bbox = np.vstack([concatVerts.min(0),
                      concatVerts.max(0)])
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
    self.procCollection.image = segImg

  def updateRegionFromVerts(self, newVerts: FRComplexVertices, offset=None):
    # Component vertices are nan-separated regions
    if offset is None:
      offset = self.bbox[0,:]
    if newVerts is None:
      newVerts = FRComplexVertices()
    # 0-center new vertices relative to FRFocusedImage image
    # Make a copy of each list first so we aren't modifying the
    # original data
    centeredVerts = newVerts.copy()
    for vertList in centeredVerts:
      vertList -= offset
    shouldUpdate = not self.region.vertsUpToDate \
                   or len(self.region.verts) != len(centeredVerts) \
                   or not np.all(np.vstack([selfLst == newLst for selfLst, newLst
                                  in zip(self.region.verts, centeredVerts)]))
    if shouldUpdate:
      self.region.updateFromVertices(centeredVerts)
      regionPos = self.region.pos().x(), self.region.pos().y()
      self.regionBuffer.update((self.region.image, regionPos))
      self.compMask = self.region.embedMaskInImg(self.imgItem.image.shape[:2])

  @FR_SINGLETON.shortcuts.registerMethod(FR_CONSTS.SHC_UNDO_MOD_REGION, [FR_ENUMS.BUFFER_UNDO])
  @FR_SINGLETON.shortcuts.registerMethod(FR_CONSTS.SHC_REDO_MOD_REGION, [FR_ENUMS.BUFFER_REDO])
  def undoRedoRegionChange(self, undoOrRedo: str):
    # Ignore requests when no region present
    if self.imgItem.image is None:
      return
    if undoOrRedo == FR_ENUMS.BUFFER_UNDO:
      newMask, offset = self.regionBuffer.undo_getObj()
    elif undoOrRedo == FR_ENUMS.BUFFER_REDO:
      newMask, offset = self.regionBuffer.redo_getObj()
    self.region.updateFromMask(newMask, offset)
    self.compMask = self.region.embedMaskInImg(self.compMask.shape)

  def clearCurDrawShape(self):
    super().clearCurDrawShape()

  def saveNewVerts(self):
    # Add in offset from main image to FRVertexRegion vertices
    if not self.region.vertsUpToDate:
      newVerts = getVertsFromBwComps(self.region.image)
      self.region.verts = newVerts.copy()
      self.region.vertsUpToDate = True
    else:
      newVerts = self.region.verts.copy()
    for vertList in newVerts:
      vertList += self.bbox[0,:]
    self.compSer.loc[TC.VERTICES] = newVerts