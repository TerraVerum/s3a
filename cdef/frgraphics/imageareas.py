from typing import Union, Tuple, Optional

import numpy as np
import pyqtgraph as pg
from pandas import DataFrame as df
from pyqtgraph import BusyCursor
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from skimage.io import imread

from cdef import FR_SINGLETON
from cdef.generalutils import getClippedBbox, FRObjUndoBuffer
from cdef.processingutils import getVertsFromBwComps, segmentComp
from cdef.projectvars import REQD_TBL_FIELDS, FR_CONSTS, FR_ENUMS
from cdef.structures import FRParam, FRVertices, FRComplexVertices
from cdef.structures import NChanImg
from .clickables import FRRightPanViewBox
from .drawopts import FRDrawOpts
from .regions import FRShapeCollection, FRVertexDefinedImg
# Required to trigger property registration
from .rois import FRExtendedROI

Signal = QtCore.pyqtSignal
Slot = QtCore.pyqtSlot
QCursor = QtGui.QCursor

@FR_SINGLETON.registerGroup(FR_CONSTS.CLS_REGION_BUF)
class FRRegionVertsUndoBuffer(FRObjUndoBuffer):

  @classmethod
  def __initEditorParams__(cls):
    cls.maxBufferLen, cls.stepsBetweenBufSave = FR_SINGLETON.generalProps.registerProps(cls,
        [FR_CONSTS.PROP_UNDO_BUF_SZ, FR_CONSTS.PROP_STEPS_BW_SAVE])

  def __init__(self):
    """
    Save buffer for region modification.
    """
    super().__init__(self.maxBufferLen, self.stepsBetweenBufSave)

class FREditableImg(pg.PlotWidget):
  def __init__(self, parent=None, allowableShapes: Tuple[FRParam,...]=None,
               allowableActions: Tuple[FRParam,...]=None, **kargs):
    super().__init__(parent, viewBox=FRRightPanViewBox(), **kargs)
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

  def handleShapeFinished(self, roi: FRExtendedROI) -> Optional[np.ndarray]:
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
    super().mousePressEvent(ev)
    if ev.buttons() == QtCore.Qt.LeftButton \
        and self.drawAction != FR_CONSTS.DRAW_ACT_PAN:
      self.shapeCollection.buildRoi(self.imgItem, ev)

  def mouseDoubleClickEvent(self, ev: QtGui.QMouseEvent):
    super().mouseDoubleClickEvent(ev)
    if ev.buttons() == QtCore.Qt.LeftButton \
        and self.drawAction != FR_CONSTS.DRAW_ACT_PAN:
      self.shapeCollection.buildRoi(self.imgItem, ev)

  def mouseMoveEvent(self, ev: QtGui.QMouseEvent):
    """
    Mouse move behavior is contingent on which shape is currently selected,
    unless we are panning
    """
    super().mouseMoveEvent(ev)
    if self.drawAction != FR_CONSTS.DRAW_ACT_PAN:
      self.shapeCollection.buildRoi(self.imgItem, ev)

  def mouseReleaseEvent(self, ev: QtGui.QMouseEvent):
    """
    Perform a processing method depending on what the current draw action is

    :return: Whether the mouse release completes the current ROI
    """
    super().mouseReleaseEvent(ev)
    # if ev.buttons() == QtCore.Qt.LeftButton:
    if self.drawAction != FR_CONSTS.DRAW_ACT_PAN:
      self.shapeCollection.buildRoi(self.imgItem, ev)

  def clearCurDrawShape(self):
    self.shapeCollection.clearAllRois()

@FR_SINGLETON.registerGroup(FR_CONSTS.CLS_MAIN_IMG_AREA)
class FRMainImage(FREditableImg):
  sigComponentCreated = Signal(object)
  # Hooked up during __init__
  sigSelectionBoundsMade = Signal(object)

  @classmethod
  def __initEditorParams__(cls):
    cls.multCompsOnCreate = FR_SINGLETON.generalProps.registerProp(cls,
                              FR_CONSTS.PROP_MK_MULT_COMPS_ON_ADD)

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

  def handleShapeFinished(self, roi: FRExtendedROI) -> Optional[np.ndarray]:
    if (self.drawAction == FR_CONSTS.DRAW_ACT_SELECT) and roi.connected:
      # Selection
      self.sigSelectionBoundsMade.emit(self.shapeCollection.shapeVerts)
    elif self.drawAction != FR_CONSTS.DRAW_ACT_PAN:
      # Component modification subject to processor
      # For now assume a single point indicates foreground where multiple indicate
      # background selection
      verts = self.shapeCollection.shapeVerts.astype(int)

      with BusyCursor():
        self.procCollection.run(fgVerts=verts, bgVerts=None)
      newVerts = self.procCollection.resultAsVerts(not self.multCompsOnCreate)
      newComps = FR_SINGLETON.tableData.makeCompDf(len(newVerts))
      newComps[REQD_TBL_FIELDS.VERTICES] = newVerts
      if len(newComps) == 0:
        return
      # TODO: Determine more robust solution for separated vertices. For now use largest component
      self.sigComponentCreated.emit(newComps)

  def mouseReleaseEvent(self, ev: QtGui.QMouseEvent):
    super().mouseReleaseEvent(ev)
    if self.drawAction == FR_CONSTS.DRAW_ACT_PAN:
      # Simulate a click-wide boundary selection so points can be selected in pan mode
      pos = self.imgItem.mapFromScene(ev.pos())
      xx, yy, = pos.x(), pos.y()
      squareCorners = FRVertices([[xx, yy], [xx, yy]], dtype=float)
      self.sigSelectionBoundsMade.emit(squareCorners)

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
      # TODO: Handle alpha channel images. For now, discard that data
      imgSrc = imread(imgSrc)
      if imgSrc.ndim < 3:
        imgSrc = imgSrc[:,:,None]
      imgSrc = imgSrc[:,:,0:3]

    if imgSrc is None:
      self.imgItem.clear()
    else:
      self.imgItem.setImage(imgSrc)
    self.procCollection.image = imgSrc

  @FR_SINGLETON.shortcuts.registerMethod(FR_CONSTS.SHC_CLEAR_SHAPE_MAIN)
  def clearCurDrawShape(self):
    super().clearCurDrawShape()

@FR_SINGLETON.registerGroup(FR_CONSTS.CLS_FOCUSED_IMG_AREA)
class FRFocusedImage(FREditableImg):

  @classmethod
  def __initEditorParams__(cls):
    cls.compCropMargin, cls.segThresh\
      = FR_SINGLETON.generalProps.registerProps(cls, [FR_CONSTS.PROP_CROP_MARGIN_PCT,
          FR_CONSTS.PROP_SEG_THRESH])

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

    self.compSer = FR_SINGLETON.tableData.makeCompDf().squeeze()
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

  def handleShapeFinished(self, roi: FRExtendedROI) -> Optional[np.ndarray]:
    if self.drawAction == FR_CONSTS.DRAW_ACT_PAN:
      return

    # Component modification subject to processor
    # For now assume a single point indicates foreground where multiple indicate
    # background selection
    verts = self.shapeCollection.shapeVerts.astype(int)
    vertsDict = {'fgVerts': None, 'bgVerts': None}
    if self.drawAction == FR_CONSTS.DRAW_ACT_ADD:
      vertsDict['fgVerts'] = verts
    elif self.drawAction == FR_CONSTS.DRAW_ACT_REM:
      vertsDict['bgVerts'] = verts
    # Check for flood fill

    newMask = self.procCollection.run(prevCompMask=self.compMask, **vertsDict)
    if not np.all(newMask == self.compMask):
      self.compMask = newMask
      self.region.updateFromMask(self.compMask)
      self.regionBuffer.update((self.compMask, (0,0)))

  def updateAll(self, mainImg: Optional[NChanImg], newComp:Optional[df]=None):
    if mainImg is None:
      self.imgItem.clear()
      self.region.updateFromVertices(FRComplexVertices())
      self.shapeCollection.clearAllRois()
      return
    newVerts: FRComplexVertices = newComp[REQD_TBL_FIELDS.VERTICES]
    # Since values INSIDE the dataframe are reset instead of modified, there is no
    # need to go through the trouble of deep copying
    self.compSer = newComp.copy(deep=False)

    # Reset the undo buffer size
    self.regionBuffer.clear()
    self.regionBuffer.resize(self.regionBuffer.maxBufferLen)

    # Propagate all resultant changesre
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
    padding = max((bbox[1,:] - bbox[0,:])*self.compCropMargin/2/100)
    self.bbox = getClippedBbox(mainImgShape, bbox, int(padding))

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
    lstLens = lambda lst: np.array([len(el) for el in lst])
    centeredVerts = newVerts.copy()
    for vertList in centeredVerts:
      vertList -= offset
    shouldUpdate = (not self.region.vertsUpToDate
                    or len(self.region.verts) != len(centeredVerts)
                    or np.any(lstLens(self.region.verts) != lstLens(centeredVerts))
                    or np.any(np.vstack([selfLst != newLst for selfLst, newLst
                                  in zip(self.region.verts, centeredVerts)])))
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
    self.compSer.loc[REQD_TBL_FIELDS.VERTICES] = newVerts