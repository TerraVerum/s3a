from typing import Union, Tuple, Optional

import numpy as np
import pandas as pd
import pyqtgraph as pg
from pyqtgraph import BusyCursor
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from s3a.procwrapper import FRImgProcWrapper
from skimage.io import imread

from s3a import FR_SINGLETON
from s3a.generalutils import getClippedBbox
from ..processingimpls import segmentComp
from s3a.projectvars import REQD_TBL_FIELDS, FR_CONSTS
from s3a.structures import FRParam, FRVertices, FRComplexVertices, FilePath, \
  BlackWhiteImg
from s3a.structures import NChanImg
from .clickables import FRRightPanViewBox
from .drawopts import FRDrawOpts
from .regions import FRShapeCollection, FRVertexDefinedImg
# Required to trigger property registration
from .rois import FRExtendedROI

Signal = QtCore.pyqtSignal
Slot = QtCore.pyqtSlot
QCursor = QtGui.QCursor

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

    # -----
    # DRAWING OPTIONS
    # -----
    self.drawAction: FRParam = FR_CONSTS.DRAW_ACT_PAN
    self.shapeCollection = FRShapeCollection(allowableShapes, self)
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

  @property
  def image(self) -> Optional[NChanImg]:
    return self.imgItem.image

  @property
  def curProcessor(self):
      return self.procCollection.curProcessor
  @curProcessor.setter
  def curProcessor(self, newProcessor: Union[str, FRImgProcWrapper]):
    self.procCollection.switchActiveProcessor(newProcessor)

  def handleShapeFinished(self, roiVerts: FRVertices) -> Optional[np.ndarray]:
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
      self.shapeCollection.buildRoi(ev, self.imgItem)

  def mouseDoubleClickEvent(self, ev: QtGui.QMouseEvent):
    super().mouseDoubleClickEvent(ev)
    if ev.buttons() == QtCore.Qt.LeftButton \
        and self.drawAction != FR_CONSTS.DRAW_ACT_PAN:
      self.shapeCollection.buildRoi(ev, self.imgItem)

  def mouseMoveEvent(self, ev: QtGui.QMouseEvent):
    """
    Mouse move behavior is contingent on which shape is currently selected,
    unless we are panning
    """
    super().mouseMoveEvent(ev)
    if self.drawAction != FR_CONSTS.DRAW_ACT_PAN:
      self.shapeCollection.buildRoi(ev, self.imgItem)

  def mouseReleaseEvent(self, ev: QtGui.QMouseEvent):
    """
    Perform a processing method depending on what the current draw action is

    :return: Whether the mouse release completes the current ROI
    """
    super().mouseReleaseEvent(ev)
    # if ev.buttons() == QtCore.Qt.LeftButton:
    if self.drawAction != FR_CONSTS.DRAW_ACT_PAN:
      self.shapeCollection.buildRoi(ev, self.imgItem)

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

  def handleShapeFinished(self, roiVerts: FRVertices) -> Optional[np.ndarray]:
    if (self.drawAction == FR_CONSTS.DRAW_ACT_SELECT) and roiVerts.connected:
      # Selection
      self.sigSelectionBoundsMade.emit(self.shapeCollection.shapeVerts)
    elif self.drawAction != FR_CONSTS.DRAW_ACT_PAN:
      # Component modification subject to processor
      # For now assume a single point indicates foreground where multiple indicate
      # background selection
      verts = self.shapeCollection.shapeVerts.astype(int)

      with BusyCursor():
        self.curProcessor.run(image=self.image, fgVerts=verts, bgVerts=None)
      newVerts = self.curProcessor.resultAsVerts(not self.multCompsOnCreate)
      # Discard entries with no real vertices
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

  def setImage(self, imgSrc: Union[FilePath, np.ndarray]=None):
    """
    Allows the user to change the main image either from a filepath or array data
    """
    if isinstance(imgSrc, FilePath.__args__):
      # TODO: Handle alpha channel images. For now, discard that data
      imgSrc = imread(imgSrc)
      if imgSrc.ndim < 3:
        imgSrc = imgSrc[:,:,None]
      imgSrc = imgSrc[:,:,0:3]

    if imgSrc is None:
      self.imgItem.clear()
    else:
      self.imgItem.setImage(imgSrc)

  @FR_SINGLETON.shortcuts.registerMethod(FR_CONSTS.SHC_CLEAR_SHAPE_MAIN)
  def clearCurDrawShape(self):
    super().clearCurDrawShape()

@FR_SINGLETON.registerGroup(FR_CONSTS.CLS_FOCUSED_IMG_AREA)
class FRFocusedImage(FREditableImg):

  @classmethod
  def __initEditorParams__(cls):
    cls.compCropMargin = FR_SINGLETON.generalProps.registerProp(cls, FR_CONSTS.PROP_CROP_MARGIN_PCT)

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

    self.compSer: pd.Series = FR_SINGLETON.tableData.makeCompDf().squeeze()

    self.bbox = np.zeros((2, 2), dtype='int32')

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

  def handleShapeFinished(self, roiVerts: FRVertices) -> Optional[np.ndarray]:
    if self.drawAction == FR_CONSTS.DRAW_ACT_PAN:
      return

    # Component modification subject to processor
    # For now assume a single point indicates foreground where multiple indicate
    # background selection
    roiVerts = roiVerts.astype(int)
    vertsDict = {'fgVerts': None, 'bgVerts': None}
    if self.drawAction == FR_CONSTS.DRAW_ACT_ADD:
      vertsDict['fgVerts'] = roiVerts
    elif self.drawAction == FR_CONSTS.DRAW_ACT_REM:
      vertsDict['bgVerts'] = roiVerts
    # Check for flood fill

    compMask = self.region.embedMaskInImg(self.image.shape[:2])
    newMask = self.curProcessor.run(image=self.image, prevCompMask=compMask, **vertsDict)
    if not np.array_equal(newMask,compMask):
      self.region.updateFromMask(newMask)

  @FR_SINGLETON.actionStack.undoable('Modify Focused Component')
  def updateAll(self, mainImg: Optional[NChanImg], newComp:Optional[pd.Series]=None,
                isAlreadyTrimmed=False):
    oldImg = self.image
    if oldImg is not None:
      oldImg = oldImg.copy()
    oldComp = self.compSer

    if mainImg is None:
      self.imgItem.clear()
      self.updateRegionFromVerts(None)
      self.shapeCollection.clearAllRois()
    else:
      newVerts: FRComplexVertices = newComp[REQD_TBL_FIELDS.VERTICES]
      # Since values INSIDE the dataframe are reset instead of modified, there is no
      # need to go through the trouble of deep copying
      self.compSer = newComp.copy(deep=False)

      # Propagate all resultant changes
      if not isAlreadyTrimmed:
        self.updateBbox(mainImg.shape, newVerts)
        bboxToUse = self.bbox
      else:
        bboxToUse = FRVertices([[0,0], mainImg.shape[:2]])
        self.bbox = bboxToUse
      self.updateCompImg(mainImg, bboxToUse)
      self.updateRegionFromVerts(newVerts, bboxToUse[0,:])
      self.autoRange()
    yield
    self.updateAll(oldImg, oldComp, True)

  def updateBbox(self, mainImgShape, newVerts: FRComplexVertices):
    concatVerts = newVerts.stack()
    # Ignore NAN entries during computation
    bbox = np.vstack([concatVerts.min(0),
                      concatVerts.max(0)])
    # Account for margins
    padding = max((bbox[1,:] - bbox[0,:])*self.compCropMargin/2/100)
    self.bbox = getClippedBbox(mainImgShape, bbox, int(padding))

  def updateCompImg(self, mainImg, bbox: FRVertices=None):
    if bbox is None:
      bbox = self.bbox
    # Account for nan entries
    newCompImg = mainImg[bbox[0,1]:bbox[1,1],
                         bbox[0,0]:bbox[1,0],
                         :]
    self.imgItem.setImage(newCompImg)

  @FR_SINGLETON.actionStack.undoable('Modify Focused Component')
  def updateRegionFromVerts(self, newVerts: FRComplexVertices=None, offset: FRVertices=None):
    # Component vertices are nan-separated regions
    oldVerts = self.region.verts
    oldRegionImg = self.region.image

    oldSelfImg = self.image
    oldSer = self.compSer

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
    if self.region.verts != centeredVerts:
      self.region.updateFromVertices(centeredVerts)
      yield
    else:
      return
    if self.compSer.loc[REQD_TBL_FIELDS.INST_ID] != oldSer.loc[REQD_TBL_FIELDS.INST_ID]:
      self.updateAll(oldSelfImg, oldSer, isAlreadyTrimmed=True)
    self.region.updateFromVertices(oldVerts, oldRegionImg)


  def saveNewVerts(self, overrideVerts: FRComplexVertices=None):
    # Add in offset from main image to FRVertexRegion vertices
    if overrideVerts is not None:
      self.compSer.loc[REQD_TBL_FIELDS.VERTICES] = overrideVerts
      return
    newVerts = self.region.verts.copy()
    for vertList in newVerts:
      vertList += self.bbox[0,:]
    self.compSer.loc[REQD_TBL_FIELDS.VERTICES] = newVerts