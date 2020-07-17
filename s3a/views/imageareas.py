from typing import Union, Optional, Tuple, Collection, Dict

import cv2 as cv
import numpy as np
import pandas as pd
import pyqtgraph as pg
from pyqtgraph import BusyCursor
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from skimage.io import imread

from s3a import FR_SINGLETON
from s3a.generalutils import getClippedBbox, frPascalCaseToTitle, cornersToFullBoundary
from s3a.projectvars import REQD_TBL_FIELDS, FR_CONSTS, MENU_OPTS_DIR
from s3a.structures import FRParam, FRVertices, FRComplexVertices, FilePath
from s3a.structures import NChanImg
from .clickables import FRRightPanViewBox
from .drawopts import FRDrawOpts, FRButtonCollection, btnCallable
from .parameditors import FRParamEditor, FRParamEditorDockGrouping
from .procwrapper import FRImgProcWrapper
from .regions import FRVertexDefinedImg, FRMouseFollowingRegionPlot

__all__ = ['FRMainImage', 'FRFocusedImage', 'FREditableImgBase']

from s3a.controls.drawctrl import FRRoiCollection

Signal = QtCore.Signal
QCursor = QtGui.QCursor

# @FR_SINGLETON.registerGroup(FR_CONSTS.CLS_IMG_AREA)
class FREditableImgBase(pg.PlotWidget):
  sigMousePosChanged = QtCore.Signal(object, object)
  """
  FRVertices() coords, [[[img pixel]]] np.ndarray. If the mouse pos is outside
  image bounds, the second param will be *None*.
  """

  @classmethod
  def __initEditorParams__(cls):
    groupName = frPascalCaseToTitle(cls.__name__)
    lowerGroupName = groupName.lower()
    cls.toolsDir = MENU_OPTS_DIR / lowerGroupName
    cls.toolsEditor = FRParamEditor(
      saveDir=cls.toolsDir, fileType=lowerGroupName.replace(' ', '') + 'tools',
      name=groupName + ' Tools', registerCls=cls, useNewInit=False
    )
    cls.procCollection = FR_SINGLETON.algParamMgr.createProcessorForClass(
      cls, editorName=groupName + ' Processor'
    )
    dockGroup = FRParamEditorDockGrouping(
      [cls.toolsEditor, cls.procCollection], dockName=groupName
    )
    FR_SINGLETON.addDocks(dockGroup)

    cls.compCropMargin, cls.treatMarginAsPct = FR_SINGLETON.generalProps.registerProps(
      cls, [FR_CONSTS.PROP_CROP_MARGIN_VAL, FR_CONSTS.PROP_TREAT_MARGIN_AS_PCT])

  def __init__(self, parent=None, drawShapes: Collection[FRParam]=(),
               drawActions: Collection[FRParam]=(),
               toolParams: Collection[FRParam]=(), toolFns: Collection[btnCallable]=(),
               **kargs):
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

    # -----
    # DRAWING OPTIONS
    # -----
    self.regionCopier = FRMouseFollowingRegionPlot(self)

    self.drawAction: FRParam = FR_CONSTS.DRAW_ACT_PAN
    self.shapeCollection = FRRoiCollection(drawShapes, self)
    self.shapeCollection.sigShapeFinished.connect(self.handleShapeFinished)

    # Make sure panning is allowed before creating draw widget
    if FR_CONSTS.DRAW_ACT_PAN not in drawActions:
      drawActions += (FR_CONSTS.DRAW_ACT_PAN,)

    def shapeAssignment(newShapeParam: FRParam):
      self.shapeCollection.curShapeParam = newShapeParam
    self.drawShapeGrp = FRButtonCollection(self, "Shapes", drawShapes, shapeAssignment)
    
    def actionAssignment(newActionParam: FRParam):
      self.drawAction = newActionParam
      if self.regionCopier.active:
        self.regionCopier.erase()
    self.drawActGrp = FRButtonCollection(self, "Actions", drawActions, actionAssignment)

    self.drawOptsWidget = FRDrawOpts(self.drawShapeGrp, self.drawActGrp, self)

    # self.toolsGrp = FRButtonCollection(self, btnParams=toolParams, btnTriggerFns=toolFns,
    #                                    exclusive=False, checkable=False)

    # self.drawOptsWidget = FRDrawOpts(parent, drawShapes, drawActions)
    # btnGroups = [self.drawOptsWidget.shapeBtnGroup, self.drawOptsWidget.actionBtnGroup]
    # for group in btnGroups:
    #   group.buttonToggled.connect(self._handleBtnToggle)
    # self.drawOptsWidget.selectOpt(self.drawAction)
    # self.drawOptsWidget.selectOpt(self.shapeCollection.curShapeParam)

    # Initialize draw shape/action buttons
    self.drawActGrp.callFuncByParam(self.drawAction)
    self.drawShapeGrp.callFuncByParam(self.shapeCollection.curShapeParam)

  def switchBtnMode(self, newMode: FRParam):
    # EAFP
    try:
      self.drawActGrp.callFuncByParam(newMode)
    except KeyError:
      self.drawShapeGrp.callFuncByParam(newMode)

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

  def mousePressEvent(self, ev: QtGui.QMouseEvent):
    super().mousePressEvent(ev)
    if (ev.buttons() == QtCore.Qt.LeftButton
        and not self.regionCopier.active
        and self.drawAction != FR_CONSTS.DRAW_ACT_PAN):
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

    posRelToImage = self.imgItem.mapFromScene(ev.pos())
    pxY = int(posRelToImage.y())
    pxX = int(posRelToImage.x())
    pxColor = None
    if (self.imgItem.image is not None
        and 0 < pxX < self.imgItem.image.shape[1]
        and 0 < pxY < self.imgItem.image.shape[0]):
      pxColor = self.imgItem.image[pxY, pxX]
      # pos = ev.pos()
    pos = FRVertices([pxX, pxY])
    self.sigMousePosChanged.emit(pos, pxColor)

  def mouseReleaseEvent(self, ev: QtGui.QMouseEvent):
    """
    Perform a processing method depending on what the current draw action is

    :return: Whether the mouse release completes the current ROI
    """
    super().mouseReleaseEvent(ev)
    if (self.drawAction != FR_CONSTS.DRAW_ACT_PAN
        and ev.button() == QtCore.Qt.LeftButton and not self.regionCopier.active):
      self.shapeCollection.buildRoi(ev, self.imgItem)

  def clearCurDrawShape(self):
    self.shapeCollection.clearAllRois()

@FR_SINGLETON.registerGroup(FR_CONSTS.CLS_MAIN_IMG_AREA)
class FRMainImage(FREditableImgBase):
  sigCompsCreated = Signal(object) # pd.DataFrame
  sigCompsUpdated = Signal(object) # pd.DataFrame
  sigCompsRemoved = Signal(object) # OneDArr
  # Hooked up during __init__
  sigSelectionBoundsMade = Signal(object) # FRVertices

  @classmethod
  def __initEditorParams__(cls):
    super().__initEditorParams__()
    (cls.mergeCompsAct, cls.splitCompsAct, cls.moveCompsAct, cls.copyCompsAct,
     cls.overrideCompVertsAct) = cls.toolsEditor.registerProps(
      cls, [FR_CONSTS.TOOL_MERGE_COMPS, FR_CONSTS.TOOL_SPLIT_COMPS,
            FR_CONSTS.TOOL_MOVE_REGIONS, FR_CONSTS.TOOL_COPY_REGIONS,
            FR_CONSTS.TOOL_OVERRIDE_VERTS_ACT], asProperty=False)
    (cls.multCompsOnCreate, cls.onlyGrowViewbox) = FR_SINGLETON.generalProps.registerProps(
      cls, [FR_CONSTS.PROP_MK_MULT_COMPS_ON_ADD, FR_CONSTS.PROP_ONLY_GROW_MAIN_VB])

  def __init__(self, parent=None, imgSrc=None, **kargs):
    allowedShapes = (FR_CONSTS.DRAW_SHAPE_RECT, FR_CONSTS.DRAW_SHAPE_POLY)
    allowedActions = (FR_CONSTS.DRAW_ACT_SELECT,FR_CONSTS.DRAW_ACT_ADD)
    super().__init__(parent, drawShapes=allowedShapes,
                     drawActions=allowedActions, **kargs)
    # -----
    # Image Item
    # -----
    self.setImage(imgSrc)
    self.compFromLastProcResult: Optional[pd.DataFrame] = None
    self.lastProcVerts: Optional[FRVertices] = None
    self.overrideCompVertsAct.sigActivated.connect(lambda: self.overrideLastProcResult())
    copier = self.regionCopier
    def startCopy():
      copier.inCopyMode = True
      copier.sigCopyStarted.emit()
    self.copyCompsAct.sigActivated.connect(startCopy)

    def startMove():
      copier.inCopyMode = False
      copier.sigCopyStarted.emit()
    self.moveCompsAct.sigActivated.connect(startMove)

    self.switchBtnMode(FR_CONSTS.DRAW_ACT_ADD)

  def handleShapeFinished(self, roiVerts: FRVertices) -> Optional[np.ndarray]:
    if self.regionCopier.active: return
    if self.drawAction in [FR_CONSTS.DRAW_ACT_SELECT] and roiVerts.connected:
      # Selection
      self.sigSelectionBoundsMade.emit(roiVerts)
    elif self.drawAction == FR_CONSTS.DRAW_ACT_ADD:
      # Component modification subject to processor
      # For now assume a single point indicates foreground where multiple indicate
      # background selection
      verts = roiVerts.astype(int)

      globalEstimate = self.multCompsOnCreate or roiVerts.empty
      namePath = ['Basic Region Operations', 'Keep Largest Comp']
      oldState = None
      if globalEstimate:
        oldState = self.curProcessor.setStageEnabled(namePath, False)

      with BusyCursor():
        self.curProcessor.run(image=self.image, fgVerts=verts.copy())
      newVerts = self.curProcessor.resultAsVerts(not globalEstimate)

      # Reset keep_largest_comp if necessary
      if globalEstimate:
        self.curProcessor.setStageEnabled(namePath, oldState)

      # Discard entries with no real vertices
      newComps = FR_SINGLETON.tableData.makeCompDf(len(newVerts))
      newComps[REQD_TBL_FIELDS.VERTICES] = newVerts
      self.compFromLastProcResult = newComps
      self.lastProcVerts = verts
      if len(newComps) == 0:
        self.compFromLastProcResult = FR_SINGLETON.tableData.makeCompDf()
        return
      # TODO: Determine more robust solution for separated vertices. For now use largest component
      self.sigCompsCreated.emit(newComps)

  def mouseReleaseEvent(self, ev: QtGui.QMouseEvent):
    super().mouseReleaseEvent(ev)
    if self.image is None: return
    pos = self.imgItem.mapFromScene(ev.pos())
    xx, yy, = pos.x(), pos.y()
    pos = FRVertices([[xx, yy]])
    if self.drawAction == FR_CONSTS.DRAW_ACT_PAN and not self.regionCopier.active:
      # Simulate a click-wide boundary selection so points can be selected in pan mode
      squareCorners = FRVertices([[xx, yy], [xx, yy]], dtype=float)
      self.sigSelectionBoundsMade.emit(squareCorners)

  def mouseDoubleClickEvent(self, ev: QtGui.QMouseEvent):
    super().mouseDoubleClickEvent(ev)
    if self.regionCopier.active:
      self.regionCopier.sigCopyStopped.emit()

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

  def overrideLastProcResult(self):
    # Nest work function so undo isn't influenced by currently selected component
    @FR_SINGLETON.actionStack.undoable('Override Last Process Result')
    def doOverride(comps=self.compFromLastProcResult, verts=self.lastProcVerts):
      if comps is None: return
      # Trim verts to image boundaries if necessary
      vMin = verts.min(0)
      if np.any(vMin < 0) or np.any(verts.max(0) > self.image.shape[:2][::-1]):
        newVerts = cornersToFullBoundary(verts, stackResult=False)
      else:
        newVerts = FRComplexVertices([verts])

      newComp = comps.iloc[[0],:].copy()
      compId = newComp.index[0]
      newComp.at[compId, REQD_TBL_FIELDS.VERTICES] = newVerts
      self.sigCompsRemoved.emit(comps.loc[:, REQD_TBL_FIELDS.INST_ID])
      if compId == -1:
        self.sigCompsCreated.emit(newComp)
      else:
        self.sigCompsUpdated.emit(newComp)
      yield
      if compId == -1:
        self.sigCompsRemoved.emit(newComp.index)
      else:
        self.sigCompsUpdated.emit(comps)
    doOverride()

  def clearCurDrawShape(self):
    super().clearCurDrawShape()
    self.regionCopier.erase()

@FR_SINGLETON.registerGroup(FR_CONSTS.CLS_FOCUSED_IMG_AREA)
class FRFocusedImage(FREditableImgBase):

  @classmethod
  def __initEditorParams__(cls):
    super().__initEditorParams__()
    (cls.resetRegionAct, cls.fillRegionAct,
     cls.clearRegionAct, cls.acceptRegionAct) = cls.toolsEditor.registerProps(
      cls, [FR_CONSTS.TOOL_RESET_FOC_REGION, FR_CONSTS.TOOL_FILL_FOC_REGION,
            FR_CONSTS.TOOL_CLEAR_FOC_REGION, FR_CONSTS.TOOL_ACCEPT_FOC_REGION],
      asProperty=False)


  def __init__(self, parent=None, **kargs):
    allowableShapes = (
      FR_CONSTS.DRAW_SHAPE_RECT, FR_CONSTS.DRAW_SHAPE_POLY, FR_CONSTS.DRAW_SHAPE_PAINT
    )
    allowableActions = (
      FR_CONSTS.DRAW_ACT_ADD, FR_CONSTS.DRAW_ACT_REM
    )
    super().__init__(parent, allowableShapes, allowableActions, **kargs)
    self.clearRegionAct.sigActivated.connect(lambda: self.updateRegionFromVerts(None))
    def fillAct():
      if self.image is None: return
      filled = np.ones(self.image.shape[:2], bool)
      self.region.updateFromMask(filled)
    self.fillRegionAct.sigActivated.connect(fillAct)
    self.resetRegionAct.sigActivated.connect(
      lambda: self.updateRegionFromVerts(self.compSer[REQD_TBL_FIELDS.VERTICES]))
    self.region = FRVertexDefinedImg()

    self.addItem(self.region)

    self.compSer: pd.Series = FR_SINGLETON.tableData.makeCompSer()

    self.bbox = np.zeros((2, 2), dtype='int32')

    self.firstRun = True

    self.switchBtnMode(FR_CONSTS.DRAW_ACT_ADD)
    self.switchBtnMode(FR_CONSTS.DRAW_SHAPE_PAINT)
    # Disable local cropping on primitive grab cut by default
    self.procCollection.nameToProcMapping['Primitive Grab Cut'].setStageEnabled(['Crop to Local Area'], False)

  def resetImage(self):
    self.updateAll(None)

  def handleShapeFinished(self, roiVerts: FRVertices) -> Optional[np.ndarray]:
    if self.drawAction == FR_CONSTS.DRAW_ACT_PAN:
      return

    # Component modification subject to processor
    # For now assume a single point indicates foreground where multiple indicate
    # background selection
    roiVerts = roiVerts.astype(int)
    vertsDict = {}
    if self.drawAction == FR_CONSTS.DRAW_ACT_ADD:
      vertsDict['fgVerts'] = roiVerts
    elif self.drawAction == FR_CONSTS.DRAW_ACT_REM:
      vertsDict['bgVerts'] = roiVerts

    compMask = self.region.embedMaskInImg(self.image.shape[:2])
    newMask = self.curProcessor.run(image=self.image, prevCompMask=compMask, **vertsDict,
                                    firstRun=self.firstRun)
    self.firstRun = False
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
      self.compSer = FR_SINGLETON.tableData.makeCompSer()
    else:
      newVerts: FRComplexVertices = newComp[REQD_TBL_FIELDS.VERTICES]
      # Since values INSIDE the dataframe are reset instead of modified, there is no
      # need to go through the trouble of deep copying
      self.compSer = newComp.copy(deep=False)
      # Remove all other fields so they don't overwrite main component fields on update
      keepCols = [REQD_TBL_FIELDS.INST_ID, REQD_TBL_FIELDS.VERTICES]
      self.compSer = self.compSer[keepCols]

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
      self.firstRun = True
    yield
    self.updateAll(oldImg, oldComp, True)

  def updateBbox(self, mainImgShape, newVerts: FRComplexVertices):
    concatVerts = newVerts.stack()
    # Ignore NAN entries during computation
    bbox = np.vstack([concatVerts.min(0),
                      concatVerts.max(0)])
    # Account for margins
    padVal = self.compCropMargin
    if self.treatMarginAsPct:
      padVal = max((bbox[1,:] - bbox[0,:])*self.compCropMargin/2/100)
    self.bbox = getClippedBbox(mainImgShape, bbox, int(padVal))

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
    """
    Updates the current focused region using the new provided vertices
    :param newVerts: Verts to use.If *None*, the image will be totally reset and the component
      will be removed. Otherwise, the provided value will be used.
    :param offset: Offset of newVerts relative to main image coordinates
    """
    if self.image is None:
      return
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
    if (self.compSer.loc[REQD_TBL_FIELDS.INST_ID] != oldSer.loc[REQD_TBL_FIELDS.INST_ID]
        or self.image is None):
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