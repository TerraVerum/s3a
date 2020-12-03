from typing import Union, Optional, Collection, Sequence

import cv2 as cv
import numpy as np
import pandas as pd
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from pyqtgraph.graphicsItems.ViewBox.ViewBoxMenu import ViewBoxMenu
from skimage.io import imread

from s3a import FR_SINGLETON
from s3a.constants import REQD_TBL_FIELDS, FR_CONSTS as FRC
from s3a.generalutils import getClippedBbox, dynamicDocstring, frParamToPgParamDict
from s3a.structures import FRParam, XYVertices, ComplexXYVertices, FilePath
from s3a.structures import NChanImg
from .buttons import DrawOpts, ButtonCollection
from .clickables import RightPanViewBox
from .regions import RegionCopierPlot
from ..parameditors import ParamEditor, TableFieldPlugin

__all__ = ['MainImage', 'FocusedImage', 'EditableImgBase']

from s3a.controls.drawctrl import RoiCollection
from ..graphicsutils import menuFromEditorActions

Signal = QtCore.Signal
QCursor = QtGui.QCursor

@FR_SINGLETON.registerGroup(FRC.CLS_IMG_AREA)
class EditableImgBase(pg.PlotWidget):
  sigMousePosChanged = Signal(object, object)
  """
  XYVertices() coords, [[[img pixel]]] np.ndarray. If the mouse pos is outside
  image bounds, the second param will be *None*.
  """

  @classmethod
  def __initEditorParams__(cls):
    cls.compCropMargin, cls.treatMarginAsPct = FR_SINGLETON.generalProps.registerProps(
      cls, [FRC.PROP_CROP_MARGIN_VAL, FRC.PROP_TREAT_MARGIN_AS_PCT])
    cls.showGuiBtns = FR_SINGLETON.generalProps.registerProp(
      cls, FRC.PROP_SHOW_GUI_TOOL_BTNS, asProperty=False
    )

  def __init__(self, parent=None, drawShapes: Collection[FRParam]=(),
               drawActions: Collection[FRParam]=(),**kargs):
    super().__init__(parent, viewBox=RightPanViewBox(), **kargs)
    vb = self.getViewBox()
    self.menu: QtWidgets.QMenu = vb.menu
    self.oldVbMenu: ViewBoxMenu = vb.menu
    # Disable default menus
    self.plotItem.ctrlMenu = None
    self.sceneObj.contextMenu = None

    self.setAspectLocked(True)
    vb.invertY()
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
    self.regionCopier = RegionCopierPlot(self)

    self.drawAction: FRParam = FRC.DRAW_ACT_PAN
    self.shapeCollection = RoiCollection(drawShapes, self)
    self.shapeCollection.sigShapeFinished.connect(self.handleShapeFinished)

    # Make sure panning is allowed before creating draw widget
    if FRC.DRAW_ACT_PAN not in drawActions:
      drawActions += (FRC.DRAW_ACT_PAN,)

    def shapeAssignment(newShapeParam: FRParam):
      self.shapeCollection.curShapeParam = newShapeParam
    self.drawShapeGrp = ButtonCollection(self, 'Shapes', drawShapes, shapeAssignment)

    def actionAssignment(newActionParam: FRParam):
      self.drawAction = newActionParam
      if self.regionCopier.active:
        self.regionCopier.erase()
    self.drawActGrp = ButtonCollection(self, 'Actions', drawActions, actionAssignment)

    self.drawOptsWidget = DrawOpts(self.drawShapeGrp, self.drawActGrp, self)

    # Don't create shortcuts since this will be done by the tool editor
    self.toolsGrp = ButtonCollection.fromToolsEditors([], self)
    self.showGuiBtns.sigValueChanged.connect(lambda _p, val: self.toolsGrp.setVisible(val))

    # Initialize draw shape/action buttons
    self.drawActGrp.callFuncByParam(self.drawAction)
    self.drawShapeGrp.callFuncByParam(self.shapeCollection.curShapeParam)

  def maybeBuildRoi(self, ev: QtGui.QMouseEvent):
    if (QtCore.Qt.LeftButton not in [ev.buttons(), ev.button()]
        or self.drawAction == FRC.DRAW_ACT_PAN
        or self.regionCopier.active):
      return
    self.shapeCollection.buildRoi(ev, self.imgItem)
    ev.accept()


  def setMenuFromEditors(self, editors: Sequence[ParamEditor]):
    vb: pg.ViewBox = self.getViewBox()
    menu = menuFromEditorActions(editors)
    menu.insertAction(menu.actions()[0], self.oldVbMenu.viewAll)
    vb.menu = menu
    self.menu = menu

  def switchBtnMode(self, newMode: FRParam):
    # EAFP
    try:
      self.drawActGrp.callFuncByParam(newMode)
    except KeyError:
      self.drawShapeGrp.callFuncByParam(newMode)

  @property
  def image(self) -> Optional[NChanImg]:
    return self.imgItem.image

  def handleShapeFinished(self, roiVerts: XYVertices) -> Optional[np.ndarray]:
    """
    Overloaded in child classes to process new regions
    """

  def mousePressEvent(self, ev: QtGui.QMouseEvent):
    self.maybeBuildRoi(ev)
    super().mousePressEvent(ev)

  def mouseDoubleClickEvent(self, ev: QtGui.QMouseEvent):
    self.maybeBuildRoi(ev)
    super().mouseDoubleClickEvent(ev)

  def mouseMoveEvent(self, ev: QtGui.QMouseEvent):
    """
    Mouse move behavior is contingent on which shape is currently selected,
    unless we are panning
    """
    super().mouseMoveEvent(ev)
    self.maybeBuildRoi(ev)
    posRelToImage = self.imgItem.mapFromScene(ev.pos())
    pxY = int(posRelToImage.y())
    pxX = int(posRelToImage.x())
    pxColor = None
    if (self.imgItem.image is not None
        and 0 < pxX < self.imgItem.image.shape[1]
        and 0 < pxY < self.imgItem.image.shape[0]):
      pxColor = self.imgItem.image[pxY, pxX]
      if pxColor.ndim == 0:
        pxColor = np.array([pxColor])
    pos = XYVertices([pxX, pxY])
    self.sigMousePosChanged.emit(pos, pxColor)

  def mouseReleaseEvent(self, ev: QtGui.QMouseEvent):
    self.maybeBuildRoi(ev)
    super().mouseReleaseEvent(ev)

  def clearCurRoi(self):
    """Clears the current ROI shape"""
    self.shapeCollection.clearAllRois()

@FR_SINGLETON.registerGroup(FRC.CLS_MAIN_IMG_AREA)
class MainImage(EditableImgBase):
  sigCompsCreated = Signal(object) # pd.DataFrame
  # Hooked up during __init__
  sigSelectionBoundsMade = Signal(object) # XYVertices

  @classmethod
  def __initEditorParams__(cls):
    super().__initEditorParams__()
    (cls.minCompSize, cls.onlyGrowViewbox) = FR_SINGLETON.generalProps.registerProps(
      cls, [FRC.PROP_MIN_COMP_SZ, FRC.PROP_ONLY_GROW_MAIN_VB])
    (cls.gridClr, cls.gridWidth, cls.showGrid) = FR_SINGLETON.colorScheme.registerProps(
      cls, [FRC.SCHEME_GRID_CLR, FRC.SCHEME_GRID_LINE_WIDTH, FRC.SCHEME_SHOW_GRID]
    )

  def __init__(self, parent=None, imgSrc=None, **kargs):
    allowedShapes = (FRC.DRAW_SHAPE_RECT, FRC.DRAW_SHAPE_POLY, FRC.DRAW_SHAPE_ELLIPSE)
    allowedActions = (FRC.DRAW_ACT_SELECT,FRC.DRAW_ACT_ADD)
    super().__init__(parent, drawShapes=allowedShapes,
                     drawActions=allowedActions, **kargs)
    self._initGrid()
    FR_SINGLETON.colorScheme.sigParamStateUpdated.connect(
      lambda: self.updateGridColor()
    )
    # plt: pg.PlotItem = self.plotItem
    # # Make sure grid lines are on top of image
    # for axDict in plt.axes.values():
    #   ax = axDict['item']
    #   ax.setZValue(500)
    # -----
    # Image Item
    # -----
    self.setImage(imgSrc)
    self.compFromLastProcResult: Optional[pd.DataFrame] = None
    self.lastProcVerts: Optional[XYVertices] = None

    self.switchBtnMode(FRC.DRAW_ACT_ADD)

  def _initGrid(self):
    pi: pg.PlotItem = self.plotItem
    pi.showGrid(alpha=1.0)
    axs = [pi.getAxis(ax) for ax in ['top', 'bottom', 'left', 'right']]
    for ax in axs: # type: pg.AxisItem
      ax.setZValue(1e9)
      # ax.setTickSpacing(5, 1)
      for evFn in 'mouseMoveEvent', 'mousePressEvent', 'mouseReleaseEvent', \
                  'mouseDoubleClickEvent', 'mouseDragEvent', 'wheelEvent':
        newEvFn = lambda ev: ev.ignore()
        setattr(ax, evFn, newEvFn)

  def updateGridColor(self):
    pi: pg.PlotItem = self.plotItem
    pi.showGrid(self.showGrid, self.showGrid)
    axs = [pi.getAxis(ax) for ax in ['top', 'bottom', 'left', 'right']]
    newPen = pg.mkPen(width=self.gridWidth, color=self.gridClr)
    for ax in axs:
      ax.setPen(newPen)


  def handleShapeFinished(self, roiVerts: XYVertices) -> Optional[np.ndarray]:
    if self.regionCopier.active or self.shapeCollection.locked: return
    if self.drawAction in [FRC.DRAW_ACT_SELECT] and roiVerts.connected:
      # Selection
      self.sigSelectionBoundsMade.emit(roiVerts)
    elif self.drawAction == FRC.DRAW_ACT_ADD:
      # Component modification subject to processor
      # For now assume a single point indicates foreground where multiple indicate
      # background selection
      # False positive from type checker
      verts = np.clip(roiVerts.astype(int), 0, self.image.shape[:2][::-1])
      self.compFromLastProcResult = FR_SINGLETON.tableData.makeCompDf()

      if cv.contourArea(verts) < self.minCompSize:
        return

      # noinspection PyTypeChecker
      verts = ComplexXYVertices([verts])
      newComps = FR_SINGLETON.tableData.makeCompDf()
      newComps[REQD_TBL_FIELDS.VERTICES] = [verts]
      self.compFromLastProcResult = newComps
      self.lastProcVerts = verts
      self.sigCompsCreated.emit(newComps)

  def mouseReleaseEvent(self, ev: QtGui.QMouseEvent):
    super().mouseReleaseEvent(ev)
    if self.image is None: return
    pos = self.imgItem.mapFromScene(ev.pos())
    xx, yy, = pos.x(), pos.y()
    if self.drawAction == FRC.DRAW_ACT_PAN and not self.regionCopier.active:
      # Simulate a click-wide boundary selection so points can be selected in pan mode
      squareCorners = XYVertices([[xx, yy], [xx, yy]], dtype=float)
      self.sigSelectionBoundsMade.emit(squareCorners)
    self.shapeCollection.removeLock(self)

  def mouseDoubleClickEvent(self, ev: QtGui.QMouseEvent):
    if self.regionCopier.active:
      self.regionCopier.sigCopyStopped.emit()
      ev.accept()
      self.shapeCollection.addLock(self)
    super().mouseDoubleClickEvent(ev)

  def switchBtnMode(self, newMode: FRParam):
    super().switchBtnMode(newMode)

  def setImage(self, imgSrc: Union[FilePath, np.ndarray]=None):
    """
    Allows the user to change the main image either from a filepath or array data
    """
    if isinstance(imgSrc, FilePath.__args__):
      # TODO: Handle alpha channel images. For now, discard that data
      imgSrc = imread(imgSrc)
      if imgSrc.ndim == 3:
        # Alpha channels cause unexpected results for most image processors. Avoid this
        # by chopping it off until otherwise needed
        imgSrc = imgSrc[:,:,0:3]

    if imgSrc is None:
      self.imgItem.clear()
    else:
      self.imgItem.setImage(imgSrc)

  @dynamicDocstring(superDoc=EditableImgBase.clearCurRoi.__doc__)
  def clearCurRoi(self):
    """{superDoc}"""
    super().clearCurRoi()
    self.regionCopier.erase()

@FR_SINGLETON.registerGroup(FRC.CLS_FOCUSED_IMG_AREA)
class FocusedImage(EditableImgBase):
  sigPluginChanged = Signal()
  sigUpdatedAll = Signal(object, object)
  """Main image, new component. Emitted during `updateAll()`"""
  sigShapeFinished = Signal(object)
  """XYVerts from roi, re-thrown from self.shapeCollection so plugins can tie into it"""

  @classmethod
  def __initEditorParams__(cls):
    super().__initEditorParams__()
    cls.toolsEditor = ParamEditor.buildClsToolsEditor(cls)


  def __init__(self, parent=None, **kargs):
    allowableShapes = (
      FRC.DRAW_SHAPE_RECT, FRC.DRAW_SHAPE_POLY, FRC.DRAW_SHAPE_PAINT, FRC.DRAW_SHAPE_ELLIPSE
    )
    allowableActions = (
      FRC.DRAW_ACT_ADD, FRC.DRAW_ACT_REM
    )
    super().__init__(parent, allowableShapes, allowableActions, **kargs)

    self.compSer: pd.Series = FR_SINGLETON.tableData.makeCompSer()
    self.bbox = np.zeros((2, 2), dtype='int32')

    self.currentPlugin: Optional[TableFieldPlugin] = None

    self.switchBtnMode(FRC.DRAW_ACT_ADD)
    self.switchBtnMode(FRC.DRAW_SHAPE_PAINT)

  def handleShapeFinished(self, roiVerts: XYVertices) -> Optional[np.ndarray]:
    if self.drawAction == FRC.DRAW_ACT_PAN:
      return
    self.sigShapeFinished.emit(roiVerts)

  @FR_SINGLETON.actionStack.undoable('Modify Focused Component')
  def updateAll(self, mainImg: NChanImg=None, newComp:Optional[pd.Series]=None,
                _isAlreadyTrimmed=False):
    """
    Updates focused image and component from provided information. Useful for creating
    a 'zoomed-in' view that allows much faster processing than applying image processing
    algorithms to the entire image each iteration.
    :param mainImg: Image from the main view
    :param newComp: New component to edit using various plugins (See :class:`TableFieldPlugin`)
    :param _isAlreadyTrimmed: Used internally during undo. Generally shouldn't be set by the
      user
    """
    oldImg = self.image
    if oldImg is not None:
      oldImg = oldImg.copy()
    oldComp = self.compSer

    if mainImg is None:
      self.imgItem.clear()
      self.shapeCollection.clearAllRois()
      self.compSer = FR_SINGLETON.tableData.makeCompSer()
    else:
      newVerts: ComplexXYVertices = newComp[REQD_TBL_FIELDS.VERTICES]
      # Since values INSIDE the dataframe are reset instead of modified, there is no
      # need to go through the trouble of deep copying
      self.compSer = newComp.copy(deep=False)

      # Propagate all resultant changes
      if not _isAlreadyTrimmed:
        self._updateBbox(mainImg.shape, newVerts)
        bboxToUse = self.bbox
      else:
        bboxToUse = XYVertices([[0,0], mainImg.shape[:2]])
        self.bbox = bboxToUse
      slices = [slice(bboxToUse[0,1], bboxToUse[1,1]),
                slice(bboxToUse[0,0], bboxToUse[1,0])]
      if mainImg.ndim > 2:
        slices.append(...)
      newCompImg = mainImg[tuple(slices)]
      if newCompImg.size == 0:
        # Empty slice
        self.imgItem.clear()
      else:
        self.imgItem.setImage(newCompImg)
        QtCore.QTimer.singleShot(0, self.autoRange)
    self.sigUpdatedAll.emit(mainImg, newComp)
    yield
    self.updateAll(oldImg, oldComp, True)

  def _updateBbox(self, mainImgShape, newVerts: ComplexXYVertices):
    concatVerts = newVerts.stack()
    # Ignore NAN entries during computation
    bbox = np.vstack([concatVerts.min(0),
                      concatVerts.max(0)])
    # Account for margins
    padVal = self.compCropMargin
    if self.treatMarginAsPct:
      padVal = max((bbox[1,:] - bbox[0,:])*self.compCropMargin/2/100)
    self.bbox = getClippedBbox(mainImgShape, bbox, int(padVal))

  def changeCurrentPlugin(self, newPlugin: TableFieldPlugin, forceActivate=True):
    if newPlugin is self.currentPlugin:
      return
    if self.currentPlugin is not None:
      self.currentPlugin.active = False
    newEditors = [self.toolsEditor]
    if newPlugin is not None:
      newEditors.append(newPlugin.toolsEditor)
    self.menu = menuFromEditorActions(newEditors, menuParent=self)
    self.getViewBox().menu = self.menu
    self.currentPlugin = newPlugin
    if forceActivate and newPlugin is not None and not newPlugin.active:
      newPlugin.active = True
    self.sigPluginChanged.emit()

  @property
  def classIdx(self):
    """
    Helper property to determine the index into all component classes of the current
    component. E.g. if all possible classes are ['resistor', 'capacitor'] and the current
    focused component is 'resistor', this function will return 0.
    """
    return FR_SINGLETON.tableData.compClasses.index(self.compSer[REQD_TBL_FIELDS.COMP_CLASS])