from functools import wraps
from typing import Union, Optional, Collection, Callable, Any, Sequence, List

import numpy as np
import pandas as pd
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from pyqtgraph.graphicsItems.ViewBox.ViewBoxMenu import ViewBoxMenu
from skimage.io import imread

from s3a import FR_SINGLETON
from s3a.constants import REQD_TBL_FIELDS as RTF, PRJ_CONSTS as CNST
from s3a.controls.drawctrl import RoiCollection
from s3a.generalutils import getCroppedImg, coerceDfTypes, serAsFrame
from s3a.models.editorbase import RunOpts
from s3a.structures import FRParam, XYVertices, ComplexXYVertices, FilePath, \
  CompositionMixin
from s3a.structures import NChanImg
from .buttons import ButtonCollection
from .clickables import RightPanViewBox
from .regions import RegionCopierPlot
from .rois import SHAPE_ROI_MAPPING
from ..graphicsutils import menuFromEditorActions
from ..parameditors import ParamEditor, EditorPropsMixin
from ..plugins.base import TableFieldPlugin

__all__ = ['MainImage']

Signal = QtCore.Signal
QCursor = QtGui.QCursor

DrawActFn = Union[Callable[[XYVertices, FRParam], Any], Callable[[XYVertices], Any]]

class MainImage(CompositionMixin, EditorPropsMixin, pg.PlotWidget):
  sigShapeFinished = Signal(object, object)
  """
  (XYVertices, FRParam) emitted when a shape is finished
  - XYVerts from roi, re-thrown from self.shapeCollection
  - Current draw action
  """

  sigMousePosChanged = Signal(object, object)
  """
  XYVertices() coords, [[[img pixel]]] np.ndarray. If the mouse pos is outside
  image bounds, the second param will be *None*.
  """

  sigUpdatedFocusedComp = Signal(object)
  """pd.Series, newly focused component"""

  @classmethod
  def __initEditorParams__(cls):
    cls.toolsEditor = ParamEditor.buildClsToolsEditor(cls)

    cls.compCropMargin, cls.treatMarginAsPct = FR_SINGLETON.generalProps.registerProps(
      [CNST.PROP_CROP_MARGIN_VAL, CNST.PROP_TREAT_MARGIN_AS_PCT])

    (cls.minCompSize, cls.onlyGrowViewbox) = FR_SINGLETON.generalProps.registerProps(
      [CNST.PROP_MIN_COMP_SZ, CNST.PROP_ONLY_GROW_MAIN_VB])

  def __init__(self, parent=None, drawShapes: Collection[FRParam]=None,
               drawActions: Collection[FRParam]=(),
               imgSrc: Union[FilePath, NChanImg]=None,
               toolbar: QtWidgets.QToolBar=None,
               **kargs):
    super().__init__(parent, viewBox=RightPanViewBox(), **kargs)

    if drawShapes is None:
      drawShapes = SHAPE_ROI_MAPPING.keys()

    vb = self.getViewBox()
    self.menu: QtWidgets.QMenu = vb.menu
    self.oldVbMenu: ViewBoxMenu = vb.menu
    # Disable default menus
    self.plotItem.ctrlMenu = None
    self.sceneObj.contextMenu = None

    self.setAspectLocked(True)
    vb.invertY()
    self.setMouseEnabled(True)
    self._initGrid()
    FR_SINGLETON.colorScheme.registerFunc(self.updateGridScheme, runOpts=RunOpts.ON_CHANGED)

    self.toolbar = toolbar

    # -----
    # IMAGE
    # -----
    self.imgItem = self.exposes(pg.ImageItem())
    self.imgItem.setZValue(-100)
    self.addItem(self.imgItem)
    self.toolsEditor.registerFunc(lambda: self.oldVbMenu.viewAll.trigger(),
                                         name='Reset Zoom', btnOpts={'guibtn':False})
    # -----
    # FOCUSED COMPONENT INFORMATION
    # -----
    self.compSer: pd.Series = FR_SINGLETON.tableData.makeCompSer()
    self._focusedTools: List[ParamEditor] = []
    """
    List of all toolsEditor that allow actions to be performed on the currently focused components
    """
    # -----
    # DRAWING OPTIONS
    # -----
    self.regionCopier = RegionCopierPlot(self)

    self.drawAction: FRParam = CNST.DRAW_ACT_PAN
    self.shapeCollection = RoiCollection(drawShapes, self)
    self.shapeCollection.sigShapeFinished.connect(
      lambda roiVerts: self.sigShapeFinished.emit(roiVerts, self.drawAction)
    )

    # Make sure panning is allowed before creating draw widget
    if CNST.DRAW_ACT_PAN not in drawActions:
      drawActions += (CNST.DRAW_ACT_PAN,)

    self.drawShapeGrp = ButtonCollection(self, 'Shapes', drawShapes, self.shapeAssignment)
    self.drawActGrp = ButtonCollection(self, 'Actions', drawActions, self.actionAssignment)

    # Initialize draw shape/action buttons
    self.drawActGrp.callFuncByParam(self.drawAction)
    self.drawShapeGrp.callFuncByParam(self.shapeCollection.curShapeParam)

    # Initialize image
    if imgSrc is not None:
      self.setImage(imgSrc)

    if toolbar is not None:
      toolbar.addWidget(self.drawShapeGrp)
      toolbar.addWidget(self.drawActGrp)
      self.addTools(self.toolsEditor)

  @property
  def compSer_asFrame(self):
      return coerceDfTypes(serAsFrame(self.compSer))

  def shapeAssignment(self, newShapeParam: FRParam):
    self.shapeCollection.curShapeParam = newShapeParam
    if self.regionCopier.active:
      self.regionCopier.erase()

  def actionAssignment(self, newActionParam: FRParam):
    self.drawAction = newActionParam
    if self.regionCopier.active:
      self.regionCopier.erase()

  def maybeBuildRoi(self, ev: QtGui.QMouseEvent):
    if (QtCore.Qt.LeftButton not in [ev.buttons(), ev.button()]
        or self.drawAction == CNST.DRAW_ACT_PAN
        or self.regionCopier.active):
      return
    self.shapeCollection.buildRoi(ev, self.imgItem)
    ev.accept()

  def switchBtnMode(self, newMode: FRParam):
    # EAFP
    try:
      self.drawActGrp.callFuncByParam(newMode)
    except KeyError:
      self.drawShapeGrp.callFuncByParam(newMode)

  def mousePressEvent(self, ev: QtGui.QMouseEvent):
    self.maybeBuildRoi(ev)
    super().mousePressEvent(ev)

  def mouseDoubleClickEvent(self, ev: QtGui.QMouseEvent):
    if self.regionCopier.active:
      self.regionCopier.sigCopyStopped.emit()
      ev.accept()
      self.shapeCollection.addLock(self)
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
    # Typical reaction is to right-click to cancel an roi
    if self.image is not None and QtCore.Qt.RightButton not in [ev.button(), ev.buttons()]:
      self.maybeBuildRoi(ev)
    super().mouseReleaseEvent(ev)
    # Special case: Panning
    if self.drawAction == CNST.DRAW_ACT_PAN and not self.regionCopier.active:
      pos = self.imgItem.mapFromScene(ev.pos())
      xx, yy, = pos.x(), pos.y()
      # Simulate a click-wide boundary selection so points can be registered in pan mode
      pt = XYVertices([[xx, yy]], dtype=float)
      self.shapeCollection.sigShapeFinished.emit(pt)
    self.shapeCollection.removeLock(self)

  def clearCurRoi(self):
    """Clears the current ROI shape"""
    self.shapeCollection.clearAllRois()
    self.regionCopier.erase()

  def widgetContainer(self, parent=None):
    """
    Though this is a PlotWidget class, it has a lot of widget children (toolsEditor group, buttons) that are
    not visible when spawning the widget. This is a convenience method that creates a new, outer widget
    from all teh graphical elements of an EditableImage.
    """
    wid = QtWidgets.QWidget(parent)
    layout = QtWidgets.QVBoxLayout()
    wid.setLayout(layout)

    layout.addWidget(self.drawActGrp)
    layout.addWidget(self.drawShapeGrp)
    layout.addWidget(self)
    return wid

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

  def updateGridScheme(self, showGrid=False, gridWidth=1, gridColor='fff'):
    """
    :param showGrid:
    :param gridWidth:
    :param gridColor:
      pType: color
    """
    pi: pg.PlotItem = self.plotItem
    pi.showGrid(showGrid, showGrid)
    axs = [pi.getAxis(ax) for ax in ['top', 'bottom', 'left', 'right']]
    newPen = pg.mkPen(width=gridWidth, color=gridColor)
    for ax in axs:
      ax.setPen(newPen)

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

  def registerDrawAction(self, actParams: Union[FRParam, Sequence[FRParam]], func: DrawActFn):
    """
    Adds specified action(s) to the list of allowable roi actions if any do not already
    exist. `func` is only triggered if a shape was finished and the current action matches
    any of the specified `actParams`

    :param actParams: Single :py:class:`~s3a.structures.FRParam` or multiple FRParams that are allowed
      to trigger this funciton
    :param func: Function to trigger when a shape is completed during the requested actions.
      If only one parameter is registered to this function, it is expected to only take
      roiVerts. If multiple are provided, it is expected to take roiVerts and the current draw action
    """
    if isinstance(actParams, FRParam):
      actParams = [actParams]

    @wraps(func)
    def wrapper(roiVerts: XYVertices, param: FRParam):
      if param in actParams:
        if len(actParams) > 1:
          func(roiVerts, param)
        else:
          func(roiVerts)
    self.sigShapeFinished.connect(wrapper)

    for actParam in actParams:
      self.drawActGrp.create_addBtn(actParam, self.actionAssignment)

  def localImage(self, margin: int=0):
    return getCroppedImg(self.image, self.compSer[RTF.VERTICES].stack(), margin, returnSlices=False)

  def addTools(self, toolsEditor: ParamEditor):
    if toolsEditor in self._focusedTools:
      return
    self._focusedTools.append(toolsEditor)
    self.menu = menuFromEditorActions(self._focusedTools, menuParent=self)
    # self.toolsGrp.clear()
    # self.toolsGrp.fromToolsEditors(self._focusedTools, checkable=False, ownerClctn=self.toolsGrp)
    retClctn = None
    if self.toolbar is not None:
      def regenTools(editor: ParamEditor, grp: ButtonCollection):
        grp.clear()
        ButtonCollection.fromToolsEditors(editor, checkable=False, ownerClctn=grp)
      retClctn = ButtonCollection.fromToolsEditors(toolsEditor, title=toolsEditor.name)
      toolsEditor.params.sigChildAdded.connect(lambda *args: regenTools(toolsEditor, retClctn))
      self.toolbar.addWidget(retClctn)
    self.getViewBox().menu = self.menu
    return retClctn

  @FR_SINGLETON.actionStack.undoable('Modify Focused Component')
  def updateFocusedComp(self, newComp: pd.Series=None):
    """
    Updates focused image and component from provided information. Useful for creating
    a 'zoomed-in' view that allows much faster processing than applying image processing
    algorithms to the entire image each iteration.
    :param mainImg: Image from the main view
    :param newComp: New component to edit using various plugins (See :class:`TableFieldPlugin`)
    """
    oldComp = self.compSer
    mainImg = self.image
    if newComp is None or mainImg is None:
      newComp = FR_SINGLETON.tableData.makeCompSer()
    else:
      # Since values INSIDE the dataframe are reset instead of modified, there is no
      # need to go through the trouble of deep copying
      newComp = newComp.copy(deep=False)

    if mainImg is None:
      self.imgItem.clear()
      self.shapeCollection.clearAllRois()
    self.compSer = newComp

    self.sigUpdatedFocusedComp.emit(newComp)
    yield
    self.updateFocusedComp(oldComp)

  # def addActionsFromMenu(self, menu: QtWidgets.QMenu):
  #   vb: pg.ViewBox = self.getViewBox()
  #   menuCopy = QtWidgets.QMenu(self)
  #   for action in menu.actions():
  #     if action.isSeparator():
  #       menuCopy.addSeparator()
  #     else:
  #       menuCopy.addAction(action.text(), action.trigger)
  #   firstAct = menuCopy.actions()[0]
  #   menuCopy.insertAction(firstAct, self.oldVbMenu.viewAll)
  #   vb.menu = menuCopy
  #   self.menu = menuCopy

# class FocusedImage(MainImage):
#   sigPluginChanged = Signal()
#   sigUpdatedAll = Signal(object, object)
#   """Main image, new component. Emitted during `updateFocusedComp()`"""
#
#   @FR_SINGLETON.actionStack.undoable('Modify Focused Component')
#   def updateAll(self, mainImg: NChanImg=None, newComp:Optional[pd.Series]=None,
#                 _isAlreadyTrimmed=False):
#     """
#     Updates focused image and component from provided information. Useful for creating
#     a 'zoomed-in' view that allows much faster processing than applying image processing
#     algorithms to the entire image each iteration.
#     :param mainImg: Image from the main view
#     :param newComp: New component to edit using various plugins (See :class:`TableFieldPlugin`)
#     :param _isAlreadyTrimmed: Used internally during undo. Generally shouldn't be set by the
#       user
#     """
#     oldImg = self.image
#     if oldImg is not None:
#       oldImg = oldImg.copy()
#     oldComp = self.compSer
#
#     if mainImg is None:
#       self.imgItem.clear()
#       self.shapeCollection.clearAllRois()
#       self.compSer = FR_SINGLETON.tableData.makeCompSer()
#     else:
#       newVerts: ComplexXYVertices = newComp[RTF.VERTICES]
#       # Since values INSIDE the dataframe are reset instead of modified, there is no
#       # need to go through the trouble of deep copying
#       self.compSer = newComp.copy(deep=False)
#
#       # Propagate all resultant changes
#       if not _isAlreadyTrimmed:
#         self._updateBbox(mainImg.shape, newVerts)
#         bboxToUse = self.bbox
#       else:
#         bboxToUse = XYVertices([[0,0], mainImg.shape[:2]])
#         self.bbox = bboxToUse
#       if np.any(np.diff(self.bbox, axis=0) == 0):
#         # Empty slice
#         self.imgItem.clear()
#       else:
#         self.imgItem.setImage(mainImg)
#     self.sigUpdatedAll.emit(mainImg, newComp)
#     yield
#     self.updateAll(oldImg, oldComp, True)
#
#   def changeCurrentPlugin(self, newPlugin: TableFieldPlugin, forceActivate=True):
#     if newPlugin is self.currentPlugin:
#       return
#     if self.currentPlugin is not None:
#       self.currentPlugin.active = False
#     newEditors = [self.toolsEditor]
#     if newPlugin is not None:
#       newEditors.append(newPlugin.toolsEditor)
#     self.menu = menuFromEditorActions(newEditors, menuParent=self)
#     self.getViewBox().menu = self.menu
#     self.currentPlugin = newPlugin
#     if forceActivate and newPlugin is not None and not newPlugin.active:
#       newPlugin.active = True
#     self.sigPluginChanged.emit()