from functools import wraps
from typing import Union, Collection, Callable, Any, Sequence, List

import numpy as np
import pandas as pd
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from s3a.constants import REQD_TBL_FIELDS as RTF, PRJ_CONSTS as CNST
from s3a.controls.drawctrl import RoiCollection
from s3a.generalutils import getCroppedImg, coerceDfTypes
from s3a.structures import XYVertices
from utilitys import ParamEditor, PrjParam, RunOpts, fns, EditorPropsMixin, DeferredActionStackMixin as DASM
from utilitys.widgets import ButtonCollection, ImageViewer, EasyWidget
from .clickables import RightPanViewBox
from .regions import RegionCopierPlot
from .rois import SHAPE_ROI_MAPPING

__all__ = ['MainImage']

from ..shared import SharedAppSettings

Signal = QtCore.Signal
QCursor = QtGui.QCursor

DrawActFn = Union[Callable[[XYVertices, PrjParam], Any], Callable[[XYVertices], Any]]

class MainImage(DASM, EditorPropsMixin, ImageViewer):
  sigShapeFinished = Signal(object, object)
  """
  (XYVertices, PrjParam) emitted when a shape is finished
  - XYVerts from roi, re-thrown from self.shapeCollection
  - Current draw action
  """

  sigUpdatedFocusedComp = Signal(object)
  """pd.Series, newly focused component"""

  sigDrawActionChanged = Signal(object)
  """New draw action (PrjParam)"""

  def __initEditorParams__(self, shared: SharedAppSettings):
    self.compSer: pd.Series = shared.tableData.makeCompSer()
    shared.colorScheme.registerFunc(self.updateGridScheme, runOpts=RunOpts.ON_CHANGED)
    self.tableData = shared.tableData

  def __init__(self, parent=None, drawShapes: Collection[PrjParam]=None,
               toolbar: QtWidgets.QToolBar=None,
               **kargs):
    super().__init__(parent, viewBox=RightPanViewBox(), **kargs)

    if drawShapes is None:
      drawShapes = SHAPE_ROI_MAPPING.keys()

    self._initGrid()
    self.lastClickPos = QtCore.QPoint()
    self.toolbar = toolbar

    self.toolsEditor.registerFunc(self.resetZoom, btnOpts=CNST.TOOL_RESET_ZOOM)

    # -----
    # FOCUSED COMPONENT INFORMATION
    # -----
    self._focusedTools: List[ParamEditor] = []
    """
    List of all toolsEditor that allow actions to be performed on the currently focused components
    """
    # -----
    # DRAWING OPTIONS
    # -----
    self.regionCopier = RegionCopierPlot(self)

    self.drawAction: PrjParam = CNST.DRAW_ACT_PAN
    self.shapeCollection = RoiCollection(drawShapes, self)
    self.shapeCollection.sigShapeFinished.connect(
      lambda roiVerts: self.sigShapeFinished.emit(roiVerts, self.drawAction)
    )

    self.drawShapeGrp = ButtonCollection(self, 'Shapes', drawShapes, self.shapeAssignment,
                                         namePath=(self.__groupingName__,),
                                         checkable=True)
    self.drawActGrp = ButtonCollection(self, 'Actions')

    # Make sure panning is allowed before creating draw widget
    self.registerDrawAction(CNST.DRAW_ACT_PAN, lambda *args: self.actionAssignment(CNST.DRAW_ACT_PAN))

    # Initialize draw shape/action buttons
    self.drawActGrp.callFuncByParam(self.drawAction)
    self.drawShapeGrp.callFuncByParam(self.shapeCollection.curShapeParam)

    self.toolsGrp = None
    if toolbar is not None:
      toolbar.addWidget(self.drawShapeGrp)
      toolbar.addWidget(self.drawActGrp)
      self.toolsGrp = self.addTools(self.toolsEditor)

  def resetZoom(self):
    """
    Reimplement viewbox zooming since padding required for scatterplot with
    dynamic uncached shapes means the viewbox overestimates required rect
    """
    if self.image is None:
      return
    imShape = self.image.shape
    vb: RightPanViewBox = self.getViewBox()
    vb.setRange(xRange=(0, imShape[1]), yRange=(0, imShape[0]))

  @property
  def compSer_asFrame(self):
      return coerceDfTypes(fns.serAsFrame(self.compSer))

  def shapeAssignment(self, newShapeParam: PrjParam):
    self.shapeCollection.curShapeParam = newShapeParam
    if self.regionCopier.active:
      self.regionCopier.erase()

  def actionAssignment(self, newActionParam: PrjParam):
    self.drawAction = newActionParam
    if self.regionCopier.active:
      self.regionCopier.erase()
    self.sigDrawActionChanged.emit(newActionParam)

  def maybeBuildRoi(self, ev: QtGui.QMouseEvent):
    ev.ignore()
    if (QtCore.Qt.MouseButton.LeftButton not in [ev.buttons(), ev.button()]
        or self.drawAction == CNST.DRAW_ACT_PAN
        or self.regionCopier.active):
      return False
    finished = self.shapeCollection.buildRoi(ev, self.imgItem)
    if not finished:
      ev.accept()
    return finished

  def switchBtnMode(self, newMode: PrjParam):
    # EAFP
    try:
      self.drawActGrp.callFuncByParam(newMode)
    except KeyError:
      self.drawShapeGrp.callFuncByParam(newMode)

  def mousePressEvent(self, ev: QtGui.QMouseEvent):
    self.maybeBuildRoi(ev)
    if not ev.isAccepted():
      super().mousePressEvent(ev)
    self.lastClickPos = ev.pos()

  def mouseDoubleClickEvent(self, ev: QtGui.QMouseEvent):
    if self.regionCopier.active:
      self.regionCopier.sigCopyStopped.emit()
      ev.accept()
      self.shapeCollection.addLock(self)
    self.maybeBuildRoi(ev)
    if not ev.isAccepted():
      super().mouseDoubleClickEvent(ev)

  def mouseMoveEvent(self, ev: QtGui.QMouseEvent):
    """
    Mouse move behavior is contingent on which shape is currently selected,
    unless we are panning
    """
    self.maybeBuildRoi(ev)
    if not ev.isAccepted():
      super().mouseMoveEvent(ev)

  def mouseReleaseEvent(self, ev: QtGui.QMouseEvent):
    # Typical reaction is to right-click to cancel an roi
    if self.image is not None and QtCore.Qt.MouseButton.RightButton not in [ev.button(), ev.buttons()]:
      self.maybeBuildRoi(ev)

      # Special case: Panning
      if (self.lastClickPos == ev.pos()
          and self.drawAction == CNST.DRAW_ACT_PAN
          and not self.regionCopier.active):
        pos = self.imgItem.mapFromScene(ev.pos())
        xx, yy, = pos.x(), pos.y()
        # Simulate a click-wide boundary selection so points can be registered in pan mode
        pt = XYVertices([[xx, yy]], dtype=float)
        self.shapeCollection.sigShapeFinished.emit(pt)
    super().mouseReleaseEvent(ev)
    self.shapeCollection.removeLock(self)

  def clearCurRoi(self):
    """Clears the current ROI shape"""
    self.shapeCollection.clearAllRois()
    self.regionCopier.erase()

  def _widgetContainerChildren(self):
    return [self.drawActGrp, self.drawShapeGrp, self]

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

  def updateGridScheme(self, showGrid=False, gridWidth=1, gridColor='#fff'):
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

  def registerDrawAction(self, actParams: Union[PrjParam, Sequence[PrjParam]], func: DrawActFn,
                         **registerOpts):
    """
    Adds specified action(s) to the list of allowable roi actions if any do not already
    exist. `func` is only triggered if a shape was finished and the current action matches
    any of the specified `actParams`

    :param actParams: Single :py:class:`~s3a.structures.PrjParam` or multiple PrjParams that are allowed
      to trigger this funciton. If empty, triggers on every parameter
    :param func: Function to trigger when a shape is completed during the requested actions.
      If only one parameter is registered to this function, it is expected to only take
      roiVerts. If multiple or none are provided, it is expected to take roiVerts and the current draw action
    :param registerOpts: Extra arguments for button registration
    """
    if isinstance(actParams, PrjParam):
      actParams = [actParams]

    @wraps(func)
    def wrapper(roiVerts: XYVertices, param: PrjParam):
      if param in actParams:
        if len(actParams) > 1:
          func(roiVerts, param)
        else:
          func(roiVerts)
    if len(actParams) == 0:
      self.sigShapeFinished.connect(func)
    else:
      self.sigShapeFinished.connect(wrapper)
    for actParam in actParams:
      self.drawActGrp.create_addBtn(actParam, self.actionAssignment, checkable=True,
                                    namePath=(self.__groupingName__,), **registerOpts)

  def focusedCompImage(self, margin: int=0):
    return getCroppedImg(self.image, self.compSer[RTF.VERTICES].stack(), margin, returnCoords=False)

  def viewboxCoords(self, margin=0):
    """
    Returns the dimensions of the viewbox as (x,y) coordinates of its boundaries
    """
    vbRange = np.array(self.getViewBox().viewRange())
    span = np.diff(vbRange).flatten()
    offset = vbRange[:, 0]
    return span*np.array([[0,0], [0,1], [1,1], [1,0]]) + offset

  def addTools(self, toolsEditor: ParamEditor):
    if toolsEditor in self._focusedTools:
      return
    toolsEditor.actionsMenuFromProcs(outerMenu=self.menu, parent=self, nest=True)
    # self.toolsGrp.clear()
    # self.toolsGrp.fromToolsEditors(self._focusedTools, checkable=False, ownerClctn=self.toolsGrp)
    retClctn = None
    if self.toolbar is not None:
      retClctn = ButtonCollection.fromToolsEditors([toolsEditor], title=toolsEditor.name, copy=False)
      toolsEditor.params.sigChildAdded.connect(
        lambda _param, child, _idx: retClctn.addByParam(child, copy=False)
      )
      self.toolbar.addWidget(retClctn)
    self.getViewBox().menu = self.menu
    return retClctn

  def updateFocusedComp(self, newComp: pd.Series=None):
    """
    Updates focused image and component from provided information. Useful for creating
    a 'zoomed-in' view that allows much faster processing than applying image processing
    algorithms to the entire image each iteration.
    :param newComp: New component to edit using various plugins (See :class:`TableFieldPlugin`)
    """
    mainImg = self.image
    if newComp is None or mainImg is None:
      newComp = self.tableData.makeCompSer()
    else:
      # Since values INSIDE the dataframe are reset instead of modified, there is no
      # need to go through the trouble of deep copying
      newComp = newComp.copy(deep=False)

    if mainImg is None:
      self.imgItem.clear()
      self.shapeCollection.clearAllRois()
    self.compSer = newComp

    self.sigUpdatedFocusedComp.emit(newComp)