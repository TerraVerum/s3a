from functools import wraps
from typing import Union, Collection, Callable, Any, Sequence, List

import numpy as np
import pandas as pd
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from pyqtgraph.graphicsItems.ViewBox.ViewBoxMenu import ViewBoxMenu
from skimage.io import imread
from utilitys import ParamEditor, EditorPropsMixin, PrjParam, RunOpts, CompositionMixin, fns

from s3a import FR_SINGLETON
from s3a.constants import REQD_TBL_FIELDS as RTF, PRJ_CONSTS as CNST
from s3a.controls.drawctrl import RoiCollection
from s3a.generalutils import getCroppedImg, coerceDfTypes
from s3a.structures import NChanImg, FilePath
from s3a.structures import XYVertices
from .buttons import ButtonCollection
from .clickables import RightPanViewBox
from .regions import RegionCopierPlot
from .rois import SHAPE_ROI_MAPPING

__all__ = ['MainImage']

Signal = QtCore.Signal
QCursor = QtGui.QCursor

DrawActFn = Union[Callable[[XYVertices, PrjParam], Any], Callable[[XYVertices], Any]]

class MainImage(CompositionMixin, EditorPropsMixin, pg.PlotWidget):
  sigShapeFinished = Signal(object, object)
  """
  (XYVertices, PrjParam) emitted when a shape is finished
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
    cls.toolsEditor = ParamEditor.buildClsToolsEditor(cls, 'Region Tools')

    cls.minCompSize, = FR_SINGLETON.generalProps.registerProps(
      [CNST.PROP_MIN_COMP_SZ])

  def __init__(self, parent=None, drawShapes: Collection[PrjParam]=None,
               imgSrc: Union[FilePath, NChanImg]=None,
               toolbar: QtWidgets.QToolBar=None,
               **kargs):
    super().__init__(parent, viewBox=RightPanViewBox(), **kargs)

    if drawShapes is None:
      drawShapes = SHAPE_ROI_MAPPING.keys()

    vb = self.getViewBox()
    self.menu: QtWidgets.QMenu = QtWidgets.QMenu(self)
    self.oldVbMenu: ViewBoxMenu = vb.menu
    # Disable default menus
    self.plotItem.ctrlMenu = None
    self.sceneObj.contextMenu = None

    self.setAspectLocked(True)
    vb.invertY()
    self.setMouseEnabled(True)
    self._initGrid()
    FR_SINGLETON.colorScheme.registerFunc(self.updateGridScheme, runOpts=RunOpts.ON_CHANGED)

    self.lastClickPos = QtCore.QPoint()

    self.toolbar = toolbar

    # -----
    # IMAGE
    # -----
    self.imgItem = self.exposes(pg.ImageItem())
    self.imgItem.setZValue(-100)
    self.addItem(self.imgItem)
    self.toolsEditor.registerFunc(self.resetZoom, btnOpts=CNST.TOOL_RESET_ZOOM)
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

    # Initialize image
    if imgSrc is not None:
      self.setImage(imgSrc)

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

  def maybeBuildRoi(self, ev: QtGui.QMouseEvent):
    ev.ignore()
    if (QtCore.Qt.LeftButton not in [ev.buttons(), ev.button()]
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

  def registerDrawAction(self, actParams: Union[PrjParam, Sequence[PrjParam]], func: DrawActFn,
                         **registerOpts):
    """
    Adds specified action(s) to the list of allowable roi actions if any do not already
    exist. `func` is only triggered if a shape was finished and the current action matches
    any of the specified `actParams`

    :param actParams: Single :py:class:`~s3a.structures.PrjParam` or multiple PrjParams that are allowed
      to trigger this funciton
    :param func: Function to trigger when a shape is completed during the requested actions.
      If only one parameter is registered to this function, it is expected to only take
      roiVerts. If multiple are provided, it is expected to take roiVerts and the current draw action
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
    self.sigShapeFinished.connect(wrapper)
    for actParam in actParams:
      self.drawActGrp.create_addBtn(actParam, self.actionAssignment, checkable=True,
                                    namePath=(self.__groupingName__,), **registerOpts)

  def focusedCompImage(self, margin: int=0):
    return getCroppedImg(self.image, self.compSer[RTF.VERTICES].stack(), margin, returnSlices=False)

  def viewboxSquare(self, margin=0):
    vbRange = np.array(self.getViewBox().viewRange())
    span = np.diff(vbRange).flatten()
    center = vbRange[:,0]+span/2
    minSpan = np.min(span) + margin
    offset = center - minSpan/2
    return minSpan*np.array([[0,0], [0,1], [1,1], [1,0]]) + offset

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

  @FR_SINGLETON.actionStack.undoable('Modify Focused Component')
  def updateFocusedComp(self, newComp: pd.Series=None):
    """
    Updates focused image and component from provided information. Useful for creating
    a 'zoomed-in' view that allows much faster processing than applying image processing
    algorithms to the entire image each iteration.
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