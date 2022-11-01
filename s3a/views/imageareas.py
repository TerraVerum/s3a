from functools import wraps
from typing import Any, Callable, Collection, Sequence, Union

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from utilitys import DeferredActionStackMixin as DASM, ParamEditor, PrjParam
from utilitys.widgets import ButtonCollection, ImageViewer

from .clickables import RightPanViewBox
from .regions import RegionMoverPlot
from .rois import SHAPE_ROI_MAPPING
from ..constants import PRJ_CONSTS as CNST
from ..controls.drawctrl import RoiCollection
from ..structures import XYVertices

__all__ = ["MainImage"]

Signal = QtCore.Signal
QCursor = QtGui.QCursor

DrawActFn = Union[Callable[[XYVertices, PrjParam], Any], Callable[[XYVertices], Any]]


class MainImage(DASM, ImageViewer):
    sigShapeFinished = Signal(object, object)
    """
    (XYVertices, PrjParam) emitted when a shape is finished
    - XYVerts from roi, re-thrown from self.shapeCollection
    - Current draw action
    """

    sigDrawActionChanged = Signal(object)
    """New draw action (PrjParam)"""

    def __init__(
        self,
        parent=None,
        drawShapes: Collection[PrjParam] = None,
        toolbar: QtWidgets.QToolBar = None,
        **kargs
    ):
        super().__init__(parent, viewBox=RightPanViewBox(), **kargs)
        self.menu.clear()

        if drawShapes is None:
            drawShapes = SHAPE_ROI_MAPPING.keys()

        self._initGrid()
        self.lastClickPos = QtCore.QPoint()
        self.toolbar = toolbar

        self.toolsEditor.registerFunc(self.resetZoom, btnOpts=CNST.TOOL_RESET_ZOOM)

        # -----
        # DRAWING OPTIONS
        # -----
        self.regionMover = RegionMoverPlot()

        self.drawAction: PrjParam = CNST.DRAW_ACT_PAN
        self.shapeCollection = RoiCollection(drawShapes, self)
        self.shapeCollection.sigShapeFinished.connect(
            lambda roiVerts: self.sigShapeFinished.emit(roiVerts, self.drawAction)
        )

        self.drawShapeGrp = ButtonCollection(
            self,
            "Shapes",
            drawShapes,
            self.shapeAssignment,
            checkable=True,
        )
        self.drawActionGroup = ButtonCollection(self, "Actions")

        # Make sure panning is allowed before creating draw widget
        self.registerDrawAction(
            CNST.DRAW_ACT_PAN, lambda *args: self.actionAssignment(CNST.DRAW_ACT_PAN)
        )

        # Initialize draw shape/action buttons
        self.drawActionGroup.callFuncByParam(self.drawAction)
        self.drawShapeGrp.callFuncByParam(self.shapeCollection.shapeParameter)

        self.toolsGroup = None
        if toolbar is not None:
            toolbar.addWidget(self.drawShapeGrp)
            toolbar.addWidget(self.drawActionGroup)
            self.toolsGroup = self.addTools(self.toolsEditor)

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

    def shapeAssignment(self, newShapeParameter: PrjParam):
        self.shapeCollection.shapeParameter = newShapeParameter
        if self.regionMover.active:
            self.regionMover.erase()

    def actionAssignment(self, newActionParameter: PrjParam):
        self.drawAction = newActionParameter
        if self.regionMover.active:
            self.regionMover.erase()
        self.sigDrawActionChanged.emit(newActionParameter)

    def maybeBuildRoi(self, ev: QtGui.QMouseEvent):
        ev.ignore()
        if (
            QtCore.Qt.MouseButton.LeftButton not in [ev.buttons(), ev.button()]
            or self.drawAction == CNST.DRAW_ACT_PAN
            or self.regionMover.active
        ):
            return False
        finished = self.shapeCollection.buildRoi(ev, self.imgItem)
        if not finished:
            ev.accept()
        return finished

    def switchButtonMode(self, newMode: PrjParam):
        # EAFP
        try:
            self.drawActionGroup.callFuncByParam(newMode)
        except KeyError:
            self.drawShapeGrp.callFuncByParam(newMode)

    def mousePressEvent(self, ev: QtGui.QMouseEvent):
        self.maybeBuildRoi(ev)
        if not ev.isAccepted():
            super().mousePressEvent(ev)
        self.lastClickPos = ev.position() if hasattr(ev, "position") else ev.localPos()

    def mouseDoubleClickEvent(self, ev: QtGui.QMouseEvent):
        if self.regionMover.active:
            self.regionMover.sigMoveStopped.emit()
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
        if self.image is not None and QtCore.Qt.MouseButton.RightButton not in [
            ev.button(),
            ev.buttons(),
        ]:
            self.maybeBuildRoi(ev)

            # Special case: Panning
            eventPos = ev.position() if hasattr(ev, "position") else ev.localPos()
            if (
                self.lastClickPos == eventPos
                and self.drawAction == CNST.DRAW_ACT_PAN
                and not self.regionMover.active
            ):
                pos = self.imgItem.mapFromScene(eventPos)
                xx, yy, = (
                    pos.x(),
                    pos.y(),
                )
                # Simulate a click-wide boundary selection so points can be registered
                # in pan mode
                pt = XYVertices([[xx, yy]], dtype=float)
                self.shapeCollection.sigShapeFinished.emit(pt)
        super().mouseReleaseEvent(ev)
        self.shapeCollection.removeLock(self)

    def clearCurrentRoi(self):
        """Clears the current ROI shape"""
        self.shapeCollection.clearAllRois()
        self.regionMover.erase()

    def _widgetContainerChildren(self):
        return [self.drawActionGroup, self.drawShapeGrp, self]

    def _initGrid(self):
        pi: pg.PlotItem = self.plotItem
        pi.showGrid(alpha=1.0)
        axs = [pi.getAxis(ax) for ax in ["top", "bottom", "left", "right"]]
        for ax in axs:  # type: pg.AxisItem
            ax.setZValue(1e9)
            # ax.setTickSpacing(5, 1)
            for evFn in (
                "mouseMoveEvent",
                "mousePressEvent",
                "mouseReleaseEvent",
                "mouseDoubleClickEvent",
                "mouseDragEvent",
                "wheelEvent",
            ):
                newEvFn = lambda ev: ev.ignore()
                setattr(ax, evFn, newEvFn)

    def updateGridScheme(self, showGrid=False, gridWidth=1, gridColor="#fff"):
        """
        Update the grid scheme for the image plot to either show a grid with
        specified width and color, or to hide the grid

        Parameters
        ----------
        showGrid
            Whether to show the grid
        gridWidth
            Width of the grid lines
        gridColor
            Color of the grid lines
            pType: color
        """
        pi: pg.PlotItem = self.plotItem
        pi.showGrid(showGrid, showGrid)
        axs = [pi.getAxis(ax) for ax in ["top", "bottom", "left", "right"]]
        newPen = pg.mkPen(width=gridWidth, color=gridColor)
        for ax in axs:
            ax.setPen(newPen)

    def registerDrawAction(
        self,
        actionParameters: Union[PrjParam, Sequence[PrjParam]],
        function: DrawActFn,
        **registerOpts
    ):
        """
        Adds specified action(s) to the list of allowable roi actions if any do not
        already exist. ``func`` is only triggered if a shape was finished and the current
        action matches any of the specified ``actParams``

        Parameters
        ----------
        actionParameters
            Single or multiple ``PrjParam``s that are allowed to trigger this funciton.
            If empty, triggers on every parameter
        function
            Function to trigger when a shape is completed during the requested actions.
            If only one parameter is registered to this function, it is expected to
            only take roiPolygon. If multiple or none are provided, it is expected to
            take roiPolygon and the current draw action
        registerOpts
            Extra arguments for button registration
        """
        if isinstance(actionParameters, PrjParam):
            actionParameters = [actionParameters]

        @wraps(function)
        def wrapper(roiPolygon: XYVertices, param: PrjParam):
            if param in actionParameters:
                if len(actionParameters) > 1:
                    function(roiPolygon, param)
                else:
                    function(roiPolygon)

        if len(actionParameters) == 0:
            self.sigShapeFinished.connect(function)
        else:
            self.sigShapeFinished.connect(wrapper)
        for actParam in actionParameters:
            self.drawActionGroup.createAndAddBtn(
                actParam,
                self.actionAssignment,
                checkable=True,
                namePath=(self.__groupingName__,),
                **registerOpts
            )

    def viewboxCoords(self, margin=0):
        """
        Returns the dimensions of the viewbox as (x,y) coordinates of its boundaries
        """
        vbRange = np.array(self.getViewBox().viewRange())
        span = np.diff(vbRange).flatten()
        offset = vbRange[:, 0]
        return span * np.array([[0, 0], [0, 1], [1, 1], [1, 0]]) + offset

    def addTools(self, toolsEditor: ParamEditor):
        toolsEditor.actionsMenuFromProcs(outerMenu=self.menu, parent=self, nest=True)
        retClctn = None
        # Define some helper functions for listening to toolsEditor changes
        def visit(param, child=None):
            if child:
                param = child
            if param.type() == "group":
                param.sigChildAdded.connect(visit)
                for ch in param:
                    visit(ch)
            else:
                retClctn.addByParam(param, copy=False)

        if self.toolbar is not None:
            retClctn = ButtonCollection(title=toolsEditor.name)
            # Regular "fromToolsEditors" doesn't listen for changes in group parameters,
            # so do a dfs node visit here
            visit(toolsEditor.params)
            self.toolbar.addWidget(retClctn)
        self.getViewBox().menu = self.menu
        return retClctn
