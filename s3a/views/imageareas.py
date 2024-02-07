from functools import wraps
from typing import Any, Callable, Collection, Sequence, Union

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from qtextras import (
    ButtonCollection,
    DeferredActionStackMixin as DASM,
    ImageViewer,
    OptionsDict,
    ParameterEditor,
    bindInteractorOptions as bind,
)

from .clickables import RightPanViewBox
from .regions import RegionMoverPlot
from .rois import SHAPE_ROI_MAPPING
from ..constants import PRJ_CONSTS as CNST
from ..controls.drawctrl import RoiCollection
from ..structures import XYVertices

__all__ = ["MainImage"]

Signal = QtCore.Signal
QCursor = QtGui.QCursor

DrawActFn = Union[Callable[[XYVertices, OptionsDict], Any], Callable[[XYVertices], Any]]


class MainImage(DASM, ImageViewer):
    sigShapeFinished = Signal(object, object)
    """
    (XYVertices, OptionsDict) emitted when a shape is finished
    - XYVerts from roi, re-thrown from self.shapeCollection
    - Current draw action
    """

    sigDrawActionChanged = Signal(object)
    """New draw action (OptionsDict)"""

    def __init__(
        self,
        parent=None,
        drawShapes: Collection[OptionsDict] = None,
        toolbar: QtWidgets.QToolBar = None,
        **kwargs,
    ):
        super().__init__(parent, viewBox=RightPanViewBox(), **kwargs)
        self.menu.clear()

        if drawShapes is None:
            drawShapes = SHAPE_ROI_MAPPING.keys()

        self._initGrid()
        self.lastClickPos = QtCore.QPoint()
        self.toolbar = toolbar

        self.toolsEditor.registerFunction(
            self.resetZoom, runActionTemplate=CNST.TOOL_RESET_ZOOM
        )

        # -----
        # DRAWING OPTIONS
        # -----
        self.regionMover = RegionMoverPlot()

        self.drawAction: OptionsDict = CNST.DRAW_ACT_PAN
        self.shapeCollection = RoiCollection(drawShapes, self)
        self.shapeCollection.sigShapeFinished.connect(
            lambda roiVerts: self.sigShapeFinished.emit(roiVerts, self.drawAction)
        )

        self.drawShapeGroup = ButtonCollection(
            self,
            "Shapes",
            drawShapes,
            self.shapeAssignment,
            checkable=True,
        )
        for options, button in self.drawShapeGroup.optionsButtonMap.items():
            if options.get("shortcut"):
                self.toolsEditor.registerObjectShortcut(button, **dict(options))
        self.drawActionGroup = ButtonCollection(self, "Actions")

        # Make sure panning is allowed before creating draw widget
        self.registerDrawAction(
            CNST.DRAW_ACT_PAN, lambda *args: self.actionAssignment(CNST.DRAW_ACT_PAN)
        )

        # Initialize draw shape/action buttons
        self.drawActionGroup.callAssociatedFunction(self.drawAction)
        self.drawShapeGroup.callAssociatedFunction(self.shapeCollection.shapeParameter)

        self.toolsGroup = None
        if toolbar is not None:
            toolbar.addWidget(self.drawShapeGroup)
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

    def shapeAssignment(self, newShapeParameter: OptionsDict):
        self.shapeCollection.shapeParameter = newShapeParameter
        if self.regionMover.active:
            self.regionMover.erase()

    def actionAssignment(self, newActionParameter: OptionsDict):
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
        finished = self.shapeCollection.buildRoi(ev, self.imageItem)
        if not finished:
            ev.accept()
        return finished

    def switchButtonMode(self, newMode: OptionsDict):
        # EAFP
        try:
            self.drawActionGroup.callAssociatedFunction(newMode)
        except KeyError:
            self.drawShapeGroup.callAssociatedFunction(newMode)

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
                pos = self.imageItem.mapFromScene(eventPos)
                (xx, yy) = (
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
        return [[self.drawActionGroup, self.drawShapeGroup], self]

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

    @bind(gridColor=dict(type="color"))
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
        """
        pi: pg.PlotItem = self.plotItem
        pi.showGrid(showGrid, showGrid)
        axs = [pi.getAxis(ax) for ax in ["top", "bottom", "left", "right"]]
        newPen = pg.mkPen(width=gridWidth, color=gridColor)
        for ax in axs:
            ax.setPen(newPen)

    def registerDrawAction(
        self,
        actionOptions: Union[OptionsDict, Sequence[OptionsDict]],
        function: DrawActFn,
        **registerOpts,
    ):
        """
        Adds specified action(s) to the list of allowable roi actions if any do not
        already exist. ``func`` is only triggered if a shape was finished and the current
        action matches any of the specified ``actParams``

        Parameters
        ----------
        actionOptions
            Single or multiple ``OptionsDict``s that are allowed to trigger this function.
            If empty, triggers on every option
        function
            Function to trigger when a shape is completed during the requested actions.
            If only one parameter is registered to this function, it is expected to
            only take roiPolygon. If multiple or none are provided, it is expected to
            take roiPolygon and the current draw action
        registerOpts
            Extra arguments for button registration
        """
        if isinstance(actionOptions, OptionsDict):
            actionOptions = [actionOptions]

        @wraps(function)
        def wrapper(roiPolygon: XYVertices, param: OptionsDict):
            if param in actionOptions:
                if len(actionOptions) > 1:
                    function(roiPolygon, param)
                else:
                    function(roiPolygon)

        if len(actionOptions) == 0:
            self.sigShapeFinished.connect(function)
        else:
            self.sigShapeFinished.connect(wrapper)
        for option in actionOptions:
            if (
                button := self.drawActionGroup.createAndAddButton(
                    option, self.actionAssignment, checkable=True, **registerOpts
                )
            ) and option.get("shortcut"):
                self.toolsEditor.registerObjectShortcut(button, **dict(option))

    def viewboxCoords(self, margin=0):
        """
        Returns the dimensions of the viewbox as (x,y) coordinates of its boundaries
        """
        vbRange = np.array(self.getViewBox().viewRange())
        span = np.diff(vbRange).flatten()
        offset = vbRange[:, 0]
        return span * np.array([[0, 0], [0, 1], [1, 1], [1, 0]]) + offset

    def addTools(self, toolsEditor: ParameterEditor):
        menu = toolsEditor.createActionsFromFunctions(stealShortcuts=False)
        self.menu.addMenu(menu)
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
                retClctn.addByParameter(param)

        if self.toolbar is not None:
            retClctn = ButtonCollection(title=toolsEditor.name)
            # Regular "fromToolsEditors" doesn't listen for changes in group parameters,
            # so do a dfs node visit here
            visit(toolsEditor.rootParameter)
            self.toolbar.addWidget(retClctn)
        self.getViewBox().menu = self.menu
        return retClctn
