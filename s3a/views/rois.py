import math
from typing import Callable, Dict, Optional

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from qtextras import DeferredActionStackMixin as DASM, OptionsDict
from skimage.draw import draw

from ..constants import PRJ_CONSTS
from ..generalutils import orderContourPoints, symbolFromVertices
from ..structures import ComplexXYVertices, XYVertices
from ..views.clickables import BoundScatterPlot

__all__ = ["RectROI", "PlotDataROI", "PolygonROI", "PointROI", "SHAPE_ROI_MAPPING"]

qe = QtCore.QEvent
_ROI_PT_DESCR = "Add ROI Points"


class PlotDataROI(DASM, BoundScatterPlot):
    connected = True

    def __init__(self, endEvType=qe.MouseButtonRelease, addIntermedPts=True):
        super().__init__(pxMode=False, size=1)
        self.setBrush(pg.mkBrush(0, 0, 0, 100))
        self.setPen(pg.mkPen("w", width=0))
        self.vertices = XYVertices()

        self.endEvType = endEvType
        self.viableEventTypes = {
            qe.MouseButtonPress,
            qe.MouseMove,
            qe.MouseButtonRelease,
            qe.MouseButtonDblClick,
        } - {endEvType}
        self.constructingRoi = False
        self.addIntermedPts = addIntermedPts
        self.firstPt = XYVertices()

    def addRoiPoints(self, pts: XYVertices):
        # noinspection PyTypeChecker
        pts = pts.astype(int)
        if self.vertices.size > 0 and np.all(pts == self.vertices[-1]):
            return
        if self.addIntermedPts or len(self.vertices) < 2:
            vertsToUse = np.vstack([self.vertices, pts])
        else:
            vertsToUse = np.vstack([self.firstPt, pts])
        refactored = self._refactorPoints(vertsToUse)
        verts = self.vertices = XYVertices(refactored)
        connectData = np.vstack([verts, verts[[0]]]).view(XYVertices)
        symbol, pos = symbolFromVertices(ComplexXYVertices([connectData]))
        self.setData(*pos.T, symbol=[symbol])

    def dataBounds(self, ax, frac=1.0, orthoRange=None):
        if frac >= 1.0 and orthoRange is None and self.bounds[ax] is not None:
            return self.bounds[ax]
        if not len(self.vertices):
            return (None, None)
        self.bounds[ax] = self.vertices[:, ax].min() - 2, self.vertices[:, ax].max() + 2
        return super().dataBounds(ax, frac, orthoRange)

    def setRoiPoints(self, points: XYVertices = None):
        if points is None:
            points = XYVertices()
        self.setData()
        self.vertices = XYVertices()
        self.firstPt = points
        if points.size > 0:
            with self.actionStack.ignoreActions():
                self.addRoiPoints(points)

    def _refactorPoints(self, vertices: np.ndarray):
        return vertices

    def updateShape(self, ev: QtGui.QMouseEvent, xyEventCoords: XYVertices):
        """
        See function signature for :func:`ExtendedROI.updateShape`
        """
        success = True
        verts = None
        constructingRoi = self.constructingRoi
        # If not left click, do nothing
        if (
            ev.buttons() & QtCore.Qt.MouseButton.LeftButton
        ) == 0 and ev.button() != QtCore.Qt.MouseButton.LeftButton:
            return self.constructingRoi, verts
        evType = ev.type()
        if evType == qe.Type.MouseButtonPress and not constructingRoi:
            # Need to start a new shape
            self.setRoiPoints(xyEventCoords)
            constructingRoi = True
        if evType in self.viableEventTypes:
            self.addRoiPoints(xyEventCoords)
            constructingRoi = True
        elif evType == self.endEvType:
            # Done drawing the ROI, complete shape, get vertices, remove old undo stack
            # entries
            verts = self.vertices
            constructingRoi = False
            self.flushBuildActions()
        else:
            success = False
            # Unable to process this event

        if success:
            ev.accept()
        self.constructingRoi = constructingRoi
        return self.constructingRoi, verts

    def flushBuildActions(self):
        if self.actionStack.locked:
            # Shouldn't flush the true action buffer
            return
        acts = self.actionStack.currentActionBuffer
        while len(acts) and acts[-1].descr == _ROI_PT_DESCR:
            acts.pop()
        while len(acts) and acts[0].descr == _ROI_PT_DESCR and not acts[0].treatAsUndo:
            acts.popleft()


class RectROI(PlotDataROI):
    def __init__(self, endEvType=qe.MouseButtonRelease):
        super().__init__(endEvType)
        self.addIntermedPts = False

    def _refactorPoints(self, vertices: np.ndarray):
        square = XYVertices(np.vstack([vertices.min(0), vertices.max(0)]))
        return np.vstack(
            [
                square[0],
                [square[0, 0], square[1, 1]],
                [square[1, 0], square[1, 1]],
                [square[1, 0], square[0, 1]],
            ]
        )


class PolygonROI(PlotDataROI):
    def __init__(self):
        super().__init__(qe.MouseButtonDblClick, True)
        # self.viableEventTypes.remove(qe.MouseMove)
        self.viableEventTypes.remove(qe.MouseButtonRelease)
        self.lastEvType = None

    def updateShape(self, ev: QtGui.QMouseEvent, xyEventCoords: XYVertices):
        self.lastEvType = ev.type()
        return super().updateShape(ev, xyEventCoords)

    @DASM.undoable(_ROI_PT_DESCR)
    def addRoiPoints(self, points: XYVertices):
        oldVerts = self.vertices.copy()
        ret = super().addRoiPoints(points)
        if self.lastEvType != qe.MouseMove and len(self.vertices) > 1:
            yield
            return super().setRoiPoints(oldVerts)
        return ret

    def _refactorPoints(self, vertices: np.ndarray):
        if self.lastEvType == qe.MouseMove and len(vertices) > 1:
            # Move without releasing last key, so update the last vertex to new position
            return np.delete(vertices, -2, axis=0)
        return vertices


class PointROI(PlotDataROI):
    roiRadius = 1

    @classmethod
    def updateRadius(cls, radius=1):
        cls.roiRadius = radius

    def __init__(self):
        super().__init__()
        self.setSize(3)
        self.constructTypes = {qe.MouseButtonPress, qe.MouseMove}

    def updateShape(self, ev: QtGui.QMouseEvent, xyEventCoords: XYVertices):
        success = False
        verts = None
        constructingRoi = False
        if ev.type() in self.constructTypes:
            success = True
            self.setData(*xyEventCoords.T, size=self.roiRadius * 2)
            verts = XYVertices(
                np.column_stack(draw.disk(xyEventCoords[0], self.roiRadius))
            )
            self.vertices = verts
            constructingRoi = True

        if success:
            ev.accept()
        return constructingRoi, verts


class EllipseROI(PlotDataROI):
    def __init__(self):
        super().__init__(endEvType=qe.MouseButtonRelease, addIntermedPts=False)

    def _refactorPoints(self, vertices: np.ndarray):
        pts = vertices[:, ::-1].astype(float)
        ma = pts.max(0)
        mi = pts.min(0)
        center = (ma - mi) / 2 + mi
        normedR = ma[0] - mi[0]
        normedC = ma[1] - mi[1]
        # Apply scaling so mouse point is on perimeter
        # angle = np.arctan(normedR/normedC)
        # normedR = abs(np.cos(angle))*normedR
        # normedC = abs(np.sin(angle))*normedC
        sqr2 = np.sqrt(2)
        perim = draw.ellipse_perimeter(
            *center.astype(int), int(normedR / sqr2), int(normedC / sqr2)
        )
        # Reorder to ensure no criss-crossing when these vertices are plotted
        perim = orderContourPoints(np.column_stack(perim[::-1]))
        return perim.view(XYVertices)


class ROIManipulator(pg.RectROI):
    """
    Pyqtgraph roi assigned to a PlotDataROI to allow various manipulations such as
    scaling, rotating, translating, etc. After the manipulation is complete,
    the wrapped PlotDataROI is transformed accordingly.
    """

    roiVerts: Optional[np.ndarray] = None
    _connectivity: Optional[np.ndarray] = None

    def __init__(self, **kwargs):
        self.handleSize = 10
        super().__init__(
            [0, 0],
            [0, 0],
            handlePen=pg.mkPen("r"),
            hoverPen=pg.mkPen("y", width=5),
            **kwargs,
        )
        self.addRotateHandle([1, 0], [0.5, 0.5])
        # For some reason, handle pens are explicitly set to 0 width. Fix that here
        for handle in self.getHandles():
            handle.pen.setWidth(3)
            handle.radius = 10
            handle.buildPath()

    def paint(self, p, opt, widget):
        super().paint(p, opt, widget)
        if self.roiVerts is None or not len(self.roiVerts):
            return
        pts = self.roiVerts.copy()
        # Draw twice, once with white and once with black pen. This ensures the outline
        # is always visible regardless of background color
        path = pg.arrayToQPath(*(pts / pts.max(0)).T, connect=self._connectivity)
        for color, width in zip("kw", (2, 1)):
            width = max(width, int(width / self.pixelLength(None)))
            p.setPen(pg.mkPen(color, width=width))
            p.drawPath(path)

    def setBaseData(self, data: XYVertices, connectivity: np.ndarray, show=True):
        """
        Entry point for transformations to a plot data roi. When
        meth:`RoiManpulator.finish` is called, whatever translations, scaling,
        and rotating done to this ROI will be applied to `dataRoi`.
        """
        self.roiVerts = data.view(np.ndarray)
        self.roiVerts = self.roiVerts - data.min(0)
        # 0 offset for easier calculations
        self.setSize(self.roiVerts.max(0))
        self._connectivity = connectivity
        if show:
            self.show()

    def getTransformedPoints(
        self, data=None, rot=None, pos=None, scale=None, dtype=int, hide=False
    ):
        """
        Applies rotation/translation/scaling to set of ROI points based on current
        pg.ROI properties. Rotation, position, and/or scale can optionally be overridden
        """
        state = self.getState()

        if data is None:
            data = self.roiVerts

        if rot is None:
            rot = state["angle"]
            rot = np.deg2rad(rot)
        if pos is None:
            pos = np.array(state["pos"]).reshape(1, 2)
        if scale is None:
            scale = np.array(state["size"] / self.roiVerts.max(0)).reshape(2, 1)

        xformMat = np.array(
            [
                [math.cos(rot), -math.sin(rot)],
                [math.sin(rot), math.cos(rot)],
            ]
        )
        useVerts = data.view(np.ndarray).T
        #          rotate                   scale   translate
        newVerts = (xformMat @ (useVerts * scale)).T + pos
        if hide:
            self.hide()
        return newVerts.astype(dtype)


class FreehandRoi(PlotDataROI):
    def __init__(self, endEventType=qe.MouseButtonDblClick, addIntermedPoints=True):
        super().__init__(endEventType, addIntermedPoints)
        self.viableEventTypes.remove(qe.MouseButtonRelease)
        self.lastEvType = None

    @DASM.undoable(_ROI_PT_DESCR)
    def addRoiPoints(self, /, points: XYVertices):
        oldVerts = self.vertices.copy()
        super().addRoiPoints(points)
        stack = self.actionStack
        if (
            self.lastEvType != qe.MouseButtonPress
            and not stack.lockedByUser
            and stack.undoDescr == _ROI_PT_DESCR
        ):
            # Moving the mouse should merge with the last undoable operation rather
            # than adding a new entry to the stack This merge shouldn't happen on a
            # redo, since the argument was already modified on the first forward pass.
            # Hence, the checking against "undoDescr"
            act = stack.actions[-1]
            # Add this point to the redo operation of the last action
            act.args = (np.r_[act.args[0], points],)
            return
        # else:
        yield
        self.setRoiPoints(oldVerts)

    def removeRoiPoints(self, removePercent: float):
        dists = np.cumsum(
            np.r_[0, np.sqrt((np.diff(self.vertices, axis=0) ** 2).sum(axis=1))]
        )
        keepAmt = dists[-1] * (1 - removePercent)
        keepAmt = np.clip(keepAmt, 0, np.inf)
        # First distance is 0, so at least one point is guaranteed to be kept
        keepIdx = np.flatnonzero(dists <= keepAmt)[-1]
        self.setRoiPoints(self.vertices[:keepIdx])

    def updateShape(self, ev: QtGui.QMouseEvent, xyEventCoords: XYVertices):
        self.lastEvType = ev.type()
        return super().updateShape(ev, xyEventCoords)


SHAPE_ROI_MAPPING: Dict[OptionsDict, Callable[[], PlotDataROI]] = {
    PRJ_CONSTS.DRAW_SHAPE_RECT: RectROI,
    PRJ_CONSTS.DRAW_SHAPE_FREE: FreehandRoi,
    PRJ_CONSTS.DRAW_SHAPE_POLY: PolygonROI,
    PRJ_CONSTS.DRAW_SHAPE_ELLIPSE: EllipseROI,
    PRJ_CONSTS.DRAW_SHAPE_POINT: PointROI,
}
