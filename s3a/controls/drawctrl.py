from typing import Collection, Dict

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui

from ..structures import OptionsDict, XYVertices
from ..views.rois import SHAPE_ROI_MAPPING, PlotDataROI

__all__ = ["RoiCollection"]


class RoiCollection(QtCore.QObject):
    # Signal(ExtendedROI)
    sigShapeFinished = QtCore.Signal(object)  # roiVertices : XYVertices

    def __init__(
        self,
        allowableShapes: Collection[OptionsDict] = (),
        parent: pg.GraphicsView = None,
    ):
        super().__init__(parent)

        if allowableShapes is None:
            allowableShapes = set()
        self.shapeVerts = XYVertices()
        # Make a new graphics item for each roi type
        self.parameterRoiMap: Dict[OptionsDict, PlotDataROI] = {}
        self._shapeParameter = (
            next(iter(allowableShapes)) if len(allowableShapes) > 0 else None
        )

        self._locks = set()
        self.addLock(self)
        self._parent = parent

        for shape in allowableShapes:
            newRoi = SHAPE_ROI_MAPPING[shape]()
            newRoi.setZValue(1000)
            self.parameterRoiMap[shape] = newRoi
            newRoi.setRoiPoints()
            newRoi.hide()
        self.addRoisToView(parent)

    def addRoisToView(self, view: pg.GraphicsView):
        self._parent = view
        if view is not None:
            for roi in self.parameterRoiMap.values():
                roi.hide()
                view.addItem(roi)

    def clearAllRois(self):
        for roi in self.parameterRoiMap.values():  # type: PlotDataROI
            roi.setRoiPoints()
            roi.hide()
            self.addLock(self)
            # If all ROIs share the same action stack, calling "flush" on one should
            # take care of everything But this is a failsafe against separate undo
            # buffers for each shape
            roi.flushBuildActions()

    def addLock(self, lock):
        """
        Allows this shape collection to be `locked`, preventing shapes from being drawn.
        Multiple locks can be applied; ROIs can only be drawn when all locks are removed.

        Parameters
        ----------
        lock
            Anything used as a lock. This will have to be manually removed later using
            ``RoiCollection.removeLock``
        """
        self._locks.add(lock)

    def removeLock(self, lock):
        try:
            self._locks.remove(lock)
        except KeyError:
            pass

    def forceUnlock(self):
        self._locks.clear()

    @property
    def locked(self):
        return len(self._locks) > 0

    def buildRoi(self, ev: QtGui.QMouseEvent, imageItem: pg.ImageItem = None):
        """
        Construct the current shape ROI depending on mouse movement and current shape
        parameters

        Parameters
        ----------
        imageItem
            Image the ROI is drawn upon, used for mapping event coordinates from a
            scene to pixel coordinates. If *None*, event coordinates are assumed to
            already be relative to pixel coordinates.
        ev
            Mouse event
        """
        # Unblock on mouse press
        # None imageItem is only the case during programmatic calls so allow this case
        if (
            (imageItem is None or imageItem.image is not None)
            and ev.type() == QtCore.QEvent.Type.MouseButtonPress
            and ev.button() == QtCore.Qt.MouseButton.LeftButton
        ):
            self.removeLock(self)
        if self.locked:
            return False
        eventPos = ev.position() if hasattr(ev, "position") else ev.localPos()
        if imageItem is not None:
            posRelToImg = imageItem.mapFromScene(eventPos)
        else:
            posRelToImg = eventPos
        # Form of rate-limiting -- only simulate click if the next pixel is at least
        # one away from the previous pixel location
        xyCoord = XYVertices([[posRelToImg.x(), posRelToImg.y()]], dtype=float)
        curRoi = self.currentShape
        constructingRoi, self.shapeVerts = self.currentShape.updateShape(ev, xyCoord)
        if self.shapeVerts is not None:
            self.sigShapeFinished.emit(self.shapeVerts)

        if not constructingRoi:
            # Vertices from the completed shape are already stored, so clean up the
            # shapes.
            curRoi.setRoiPoints()
            curRoi.hide()
        else:
            # Still constructing ROI. Show it
            curRoi.show()
        return constructingRoi

    @property
    def shapeParameter(self):
        return self._shapeParameter

    @shapeParameter.setter
    def shapeParameter(self, newShape: OptionsDict):
        """
        When the shape is changed, be sure to reset the underlying ROIs
        """
        # Reset the underlying ROIs for a different shape than we currently are using
        if newShape != self._shapeParameter:
            self.clearAllRois()
        self._shapeParameter = newShape

    @property
    def currentShape(self):
        return self.parameterRoiMap[self._shapeParameter]
