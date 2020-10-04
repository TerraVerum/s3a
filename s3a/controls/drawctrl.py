from typing import Tuple, Dict, Union, Collection

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui

from s3a import FR_SINGLETON
from s3a.constants import FR_CONSTS
from s3a.structures import FRParam, FRVertices
from s3a.views.rois import FRExtendedROI, SHAPE_ROI_MAPPING

__all__ = ['FRRoiCollection']

@FR_SINGLETON.registerGroup(FR_CONSTS.CLS_ROI_CLCTN)
class FRRoiCollection(QtCore.QObject):
  # Signal(FRExtendedROI)
  sigShapeFinished = QtCore.Signal(object)

  def __init__(self, allowableShapes: Collection[FRParam]=(), parent: pg.GraphicsView=None):
    super().__init__(parent)
    if allowableShapes is None:
      allowableShapes = set()
    self.shapeVerts = FRVertices()
    # Make a new graphics item for each roi type
    self.roiForShape: Dict[FRParam, Union[pg.ROI, FRExtendedROI]] = {}
    self._curShape = allowableShapes[0] if len(allowableShapes) > 0 else None

    self._locks = set()
    self.addLock(self)
    self._parent = parent

    for shape in allowableShapes:
      newRoi = SHAPE_ROI_MAPPING[shape]()
      newRoi.setZValue(1000)
      self.roiForShape[shape] = newRoi
      newRoi.hide()
    self.addRoisToView(parent)

  def addRoisToView(self, view: pg.GraphicsView):
    self._parent = view
    if view is not None:
      for roi in self.roiForShape.values():
        roi.hide()
        view.addItem(roi)

  def clearAllRois(self):
    for roi in self.roiForShape.values(): # type: FRExtendedROI
      roi.clear()
      roi.hide()
      self.addLock(self)

  def addLock(self, lock):
    """
    Allows this shape collection to be `locked`, preventing shapes from being drawn.
    Multiple locks can be applied; ROIs can only be drawn when all locks are removed.

    :param lock: Anything used as a lock. This will have to be manually removed later
      using `FRRoiCollection.removeLock`
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

  def buildRoi(self, ev: QtGui.QMouseEvent, imgItem: pg.ImageItem=None):
    """
    Construct the current shape ROI depending on mouse movement and current shape parameters
    :param imgItem: Image the ROI is drawn upon, used for mapping event coordinates
      from a scene to pixel coordinates. If *None*, event coordinates are assumed
      to already be relative to pixel coordinates.
    :param ev: Mouse event
    """
    # Unblock on mouse press
    # None imgitem is only the case during programmatic calls so allow this case
    if ((imgItem is None or imgItem.image is not None)
        and ev.type() == ev.MouseButtonPress
        and ev.button() == QtCore.Qt.LeftButton):
      self.removeLock(self)
    if self.locked: return
    if imgItem is not None:
      posRelToImg = imgItem.mapFromScene(ev.pos())
    else:
      posRelToImg = ev.pos()
    # Form of rate-limiting -- only simulate click if the next pixel is at least one away
    # from the previous pixel location
    xyCoord = FRVertices([[posRelToImg.x(), posRelToImg.y()]], dtype=float)
    curRoi = self.curShape
    constructingRoi, self.shapeVerts = self.curShape.updateShape(ev, xyCoord)
    if self.shapeVerts is not None:
      self.sigShapeFinished.emit(self.shapeVerts)

    if not constructingRoi:
      # Vertices from the completed shape are already stored, so clean up the shapes.
      curRoi.hide()
      self.curShape.clear()
    else:
      # Still constructing ROI. Show it
      curRoi.show()

  @property
  def curShapeParam(self): return self._curShape
  @curShapeParam.setter
  def curShapeParam(self, newShape: FRParam):
    """
    When the shape is changed, be sure to reset the underlying ROIs
    :param newShape: New shape
    :return: None
    """
    # Reset the underlying ROIs for a different shape than we currently are using
    if newShape != self._curShape:
      self.clearAllRois()
    self._curShape = newShape

  @property
  def curShape(self):
      return self.roiForShape[self._curShape]