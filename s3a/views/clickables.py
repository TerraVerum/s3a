from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from pyqtgraph.GraphicsScene.mouseEvents import MouseDragEvent
from pyqtgraph.Qt import QtCore, QtGui
from skimage.measure import points_in_poly

from s3a.structures import XYVertices

Signal = QtCore.Signal

__all__ = ['BoundScatterPlot', 'RightPanViewBox']

class BoundScatterPlot(pg.ScatterPlotItem):
  def __init__(self, *args, boundaryOnly=False, **kwargs):
    super().__init__(*args, **kwargs)
    # TODO: Find out where the mouse is and make sure it's above a point before changing
    # the mouse cursor

    self.hoverCursor = QtCore.Qt.CursorShape.PointingHandCursor
    self.boundaryOnly = boundaryOnly

  # Not working at the moment :/
  # def mouseMoveEvent(self, ev):
  #   if self.pointsAt(ev.pos()):
  #     self.setCursor(self.hoverCursor)
  #   else:
  #     self.unsetCursor()

  def boundsWithin(self, selection: XYVertices):
    # TODO: Optimize for rectangular selections
    # polyPoints = [QtCore.QPointF(*row) for row in selection]
    # selectionPoly = QtGui.QPolygonF(polyPoints)
    if self.data['x'] is None:
      return np.array([])
    pointLocs = np.column_stack(self.getData())
    # tfIsInSelection = (pointLocs[0] >= bbox[0]) \
    #   & (pointLocs[0] <= bbox[2]) \
    #   & (pointLocs[1] >= bbox[1]) \
    #   & (pointLocs[1] <= bbox[3])
    tfIsInSelection = points_in_poly(pointLocs, selection)
    # tfIsInSelection = np.array([selectionPoly.containsPoint(QtCore.QPointF(*row), QtCore.Qt.WindingFill)
    #                             for row in pointLocs], dtype=bool)
    return np.array([point.data() for point in self.points()[tfIsInSelection]])

  def _maskAt(self, pos: QtCore.QPointF | QtCore.QRectF):
    """
    The default implementation only checks a square around each spot. However, this is not
    precise enough for my needs. The overloaded pointsAt checks any polygonal area as defined
    by the spot boundary. It also triggers when clicking *inside* the spot boundary,
    which I don't want.

    :param pos: Where to search for points
    """
    if not isinstance(pos, QtCore.QPointF):
      return super()._maskAt(pos)
    spots = self.points()
    # super() impl. finds ponts within the square, which is a subset of exact
    # behavior. In other words, if super() does *not* detect a point, there's no
    # way it is selected. So, only carefully check points whose rect is
    # in containable range
    # However, this will only work when proper sizing is implemented for
    # potentially wonky shapes
    # spotsAtPos = super()._maskAt(pos)
    spotsAtPos = np.ones(len(spots), bool)
    checkIdxs = np.flatnonzero(spotsAtPos)
    if not len(checkIdxs):
      return spotsAtPos
    strokerWidth = spots[0].pen().width()
    for ii in checkIdxs: # type: pg.SpotItem
      spot = spots[ii]
      symb = QtGui.QPainterPath(spot.symbol())
      symb.translate(spot.pos())
      stroker = QtGui.QPainterPathStroker()
      stroker.setWidth(strokerWidth)
      checkSymb = symb
      if self.boundaryOnly:
        mousePath = stroker.createStroke(symb)
        checkSymb = mousePath
      # Only trigger when clicking a boundary, not the inside of the shape
      spotsAtPos[ii] = checkSymb.contains(pos)
    return spotsAtPos

class RightPanViewBox(pg.ViewBox):
  def mouseDragEvent(self, ev: MouseDragEvent, axis=None):
    btns = QtCore.Qt.MouseButton
    if ev.buttons() == btns.RightButton \
        or ev.button() == btns.RightButton:
      ev.buttons = lambda: btns.LeftButton
      ev.button = ev.buttons
    super().mouseDragEvent(ev)
