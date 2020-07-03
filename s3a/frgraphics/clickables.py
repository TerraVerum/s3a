import numpy as np
import pyqtgraph as pg
from pyqtgraph.GraphicsScene.mouseEvents import MouseDragEvent
from pyqtgraph.Qt import QtCore, QtGui
from skimage.measure import points_in_poly

from s3a.structures import FRVertices

Signal = QtCore.Signal


class FRBoundScatterPlot(pg.ScatterPlotItem):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    # TODO: Find out where the mouse is and make sure it's above a point before changing
    # the mouse cursor

    self.hoverCursor = QtCore.Qt.PointingHandCursor

  # Not working at the moment :/
  # def mouseMoveEvent(self, ev):
  #   if self.pointsAt(ev.pos()):
  #     self.setCursor(self.hoverCursor)
  #   else:
  #     self.unsetCursor()

  def boundsWithin(self, selection: FRVertices):
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

  def pointsAt(self, pos: QtCore.QPointF):
    """
    The default implementation only checks a square around each spot. However, this is not
    precise enough for my needs. The overloaded pointsAt checks any polygonal area as defined
    by the spot boundary. It also triggers when clicking *inside* the spot boundary,
    which I don't want.
    """
    pts = []
    spots = self.points()
    if len(spots) == 0:
      return spots
    strokerWidth = spots[0].pen().width()
    for spot in spots: # type: pg.SpotItem
      symb = QtGui.QPainterPath(spot.symbol())
      symb.translate(spot.pos())
      stroker = QtGui.QPainterPathStroker()
      stroker.setWidth(strokerWidth)
      mousePath = stroker.createStroke(symb)
      # Only trigger when clicking a boundary, not the inside of the shape
      if mousePath.contains(pos):
        pts.append(spot)
    return pts[::-1]

  def measureSpotSizes(self, dataSet):
    """
    Override the method so that it takes symbol size into account
    """
    for rec in dataSet:
      ## keep track of the maximum spot size and pixel size
      symbol, size, pen, brush = self.getSpotOpts(rec)
      br = symbol.boundingRect()
      size = max(br.width(), br.height())*2
      width = 0
      pxWidth = 0
      if self.opts['pxMode']:
        pxWidth = size + pen.widthF()
      else:
        width = size
        if pen.isCosmetic():
          pxWidth += pen.widthF()
        else:
          width += pen.widthF()
      self._maxSpotWidth = max(self._maxSpotWidth, width)
      self._maxSpotPxWidth = max(self._maxSpotPxWidth, pxWidth)
    self.bounds = [None, None]

class FRRightPanViewBox(pg.ViewBox):
  def mouseDragEvent(self, ev: MouseDragEvent, axis=None):
    if ev.buttons() == QtCore.Qt.RightButton \
        or ev.button() == QtCore.Qt.RightButton:
      ev.buttons = lambda: QtCore.Qt.LeftButton
      ev.button = ev.buttons
    super().mouseDragEvent(ev)