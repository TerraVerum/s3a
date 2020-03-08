import pyqtgraph as pg
from pyqtgraph.GraphicsScene.mouseEvents import MouseDragEvent
from pyqtgraph.Qt import QtCore, QtGui

from ..structures import FRVertices

Signal = QtCore.pyqtSignal

import numpy as np

from .parameditors import FR_SINGLETON
from cdef.projectvars import FR_CONSTS

class ClickableImageItem(pg.ImageItem):
  sigClicked = Signal(object)

  clickable = True
  requireCtrlKey = True

  def mouseClickEvent(self, ev: QtGui.QMouseEvent):
    # Capture clicks only if component is present and user allows it
    # And user pressed control
    keyMods = ev.modifiers()
    if not ev.isAccepted() and ev.button() == QtCore.Qt.LeftButton \
       and self.clickable and self.image is not None \
       and (keyMods == QtCore.Qt.ControlModifier or not self.requireCtrlKey):
      xyCoord = np.round(np.array([[ev.pos().x(), ev.pos().y()]], dtype='int'))
      self.sigClicked.emit(xyCoord)
    super().mouseClickEvent(ev)

class ClickableScatterItem(pg.ScatterPlotItem):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    # TODO: Find out where the mouse is and make sure it's above a point before changing
    # the mouse cursor

    self.hoverCursor = QtCore.Qt.PointingHandCursor

  def mouseMoveEvent(self, ev):
    if self.pointsAt(ev.pos()):
      self.setCursor(self.hoverCursor)
    else:
      self.unsetCursor()

  def idsWithin(self, selection: FRVertices):
    # TODO: Optimize for rectangular selections
    polyPoints = [QtCore.QPointF(*row) for row in selection]
    selectionPoly = QtGui.QPolygonF(polyPoints)
    pointLocs = np.column_stack(self.getData())
    # tfIsInSelection = (pointLocs[0] >= bbox[0]) \
    #   & (pointLocs[0] <= bbox[2]) \
    #   & (pointLocs[1] >= bbox[1]) \
    #   & (pointLocs[1] <= bbox[3])
    tfIsInSelection = np.array([selectionPoly.containsPoint(QtCore.QPointF(*row), QtCore.Qt.WindingFill)
                                for row in pointLocs], dtype=bool)
    return [point.data() for point in self.points()[tfIsInSelection]]

class ClickableTextItem(pg.TextItem):

  @FR_SINGLETON.scheme.registerProp(FR_CONSTS.SCHEME_BOUNDARY_COLOR)
  def boundClr(self): pass
  @FR_SINGLETON.scheme.registerProp(FR_CONSTS.SCHEME_VALID_ID_COLOR)
  def validIdClr(self): pass
  @FR_SINGLETON.scheme.registerProp(FR_CONSTS.SCHEME_NONVALID_ID_COLOR)
  def invalidIdClr(self): pass



  sigClicked = Signal()
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

    self.origCursor = self.cursor()
    self.hoverCursor = QtCore.Qt.PointingHandCursor
    self.setAnchor((0.5,0.5))
    self.setAcceptHoverEvents(True)

  def hoverEnterEvent(self, ev):
    self.setCursor(self.hoverCursor)

  def hoverLeaveEvent(self, ev):
    self.unsetCursor()

  def mousePressEvent(self, ev: QtGui.QMouseEvent):
    self.sigClicked.emit()
    ev.accept()

  def setText(self, newText: str, validated: bool = False):
    """
    Overload setting text to utilize scheme editor
    """
    schemeClrProp = self.invalidIdClr
    if validated:
      schemeClrProp = self.validIdClr

    curFont = self.textItem.font()
    curFont.setPointSize(self.txtSize)
    self.setFont(curFont)

    self.setColor(schemeClrProp)

    super().setText(newText)

  def update(self, newText, newVerts, newValid):
    # Case when verts is empty or all NaN. Assume it is not possible for one vertex to
    # be NaN while the other is a real number
    if np.any(np.isfinite(newVerts)):
      newPos = np.mean(newVerts, axis=0)
    else:
      newPos = np.ones(2)*np.nan
    self.setPos(newPos[0], newPos[1])
    self.setText(newText, newValid)

class RightPanViewBox(pg.ViewBox):
  def mouseDragEvent(self, ev: MouseDragEvent, axis=None):
    if ev.buttons() == QtCore.Qt.RightButton \
        or ev.button() == QtCore.Qt.RightButton:
      ev.buttons = lambda: QtCore.Qt.LeftButton
      ev.button = ev.buttons
    super().mouseDragEvent(ev)