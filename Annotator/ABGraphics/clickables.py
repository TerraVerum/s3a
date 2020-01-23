import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

Signal = QtCore.pyqtSignal

import numpy as np
import warnings

from .parameditors import SCHEME_HOLDER
from ..constants import TEMPLATE_SCHEME_VALUES as SV
from .. import appInst


class ClickableImageItem(pg.ImageItem):
  sigClicked = Signal(object)

  clickable = True
  requireCtrlKey = True

  def mouseClickEvent(self, ev: QtGui.QMouseEvent):
    # Capture clicks only if component is present and user allows it
    # And user pressed control
    keyMods = appInst.keyboardModifiers()
    if not ev.isAccepted() and ev.button() == QtCore.Qt.LeftButton \
       and self.clickable and self.image is not None \
       and (keyMods == QtCore.Qt.ControlModifier or not self.requireCtrlKey):
      xyCoord = np.round(np.array([[ev.pos().x(), ev.pos().y()]], dtype='int'))
      self.sigClicked.emit(xyCoord)

class ClickableScatterItem(pg.ScatterPlotItem):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    # TODO: Find out where the mouse is and make sure it's above a point before changing
    # the mouse cursor
    #self.setAcceptHoverEvents(True)

    self.hoverCursor = QtCore.Qt.PointingHandCursor


  def hoverEnterEvent(self, ev):
    self.setCursor(self.hoverCursor)

  def hoverLeaveEvent(self, ev):
    self.unsetCursor()

# noinspection PyUnusedLocal
class ClickableTextItem(pg.TextItem):
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
    schemeClrProp = SV.NONVALID_ID_COLOR
    if validated:
      schemeClrProp = SV.VALID_ID_COLOR
    txtSize, txtClr = SCHEME_HOLDER.scheme.getCompProps(
        (SV.ID_FONT_SIZE, schemeClrProp))

    curFont = self.textItem.font()
    curFont.setPointSize(txtSize)
    self.setFont(curFont)

    self.setColor(txtClr)

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