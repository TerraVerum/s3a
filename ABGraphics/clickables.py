import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
Signal = QtCore.pyqtSignal

import numpy as np
import warnings

from ABGraphics.parameditors import SchemeEditor
from constants import SchemeValues as SV


class ClickableImageItem(pg.ImageItem):
  sigClicked = Signal(object)

  clickable = True

  def mouseClickEvent(self, ev):
    # Capture clicks only if component is present and user allows it
    if ev.button() == QtCore.Qt.LeftButton \
       and self.clickable and self.image is not None:
      xyCoord = np.round(np.array([[ev.pos().x(), ev.pos().y()]], dtype='int'))
      self.sigClicked.emit(xyCoord)

class ClickableTextItem(pg.TextItem):
  sigClicked = Signal()

  scheme = SchemeEditor()

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.origCursor = self.cursor()
    self.hoverCursor = QtCore.Qt.PointingHandCursor
    self.setAnchor((0.5,0.5))
    self.setAcceptHoverEvents(True)

  def hoverEnterEvent(self, ev):
    self.setCursor(self.hoverCursor)

  def hoverLeaveEvent(self, ev):
    #self.setCursor(self.origCursor)
    self.unsetCursor()

  def mousePressEvent(self, ev: QtGui.QMouseEvent):
    self.sigClicked.emit()
    ev.accept()

  def setText(self, newText: str, validated: bool = False):
    '''
    Overload setting text to utilize scheme editor
    '''
    schemeClrProp = SV.NONVALID_ID_COLOR
    if validated:
      schemeClrProp = SV.VALID_ID_COLOR
    txtSize, txtClr = ClickableTextItem.scheme.getCompProps(
        (SV.ID_FONT_SIZE, schemeClrProp))

    curFont = self.textItem.font()
    curFont.setPointSize(txtSize)
    self.setFont(curFont)

    self.setColor(txtClr)

    super().setText(newText)

  def update(self, newText, newVerts, newValid):
    # It is OK for NaN mean values, since this will hide the text
    with warnings.catch_warnings():
      warnings.simplefilter("ignore", category=RuntimeWarning)
      newPos = np.mean(newVerts, axis=0)
      self.setPos(newPos[0], newPos[1])
    self.setText(newText, newValid)

  @staticmethod
  def setScheme(scheme):
    ClickableTextItem.scheme = scheme