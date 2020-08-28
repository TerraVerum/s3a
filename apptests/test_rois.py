import pytest
from typing import Tuple

from pyqtgraph.Qt import QtWidgets, QtCore, QtGui
import numpy as np

from s3a.views.imageareas import FREditableImgBase
from s3a.views.rois import SHAPE_ROI_MAPPING
from s3a.constants import FR_CONSTS

shapes = tuple(SHAPE_ROI_MAPPING.keys())
editableImg = FREditableImgBase(drawShapes=shapes,
                                drawActions=(FR_CONSTS.DRAW_ACT_SELECT,))
clctn = editableImg.shapeCollection

def leftClick(pt: Tuple[int, int]):
  event = QtGui.QMouseEvent(
    QtCore.QEvent.MouseButtonPress,
    QtCore.QPoint(*pt),
    QtCore.Qt.LeftButton,
    QtCore.Qt.LeftButton,
    QtCore.Qt.NoModifier,
  )
  clctn.buildRoi(event)

@pytest.fixture
def mouseDragFactory(qtbot):
  def mouseDrag(widget: FREditableImgBase, startPos, endPos):
    startPos = widget.imgItem.mapToScene(startPos)
    endPos = widget.imgItem.mapToScene(endPos)
    press = QtGui.QMouseEvent(QtGui.QMouseEvent.MouseButtonPress,
                              startPos, QtCore.Qt.LeftButton, QtCore.Qt.LeftButton,
                              QtCore.Qt.NoModifier)
    widget.mousePressEvent(press)

    move = QtGui.QMouseEvent(QtCore.QEvent.MouseMove,
                             endPos, QtCore.Qt.LeftButton, QtCore.Qt.LeftButton,
                             QtCore.Qt.NoModifier,)
    widget.mouseMoveEvent(move)

    release = QtGui.QMouseEvent(QtGui.QMouseEvent.MouseButtonRelease,
                                endPos, QtCore.Qt.LeftButton, QtCore.Qt.LeftButton,
                                QtCore.Qt.NoModifier)
    widget.mouseReleaseEvent(release)
  return mouseDrag

def test_simple_click():
  # For now just make sure no errors occur when dragging one vertex, since
  # this is a legal and expected op for every one
  pt = (0,0)
  clctn.forceBlockRois = False
  editableImg.drawAction = FR_CONSTS.DRAW_ACT_SELECT
  for curShape in shapes:
    clctn.curShapeParam = curShape
    leftClick(pt)
    assert np.all(pt in clctn.curShape.vertices)

def test_drag_pt(mouseDragFactory):
  editableImg.drawAction = FR_CONSTS.DRAW_ACT_SELECT
  for curShape in shapes:
    clctn.curShapeParam = curShape
    clctn.forceBlockRois = False
    mouseDragFactory(editableImg, QtCore.QPoint(10,10), QtCore.QPoint(100,100))
    # Shapes need real mouse events to properly form, so all this can really do is
    # ensure nothing breaks