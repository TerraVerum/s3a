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


def test_roi_vert_drag():
  # For now just make sure no errors occur when dragging one vertex, since
  # this is a legal and expected op for every one
  pt = (0,0)
  clctn.forceBlockRois = False
  for curShape in shapes:
    clctn.curShapeParam = curShape
    leftClick(pt)
    assert np.all(pt in clctn.curShape.vertices)