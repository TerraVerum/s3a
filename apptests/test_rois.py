from typing import Tuple

import numpy as np
import pyqtgraph as pg
import pytest
from pyqtgraph.Qt import QtCore, QtGui
from utilitys import EditorPropsMixin

from s3a import XYVertices
from s3a.constants import PRJ_CONSTS
from s3a.shared import SharedAppSettings
from s3a.views.imageareas import MainImage
from s3a.views.rois import SHAPE_ROI_MAPPING

shapes = tuple(SHAPE_ROI_MAPPING.keys())
with EditorPropsMixin.setEditorPropertyOpts(shared=SharedAppSettings()):
    editableImg = MainImage(
        drawShapes=shapes,
        drawActions=(PRJ_CONSTS.DRAW_ACT_SELECT,),
    )
clctn = editableImg.shapeCollection


def leftClick(pt: Tuple[int, int]):
    btns = QtCore.Qt.MouseButton
    event = QtGui.QMouseEvent(
        QtCore.QEvent.Type.MouseButtonPress,
        QtCore.QPoint(*pt),
        btns.LeftButton,
        btns.LeftButton,
        QtCore.Qt.KeyboardModifier.NoModifier,
    )
    clctn.buildRoi(event)


@pytest.fixture
def mouseDragFactory(qtbot):
    def mouseDrag(widget: MainImage, startPos, endPos):
        btns = QtCore.Qt.MouseButton
        startPos = widget.imgItem.mapToScene(pg.Point(startPos))
        endPos = widget.imgItem.mapToScene(pg.Point(endPos))
        press = QtGui.QMouseEvent(
            QtGui.QMouseEvent.Type.MouseButtonPress,
            startPos,
            btns.LeftButton,
            btns.LeftButton,
            QtCore.Qt.KeyboardModifier.NoModifier,
        )
        widget.mousePressEvent(press)

        move = QtGui.QMouseEvent(
            QtCore.QEvent.Type.MouseMove,
            endPos,
            btns.LeftButton,
            btns.LeftButton,
            QtCore.Qt.KeyboardModifier.NoModifier,
        )
        widget.mouseMoveEvent(move)

        release = QtGui.QMouseEvent(
            QtGui.QMouseEvent.Type.MouseButtonRelease,
            endPos,
            btns.LeftButton,
            btns.LeftButton,
            QtCore.Qt.KeyboardModifier.NoModifier,
        )
        widget.mouseReleaseEvent(release)

    return mouseDrag


def test_simple_click():
    # For now just make sure no errors occur when dragging one vertex, since
    # this is a legal and expected op for every one
    editableImg.shapeCollection.forceUnlock()
    pt = (0, 0)
    editableImg.drawAction = PRJ_CONSTS.DRAW_ACT_SELECT
    for curShape in shapes:
        clctn.curShapeParam = curShape
        leftClick(pt)
        assert np.all(pt in clctn.curShape.vertices)


def test_drag_pt(mouseDragFactory):
    editableImg.drawAction = PRJ_CONSTS.DRAW_ACT_SELECT
    for curShape in shapes:
        clctn.curShapeParam = curShape
        mouseDragFactory(editableImg, (10, 10), (100, 100))
        # Shapes need real mouse events to properly form, so all this can really do is
        # ensure nothing breaks


@pytest.mark.parametrize(
    ["polyparam", "initial", "expectedAfterUndo"],
    ([PRJ_CONSTS.DRAW_SHAPE_POLY, 4, 3], [PRJ_CONSTS.DRAW_SHAPE_FREE, 103, 2]),
)
def test_poly_undo(mouseDragFactory, polyparam, initial, expectedAfterUndo):
    editableImg.drawAction = PRJ_CONSTS.DRAW_ACT_SELECT
    editableImg.shapeCollection.curShapeParam = polyparam
    poly = editableImg.shapeCollection.curShape
    poly.lastEvType = QtCore.QEvent.MouseButtonPress
    poly.setRoiPoints()
    poly.addRoiPoints(XYVertices([[0, 0], [10, 0]]))
    poly.addRoiPoints(XYVertices([[10, 10]]))
    poly.lastEvType = QtCore.QEvent.MouseMove
    for pt in np.c_[np.arange(100), np.arange(100)]:
        if np.array_equal(pt, [99, 99]) and polyparam is PRJ_CONSTS.DRAW_SHAPE_POLY:
            # Last point; simulate mouse release
            poly.lastEvType = QtCore.QEvent.MouseButtonPress
        poly.addRoiPoints(pt.reshape(1, -1))
    assert len(poly.vertices) == initial
    poly.actionStack.undo()
    assert len(poly.vertices) == expectedAfterUndo
    poly.actionStack.redo()
    assert len(poly.vertices) == initial
