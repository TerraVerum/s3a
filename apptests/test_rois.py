from typing import Tuple

import numpy as np
import pyqtgraph as pg
import pytest
from pyqtgraph.Qt import QtCore, QtGui

from s3a import XYVertices
from s3a.constants import PRJ_CONSTS
from s3a.views.imageareas import MainImage
from s3a.views.rois import SHAPE_ROI_MAPPING

shapes = tuple(SHAPE_ROI_MAPPING.keys())
editableImg = MainImage(
    drawShapes=shapes,
    drawActions=(PRJ_CONSTS.DRAW_ACT_SELECT,),
)
clctn = editableImg.shapeCollection


def leftClick(pt: Tuple[int, int], widget):
    btns = QtCore.Qt.MouseButton
    globalPt = QtCore.QPointF(widget.mapToGlobal(QtCore.QPoint(*pt)))
    event = QtGui.QMouseEvent(
        QtCore.QEvent.Type.MouseButtonPress,
        QtCore.QPoint(*pt),
        globalPt,
        btns.LeftButton,
        btns.LeftButton,
        QtCore.Qt.KeyboardModifier.NoModifier,
    )
    clctn.buildRoi(event)


@pytest.fixture
def mouseDragFactory(qtbot):
    def mouseDrag(widget: MainImage, startPos, endPos):
        gblEndPos = QtCore.QPointF(widget.mapToGlobal(QtCore.QPoint(*endPos)))
        gblStartPos = QtCore.QPointF(widget.mapToGlobal(QtCore.QPoint(*startPos)))

        btns = QtCore.Qt.MouseButton
        startPos = widget.imageItem.mapToScene(pg.Point(startPos))
        endPos = widget.imageItem.mapToScene(pg.Point(endPos))
        press = QtGui.QMouseEvent(
            QtGui.QMouseEvent.Type.MouseButtonPress,
            startPos,
            gblStartPos,
            btns.LeftButton,
            btns.LeftButton,
            QtCore.Qt.KeyboardModifier.NoModifier,
        )
        widget.mousePressEvent(press)

        move = QtGui.QMouseEvent(
            QtCore.QEvent.Type.MouseMove,
            endPos,
            gblEndPos,
            btns.LeftButton,
            btns.LeftButton,
            QtCore.Qt.KeyboardModifier.NoModifier,
        )
        widget.mouseMoveEvent(move)

        release = QtGui.QMouseEvent(
            QtGui.QMouseEvent.Type.MouseButtonRelease,
            endPos,
            gblEndPos,
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
        clctn.shapeParameter = curShape
        leftClick(pt, editableImg)
        assert np.all(pt in clctn.currentShape.vertices)


def test_drag_pt(mouseDragFactory):
    editableImg.drawAction = PRJ_CONSTS.DRAW_ACT_SELECT
    for curShape in shapes:
        clctn.shapeParameter = curShape
        mouseDragFactory(editableImg, (10, 10), (100, 100))
        # Shapes need real mouse events to properly form, so all this can really do is
        # ensure nothing breaks


@pytest.mark.parametrize(
    ["polyparam", "initial", "expectedAfterUndo"],
    ([PRJ_CONSTS.DRAW_SHAPE_POLY, 4, 3], [PRJ_CONSTS.DRAW_SHAPE_FREE, 103, 2]),
)
def test_poly_undo(mouseDragFactory, polyparam, initial, expectedAfterUndo):
    editableImg.drawAction = PRJ_CONSTS.DRAW_ACT_SELECT
    editableImg.shapeCollection.shapeParameter = polyparam
    poly = editableImg.shapeCollection.currentShape
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
