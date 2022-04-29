import numpy as np
import pytest
from pyqtgraph.Qt import QtGui, QtCore

from apptests.testingconsts import RND
from testingconsts import NUM_COMPS
from s3a import REQD_TBL_FIELDS
from s3a.constants import PRJ_CONSTS
from s3a.controls.drawctrl import RoiCollection
from s3a.parameditors.algcollection import AlgParamEditor, AlgCollection
from s3a.processing import ProcessIO, ImageProcess, ImgProcWrapper
from s3a.structures import XYVertices, ComplexXYVertices, PrjParam, NChanImg


def leftClickGen(pos: XYVertices, dbclick=False):
    Ev = QtCore.QEvent
    Qt = QtCore.Qt
    if dbclick:
        typ = Ev.MouseButtonDblClick
    else:
        typ = Ev.MouseButtonPress
    pos = QtCore.QPointF(*pos.flatten())
    out = QtGui.QMouseEvent(typ, pos, Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)
    return out


@pytest.fixture
def roiFactory(app):
    clctn = RoiCollection(
        (PRJ_CONSTS.DRAW_SHAPE_POLY, PRJ_CONSTS.DRAW_SHAPE_RECT), app.mainImg
    )

    def _polyRoi(pts: XYVertices, shape: PrjParam = PRJ_CONSTS.DRAW_SHAPE_RECT):
        clctn.curShapeParam = shape
        for pt in pts:
            ev = leftClickGen(pt)
            clctn.buildRoi(ev)
        return clctn.curShape

    return _polyRoi


def test_update(app, mgr, vertsPlugin, sampleComps):
    # Disable region simplification for accurate testing
    vertsPlugin.props[PRJ_CONSTS.PROP_REG_APPROX_EPS] = -1
    mImg = app.mainImg
    newCompSer = sampleComps.iloc[-1].copy()

    masks = RND.random((3, 500, 500)) > 0.5

    def regionCmp(mask):
        assert np.array_equal(mask > 0, vertsPlugin.region.toGrayImg(mask.shape) > 0)

    changeDict = app.addAndFocusComps(newCompSer.to_frame().T)
    newCompSer[REQD_TBL_FIELDS.INST_ID] = changeDict["added"][0]
    assert mImg.compSer.equals(newCompSer)
    # Add two updates so one is undoable and still comparable
    for ii in range(2):
        vertsPlugin.updateRegionFromMask(masks[ii])
        regionCmp(masks[ii])

    newerSer = sampleComps.iloc[0].copy()
    changeDict = app.addAndFocusComps(newerSer.to_frame().T)
    newerSer[REQD_TBL_FIELDS.INST_ID] = changeDict["added"][0]
    oldMask = vertsPlugin.region.toGrayImg()
    vertsPlugin.updateRegionFromMask(masks[2])
    regionCmp(masks[2])

    # Test undos for comp change and non-comp changes
    vertsPlugin.actionStack.undo()
    regionCmp(oldMask)
    assert mImg.compSer.equals(newerSer)

    vertsPlugin.actionStack.undo()
    assert mImg.compSer[REQD_TBL_FIELDS.INST_ID] == newCompSer[REQD_TBL_FIELDS.INST_ID]
    # Undo create new comp, nothing is selected

    app.sharedAttrs.actionStack.undo()
    # Undo earlier region edit, region should now equal the second-to-last edit
    regionCmp(masks[0])
    assert mImg.compSer[REQD_TBL_FIELDS.INST_ID] == newCompSer[REQD_TBL_FIELDS.INST_ID]

    app.sharedAttrs.actionStack.redo()
    regionCmp(masks[1])
    app.sharedAttrs.actionStack.redo()
    app.sharedAttrs.actionStack.redo()
    regionCmp(masks[2])
    assert mImg.compSer.equals(newerSer)
    assert not app.sharedAttrs.actionStack.canRedo


def test_region_modify(sampleComps, app, mgr, vertsPlugin):
    # Disable region simplification for accurate testing
    vertsPlugin.props[PRJ_CONSTS.PROP_REG_APPROX_EPS] = -1
    vertsPlugin.procEditor.changeActiveProcessor("Basic Shapes")
    mImg = app.mainImg
    app.addAndFocusComps(sampleComps)
    shapeBnds = mImg.image.shape[:2]
    reach = np.min(shapeBnds)
    oldData = vertsPlugin.region.regionData.copy()
    mImg.shapeCollection.curShapeParam = PRJ_CONSTS.DRAW_SHAPE_POLY
    mImg.drawAction = PRJ_CONSTS.DRAW_ACT_CREATE
    imsum = lambda: vertsPlugin.region.toGrayImg(shapeBnds).sum()

    # 1st action
    app.mainImg.updateFocusedComp(None)
    assert imsum() == 0

    newVerts = XYVertices([[5, 5], [reach, reach], [reach, 5], [5, 5]])

    # 2nd action
    app.mainImg.updateFocusedComp(sampleComps.iloc[-1])
    mImg.shapeCollection.sigShapeFinished.emit(newVerts)
    checkpointMask = vertsPlugin.region.toGrayImg(shapeBnds)
    assert np.any(checkpointMask)

    mImg.drawAction = PRJ_CONSTS.DRAW_ACT_ADD

    app.sharedAttrs.actionStack.undo()
    # app.sharedAttrs.actionStack.undo()
    # Cmp to first action
    assert imsum() == 0
    app.sharedAttrs.actionStack.undo()
    # Cmp to original
    assert vertsPlugin.region.regionData[REQD_TBL_FIELDS.VERTICES].equals(
        oldData[REQD_TBL_FIELDS.VERTICES]
    )

    app.sharedAttrs.actionStack.redo()
    assert imsum() == 0
    app.sharedAttrs.actionStack.redo()
    assert imsum() == checkpointMask.sum()


@pytest.mark.withcomps
def test_selectionbounds_all(app, mgr):
    imBounds = app.mainImg.image.shape[:2][::-1]
    bounds = XYVertices(
        [[0, 0], [0, imBounds[1]], [imBounds[0], imBounds[1]], [imBounds[0], 0]]
    )
    app.compDisplay.reflectSelectionBoundsMade(bounds)
    assert len(app.compDisplay.selectedIds) == len(mgr.compDf)


@pytest.mark.withcomps
def test_selectionbounds_none(app):
    app.compTbl.clearSelection()
    app.compDisplay.selectedIds = np.array([], dtype=int)
    # Selection in negative area ensures no comps will be selected
    app.compDisplay.reflectSelectionBoundsMade(XYVertices([[-100, -100]]))
    assert len(app.compDisplay.selectedIds) == 0


def test_proc_err(tmp_path):
    def badProc(image: NChanImg):
        return ProcessIO(image=image, extra=1 / 0)

    proc = ImageProcess.fromFunction(badProc, name="Bad")
    clctn = AlgCollection(ImgProcWrapper, ImageProcess)
    algEditor = AlgParamEditor(clctn, saveDir=tmp_path)
    clctn.addProcess(proc, top=True)

    algEditor.changeActiveProcessor("Bad")
    kwargs = dict(image=np.array([[True]], dtype=bool), fgVerts=XYVertices([[0, 0]]))
    with pytest.warns(UserWarning):
        algEditor.curProcessor.run(errorsToWarnings=True, **kwargs)
    with pytest.raises(ZeroDivisionError):
        algEditor.curProcessor.run(**kwargs)
