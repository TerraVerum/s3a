import numpy as np
import pytest
from pyqtgraph.Qt import QtCore, QtGui

from apptests.testingconsts import RND
from s3a import REQD_TBL_FIELDS
from s3a.constants import PRJ_CONSTS
from s3a.controls.drawctrl import RoiCollection
from s3a.parameditors.algcollection import AlgorithmCollection, AlgorithmEditor
from s3a.processing import ImagePipeline
from s3a.structures import NChanImg, OptionsDict, XYVertices


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
        (PRJ_CONSTS.DRAW_SHAPE_POLY, PRJ_CONSTS.DRAW_SHAPE_RECT), app.mainImage
    )

    def _polyRoi(pts: XYVertices, shape: OptionsDict = PRJ_CONSTS.DRAW_SHAPE_RECT):
        clctn.shapeParameter = shape
        for pt in pts:
            ev = leftClickGen(pt)
            clctn.buildRoi(ev)
        return clctn.currentShape

    return _polyRoi


def test_update(app, mgr, vertsPlugin, sampleComps):
    # Disable region simplification for accurate testing
    vertsPlugin.props[PRJ_CONSTS.PROP_REG_APPROX_EPS] = -1
    tblModel = app.componentManager
    newCompSer = sampleComps.iloc[-1].copy()

    masks = RND.random((3, 500, 500)) > 0.5

    def regionCmp(mask):
        assert np.array_equal(mask > 0, vertsPlugin.region.toGrayImage(mask.shape) > 0)

    changeDict = app.addAndFocusComponents(newCompSer.to_frame().T)
    newCompSer[REQD_TBL_FIELDS.ID] = changeDict["added"][0]
    assert tblModel.focusedComponent.equals(newCompSer)
    # Add two updates so one is undoable and still comparable
    for ii in range(2):
        vertsPlugin.updateRegionFromMask(masks[ii])
        regionCmp(masks[ii])

    newerSer = sampleComps.iloc[0].copy()
    changeDict = app.addAndFocusComponents(newerSer.to_frame().T)
    newerSer[REQD_TBL_FIELDS.ID] = changeDict["added"][0]
    oldMask = vertsPlugin.region.toGrayImage()
    vertsPlugin.updateRegionFromMask(masks[2])
    regionCmp(masks[2])

    # Test undos for comp change and non-comp changes
    vertsPlugin.actionStack.undo()
    regionCmp(oldMask)
    assert tblModel.focusedComponent.equals(newerSer)

    vertsPlugin.actionStack.undo()
    assert (
        tblModel.focusedComponent[REQD_TBL_FIELDS.ID] == newCompSer[REQD_TBL_FIELDS.ID]
    )
    # Undo create new comp, nothing is selected

    app.actionStack.undo()
    # Undo earlier region edit, region should now equal the second-to-last edit
    regionCmp(masks[0])
    assert (
        tblModel.focusedComponent[REQD_TBL_FIELDS.ID] == newCompSer[REQD_TBL_FIELDS.ID]
    )

    app.actionStack.redo()
    regionCmp(masks[1])
    app.actionStack.redo()
    app.actionStack.redo()
    regionCmp(masks[2])
    assert tblModel.focusedComponent.equals(newerSer)
    assert not app.actionStack.canRedo


def test_region_modify(sampleComps, app, mgr, vertsPlugin):
    # Disable region simplification for accurate testing
    vertsPlugin.props[PRJ_CONSTS.PROP_REG_APPROX_EPS] = -1
    vertsPlugin.processEditor.changeActiveProcessor("Basic Shapes")
    mImg = app.mainImage
    app.addAndFocusComponents(sampleComps)
    shapeBnds = mImg.image.shape[:2]
    reach = np.min(shapeBnds)
    oldData = vertsPlugin.region.regionData.copy()
    mImg.shapeCollection.shapeParameter = PRJ_CONSTS.DRAW_SHAPE_POLY
    mImg.drawAction = PRJ_CONSTS.DRAW_ACT_CREATE
    imsum = lambda: vertsPlugin.region.toGrayImage(shapeBnds).sum()

    # 1st action
    app.componentManager.updateFocusedComponent(None)
    assert imsum() == 0

    newVerts = XYVertices([[5, 5], [reach, reach], [reach, 5], [5, 5]])

    # 2nd action
    app.componentManager.updateFocusedComponent(sampleComps.iloc[-1])
    mImg.shapeCollection.sigShapeFinished.emit(newVerts)
    checkpointMask = vertsPlugin.region.toGrayImage(shapeBnds)
    assert np.any(checkpointMask)

    mImg.drawAction = PRJ_CONSTS.DRAW_ACT_ADD

    app.actionStack.undo()
    # app.actionStack.undo()
    # Cmp to first action
    assert imsum() == 0
    app.actionStack.undo()
    # Cmp to original
    assert vertsPlugin.region.regionData[REQD_TBL_FIELDS.VERTICES].equals(
        oldData[REQD_TBL_FIELDS.VERTICES]
    )

    app.actionStack.redo()
    assert imsum() == 0
    app.actionStack.redo()
    assert imsum() == checkpointMask.sum()


@pytest.mark.withcomps
def test_selectionbounds_all(app, mgr):
    imBounds = app.mainImage.image.shape[:2][::-1]
    bounds = XYVertices(
        [[0, 0], [0, imBounds[1]], [imBounds[0], imBounds[1]], [imBounds[0], 0]]
    )
    app.componentController.reflectSelectionBoundsMade(bounds)
    assert len(app.componentController.selectedIds) == len(mgr.compDf)


@pytest.mark.withcomps
def test_selectionbounds_none(app):
    app.tableView.clearSelection()
    app.componentController.selectedIds = np.array([], dtype=int)
    # Selection in negative area ensures no components will be selected
    app.componentController.reflectSelectionBoundsMade(XYVertices([[-100, -100]]))
    assert len(app.componentController.selectedIds) == 0


def test_proc_err(tmp_path):
    def badProc(image: NChanImg):
        return dict(image=image, extra=1 / 0)

    proc = ImagePipeline(name="Bad")
    proc.addStage(badProc)
    clctn = AlgorithmCollection(ImagePipeline)
    algEditor = AlgorithmEditor(clctn, directory=tmp_path)
    clctn.addProcess(proc, top=True)

    algEditor.changeActiveProcessor("Bad")
    kwargs = dict(
        image=np.array([[True]], dtype=bool), foregroundVertices=XYVertices([[0, 0]])
    )
    with pytest.raises(ZeroDivisionError):
        algEditor.currentProcessor.activate(**kwargs)
