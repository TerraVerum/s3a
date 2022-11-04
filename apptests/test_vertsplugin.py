"""Most other functionality is tested under procimpls, this is for other registered functions"""
import numpy as np
import pytest
from skimage import draw

from apptests.testingconsts import SAMPLE_SMALL_IMG
from s3a import PRJ_CONSTS, REQD_TBL_FIELDS, XYVertices
from s3a.generalutils import imageCornerVertices
from s3a.plugins.file import NewProjectWizard
from s3a.processing.algorithms import imageproc


@pytest.mark.withcomps
@pytest.mark.smallimage
def test_registered_verts_funcs(vertsPlugin, app):
    # Disable region simplification for accurate testing
    vertsPlugin.props[PRJ_CONSTS.PROP_REG_APPROX_EPS] = -1
    togray = lambda: vertsPlugin.region.toGrayImage(SAMPLE_SMALL_IMG.shape)
    imsize = np.prod(SAMPLE_SMALL_IMG.shape[:2])
    editor = vertsPlugin.processEditor
    editor.changeActiveProcessor("Basic Shapes")
    foregroundVertices = XYVertices([[0, 0], [10, 0], [10, 10]])
    vertsPlugin.run(foregroundVertices=foregroundVertices, updateGui=True)
    assert imageproc.procCache["mask"].sum()

    vertsPlugin.clearProcessorHistory()
    assert imageproc.procCache["mask"].sum() == 0

    vertsPlugin.fillRegionMask()
    img = togray()
    assert (img > 0).sum() == imsize

    vertsPlugin.clearFocusedRegion()
    assert togray().sum() == 0

    vertsPlugin.actionStack.undo()
    assert (img > 0).sum() == imsize

    app.changeFocusedComponent(app.componentManager.compDf.index[0])
    img = togray()
    vertsPlugin.clearFocusedRegion()
    vertsPlugin.resetFocusedRegion()
    assert np.array_equal(togray(), img)

    coords = draw.disk((0, 0), 50, shape=(100, 100))
    img = np.zeros((100, 100), bool)
    img[coords] = True
    vertsPlugin.updateRegionFromMask(img)
    # Reassign image so the size is right
    img = togray() > 0
    vertsPlugin.invertRegion()
    assert np.all((togray() > 0) == (~img))

    # No sensible inverse of empty image
    vertsPlugin.clearFocusedRegion()
    assert (togray() > 0).sum() == 0


def test_region_offset(vertsPlugin, sampleComps):
    vertsPlugin.updateRegionFromDf(
        sampleComps.iloc[[0]], offset=XYVertices([10000, 10000])
    )
    vMax = vertsPlugin.region.regionData[REQD_TBL_FIELDS.VERTICES].s3averts.max()
    assert max(vMax) > max(sampleComps[REQD_TBL_FIELDS.VERTICES].s3averts.max())


@pytest.mark.withcomps
def test_accept_region(app, vertsPlugin):
    comp = app.componentManager.compDf.iloc[[0]]
    app.changeFocusedComponent(comp.index)
    vertsPlugin.clearFocusedRegion()

    app.acceptFocusedRegion()
    assert comp.index[0] not in app.componentManager.compDf.index

    comp = app.componentManager.compDf.iloc[[0]]
    app.changeFocusedComponent(comp.index)
    vertsPlugin.fillRegionMask()
    verts = vertsPlugin.region.regionData[REQD_TBL_FIELDS.VERTICES]
    app.acceptFocusedRegion()
    assert np.array_equal(
        verts, app.componentManager.compDf[REQD_TBL_FIELDS.VERTICES][comp.index]
    )


@pytest.mark.smallimage
@pytest.mark.withcomps
def test_region_history(vertsPlugin, app, monkeypatch):
    comp = app.componentManager.compDf.iloc[[0]]
    app.changeFocusedComponent(comp.index)

    vertsPlugin.run(
        backgroundVertices=imageCornerVertices(SAMPLE_SMALL_IMG), updateGui=True
    )
    vertsPlugin.run(updateGui=True)

    initial, history = vertsPlugin.getRegionHistory()
    assert np.array_equal(initial, SAMPLE_SMALL_IMG)
    assert len(history)
    assert np.array_equal(
        history[-1], vertsPlugin.region.toGrayImage(SAMPLE_SMALL_IMG.shape) > 0
    )

    with monkeypatch.context() as m:
        m.setattr(vertsPlugin.playbackWindow, "show", lambda *args: None)
        m.setattr(vertsPlugin.playbackWindow, "raise_", lambda *args: None)
        vertsPlugin.playbackRegionHistory()
        winImg = vertsPlugin.playbackWindow.displayPlot.imageItem.image
        assert np.array_equal(winImg, initial)


def test_proj_wizard(filePlugin):
    # Much is gui, just make sure mechanics work
    npw = NewProjectWizard(filePlugin)
    for fileLst in npw.fileLists.values():
        assert not fileLst.files
