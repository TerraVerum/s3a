from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from apptests.conftest import NUM_COMPS, dfTester
from apptests.testingconsts import RND, SAMPLE_IMG, SAMPLE_IMG_FNAME
from s3a import mkQApp
from s3a.constants import LAYOUTS_DIR, PRJ_CONSTS as CNST, REQD_TBL_FIELDS
from s3a.plugins.mainimage import MainImagePlugin
from s3a.structures import ComplexXYVertices, XYVertices


def test_change_img(app):
    im2 = RND.integers(0, 255, SAMPLE_IMG.shape, "uint8")
    name = Path("./testfile.png").absolute()
    app.setMainImage(name, im2)
    assert (
        name == app.sourceImagePath
    ), "Annotation source not set after loading image on start"

    np.testing.assert_array_equal(
        app.mainImage.image, im2, "Main image doesn't match sample image"
    )


def test_change_img_none(app):
    app.setMainImage()
    assert app.mainImage.image is None
    assert app.sourceImagePath is None


@pytest.mark.withcomps
def test_clear_bounds(app, vertsPlugin):
    assert len(app.componentManager.compDf) > 0
    app.clearBoundaries()
    assert (
        len(app.componentManager.compDf) == 0
    ), "Comps not cleared after clearing boundaries"


@pytest.mark.withcomps
def test_export_all_comps(tmp_path, app):
    compFile = tmp_path / "tmp.csv"
    app.exportCurrentAnnotation(str(compFile))
    assert compFile.exists(), "All-comp export didn't produce a component list"


def test_load_comps_merge(tmp_path, app, sampleComps):
    compFile = tmp_path / "tmp.csv"

    app.componentManager.addComponents(sampleComps)
    app.exportCurrentAnnotation(str(compFile))
    app.clearBoundaries()

    app.openAnnotations(str(compFile))
    equalCols = np.setdiff1d(
        dfTester.compDf.columns, [REQD_TBL_FIELDS.ID, REQD_TBL_FIELDS.IMAGE_FILE]
    )
    dfCmp = (
        app.componentManager.compDf[equalCols].values
        == dfTester.compDf[equalCols].values
    )
    assert np.all(dfCmp), "Loaded dataframe doesn't match daved dataframe"


def test_import_large_verts(sampleComps, tmp_path, app):
    sampleComps = sampleComps.copy()
    sampleComps[REQD_TBL_FIELDS.ID] = np.arange(len(sampleComps))
    sampleComps.at[0, REQD_TBL_FIELDS.VERTICES] = ComplexXYVertices(
        [XYVertices([[50e3, 50e3]])]
    )
    io = app.componentIo
    io.exportCsv(sampleComps, tmp_path / "Bad Verts.csv")
    with pytest.warns(UserWarning):
        io.importCsv(tmp_path / "Bad Verts.csv", imageShape=app.mainImage.image.shape)


def test_change_comp(app, mgr):
    stack = app.actionStack
    tblModel = app.componentManager
    mgr.addComponents(dfTester.compDf.copy())
    comp = mgr.compDf.loc[[RND.integers(NUM_COMPS)]]
    app.changeFocusedComponent(comp.index)
    assert tblModel.focusedComponent.equals(comp.squeeze())
    stack.undo()
    assert tblModel.focusedComponent[REQD_TBL_FIELDS.ID] == -1


def test_save_layout(app):
    app.saveLayout("tmp")
    savePath = LAYOUTS_DIR / f"tmp.dockstate"
    assert savePath.exists()
    savePath.unlink()


def test_autosave(tmp_path, app, filePlugin):
    appInst = mkQApp()
    interval = 0.01
    # Wrap in path otherwise some path ops don't work as expected
    filePlugin.startAutosave(interval, tmp_path, "autosave")
    testComps1 = dfTester.compDf.copy()
    app.componentManager.addComponents(testComps1)
    filePlugin.autosaveTimer.timeout.emit()
    appInst.processEvents()
    dfTester.fillRandomVerts()
    testComps2 = pd.concat([testComps1, dfTester.compDf.copy()])
    app.componentManager.addComponents(testComps2)
    filePlugin.autosaveTimer.timeout.emit()
    appInst.processEvents()

    testComps3 = pd.concat([testComps2, dfTester.compDf.copy()])
    app.componentManager.addComponents(testComps3)
    filePlugin.autosaveTimer.timeout.emit()
    filePlugin.stopAutosave()
    savedFiles = list(tmp_path.glob("autosave*.csv"))
    assert len(savedFiles) >= 3, "Not enough autosaves generated"


@pytest.mark.withcomps
def test_stage_plotting(monkeypatch, app, vertsPlugin):
    mainImg = app.mainImage
    mainImg.drawActionGroup.callAssociatedFunction(CNST.DRAW_ACT_CREATE)
    vertsPlugin.processEditor.changeActiveProcessor("Basic Shapes")
    mainImgProps = app.classPluginMap[MainImagePlugin].props
    oldSz = mainImgProps[CNST.PROP_MIN_COMP_SZ]
    mainImgProps[CNST.PROP_MIN_COMP_SZ] = 0
    mainImg.shapeCollection.sigShapeFinished.emit(XYVertices([[0, 0], [5, 5]]))
    assert len(app.componentManager.compDf) > 0
    app.changeFocusedComponent(app.componentManager.compDf.index[0])
    assert app.componentManager.focusedComponent.loc[REQD_TBL_FIELDS.ID] >= 0
    mainImgProps[CNST.PROP_MIN_COMP_SZ] = oldSz

    vertsPlugin.processEditor.changeActiveProcessor("Basic Shapes")

    vertsPlugin.runFromDrawAction(XYVertices([[0, 0], [10, 10]]), CNST.DRAW_ACT_ADD)
    proc = vertsPlugin.currentProcessor
    assert proc.result is not None, "Processor result not set"
    oldMakeWidget = proc._stageSummaryWidget

    def patchedWidget():
        widget = oldMakeWidget()
        widget.showMaximized = lambda: None
        return widget

    with monkeypatch.context() as m:
        m.setattr(proc, "_stageSummaryWidget", patchedWidget)
        vertsPlugin.processorAnalytics()


def test_unsaved_changes(sampleComps, tmp_path, app):
    app.componentManager.addComponents(sampleComps)
    assert app.hasUnsavedChanges
    app.saveCurrentAnnotation()
    assert not app.hasUnsavedChanges


def test_set_colorinfo(app):
    # various number of channels in image
    for clr in [[5], [5, 5, 5], [4, 4, 4, 4]]:
        clr = np.array(clr)
        app.mainImage.updateCursorInfo((100, 100), clr)
        assert "100, 100" in app.mousePosLabel.text()
        assert f"{clr}" in app.pixelColorLabel.text()


@pytest.mark.withcomps
def test_quickload_profile(tmp_path, app):
    outfile = tmp_path / "tmp.csv"
    app.exportCurrentAnnotation(outfile)
    app.appStateEditor.loadParameterValues(
        stateDict=dict(
            annotations=str(outfile),
            mainimageprocessor="Default",
            focusedimageprocessor="Default",
            colorscheme="Default",
            tablefilter="Default",
            mainimagetools="Default",
            focusedimagetools="Default",
            generalproperties="Default",
            shortcuts="Default",
        )
    )


def test_load_last_settings(tmp_path, sampleComps, app):
    oldSaveDir = app.appStateEditor.directory
    app.appStateEditor.stateManager.directory = tmp_path
    app.setMainImage(SAMPLE_IMG_FNAME, SAMPLE_IMG)
    app.addAndFocusComponents(sampleComps)
    app.appStateEditor.saveParameterValues()
    app.forceClose()
    app.appStateEditor.loadParameterValues()
    app.appStateEditor.stateManager.directory = oldSaveDir
    assert np.array_equal(app.mainImage.image, SAMPLE_IMG)
    sampleComps[REQD_TBL_FIELDS.IMAGE_FILE] = SAMPLE_IMG_FNAME.name
    sampleComps[REQD_TBL_FIELDS.ID] = sampleComps.index
    assert np.array_equal(sampleComps, app.componentDf)
