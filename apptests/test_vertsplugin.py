"""Most other functionality is tested under procimpls, this is for other registered functions"""
import pytest
import numpy as np

from apptests.testingconsts import SAMPLE_SMALL_IMG
from s3a import XYVertices, REQD_TBL_FIELDS, ComplexXYVertices
from s3a.generalutils import imgCornerVertices
from s3a.plugins.file import NewProjectWizard
from s3a.processing.algorithms.imageproc import _historyMaskHolder

@pytest.mark.withcomps
@pytest.mark.smallimage
def test_registered_verts_funcs(vertsPlugin, app):
  togray = lambda: vertsPlugin.region.toGrayImg(SAMPLE_SMALL_IMG.shape)
  imsize = np.prod(SAMPLE_SMALL_IMG.shape[:2])
  editor = vertsPlugin.procEditor
  editor.changeActiveProcessor('Basic Shapes')
  vertsPlugin.run()
  assert _historyMaskHolder[0].sum()

  vertsPlugin.clearProcessorHistory()
  assert _historyMaskHolder[0].sum() == 0

  vertsPlugin.fillRegionMask()
  img = togray()
  assert (img > 0).sum() == imsize

  vertsPlugin.clearFocusedRegion()
  assert togray().sum() == 0

  vertsPlugin.actionStack.undo()
  assert (img > 0).sum() == imsize

  app.changeFocusedComp(app.compMgr.compDf.index[0])
  img = togray()
  vertsPlugin.clearFocusedRegion()
  vertsPlugin.resetFocusedRegion()
  assert np.array_equal(togray(), img)

def test_region_offset(vertsPlugin, sampleComps):
  vertsPlugin.updateRegionFromDf(sampleComps.iloc[[0]], offset=XYVertices([10000,10000]))
  vMax = ComplexXYVertices.stackedMax(vertsPlugin.region.regionData[REQD_TBL_FIELDS.VERTICES])
  assert max(vMax) > max(ComplexXYVertices.stackedMax(sampleComps[REQD_TBL_FIELDS.VERTICES]))

@pytest.mark.withcomps
def test_accept_region(app, vertsPlugin):
  comp = app.compMgr.compDf.iloc[[0]]
  app.changeFocusedComp(comp.index)
  vertsPlugin.clearFocusedRegion()

  app.acceptFocusedRegion()
  assert comp.index[0] not in app.compMgr.compDf.index

  comp = app.compMgr.compDf.iloc[[0]]
  app.changeFocusedComp(comp.index)
  vertsPlugin.fillRegionMask()
  verts = vertsPlugin.region.regionData[REQD_TBL_FIELDS.VERTICES]
  app.acceptFocusedRegion()
  assert np.array_equal(verts, app.compMgr.compDf[REQD_TBL_FIELDS.VERTICES][comp.index])

@pytest.mark.smallimage
@pytest.mark.withcomps
def test_region_history(vertsPlugin, app, monkeypatch):
  comp = app.compMgr.compDf.iloc[[0]]
  app.changeFocusedComp(comp.index)

  vertsPlugin.run(bgVerts=imgCornerVertices(SAMPLE_SMALL_IMG))
  vertsPlugin.run()

  initial, history = vertsPlugin.getRegionHistory()
  assert np.array_equal(initial, SAMPLE_SMALL_IMG)
  assert len(history)
  assert np.array_equal(history[-1], vertsPlugin.region.toGrayImg(SAMPLE_SMALL_IMG.shape) > 0)

  with monkeypatch.context() as m:
    m.setattr(vertsPlugin.playbackWindow, 'show', lambda *args: None)
    m.setattr(vertsPlugin.playbackWindow, 'raise_', lambda *args: None)
    vertsPlugin.playbackRegionHistory()
    winImg = vertsPlugin.playbackWindow.displayPlt.imgItem.image
    assert np.array_equal(winImg, initial)


def test_proj_wizard(filePlg):
  # Much is gui, just make sure mechanics work
  npw = NewProjectWizard(filePlg)
  for fileLst in npw.fileLists.values():
    assert not fileLst