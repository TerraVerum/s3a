import re
from ast import literal_eval
from pathlib import Path

import numpy as np
import pytest

from conftest import NUM_COMPS, _block_pltShow, app, mgr, dfTester
from s3a import FR_SINGLETON, appInst, S3A
from s3a.models.s3abase import S3ABase
from s3a.generalutils import resolveAuthorName
from s3a.projectvars import REQD_TBL_FIELDS, LAYOUTS_DIR, ANN_AUTH_DIR
from s3a.structures import FRAlgProcessorError, FRS3AException, FRVertices, \
  FRComplexVertices, FRS3AWarning
from testingconsts import FIMG_SER_COLS, RND, SAMPLE_IMG, SAMPLE_IMG_FNAME


def test_change_img():
  im2 = RND.integers(0, 255, SAMPLE_IMG.shape, 'uint8')
  name = Path('./testfile').absolute()
  app.setMainImg(name, im2)
  assert name == app.srcImgFname, 'Annotation source not set after loading image on start'

  np.testing.assert_array_equal(app.mainImg.image, im2,
                                'Main image doesn\'t match sample image')

def test_change_img_none():
  app.setMainImg()
  assert app.mainImg.image is None
  assert app.srcImgFname is None

def test_est_bounds_no_img():
  app.setMainImg()
  with pytest.raises(FRAlgProcessorError):
    app.estimateBoundaries()

def test_est_clear_bounds():
  # Change to easy processor first for speed
  clctn = app.mainImg.procCollection
  prevProc = clctn.curProcessor
  clctn.switchActiveProcessor('Basic Shapes')
  app.estimateBoundaries()
  assert len(app.compMgr.compDf) > 0, 'Comp not created after global estimate'
  app.clearBoundaries()
  assert len(
    app.compMgr.compDf) == 0, 'Comps not cleared after clearing boundaries'
  # Restore state
  clctn.switchActiveProcessor(prevProc)

def test_export_all_comps(tmpdir):
  compFile = tmpdir/'tmp.csv'
  app.exportCompList(str(compFile))
  assert compFile.exists(), 'All-comp export didn\'t produce a component list'

def test_load_comps_merge(tmpdir):
  compFile = tmpdir/'tmp.csv'

  app.compMgr.addComps(dfTester.compDf)
  app.exportCompList(str(compFile))
  app.clearBoundaries()

  app.loadCompList(str(compFile))
  equalCols = np.setdiff1d(dfTester.compDf.columns, [REQD_TBL_FIELDS.INST_ID,
                                                     REQD_TBL_FIELDS.SRC_IMG_FILENAME])
  dfCmp = app.compMgr.compDf[equalCols].values == dfTester.compDf[equalCols].values
  assert np.all(dfCmp), 'Loaded dataframe doesn\'t match daved dataframe'

def test_import_large_verts(sampleComps, tmpdir):
  sampleComps.loc[:, REQD_TBL_FIELDS.INST_ID] = np.arange(len(sampleComps))
  sampleComps.at[0, REQD_TBL_FIELDS.VERTICES] = FRComplexVertices([FRVertices([[50e3, 50e3]])])
  io = app.compIo
  io.prepareDf(sampleComps)
  io.exportCsv(tmpdir/'Bad Verts.csv')
  with pytest.warns(FRS3AWarning):
    io.buildFromCsv(tmpdir/'Bad Verts.csv', app.mainImg.image.shape)

def test_change_comp():
  stack = FR_SINGLETON.actionStack
  fImg = app.focusedImg
  mgr.addComps(dfTester.compDf.copy())
  comp = mgr.compDf.loc[[RND.integers(NUM_COMPS)]]
  app.changeFocusedComp(comp)
  assert app.focusedImg.compSer.equals(comp.squeeze()[FIMG_SER_COLS])
  assert fImg.image is not None
  stack.undo()
  assert fImg.image is None

def test_save_layout():
  with pytest.raises(FRS3AException):
    app.saveLayout('default')
  app.saveLayout('tmp')
  savePath = LAYOUTS_DIR/f'tmp.dockstate'
  assert savePath.exists()
  savePath.unlink()

def test_autosave(tmpdir):
  interval = 0.01
  # Wrap in path otherwise some path ops don't work as expected
  tmpdir = Path(tmpdir)
  app.startAutosave(interval, tmpdir, 'autosave')
  testComps1 = dfTester.compDf.copy()
  app.compMgr.addComps(testComps1)
  app.autosaveTimer.timeout.emit()
  appInst.processEvents()
  dfTester.fillRandomVerts()
  testComps2 = testComps1.append(dfTester.compDf.copy())
  app.compMgr.addComps(testComps2)
  app.autosaveTimer.timeout.emit()
  appInst.processEvents()

  dfTester.fillRandomClasses()
  testComps3 = testComps2.append(dfTester.compDf.copy())
  app.compMgr.addComps(testComps3)
  app.autosaveTimer.timeout.emit()
  app.stopAutosave()
  savedFiles = list(tmpdir.glob('autosave*.csv'))
  assert len(savedFiles) >= 3, 'Not enough autosaves generated'

def test_stage_plotting():
  for editableImg in app.mainImg, app.focusedImg:
    editableImg.procCollection = FR_SINGLETON.algParamMgr.createProcessorForClass(editableImg)
  with _block_pltShow():
    with pytest.raises(FRAlgProcessorError):
      app.showModCompAnalytics()
    with pytest.raises(FRAlgProcessorError):
      app.showNewCompAnalytics()
    # Make a component so modofications can be tested
    mainImg = app.mainImg
    mainImg.procCollection.switchActiveProcessor('Basic Shapes')
    mainImg.handleShapeFinished(FRVertices())
    app.showNewCompAnalytics()
    assert app.focusedImg.compSer.loc[REQD_TBL_FIELDS.INST_ID] >= 0

    focImg = app.focusedImg
    focImg.procCollection.switchActiveProcessor('Basic Shapes')
    focImg.handleShapeFinished(FRVertices())
    app.showModCompAnalytics()

def test_no_author():
  p = Path(ANN_AUTH_DIR/'defaultAuthor.txt')
  p.unlink()
  with pytest.raises(SystemExit):
    S3ABase()
  resolveAuthorName('testauthor')

def test_unsaved_changes(sampleComps, tmpdir):
  app.compMgr.addComps(sampleComps)
  assert app.hasUnsavedChanges
  app.exportCompList(tmpdir/'export.csv')
  assert not app.hasUnsavedChanges

def test_set_colorinfo():
  # various number of channels in image
  for clr in [[5], [5,5,5], [4,4,4,4]]:
    clr = np.array(clr)
    app.setInfo(((100,100), clr))
    assert '100, 100' in app.mouseCoords.text()
    assert f'{clr}' in app.pxColor.text()
    bgText = app.pxColor.styleSheet()
    bgColor = re.search(r'\((.*)\)', bgText).group()
    # literal_eval turns str to tuple
    bgColor = np.array(literal_eval(bgColor))
    assert bgColor.size == 4
    assert np.all(np.isin(clr, bgColor))

@pytest.mark.withcomps
def test_quickload_profile(tmpdir):
  outfile = Path(tmpdir)/'tmp.csv'
  app.exportCompList(outfile)
  app.appStateEditor.loadParamState(
    stateDict=dict(image=str(SAMPLE_IMG_FNAME), layout='Default', annotations=str(outfile),
    mainimageprocessor='Default', focusedimageprocessor='Default',
    colorscheme='Default', tablefilter='Default', mainimagetools='Default',
    focusedimagetools='Default', generalproperties='Default', shortcuts='Default'
  ))