import re
from ast import literal_eval
from pathlib import Path

import numpy as np
import pytest

from conftest import NUM_COMPS, dfTester
from s3a import FR_SINGLETON, appInst, PRJ_CONSTS, S3A
from s3a.constants import REQD_TBL_FIELDS, LAYOUTS_DIR, ANN_AUTH_DIR
from s3a.generalutils import resolveAuthorName
from s3a.structures import XYVertices, ComplexXYVertices
from testingconsts import RND, SAMPLE_IMG, SAMPLE_IMG_FNAME


def test_change_img(app):
  im2 = RND.integers(0, 255, SAMPLE_IMG.shape, 'uint8')
  name = Path('./testfile.png').absolute()
  app.setMainImg(name, im2)
  assert name == app.srcImgFname, 'Annotation source not set after loading image on start'

  np.testing.assert_array_equal(app.mainImg.image, im2,
                                'Main image doesn\'t match sample image')

def test_change_img_none(app):
  app.setMainImg()
  assert app.mainImg.image is None
  assert app.srcImgFname is None

"""For some reason, the test below works if performed manually. However, I can't
seem to get the programmatically allocated keystrokes to work."""
# def test_ambig_shc(qtbot):
#   param = PrjParam('Dummy', 'T', 'registeredaction')
#
#   p2 = copy(param)
#   p2.name = 'dummy2'
#   FR_SINGLETON.shortcuts.createRegisteredButton(param, app.mainImg)
#   FR_SINGLETON.shortcuts.createRegisteredButton(p2, app.mainImg)
#   keypress = QtGui.QKeyEvent(QtGui.QKeyEvent.KeyPress, QtCore.Qt.Key_T, QtCore.Qt.NoModifier, "T")
#   with pytest.warns(UserWarning):
#     QtGui.QGuiApplication.sendEvent(app.mainImg, keypress)
#     appInst.processEvents()


@pytest.mark.withcomps
def test_clear_bounds(app, vertsPlugin):
  assert len(app.compMgr.compDf) > 0
  app.clearBoundaries()
  assert len(app.compMgr.compDf) == 0, \
    'Comps not cleared after clearing boundaries'

@pytest.mark.withcomps
def test_export_all_comps(tmp_path, app):
  compFile = tmp_path/'tmp.csv'
  app.exportCurAnnotation(str(compFile))
  assert compFile.exists(), 'All-comp export didn\'t produce a component list'

def test_load_comps_merge(tmp_path, app, sampleComps):
  compFile = tmp_path/'tmp.csv'

  app.compMgr.addComps(sampleComps)
  app.exportCurAnnotation(str(compFile))
  app.clearBoundaries()

  app.openAnnotations(str(compFile))
  equalCols = np.setdiff1d(dfTester.compDf.columns, [REQD_TBL_FIELDS.INST_ID,
                                                     REQD_TBL_FIELDS.SRC_IMG_FILENAME])
  dfCmp = app.compMgr.compDf[equalCols].values == dfTester.compDf[equalCols].values
  assert np.all(dfCmp), 'Loaded dataframe doesn\'t match daved dataframe'

def test_import_large_verts(sampleComps, tmp_path, app):
  sampleComps = sampleComps.copy()
  sampleComps[REQD_TBL_FIELDS.INST_ID] = np.arange(len(sampleComps))
  sampleComps.at[0, REQD_TBL_FIELDS.VERTICES] = ComplexXYVertices([XYVertices([[50e3, 50e3]])])
  io = app.compIo
  io.exportCsv(sampleComps, tmp_path/'Bad Verts.csv')
  with pytest.warns(UserWarning):
    io.buildFromCsv(tmp_path/'Bad Verts.csv', app.mainImg.image.shape)

def test_change_comp(app, mgr):
  stack = FR_SINGLETON.actionStack
  mImg = app.mainImg
  mgr.addComps(dfTester.compDf.copy())
  comp = mgr.compDf.loc[[RND.integers(NUM_COMPS)]]
  app.changeFocusedComp(comp)
  assert app.mainImg.compSer.equals(comp.squeeze())
  assert mImg.image is not None
  stack.undo()
  assert app.mainImg.compSer[REQD_TBL_FIELDS.INST_ID] == -1

def test_save_layout(app):
  with pytest.raises(IOError):
    app.saveLayout('default')
  app.saveLayout('tmp')
  savePath = LAYOUTS_DIR/f'tmp.dockstate'
  assert savePath.exists()
  savePath.unlink()

def test_autosave(tmp_path, app, filePlg):
  interval = 0.01
  # Wrap in path otherwise some path ops don't work as expected
  filePlg.startAutosave(interval, tmp_path, 'autosave')
  testComps1 = dfTester.compDf.copy()
  app.compMgr.addComps(testComps1)
  filePlg.autosaveTimer.timeout.emit()
  appInst.processEvents()
  dfTester.fillRandomVerts()
  testComps2 = testComps1.append(dfTester.compDf.copy())
  app.compMgr.addComps(testComps2)
  filePlg.autosaveTimer.timeout.emit()
  appInst.processEvents()

  testComps3 = testComps2.append(dfTester.compDf.copy())
  app.compMgr.addComps(testComps3)
  filePlg.autosaveTimer.timeout.emit()
  filePlg.stopAutosave()
  savedFiles = list(tmp_path.glob('autosave*.pkl'))
  assert len(savedFiles) >= 3, 'Not enough autosaves generated'

@pytest.mark.withcomps
def test_stage_plotting(monkeypatch, app, vertsPlugin):
  mainImg = app.mainImg
  mainImg.drawActGrp.callFuncByParam(PRJ_CONSTS.DRAW_ACT_CREATE)
  vertsPlugin.procCollection.switchActiveProcessor('Basic Shapes')
  oldSz = mainImg.minCompSize
  mainImg.minCompSize = 0
  mainImg.shapeCollection.sigShapeFinished.emit(XYVertices([[0, 0], [5, 5]]))
  assert len(app.compMgr.compDf) > 0
  app.changeFocusedComp(app.compMgr.compDf.iloc[[0]])
  assert app.mainImg.compSer.loc[REQD_TBL_FIELDS.INST_ID] >= 0
  mainImg.minCompSize = oldSz

  vertsPlugin.procCollection.switchActiveProcessor('Basic Shapes')
  mainImg.drawActGrp.callFuncByParam(PRJ_CONSTS.DRAW_ACT_ADD)

  mainImg.shapeCollection.sigShapeFinished.emit(XYVertices([[0, 0], [10, 10]]))
  proc = vertsPlugin.curProcessor.processor
  oldMakeWidget = proc._stageSummaryWidget
  def patchedWidget():
    widget = oldMakeWidget()
    widget.showMaximized = lambda: None
    return widget
  with monkeypatch.context() as m:
    m.setattr(proc, '_stageSummaryWidget', patchedWidget)
    vertsPlugin.processorAnalytics()

def test_no_author(app):
  p = Path(ANN_AUTH_DIR/'defaultAuthor.txt')
  p.unlink()
  with pytest.raises(SystemExit):
    S3A(guiMode=False)
  # Now plugin s3a refs are screwed up, so fix them
  for plg in FR_SINGLETON.clsToPluginMapping.values():
    plg.win = app
  resolveAuthorName('testauthor')

def test_unsaved_changes(sampleComps, tmp_path, app):
  app.compMgr.addComps(sampleComps)
  assert app.hasUnsavedChanges
  app.saveCurAnnotation()
  assert not app.hasUnsavedChanges

def test_set_colorinfo(app):
  # various number of channels in image
  for clr in [[5], [5,5,5], [4,4,4,4]]:
    clr = np.array(clr)
    app.setInfo((100,100), clr)
    assert '100, 100' in app.mouseCoords.text()
    assert f'{clr}' in app.pxColor.text()
    bgText = app.pxColor.styleSheet()
    bgColor = re.search(r'\((.*)\)', bgText).group()
    # literal_eval turns str to tuple
    bgColor = np.array(literal_eval(bgColor))
    assert bgColor.size == 4
    assert np.all(np.isin(clr, bgColor))

@pytest.mark.withcomps
def test_quickload_profile(tmp_path, app):
  outfile = tmp_path/'tmp.csv'
  app.exportCurAnnotation(outfile)
  app.appStateEditor.loadParamValues(
    stateDict=dict(layout='Default', annotations=str(outfile),
    mainimageprocessor='Default', focusedimageprocessor='Default',
    colorscheme='Default', tablefilter='Default', mainimagetools='Default',
    focusedimagetools='Default', generalproperties='Default', shortcuts='Default'
  ))

def test_load_last_settings(tmp_path, sampleComps, app):
  oldSaveDir = app.appStateEditor.saveDir
  app.appStateEditor.saveDir = tmp_path
  app.setMainImg(SAMPLE_IMG_FNAME, SAMPLE_IMG)
  app.add_focusComps(sampleComps)
  app.appStateEditor.saveParamValues()
  app.forceClose()
  app.appStateEditor.loadParamValues()
  app.appStateEditor.saveDir = oldSaveDir
  assert np.array_equal(app.mainImg.image, SAMPLE_IMG)
  sampleComps[REQD_TBL_FIELDS.SRC_IMG_FILENAME] = SAMPLE_IMG_FNAME.name
  assert np.array_equal(sampleComps, app.exportableDf)

