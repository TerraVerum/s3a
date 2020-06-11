from pathlib import Path

import numpy as np
import pytest

from appsetup import (NUM_COMPS, SAMPLE_IMG,
                      TESTS_DIR, SAMPLE_IMG_FNAME, clearTmpFiles, RND, defaultApp_tester)
from s3a import FR_SINGLETON
from s3a.projectvars import REQD_TBL_FIELDS, LAYOUTS_DIR
from s3a.structures import FRAlgProcessorError, FRS3AException

EXPORT_DIR = TESTS_DIR/'files'

app, dfTester = defaultApp_tester()
mgr = app.compMgr

@pytest.fixture
def clearedApp():
  clearTmpFiles()
  app.clearBoundaries()
  app.resetMainImg(SAMPLE_IMG_FNAME, SAMPLE_IMG)
  FR_SINGLETON.actionStack.clear()
  return app

def test_change_img(clearedApp):
  im2 = RND.integers(0, 255, SAMPLE_IMG.shape, 'uint8')
  name = Path('./testfile').absolute()
  clearedApp.resetMainImg(name, im2)
  assert name == app.srcImgFname, 'Annotation source not set after loading image on start'

  np.testing.assert_array_equal(clearedApp.mainImg.image, im2,
                                'Main image doesn\'t match sample image')

def test_change_img_none(clearedApp):
  clearedApp.resetMainImg()
  assert clearedApp.mainImg.image is None
  assert app.srcImgFname is None

def test_est_bounds_no_img(clearedApp):
  clearedApp.resetMainImg()
  with pytest.raises(FRAlgProcessorError):
    clearedApp.estimateBoundaries()

def test_est_clear_bounds(clearedApp):
  # Change to easy processor first for speed
  clctn = clearedApp.mainImg.procCollection
  prevProc = clctn.curProcessor
  clctn.switchActiveProcessor('Basic Shapes')
  clearedApp.estimateBoundaries()
  assert len(clearedApp.compMgr.compDf) > 0, 'Comp not created after global estimate'
  clearedApp.clearBoundaries()
  assert len(clearedApp.compMgr.compDf) == 0, 'Comps not cleared after clearing boundaries'
  # Restore state
  clctn.switchActiveProcessor(prevProc)

def test_export_all_comps(clearedApp):
  compFile = EXPORT_DIR/'tmp.csv'
  clearedApp.exportCompList(str(compFile))
  assert compFile.exists(), 'All-comp export didn\'t produce a component list'

def test_load_comps_merge(clearedApp):
  compFile = EXPORT_DIR/'tmp.csv'

  clearedApp.compMgr.addComps(dfTester.compDf)
  clearedApp.exportCompList(str(compFile))
  clearedApp.clearBoundaries()

  clearedApp.loadCompList(str(compFile))
  equalCols = np.setdiff1d(dfTester.compDf.columns, [REQD_TBL_FIELDS.INST_ID,
                                                     REQD_TBL_FIELDS.SRC_IMG_FILENAME])
  dfCmp = clearedApp.compMgr.compDf[equalCols].values == dfTester.compDf[equalCols].values
  assert np.all(dfCmp), 'Loaded dataframe doesn\'t match daved dataframe'

def test_change_comp(clearedApp):
  stack = FR_SINGLETON.actionStack
  fImg = clearedApp.focusedImg
  mgr.addComps(dfTester.compDf.copy())
  comp = mgr.compDf.loc[[RND.integers(NUM_COMPS)]]
  clearedApp.changeFocusedComp(comp)
  assert clearedApp.focusedImg.compSer.equals(comp.squeeze())
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
