from pathlib import Path
import stat

import numpy as np

from cdef import FRCdefApp, FR_SINGLETON
from cdef.generalutils import augmentException
from cdef.projectvars import FR_ENUMS
from cdef.projectvars import REQD_TBL_FIELDS
from appsetup import (CompDfTester, makeCompDf, NUM_COMPS, SAMPLE_IMG,
                      TESTS_DIR, SAMPLE_IMG_DIR, clearTmpFiles, RND)

from unittest import TestCase

from cdef.structures import FRComplexVertices, FRAlgProcessorError
from cdef.tablemodel import FRComponentIO

EXPORT_DIR = TESTS_DIR/'files'

app = FRCdefApp(Image=SAMPLE_IMG_DIR)

dfTester = CompDfTester(NUM_COMPS)
dfTester.fillRandomVerts(imShape=SAMPLE_IMG.shape)
dfTester.fillRandomClasses()

class CdefAppTestCases(TestCase):
  def setUp(self):
    clearTmpFiles()
    app.clearBoundaries()
    app.resetMainImg(SAMPLE_IMG_DIR, SAMPLE_IMG)

  def test_change_img(self):
    im2 = RND.integers(0, 255, SAMPLE_IMG.shape, 'uint8')
    name = 'testfile'
    app.resetMainImg(name, im2)
    self.assertSequenceEqual(name, FR_SINGLETON.tableData.annFile,
                             'Annotation source not set after loading image on start')

    np.testing.assert_array_equal(app.mainImg.image, im2,
                                  'Main image doesn\'t match sample image')

  def test_change_img_none(self):
    app.resetMainImg()
    self.assertIsNone(app.mainImg.image)
    self.assertIsNone(FR_SINGLETON.tableData.annFile)

  def test_est_bounds_no_img(self):
    app.resetMainImg()
    self.assertRaises(FRAlgProcessorError, app.estimateBoundaries)

  def test_est_clear_bounds(self):
    # Change to easy processor first for speed
    clctn = app.mainImg.procCollection
    prevProc = clctn.curProcessor
    basicProc = clctn.nameToProcMapping['Basic Shapes']
    clctn.changeActiveAlg(basicProc)
    app.estimateBoundaries()
    self.assertTrue(len(app.compMgr.compDf) > 0, 'Comp not created after'
                                                 ' global estimate')
    app.clearBoundaries()
    self.assertTrue(len(app.compMgr.compDf) == 0, 'Comps not cleared after'
                                                  ' clearing boundaries')
    # Restore state
    clctn.changeActiveAlg(prevProc)

  def test_export_all_comps(self):
    compFile = EXPORT_DIR/'tmp.csv'
    app.exportCompList(str(compFile))
    self.assertTrue(compFile.exists(),
                    'All-comp export didn\'t produce a component list')

  def test_load_comps_merge(self):
    compFile = EXPORT_DIR/'tmp.csv'

    app.compMgr.addComps(dfTester.compDf)
    app.exportCompList(str(compFile))
    app.clearBoundaries()

    app.loadCompList(str(compFile))
    equalCols = np.setdiff1d(dfTester.compDf.columns, [REQD_TBL_FIELDS.INST_ID,
                                                       REQD_TBL_FIELDS.ANN_FILENAME])
    dfCmp = app.compMgr.compDf[equalCols].values == dfTester.compDf[equalCols].values
    self.assertTrue(np.all(dfCmp), 'Loaded dataframe doesn\'t match daved dataframe')

