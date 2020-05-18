import numpy as np

from cdef import FRCdefApp, FR_SINGLETON
from cdef.projectvars import FR_ENUMS
from cdef.projectvars import REQD_TBL_FIELDS
from appsetup import (CompDfTester, makeCompDf, NUM_COMPS, SAMPLE_IMG,
                       TESTS_DIR, SAMPLE_IMG_DIR)

from unittest import TestCase

class TableModelTestCases(TestCase):
  def setUp(self):
    super().setUp()
    self.app = FRCdefApp(Image=SAMPLE_IMG_DIR)
    self.mgr = self.app.compMgr

    dfTester = CompDfTester(NUM_COMPS)
    dfTester.fillRandomVerts(imShape=SAMPLE_IMG.shape)
    dfTester.fillRandomClasses()

    self.sampleComps = dfTester.compDf
    self.emptyArr = np.array([], int)

class CompMgrTester(TableModelTestCases):
  def test_add_comps(self):
    # Standard add
    oldIds = np.arange(NUM_COMPS, dtype=int)
    comps = self.sampleComps.copy(deep=True)
    changeList = self.mgr.addComps(comps)
    self.cmpChangeList(changeList, oldIds)

    # Empty add
    changeList = self.mgr.addComps(makeCompDf(0))
    self.cmpChangeList(changeList)

    # Add existing should change during merge
    changeList = self.mgr.addComps(comps, FR_ENUMS.COMP_ADD_AS_MERGE)
    self.cmpChangeList(changeList, changed=oldIds)

    # Should be new IDs during 'add as new'
    changeList = self.mgr.addComps(comps)
    self.cmpChangeList(changeList, added=oldIds + NUM_COMPS)
  def test_rm_comps(self):
    comps = self.sampleComps.copy(deep=True)
    ids = np.arange(NUM_COMPS, dtype=int)

    # Standard remove
    self.mgr.addComps(comps)
    changeList = self.mgr.rmComps(ids)
    self.cmpChangeList(changeList, deleted=ids)

    # Remove when ids don't exist
    changeList = self.mgr.rmComps(ids)
    self.cmpChangeList(changeList)

    # Remove all
    for _ in range(10):
      self.mgr.addComps(comps)
    oldIds = self.mgr.compDf[REQD_TBL_FIELDS.INST_ID].values
    changeList = self.mgr.rmComps('all')
    self.cmpChangeList(changeList,deleted=oldIds)


  @staticmethod
  def cmpChangeList(changeList: dict, added: np.ndarray=None, deleted: np.ndarray=None,
                    changed: np.ndarray=None):
    emptyArr = np.array([], int)
    arrs = locals()
    for name in 'added', 'deleted', 'changed':
      if arrs[name] is None:
        arrCmp = emptyArr
      else:
        arrCmp = arrs[name]
      np.testing.assert_equal(changeList[name], arrCmp)

class CompIOTester(TableModelTestCases):
  def test_normal_export(self):
    io = self.app.compExporter
    io.exportOnlyVis = False
    curPath = TESTS_DIR/'files'/'normalExport - All IDs.csv'
    io.prepareDf(self.sampleComps)
    outDf, errMsg = io.exportCsv(str(curPath))
    self.assertTrue(curPath.exists(), 'Normal export with all IDs not successful.\n'
                                      'Error message from save:\n'
                                      f'{errMsg}')

  def test_filter_export(self):
    io = self.app.compExporter

    curPath = TESTS_DIR/'files'/'normalExport - Filtered IDs.csv'
    filterIds = np.array([0,3,2])
    io.exportOnlyVis = False
    io.prepareDf(self.sampleComps, filterIds)
    np.testing.assert_array_equal(io.compDf.index, self.sampleComps.index,
                                  'Export DF should not use only filtered IDs'
                                  ' when not exporting only visible, but'
                                  ' ID lists don\'t match.')
    # With export only visible false, should still export whole frame
    outDf, errMsg = io.exportCsv(str(curPath))
    self.assertTrue(curPath.exists(), 'Normal export with filter ids passed not successful.\n'
                                      'Error message from save:\n'
                                      f'{errMsg}')

    io.exportOnlyVis = True
    io.prepareDf(self.sampleComps, filterIds)
    np.testing.assert_array_equal(io.compDf.index, filterIds,
                                  'Export DF should use only filtered IDswhen exporting only '
                                  ' visible, but ID lists don\'t match.')
    # With export only visible false, should still export whole frame
    outDf, errMsg = io.exportCsv(str(curPath))
    self.assertTrue(curPath.exists(), 'Filtered IDs export not successful.\n'
                                      'Error message from save:\n'
                                      f'{errMsg}')