from pathlib import Path
import stat

import numpy as np

from cdef import FRCdefApp, FR_SINGLETON
from cdef.generalutils import augmentException
from cdef.projectvars import FR_ENUMS
from cdef.projectvars import REQD_TBL_FIELDS
from appsetup import (CompDfTester, makeCompDf, NUM_COMPS, SAMPLE_IMG,
                      SAMPLE_IMG_DIR, EXPORT_DIR, clearTmpFiles, RND)

from unittest import TestCase

from cdef.structures import FRComplexVertices
from cdef.tablemodel import FRComponentIO

# Construct app outside setUp to drastically reduce loading times
app = FRCdefApp(Image=SAMPLE_IMG_DIR)
mgr = app.compMgr

dfTester = CompDfTester(NUM_COMPS)
dfTester.fillRandomVerts(imShape=SAMPLE_IMG.shape)
dfTester.fillRandomClasses()

class TableModelTestCases(TestCase):
  def setUp(self):
    super().setUp()

    self.sampleComps = dfTester.compDf.copy(deep=True)
    mgr.rmComps()

class CompMgrTester(TableModelTestCases):
  def setUp(self):
    super().setUp()
    self.oldIds = np.arange(NUM_COMPS, dtype=int)

  def test_normal_add(self):
    oldIds = np.arange(NUM_COMPS, dtype=int)
    changeList = mgr.addComps(self.sampleComps)
    self.cmpChangeList(changeList, oldIds)

  def test_undo_add(self):
    oldIds = np.arange(NUM_COMPS, dtype=int)
    mgr.addComps(self.sampleComps)
    FR_SINGLETON.actionStack.undo()
    self.assertTrue(len(mgr.compDf) == 0)

  def test_empty_add(self):
    changeList = mgr.addComps(makeCompDf(0))
    self.cmpChangeList(changeList)

  def test_rm_by_empty_vert_add(self):
    numDeletions = NUM_COMPS//3
    perm = RND.permutation(NUM_COMPS)
    deleteIdxs = np.sort(perm[:numDeletions])
    changeIdxs = np.sort(perm[numDeletions:])
    mgr.addComps(self.sampleComps)

    # List assignment behaves poorly for list-inherited objs (like frcomplexverts) so
    # use individual assignment
    for idx in deleteIdxs:
      self.sampleComps.at[idx, REQD_TBL_FIELDS.VERTICES] = FRComplexVertices()
    changeList = mgr.addComps(self.sampleComps, FR_ENUMS.COMP_ADD_AS_MERGE)
    self.cmpChangeList(changeList, deleted=deleteIdxs, changed=changeIdxs)


  def test_double_add(self):
    changeList = mgr.addComps(self.sampleComps, FR_ENUMS.COMP_ADD_AS_NEW)
    self.cmpChangeList(changeList, added=self.oldIds)

    # Should be new IDs during 'add as new'
    changeList = mgr.addComps(self.sampleComps, FR_ENUMS.COMP_ADD_AS_NEW)
    self.cmpChangeList(changeList, added=self.oldIds + NUM_COMPS)

  def test_change_comps(self):
    changeList = mgr.addComps(self.sampleComps, FR_ENUMS.COMP_ADD_AS_NEW)
    self.cmpChangeList(changeList, added=self.oldIds)

    newValids = dfTester.fillRandomValids(self.sampleComps)
    newClasses = dfTester.fillRandomClasses(self.sampleComps)
    changeList = mgr.addComps(self.sampleComps, FR_ENUMS.COMP_ADD_AS_MERGE)
    self.cmpChangeList(changeList, changed=self.oldIds)
    np.testing.assert_array_equal(mgr.compDf[REQD_TBL_FIELDS.VALIDATED].values,
                                  newValids, '"Validated" list doesn\'t match during'
                                             ' test_change_comps')
    np.testing.assert_array_equal(newClasses,
                                  mgr.compDf[REQD_TBL_FIELDS.COMP_CLASS].values,
                                  '"Class" list doesn\'t match during test_change_comps')

  def test_rm_comps(self):
    comps = self.sampleComps.copy(deep=True)
    ids = np.arange(NUM_COMPS, dtype=int)

    # Standard remove
    mgr.addComps(comps)
    changeList = mgr.rmComps(ids)
    self.cmpChangeList(changeList, deleted=ids)

    # Remove when ids don't exist
    changeList = mgr.rmComps(ids)
    self.cmpChangeList(changeList)

    # Remove all
    for _ in range(10):
      mgr.addComps(comps)
    oldIds = mgr.compDf[REQD_TBL_FIELDS.INST_ID].values
    changeList = mgr.rmComps(FR_ENUMS.COMP_RM_ALL)
    self.cmpChangeList(changeList,deleted=oldIds)

  def test_rm_undo(self):
    comps = self.sampleComps.copy(deep=True)
    ids = np.arange(NUM_COMPS, dtype=int)

    # Standard remove
    mgr.addComps(comps)
    mgr.rmComps(ids)
    FR_SINGLETON.actionStack.undo()
    self.assertTrue(mgr.compDf.equals(comps))


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
      np.testing.assert_equal(changeList[name], arrCmp, f'Mismatch in "{name}"'
                                                        f' of change list')

class CompIOTester(TableModelTestCases):
  def setUp(self):
    super().setUp()
    # Clear exports from previous runs
    clearTmpFiles()

  def test_normal_export(self):
    io = app.compExporter
    io.exportOnlyVis = False
    curPath = EXPORT_DIR / 'normalExport - All IDs.csv'
    io.prepareDf(self.sampleComps)
    self.doAndAssertExport(curPath, io, 'Normal export with all IDs not successful.')

  def test_filter_export(self):
    io = app.compExporter

    curPath = EXPORT_DIR / 'normalExport - Filtered IDs export all.csv'
    filterIds = np.array([0,3,2])
    io.exportOnlyVis = False
    io.prepareDf(self.sampleComps, filterIds)
    np.testing.assert_array_equal(io.compDf.index, self.sampleComps.index,
                                  'Export DF should not use only filtered IDs'
                                  ' when not exporting only visible, but'
                                  ' ID lists don\'t match.')
    # With export only visible false, should still export whole frame
    self.doAndAssertExport(curPath, io, 'Normal export with filter ids passed not successful.')

    curPath = EXPORT_DIR / 'normalExport - Filtered IDs export filtered.csv'
    io.exportOnlyVis = True
    io.prepareDf(self.sampleComps, filterIds)
    np.testing.assert_array_equal(io.compDf.index, filterIds,
                                  'Export DF should use only filtered IDs when exporting only '
                                  'visible, but ID lists don\'t match.')
    # With export only visible false, should still export whole frame
    self.doAndAssertExport(curPath, io, 'Export with filtered ids not successful.')

  def doAndAssertExport(self, fpath: Path, io: FRComponentIO, failMsg: str):
    try:
      io.exportCsv(str(fpath))
    except Exception as ex:
      augmentException(ex, f'{failMsg}\n')
      raise
    self.assertTrue(fpath.exists(), 'Csv file doesn\'t exist despite export')