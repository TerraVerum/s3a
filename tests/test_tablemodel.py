import numpy as np
import pytest

from appsetup import (CompDfTester, makeCompDf, NUM_COMPS, SAMPLE_IMG,
                      SAMPLE_IMG_DIR, clearTmpFiles, RND, defaultApp_tester)
from cdef import FRCdefApp, FR_SINGLETON
from cdef.projectvars import FR_ENUMS
from cdef.projectvars import REQD_TBL_FIELDS
from cdef.structures import FRComplexVertices

# Construct app outside setUp to drastically reduce loading times
app, dfTester = defaultApp_tester()
mgr = app.compMgr
stack = FR_SINGLETON.actionStack

oldIds = np.arange(NUM_COMPS, dtype=int)

@pytest.fixture
def sampleComps():
  app.clearBoundaries()
  clearTmpFiles()
  return dfTester.compDf.copy()


def test_normal_add(sampleComps):
  changeList = mgr.addComps(sampleComps)
  cmpChangeList(changeList, oldIds)

def test_undo_add(sampleComps):
  mgr.addComps(sampleComps)
  stack.undo()
  assert len(mgr.compDf) == 0
  stack.redo()
  assert mgr.compDf.equals(sampleComps)

def test_empty_add():
  changeList = mgr.addComps(makeCompDf(0))
  cmpChangeList(changeList)

def test_rm_by_empty_vert_add(sampleComps):
  numDeletions = NUM_COMPS//3
  perm = RND.permutation(NUM_COMPS)
  deleteIdxs = np.sort(perm[:numDeletions])
  changeIdxs = np.sort(perm[numDeletions:])
  mgr.addComps(sampleComps)

  # List assignment behaves poorly for list-inherited objs (like frcomplexverts) so
  # use individual assignment
  for idx in deleteIdxs:
    sampleComps.at[idx, REQD_TBL_FIELDS.VERTICES] = FRComplexVertices()
  changeList = mgr.addComps(sampleComps, FR_ENUMS.COMP_ADD_AS_MERGE)
  cmpChangeList(changeList, deleted=deleteIdxs, changed=changeIdxs)


def test_double_add(sampleComps):
  changeList = mgr.addComps(sampleComps, FR_ENUMS.COMP_ADD_AS_NEW)
  cmpChangeList(changeList, added=oldIds)

  # Should be new IDs during 'add as new'
  changeList = mgr.addComps(sampleComps, FR_ENUMS.COMP_ADD_AS_NEW)
  cmpChangeList(changeList, added=oldIds + NUM_COMPS)

def test_change_comps(sampleComps):
  changeList = mgr.addComps(sampleComps, FR_ENUMS.COMP_ADD_AS_NEW)
  cmpChangeList(changeList, added=oldIds)

  newValids = dfTester.fillRandomValids(sampleComps)
  newClasses = dfTester.fillRandomClasses(sampleComps)
  changeList = mgr.addComps(sampleComps, FR_ENUMS.COMP_ADD_AS_MERGE)
  cmpChangeList(changeList, changed=oldIds)
  np.testing.assert_array_equal(mgr.compDf[REQD_TBL_FIELDS.VALIDATED].values,
                                newValids, '"Validated" list doesn\'t match during'
                                           ' test_change_comps')
  np.testing.assert_array_equal(newClasses,
                                mgr.compDf[REQD_TBL_FIELDS.COMP_CLASS].values,
                                '"Class" list doesn\'t match during test_change_comps')

def test_rm_comps(sampleComps):
  ids = np.arange(NUM_COMPS, dtype=int)

  # Standard remove
  mgr.addComps(sampleComps)
  changeList = mgr.rmComps(ids)
  cmpChangeList(changeList, deleted=ids)

  # Remove when ids don't exist
  changeList = mgr.rmComps(ids)
  cmpChangeList(changeList)

  # Remove all
  for _ in range(10):
    mgr.addComps(sampleComps)
  prevIds = mgr.compDf[REQD_TBL_FIELDS.INST_ID].values
  changeList = mgr.rmComps(FR_ENUMS.COMP_RM_ALL)
  cmpChangeList(changeList,deleted=prevIds)

def test_rm_undo(sampleComps):
  ids = np.arange(NUM_COMPS, dtype=int)

  # Standard remove
  mgr.addComps(sampleComps)
  mgr.rmComps(ids)
  FR_SINGLETON.actionStack.undo()
  assert mgr.compDf.equals(sampleComps)


# def test_table_setdata(sampleComps):
#   mgr.addComps(sampleComps)
#
#   _ = REQD_TBL_FIELDS
#   colVals = {
#     _.VERTICES: [FRComplexVertices(FRVertices([[1,2], [3,4]]))],
#     _.COMP_CLASS: FR_SINGLETON.tableData.compClasses[4],
#     _.VALIDATED: True,
#     _.ANN_AUTHOR: 'Hi There',
#     _.ANN_FILENAME: 'newfilename'
#   }
#   intColMapping = {FR_SINGLETON.tableData.allFields.index(k):v
#                    for k, v in colVals.items()}
#
#   for col, newVal in intColMapping.items():
#     row = RND.integers(NUM_COMPS)
#     idx = app.compTbl.model().index(row, col)
#     oldVal = mgr.data(idx, QtCore.Qt.EditRole)
#     mgr.setData(idx, newVal)
#     assert mgr.compDf.iloc[row, col] == newVal
#     stack.undo()
#     assert mgr.compDf.iloc[row, col] == oldVal

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