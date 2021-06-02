import numpy as np
import pytest
from pyqtgraph.Qt import QtCore

from conftest import NUM_COMPS, dfTester
from helperclasses import clearTmpFiles
from testingconsts import RND
from s3a.constants import PRJ_ENUMS
from s3a.constants import REQD_TBL_FIELDS
from s3a.structures import ComplexXYVertices, XYVertices

oldIds = np.arange(NUM_COMPS, dtype=int)

@pytest.fixture
def sampleComps(app):
  app.clearBoundaries()
  return dfTester.compDf.copy()


def test_normal_add(sampleComps, mgr):
  mgr.rmComps()
  changeList = mgr.addComps(sampleComps)
  cmpChangeList(changeList, oldIds)

def test_undo_add(sampleComps, mgr):
  mgr.addComps(sampleComps)
  mgr.actionStack.undo()
  assert len(mgr.compDf) == 0
  mgr.actionStack.redo()
  assert len(mgr.compDf) > 0

def test_empty_add(mgr):
  changeList = mgr.addComps(mgr.tableData.makeCompDf(0))
  cmpChangeList(changeList)

def test_rm_by_empty_vert_add(sampleComps, mgr):
  numDeletions = NUM_COMPS//3
  perm = RND.permutation(NUM_COMPS)
  deleteIdxs = np.sort(perm[:numDeletions])
  changeIdxs = np.sort(perm[numDeletions:])
  mgr.addComps(sampleComps)

  # List assignment behaves poorly for list-inherited objs (like frcomplexverts) so
  # use individual assignment
  for idx in deleteIdxs:
    sampleComps.at[idx, REQD_TBL_FIELDS.VERTICES] = ComplexXYVertices()
  changeList = mgr.addComps(sampleComps, PRJ_ENUMS.COMP_ADD_AS_MERGE)
  cmpChangeList(changeList, deleted=deleteIdxs, changed=changeIdxs)


def test_double_add(sampleComps, mgr):
  changeList = mgr.addComps(sampleComps, PRJ_ENUMS.COMP_ADD_AS_NEW)
  cmpChangeList(changeList, added=oldIds)

  # Should be new IDs during 'add as new'
  changeList = mgr.addComps(sampleComps, PRJ_ENUMS.COMP_ADD_AS_NEW)
  cmpChangeList(changeList, added=oldIds + NUM_COMPS)

def test_change_comps(sampleComps, mgr):
  changeList = mgr.addComps(sampleComps, PRJ_ENUMS.COMP_ADD_AS_NEW)
  cmpChangeList(changeList, added=oldIds)

  newVerts = dfTester.fillRandomVerts(compDf=sampleComps)
  changeList = mgr.addComps(sampleComps, PRJ_ENUMS.COMP_ADD_AS_MERGE)
  cmpChangeList(changeList, changed=oldIds)
  assert newVerts == mgr.compDf[REQD_TBL_FIELDS.VERTICES].to_list(), \
                                '"Vertices" list doesn\'t match during test_change_comps'

def test_rm_comps(sampleComps, mgr):
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
  changeList = mgr.rmComps()
  cmpChangeList(changeList,deleted=prevIds)

  # Remove single
  mgr.addComps(sampleComps)
  mgr.rmComps(sampleComps.index[0])

def test_rm_undo(sampleComps, mgr):
  ids = np.arange(NUM_COMPS, dtype=int)

  # Standard remove
  mgr.addComps(sampleComps)
  mgr.rmComps(ids)
  mgr.actionStack.undo()
  assert np.setdiff1d(ids, mgr.compDf.index).size == 0

def test_merge_comps(sampleComps, mgr):
  mgr.addComps(sampleComps)
  mgr.mergeCompVertsById(sampleComps.index)
  assert len(mgr.compDf) == 1
  mgr.actionStack.undo()
  assert len(mgr.compDf) == len(sampleComps)

def test_bad_merge(sampleComps, mgr):
  mgr.addComps(sampleComps)
  with pytest.warns(UserWarning):
    mgr.mergeCompVertsById([0])
  with pytest.warns(UserWarning):
    mgr.mergeCompVertsById([])

def test_table_setdata(sampleComps, app, mgr):
  mgr.addComps(sampleComps)

  _ = REQD_TBL_FIELDS
  colVals = {
    _.VERTICES: ComplexXYVertices([XYVertices([[1, 2], [3, 4]])]),
    _.SRC_IMG_FILENAME: 'newfilename'
  }
  intColMapping = {mgr.tableData.allFields.index(k):v
                   for k, v in colVals.items()}

  for col, newVal in intColMapping.items():
    row = RND.integers(NUM_COMPS)
    idx = app.compTbl.model().index(row, col)
    oldVal = mgr.data(idx, QtCore.Qt.EditRole)
    mgr.setData(idx, newVal)
    # Test with no change
    mgr.setData(idx, newVal)
    assert mgr.compDf.iloc[row, col] == newVal
    mgr.actionStack.undo()
    assert mgr.compDf.iloc[row, col] == oldVal

def test_table_getdata(sampleComps, mgr):
  mgr.addComps(sampleComps)
  idx = mgr.index(0, list(REQD_TBL_FIELDS).index(REQD_TBL_FIELDS.SRC_IMG_FILENAME))
  dataVal = sampleComps.iat[0, idx.column()]
  assert mgr.data(idx, QtCore.Qt.EditRole) == dataVal
  assert mgr.data(idx, QtCore.Qt.DisplayRole) == str(dataVal)
  assert mgr.data(idx, 854) is None

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