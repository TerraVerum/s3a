from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import cv2 as cv
import pytest

from conftest import stack
from helperclasses import CompDfTester
from s3a import appInst, FR_SINGLETON, S3A
from s3a.constants import REQD_TBL_FIELDS, PRJ_CONSTS
from s3a.structures import ComplexXYVertices, PrjParam, XYVertices, S3AWarning
from s3a.views.tableview import CompTableView


@pytest.mark.withcomps
def test_merge_selected_comps(app, mgr):
  oldLen = len(mgr.compDf)
  app.compTbl.selectAll()
  appInst.processEvents()
  app.compDisplay.mergeSelectedComps()
  assert len(mgr.compDf) == 1
  # # Undo swap current comp, undo merge
  stack.undo()
  stack.undo()
  assert len(mgr.compDf) == oldLen
  app.compTbl.clearSelection()
  # Nothing should happen
  app.compDisplay.mergeSelectedComps()


def test_split_selected_comps(app, mgr):
  compMask = np.zeros((100,100), 'uint8')
  cv.rectangle(compMask, (0, 0), (5, 5), 1, -1)
  cv.rectangle(compMask, (9, 9), (16, 16), 1, -1)
  cv.rectangle(compMask, (21, 21), (30, 30), 1, -1)
  cv.rectangle(compMask, (46, 46), (60, 60), 1, -1)
  verts = ComplexXYVertices.fromBwMask(compMask > 0)
  comp = FR_SINGLETON.tableData.makeCompDf()
  comp.at[comp.index[0], REQD_TBL_FIELDS.VERTICES] = verts
  app.add_focusComps(comp)

  app.compTbl.selectAll()
  app.compDisplay.splitSelectedComps()
  assert len(mgr.compDf) == 4
  # Once for focused comp, once for splitting
  stack.undo()
  stack.undo()
  assert len(mgr.compDf) == 1
  # Nothing should happen
  app.compDisplay.splitSelectedComps()

@pytest.mark.withcomps
def test_set_cells_as(app, mgr):
  oldCls = FR_SINGLETON.tableData.compClasses[0]
  # Even amount of comps for easy comparison
  if (len(mgr.compDf) % 2) == 1:
    mgr.rmComps(mgr.compDf.index[-1])
  mgr.compDf[REQD_TBL_FIELDS.COMP_CLASS] = oldCls
  # Ensure the overwrite data will be different from what it's overwriting
  newCls = FR_SINGLETON.tableData.compClasses[1]
  newDf = mgr.compDf.loc[[0]]
  compClsIdx = FR_SINGLETON.tableData.allFields.index(REQD_TBL_FIELDS.COMP_CLASS)
  newDf.iat[0, compClsIdx] = newCls
  oldMode = app.compTbl.selectionMode()
  app.compTbl.setSelectionMode(app.compTbl.MultiSelection)
  for row in mgr.compDf.index[::2]:
    app.compTbl.selectRow(row)
  app.compTbl.setSelectionMode(oldMode)
  selection = app.compTbl.ids_rows_colsFromSelection()
  app.compTbl.setSelectedCellsAs(selection, newDf)
  matchList = np.tile([newCls, oldCls], len(mgr.compDf)//2)
  assert np.array_equal(mgr.compDf[REQD_TBL_FIELDS.COMP_CLASS], matchList)

def test_set_as_gui(sampleComps):
  # Monkeypatch gui for testing
  view = CompTableView()
  mgr = view.mgr
  mgr.addComps(sampleComps)
  view.popup.exec = lambda: True
  allCols = np.arange(len(view.mgr.colTitles))
  editCols = np.setdiff1d(allCols, mgr.noEditColIdxs)

  oldSetData = view.popup.setData
  def patchedSetData(*args, **kwargs):
    oldSetData(*args, **kwargs)
    view.popup.dirtyColIdxs = editCols
  view.popup.setData = patchedSetData
  numEditCols = len(editCols)
  selectionIdxs = np.tile(np.arange(len(mgr.compDf))[:,None], (numEditCols,3))
  selectionIdxs[:,2] = np.tile(editCols, len(mgr.compDf))
  overwriteData = mgr.compDf.iloc[[0]]
  view.setSelectedCellsAs(selectionIdxs, overwriteData)
  editableDf = view.mgr.compDf.iloc[:, editCols]
  cmpDf = pd.concat([mgr.compDf.iloc[[0], editCols]]*len(mgr.compDf))
  assert np.array_equal(editableDf.values, cmpDf.values)

@pytest.mark.withcomps
def test_move_comps(app, mgr, copyHelper):
  copyHelper(copyMode=False)
  oldComps = mgr.compDf.copy()
  app.compDisplay.finishRegionCopier(True)
  compCopiedCompDfs(oldComps, mgr.compDf)


@pytest.mark.withcomps
def test_copy_comps(app, mgr, copyHelper):
  copyHelper(copyMode=True)
  oldComps = mgr.compDf.copy()
  app.compDisplay.finishRegionCopier(True)
  assert len(mgr.compDf) == 2*len(oldComps)
  compCopiedCompDfs(oldComps, mgr.compDf, newStartIdx=len(oldComps))

def test_impossible_filter(tmp_path):
  tmpFile = tmp_path/'testCfg.yml'
  dummyCfgStr = '''
  table-cfg:
    fields:
      dummy:
        value: {}
        pType: nopossiblefilter
  '''
  tmpFile.write_text(dummyCfgStr)

  with pytest.warns(S3AWarning):
    FR_SINGLETON.tableData.loadCfg(tmpFile)

def compCopiedCompDfs(old: pd.DataFrame, new: pd.DataFrame, newStartIdx=0):
  for ii in range(len(old)):
    oldComp = old.iloc[ii, :]
    for jj in range(len(oldComp[REQD_TBL_FIELDS.VERTICES])):
      oldComp[REQD_TBL_FIELDS.VERTICES][jj] += 50
    oldComp.at[REQD_TBL_FIELDS.INST_ID] += newStartIdx
    assert np.array_equal(oldComp, new.iloc[newStartIdx+ii, :])

@pytest.fixture
def copyHelper(app, mgr):
  def copyHelper(copyMode=True):
    copier = app.mainImg.regionCopier
    copier.offset = XYVertices([[50, 50]])
    copier.regionIds = mgr.compDf.index
    copier.inCopyMode = copyMode
  return copyHelper