import numpy as np
import pandas as pd
import cv2 as cv
import pytest

from conftest import app, mgr, stack
from s3a import appInst, FR_SINGLETON
from s3a.projectvars import REQD_TBL_FIELDS
from s3a.structures import FRComplexVertices
from s3a.views.tableview import FRCompTableView


@pytest.mark.withcomps
def test_merge_selected_comps():
  oldLen = len(mgr.compDf)
  app.compTbl.selectAll()
  appInst.processEvents()
  app.compDisplay.mergeSelectedComps()
  assert len(mgr.compDf) == 1
  # # Undo swap current comp, undo merge
  stack.undo()
  stack.undo()
  assert len(mgr.compDf) == oldLen

def test_split_selected_comps():
  compMask = np.zeros((100,100), 'uint8')
  cv.rectangle(compMask, (0, 0), (5, 5), 1, -1)
  cv.rectangle(compMask, (9, 9), (16, 16), 1, -1)
  cv.rectangle(compMask, (21, 21), (30, 30), 1, -1)
  cv.rectangle(compMask, (46, 46), (60, 60), 1, -1)
  verts = FRComplexVertices.fromBwMask(compMask > 0)
  comp = FR_SINGLETON.tableData.makeCompDf()
  comp.at[comp.index[0], REQD_TBL_FIELDS.VERTICES] = verts
  app.add_focusComp(comp)

  app.compTbl.selectAll()
  app.compDisplay.splitSelectedComps()
  assert len(mgr.compDf) == 4
  stack.undo()
  assert len(mgr.compDf) == 1

@pytest.mark.withcomps
def test_set_cells_as():
  oldCls = FR_SINGLETON.tableData.compClasses[0]
  # Even amount of comps for easy comparison
  if (len(mgr.compDf) % 2) == 1:
    mgr.rmComps(mgr.compDf.index[-1])
  mgr.compDf.loc[:, REQD_TBL_FIELDS.COMP_CLASS] = oldCls
  newCls = FR_SINGLETON.tableData.compClasses[1]
  newDf = mgr.compDf.loc[[0]]
  compClsIdx = FR_SINGLETON.tableData.allFields.index(REQD_TBL_FIELDS.COMP_CLASS)
  newDf.iat[0, compClsIdx] = newCls
  app.compTbl.setCellsAs(mgr.compDf.index[::2], [compClsIdx], newDf)
  matchList = np.tile([newCls, oldCls], len(mgr.compDf)//2)
  assert np.array_equal(mgr.compDf[REQD_TBL_FIELDS.COMP_CLASS], matchList)

def test_set_as_gui(sampleComps):
  # Monkeypatch gui for testing
  view = FRCompTableView()
  view.mgr.addComps(sampleComps)
  view.popup.exec = lambda: True
  allCols = np.arange(len(view.mgr.colTitles))
  editCols = np.setdiff1d(allCols, mgr.noEditColIdxs)

  oldSetData = view.popup.setData
  def patchedSetData(*args, **kwargs):
    oldSetData(*args, **kwargs)
    view.popup.dirtyColIdxs = editCols
  view.popup.setData = patchedSetData
  view.setSelectedCellsAs_gui(view.mgr.compDf.index, allCols)
  editableDf = view.mgr.compDf.iloc[:, editCols]
  cmpDf = pd.concat([view.mgr.compDf.iloc[[0], editCols]]*len(view.mgr.compDf))
  assert np.array_equal(editableDf.values, cmpDf.values)