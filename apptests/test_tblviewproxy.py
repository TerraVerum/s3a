import warnings

import cv2 as cv
import numpy as np
import pandas as pd
import pytest
from pyqtgraph.Qt import QtCore

from s3a import appInst
from s3a.constants import REQD_TBL_FIELDS
from s3a.generalutils import imgCornerVertices, cvImread_rgb
from s3a.structures import ComplexXYVertices, XYVertices
from s3a.views.tableview import CompTableView, PopupTableDialog


@pytest.mark.withcomps
def test_merge_selected_comps(app, mgr):
  oldLen = len(mgr.compDf)
  app.compTbl.selectAll()
  appInst.processEvents()
  assert len(app.compDisplay.selectedIds) > 0
  app.compDisplay.mergeSelectedComps()
  assert len(mgr.compDf) == 1
  mgr.actionStack.undo()
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
  verts = ComplexXYVertices.fromBinaryMask(compMask > 0)
  comp = mgr.tableData.makeCompDf()
  comp.at[comp.index[0], REQD_TBL_FIELDS.VERTICES] = verts
  app.clearBoundaries()
  app.add_focusComps(comp)

  app.compTbl.selectAll()
  app.compDisplay.splitSelectedComps()
  assert len(mgr.compDf) == 4
  mgr.actionStack.undo()
  assert len(mgr.compDf) == 1
  # Nothing should happen
  app.compDisplay.selectedIds = mgr.compDf.index
  app.compDisplay.splitSelectedComps()

def test_rm_overlap(app):
  verts = [imgCornerVertices(np.zeros(shape)) for shape in ([100,100], [200,200])]
  comps = app.sharedAttrs.tableData.makeCompDf(2)
  comps[REQD_TBL_FIELDS.VERTICES] = [ComplexXYVertices([v]) for v in verts]
  cd = app.compDisplay
  changeDict = app.add_focusComps(comps)
  comps[REQD_TBL_FIELDS.INST_ID] = changeDict['ids']
  old = comps.copy()
  cd.selectedIds = changeDict['ids']
  cd.removeSelectedCompOverlap()
  assert len(app.compMgr.compDf) == 1

  app.clearBoundaries()
  changeDict = app.add_focusComps(old)
  cd.selectedIds = changeDict['ids'][::-1]
  cd.removeSelectedCompOverlap()
  assert len(app.compMgr.compDf) == 2
  checkVerts = app.compMgr.compDf.loc[changeDict['ids'][-1], REQD_TBL_FIELDS.VERTICES]
  assert len(checkVerts[0]) > 4

@pytest.mark.withcomps
def test_set_cells_as(app, mgr):
  oldSrcFile = app.srcImgFname
  # Even amount of comps for easy comparison
  if (len(mgr.compDf) % 2) == 1:
    mgr.rmComps(mgr.compDf.index[-1])
  mgr.compDf[REQD_TBL_FIELDS.IMG_FILE] = oldSrcFile
  # Ensure the overwrite data will be different from what it's overwriting
  newFile = 'TestFile.png'
  newDf = mgr.compDf.loc[[0]]
  srcFileIdx = mgr.tableData.allFields.index(REQD_TBL_FIELDS.IMG_FILE)
  newDf.iat[0, srcFileIdx] = newFile
  selection = np.column_stack([mgr.compDf.index[::2], mgr.compDf.index[::2], np.tile(srcFileIdx, len(mgr.compDf)//2)])
  # Sometimes Qt doesn't process selections programmatically. Not sure what to do about that
  if len(selection) == 0:
    return
  app.compTbl.setSelectedCellsAs(selection, newDf)
  matchList = np.tile([newFile, oldSrcFile], len(mgr.compDf)//2)
  # Irritating that sometimes windows path comparisons fail despite having the same str
  # representations
  for entryA, entryB in zip(mgr.compDf[REQD_TBL_FIELDS.IMG_FILE].to_list(),
                            matchList):
    assert str(entryA) == str(entryB)

def test_set_as_gui(sampleComps):
  # Monkeypatch gui for testing
  view = CompTableView()
  mgr = view.mgr
  mgr.addComps(sampleComps)
  view.popup.exec = lambda: True
  allCols = np.arange(len(view.mgr.colTitles))
  editCols = np.setdiff1d(allCols, mgr.noEditColIdxs)

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
  compareCopiedCompDfs(oldComps, mgr.compDf, app.compDisplay.regionMover.dataMin)

@pytest.mark.withcomps
def test_copy_comps(app, mgr, copyHelper):
  copyHelper(copyMode=True)
  oldComps = mgr.compDf.copy()
  app.compDisplay.finishRegionCopier(True)
  assert len(mgr.compDf) == 2*len(oldComps)
  compareCopiedCompDfs(oldComps, mgr.compDf,
                       app.compDisplay.regionMover.dataMin, newStartIdx=len(oldComps))

def test_scale_viewbox(app, mgr):
  verts = ComplexXYVertices([XYVertices([[0, 0], [35, 35]])])
  comps = mgr.tableData.makeCompDf(1)
  comps[REQD_TBL_FIELDS.VERTICES] = [verts]
  changeDict = mgr.addComps(comps)

  app.compDisplay.scaleViewboxToSelectedIds(changeDict['ids'], paddingPct=0)
  bounds = np.array(app.mainImg.getViewBox().targetRange())
  assert np.array_equal(bounds, [[0, 35], [0, 35]])

@pytest.mark.smallimage
def test_export_overlay(app, mgr, tmp_path):
  verts = imgCornerVertices(app.mainImg.image)
  comps = mgr.tableData.makeCompDf(1)
  comps[REQD_TBL_FIELDS.VERTICES] = [ComplexXYVertices([verts])]
  app.compDisplay.regionPlot.showFocused = True
  app.add_focusComps(comps)
  app.compDisplay.regionPlot.updateColors(labelColormap='Reds', fillAlpha=1.0)
  exportLoc = str(tmp_path/'export.png')
  app.compDisplay.exportCompOverlay(file=exportLoc)
  app.compDisplay.regionPlot.showFocused = False
  img = cvImread_rgb(exportLoc)
  checkPix = img[20,20,:]
  # Red channel should be largest for overlay export and red colormap
  assert np.array_equal(checkPix, [255,0,0])

@pytest.mark.withcomps
def test_sorted_tbl(app, mgr):
  proxy = app.sortFilterProxy
  proxy.sort(0, QtCore.Qt.SortOrder.DescendingOrder)
  ordering = [proxy.index(ii,0).data(QtCore.Qt.ItemDataRole.EditRole) for ii in range(len(mgr.compDf))]
  assert np.array_equal(mgr.compDf.index[::-1], ordering)


def compareCopiedCompDfs(old: pd.DataFrame, new: pd.DataFrame, newOffset, newStartIdx=0):
  newOffset = newOffset.astype(int)
  for ii in range(len(old)):
    oldComp = old.iloc[ii, :].copy()
    for jj in range(len(oldComp[REQD_TBL_FIELDS.VERTICES])):
      oldComp[REQD_TBL_FIELDS.VERTICES][jj] += newOffset
    oldComp.at[REQD_TBL_FIELDS.INST_ID] += newStartIdx
    assert np.array_equal(oldComp, new.iloc[newStartIdx+ii, :])

@pytest.fixture
def copyHelper(app, mgr):
  def copyHelper(copyMode=True):
    copier = app.mainImg.regionMover
    copier.resetBaseData(mgr.compDf.copy())
    copier.inCopyMode = copyMode
  return copyHelper

def test_minimal_model(sampleComps):
  view = PopupTableDialog()
  comp = sampleComps.iloc[[0]]
  numCols = len(comp.columns)
  idxs = np.arange(numCols)
  view.setData(comp, idxs, idxs[[-1]])
  model = view.model
  csrole = QtCore.Qt.ItemDataRole.CheckStateRole
  for colIdx in idxs:
    modelIdx = model.index(0, colIdx)
    state = model.data(modelIdx, csrole)
    assert state == model.csMap[colIdx == idxs[-1]]

  colIdx = comp.columns.get_loc(REQD_TBL_FIELDS.INST_ID)
  modelIdx = model.index(0, colIdx)
  model.setData(modelIdx, model.csMap[True], csrole)
  assert colIdx in model.checkedColIdxs
  model.setData(modelIdx, model.csMap[False], csrole)
  assert colIdx not in model.checkedColIdxs
