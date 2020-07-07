import numpy as np
import cv2 as cv
import pytest

from conftest import app, mgr, stack
from s3a import appInst, FR_SINGLETON
from s3a.projectvars import REQD_TBL_FIELDS
from s3a.structures import FRComplexVertices

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

