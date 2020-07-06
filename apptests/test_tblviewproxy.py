import pytest
from pyqtgraph.Qt import QtCore
from s3a import appInst

from conftest import app, mgr, stack


@pytest.mark.noclear
def test_merge_selected_comps(sampleComps):
  mgr.addComps(sampleComps)
  app.compTbl.selectAll()
  appInst.processEvents()
  app.compDisplay.mergeSelectedComps()
  assert len(mgr.compDf) == 1
  # # Undo swap current comp, undo merge
  stack.undo()
  stack.undo()
  assert len(mgr.compDf) == len(sampleComps)