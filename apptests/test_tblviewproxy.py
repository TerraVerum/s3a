from conftest import app, mgr, stack
from s3a import appInst


def test_merge_selected_comps(sampleComps):
  mgr.addComps(sampleComps)
  app.compTbl.selectAll()
  app.compDisplay.mergeSelectedComps()
  assert len(mgr.compDf) == 1
  # Undo swap current comp, undo merge
  stack.undo()
  stack.undo()
  assert len(mgr.compDf) == len(sampleComps)