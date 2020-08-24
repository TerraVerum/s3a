from pathlib import Path

import numpy as np
import pytest
from pyqtgraph.Qt import QtGui, QtCore

from conftest import NUM_COMPS, app, mgr, dfTester
from s3a import FR_SINGLETON
from s3a.controls.drawctrl import FRRoiCollection
from s3a.constants import FR_CONSTS, REQD_TBL_FIELDS as RTF
from s3a.processing import FRProcessIO, FRImageProcess
from s3a.structures import FRVertices, FRComplexVertices, FRParam, FRS3AWarning, NChanImg
from s3a.parameditors.algcollection import FRAlgCollectionEditor
from testingconsts import FIMG_SER_COLS

# Construct app outside setUp to drastically reduce loading times
# Make the processor wellformed
app.focusedImg.procCollection.switchActiveProcessor('Basic Shapes')
proc = app.focusedImg.curProcessor
for stage in proc.processor.stages:
  if stage.allowDisable:
    proc.setStageEnabled([stage.name], False)
stack = FR_SINGLETON.actionStack

fImg = app.focusedImg

mgr.addComps(dfTester.compDf)

def leftClickGen(pos: FRVertices, dbclick=False):
  Ev = QtCore.QEvent
  Qt = QtCore.Qt
  if dbclick:
    typ = Ev.MouseButtonDblClick
  else:
    typ = Ev.MouseButtonPress
  pos = QtCore.QPointF(*pos.asPoint())
  out = QtGui.QMouseEvent(typ, pos, Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)
  return out

@pytest.fixture
def roiFactory():
  clctn = FRRoiCollection((FR_CONSTS.DRAW_SHAPE_POLY, FR_CONSTS.DRAW_SHAPE_RECT),
                          app.focusedImg)
  def _polyRoi(pts: FRVertices, shape: FRParam=FR_CONSTS.DRAW_SHAPE_RECT):
    clctn.curShapeParam = shape
    for pt in pts:
      ev = leftClickGen(pt)
      clctn.buildRoi(ev)
    return clctn.curShape

  return _polyRoi

@pytest.mark.withcomps
def test_update():
  assert fImg.image is None
  focusedId = NUM_COMPS-1
  newCompSer = mgr.compDf.loc[focusedId]
  # Action 1
  fImg.updateAll(app.mainImg.image, newCompSer)
  newCompSer = newCompSer[FIMG_SER_COLS]
  assert fImg.image is not None
  assert fImg.compSer.equals(newCompSer)
  assert np.array_equal(fImg.bbox[1,:] - fImg.bbox[0,:], fImg.image.shape[:2][::-1])

  # Action 2
  newerSer = mgr.compDf.loc[0]
  fImg.updateAll(app.mainImg.image, newerSer)

  FR_SINGLETON.actionStack.undo()
  assert fImg.compSer.equals(newCompSer)
  FR_SINGLETON.actionStack.undo()
  assert fImg.image is None

  FR_SINGLETON.actionStack.redo()
  assert fImg.compSer.equals(newCompSer)
  FR_SINGLETON.actionStack.redo()
  assert fImg.compSer.equals(newerSer[FIMG_SER_COLS])

def test_region_modify(sampleComps):
  app.add_focusComp(sampleComps)
  shapeBnds = fImg.image.shape[:2]
  reach = np.min(shapeBnds)
  oldVerts = fImg.region.verts
  fImg.shapeCollection.curShapeParam = FR_CONSTS.DRAW_SHAPE_POLY
  fImg.drawAction = FR_CONSTS.DRAW_ACT_ADD
  imsum = lambda: fImg.region.image.sum()

  # 1st action
  fImg.updateRegionFromVerts(None)
  assert imsum() == 0

  newVerts = FRVertices([[5,5], [reach, reach], [reach, 5], [5,5]])
  cplxVerts = FRComplexVertices([newVerts])
  newMask = cplxVerts.toMask(shapeBnds)
  newMask = newMask > 0

  # 2nd action
  fImg.handleShapeFinished(newVerts)
  assert np.array_equal(fImg.region.embedMaskInImg(shapeBnds), newMask)

  FR_SINGLETON.actionStack.undo()
  # Cmp to first action
  assert imsum() == 0
  FR_SINGLETON.actionStack.undo()
  # Cmp to original
  assert fImg.region.verts == oldVerts

  FR_SINGLETON.actionStack.redo()
  assert imsum() == 0
  FR_SINGLETON.actionStack.redo()
  assert np.array_equal(fImg.region.embedMaskInImg(shapeBnds), newMask)


@pytest.mark.withcomps
def test_selectionbounds_all():
  imBounds = app.mainImg.image.shape[:2][::-1]
  bounds = FRVertices([[0,0],
                       [0, imBounds[1]],
                       [imBounds[0], imBounds[1]],
                        [imBounds[0], 0]])
  app.mainImg.sigSelectionBoundsMade.emit(bounds)
  assert len(app.compDisplay.selectedIds) == len(mgr.compDf)

@pytest.mark.withcomps
def test_selectionbounds_none():
  app.compTbl.clearSelection()
  app.compDisplay.selectedIds = np.array([], dtype=int)
  # Selection in negative area ensures no comps will be selected
  app.mainImg.sigSelectionBoundsMade.emit(FRVertices([[-100,-100]]))
  assert len(app.compDisplay.selectedIds) == 0

# def test_override_comp():
#   mImg = app.mainImg
#   mImg.procCollection.switchActiveProcessor('Basic Shapes')
#   rmSmallCompsParam = mImg.procCollection.params.child(
#     'Basic Shapes', 'Basic Region Operations', 'Rm Small Comps', 'minSzThreshold')
#   # Make sure the drawn comp is deleted
#   imShape = np.asarray(mImg.image.shape[:2])
#   rmSmallCompsParam.setValue(np.prod(imShape))
#   newVerts = FRVertices([
#     [0, 0], [1, 0], [1, 1], [0, 1]
#   ])*imShape//2
#   mImg.handleShapeFinished(newVerts)
#   assert len(mgr.compDf) == 0
#   mImg.overrideCompVertsAct.activate()
#   assert len(mgr.compDf) == 1
#   assert mgr.compDf.at[0, RTF.VERTICES] == FRComplexVertices([newVerts])
#
#   stack.undo()
#   assert len(mgr.compDf) == 0

def test_proc_err(tmpdir):
  def badProc(image: NChanImg):
    return FRProcessIO(image=image, extra=1/0)
  newCtor = lambda: FRImageProcess.fromFunction(badProc, 'Bad')
  newClctn = FRAlgCollectionEditor(Path(tmpdir),[newCtor])

  newClctn.switchActiveProcessor('Bad')
  with pytest.warns(FRS3AWarning):
    newClctn.curProcessor.run(image=np.array([[True]], dtype=bool), fgVerts=FRVertices([[0,0]]))
