import numpy as np
import pytest
from pyqtgraph.Qt import QtGui, QtCore

from testingconsts import NUM_COMPS
from s3a import PRJ_SINGLETON, REQD_TBL_FIELDS
from s3a.constants import PRJ_CONSTS
from s3a.controls.drawctrl import RoiCollection
from s3a.parameditors.algcollection import AlgParamEditor
from s3a.processing import ProcessIO, ImageProcess, ImgProcWrapper
from s3a.structures import XYVertices, ComplexXYVertices, PrjParam, NChanImg

def leftClickGen(pos: XYVertices, dbclick=False):
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
def roiFactory(app):
  clctn = RoiCollection((PRJ_CONSTS.DRAW_SHAPE_POLY, PRJ_CONSTS.DRAW_SHAPE_RECT),
                        app.mainImg)
  def _polyRoi(pts: XYVertices, shape: PrjParam=PRJ_CONSTS.DRAW_SHAPE_RECT):
    clctn.curShapeParam = shape
    for pt in pts:
      ev = leftClickGen(pt)
      clctn.buildRoi(ev)
    return clctn.curShape

  return _polyRoi

@pytest.mark.withcomps
def test_update(app, mgr, vertsPlugin):
  mImg = app.mainImg
  focusedId = NUM_COMPS-1
  newCompSer = mgr.compDf.loc[focusedId]
  # Action 1
  mImg.updateFocusedComp(newCompSer)
  assert mImg.compSer.equals(newCompSer)

  # Action 2
  newerSer = mgr.compDf.loc[0]
  mImg.updateFocusedComp(newerSer)

  PRJ_SINGLETON.actionStack.undo()
  assert mImg.compSer.equals(newCompSer)
  PRJ_SINGLETON.actionStack.undo()
  assert mImg.compSer[REQD_TBL_FIELDS.INST_ID] == -1
  PRJ_SINGLETON.actionStack.redo()
  assert mImg.compSer.equals(newCompSer)
  PRJ_SINGLETON.actionStack.redo()
  assert mImg.compSer.equals(newerSer)

def test_region_modify(sampleComps, app, mgr, vertsPlugin):
  mImg = app.mainImg
  app.add_focusComps(sampleComps)
  shapeBnds = mImg.image.shape[:2]
  reach = np.min(shapeBnds)
  oldData = vertsPlugin.region.regionData
  mImg.shapeCollection.curShapeParam = PRJ_CONSTS.DRAW_SHAPE_POLY
  mImg.drawAction = PRJ_CONSTS.DRAW_ACT_ADD
  imsum = lambda: vertsPlugin.region.toGrayImg(shapeBnds).sum()

  # 1st action
  app.mainImg.updateFocusedComp(None)
  assert imsum() == 0

  newVerts = XYVertices([[5,5], [reach, reach], [reach, 5], [5,5]])
  cplxVerts = ComplexXYVertices([newVerts])
  newMask = cplxVerts.toMask(shapeBnds, asBool=False) > 0

  # 2nd action
  app.mainImg.updateFocusedComp(sampleComps.iloc[-1])
  mImg.shapeCollection.sigShapeFinished.emit(newVerts)
  checkpointMask = vertsPlugin.region.toGrayImg(shapeBnds) > 0
  assert np.any(checkpointMask)

  PRJ_SINGLETON.actionStack.undo()
  PRJ_SINGLETON.actionStack.undo()
  # Once for the shape, again for the focus
  # Cmp to first action
  assert imsum() == 0
  PRJ_SINGLETON.actionStack.undo()
  # Cmp to original
  assert vertsPlugin.region.regionData[REQD_TBL_FIELDS.VERTICES].equals(oldData[REQD_TBL_FIELDS.VERTICES])

  PRJ_SINGLETON.actionStack.redo()
  assert imsum() == 0
  PRJ_SINGLETON.actionStack.redo()
  PRJ_SINGLETON.actionStack.redo()
  pluginMask = vertsPlugin.region.toGrayImg(shapeBnds)
  assert np.array_equal(pluginMask, checkpointMask)

@pytest.mark.withcomps
def test_selectionbounds_all(app, mgr):
  imBounds = app.mainImg.image.shape[:2][::-1]
  bounds = XYVertices([[0,0],
                       [0, imBounds[1]],
                       [imBounds[0], imBounds[1]],
                        [imBounds[0], 0]])
  app.compDisplay.reflectSelectionBoundsMade(bounds)
  assert len(app.compDisplay.selectedIds) == len(mgr.compDf)

@pytest.mark.withcomps
def test_selectionbounds_none(app):
  app.compTbl.clearSelection()
  app.compDisplay.selectedIds = np.array([], dtype=int)
  # Selection in negative area ensures no comps will be selected
  app.compDisplay.reflectSelectionBoundsMade(XYVertices([[-100,-100]]))
  assert len(app.compDisplay.selectedIds) == 0

def test_proc_err(tmp_path):
  def badProc(image: NChanImg):
    return ProcessIO(image=image, extra=1 / 0)
  newCtor = lambda: ImageProcess.fromFunction(badProc, name='Bad')
  newClctn = AlgParamEditor(tmp_path, [newCtor], ImgProcWrapper)

  newClctn.switchActiveProcessor('Bad')
  with pytest.warns(UserWarning):
    newClctn.curProcessor.run(image=np.array([[True]], dtype=bool), fgVerts=XYVertices([[0,0]]))
