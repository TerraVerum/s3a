import cv2 as cv
import numpy as np
import pytest
from pyqtgraph.Qt import QtTest, QtGui, QtCore
from skimage.measure import points_in_poly

from s3a.frgraphics.regions import FRShapeCollection
from s3a.structures import FRVertices, FRComplexVertices

QTest = QtTest.QTest

from appsetup import (NUM_COMPS, RND, defaultApp_tester)
from s3a import FR_SINGLETON
from s3a.frgraphics.imageareas import FRFocusedImage
from s3a.projectvars import REQD_TBL_FIELDS, FR_CONSTS

# Construct app outside setUp to drastically reduce loading times
app, dfTester = defaultApp_tester()

mgr = app.compMgr
# Make the processor wellformed
app.focusedImg.procCollection.switchActiveProcessor('Basic Shapes')
proc = app.focusedImg.curProcessor
for stage in proc.processor.stages:
  if stage.allowDisable:
    proc.setStageEnabled([stage.name], False)
stack = FR_SINGLETON.actionStack

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
def clearFImg():
  app.focusedImg.resetImage()
  return app.focusedImg

@pytest.fixture
def fImg(clearFImg):
  app.changeFocusedComp(mgr.compDf.iloc[[0], :])
  return app.focusedImg

@pytest.fixture
def roiFactory():
  clctn = FRShapeCollection((FR_CONSTS.DRAW_SHAPE_POLY, FR_CONSTS.DRAW_SHAPE_RECT),
                            app.focusedImg)
  def _polyRoi(pts: FRVertices):
    clctn.curShape = FR_CONSTS.DRAW_SHAPE_RECT
    for pt in pts:
      ev = leftClickGen(pt)
      clctn.buildRoi(ev)

  return _polyRoi


def test_update(clearFImg: FRFocusedImage):
  assert clearFImg.image is None
  mgr.addComps(dfTester.compDf.copy())
  focusedId = NUM_COMPS
  newCompSer = mgr.compDf.loc[focusedId]
  # Action 1
  clearFImg.updateAll(app.mainImg.image, newCompSer)
  assert clearFImg.image is not None
  assert clearFImg.compSer.equals(newCompSer)
  assert np.array_equal(clearFImg.bbox[1,:] - clearFImg.bbox[0,:], clearFImg.image.shape[:2][::-1])

  # Action 2
  newerSer = mgr.compDf.loc[0]
  clearFImg.updateAll(app.mainImg.image, newerSer)

  FR_SINGLETON.actionStack.undo()
  assert clearFImg.compSer.equals(newCompSer)
  FR_SINGLETON.actionStack.undo()
  assert clearFImg.image is None

  FR_SINGLETON.actionStack.redo()
  assert clearFImg.compSer.equals(newCompSer)
  FR_SINGLETON.actionStack.redo()
  assert clearFImg.compSer.equals(newerSer)




def test_region_modify(fImg: FRFocusedImage):
  shapeBnds = fImg.image.shape[:2]
  reach = np.min(shapeBnds)
  oldVerts = fImg.region.verts
  fImg.shapeCollection.curShape = FR_CONSTS.DRAW_SHAPE_POLY
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
