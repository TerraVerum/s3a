from typing import Sequence

import numpy as np
import pytest

from pyqtgraph.Qt import QtTest, QtGui, QtCore

from cdef.frgraphics.rois import FRPolygonROI

QTest = QtTest.QTest

from appsetup import (NUM_COMPS, RND, defaultApp_tester)
from cdef import FR_SINGLETON
from cdef.frgraphics.imageareas import FRFocusedImage
from cdef.projectvars import REQD_TBL_FIELDS, FR_CONSTS

# Construct app outside setUp to drastically reduce loading times
app, dfTester = defaultApp_tester()

mgr = app.compMgr
moduleFImg = app.focusedImg
# Make the processor wellformed
moduleFImg.procCollection.switchActiveProcessor('Basic Shapes')
proc = moduleFImg.curProcessor
for stage in proc.processor.stages:
  if stage.allowDisable:
    proc.setStageEnabled([stage.name], False)
stack = FR_SINGLETON.actionStack

mgr.addComps(dfTester.compDf)

def leftClickGen(pos: Sequence, dbclick=False):
  Ev = QtCore.QEvent
  Qt = QtCore.Qt
  if dbclick:
    typ = Ev.MouseButtonDblClick
  else:
    typ = Ev.MouseButtonPress
  pos = QtCore.QPointF(*pos)
  out = QtGui.QMouseEvent(typ, moduleFImg.imgItem.mapToScene(pos),
                          Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)
  return out

@pytest.fixture
def clearFImg():
  moduleFImg.resetImage()
  return moduleFImg

@pytest.fixture
def fImg(clearFImg):
  app.updateCurComp(mgr.compDf.iloc[[0],:])
  return moduleFImg

@pytest.fixture
def polyRoi():
  roi = FRPolygonROI()

def test_update(clearFImg):
  assert clearFImg.image is None
  mgr.addComps(dfTester.compDf.copy())
  focusedId = RND.integers(NUM_COMPS)
  newCompDf = mgr.compDf.loc[focusedId]
  clearFImg.updateAll(app.mainImg.image, newCompDf)
  assert clearFImg.image is not None
  assert np.array_equal(clearFImg.compSer[REQD_TBL_FIELDS.VERTICES],
                        newCompDf[REQD_TBL_FIELDS.VERTICES])

def test_region_modify(fImg: FRFocusedImage):
  shapeBnds = fImg.image.shape[:2]
  reach = np.min(shapeBnds)
  fImg.shapeCollection.curShape = FR_CONSTS.DRAW_SHAPE_POLY
  fImg.drawAction = FR_CONSTS.DRAW_ACT_ADD
  fImg.mousePressEvent(leftClickGen((5,5)))
  fImg.mousePressEvent(leftClickGen((reach,reach)))
  fImg.mousePressEvent(leftClickGen((reach,5)))
  fImg.mousePressEvent(leftClickGen((5,5)))
  assert fImg.region.image.sum() > 0