import numpy as np
import pytest

from appsetup import (NUM_COMPS, RND, defaultApp_tester)
from cdef import FR_SINGLETON
from cdef.projectvars import REQD_TBL_FIELDS

# Construct app outside setUp to drastically reduce loading times
app, dfTester = defaultApp_tester()

mgr = app.compMgr
moduleFImg = app.focusedImg
stack = FR_SINGLETON.actionStack

mgr.addComps(dfTester.compDf)

@pytest.fixture
def fImg():
  moduleFImg.resetImage()
  return moduleFImg

def test_update(fImg):
  assert fImg.image is None
  mgr.addComps(dfTester.compDf.copy())
  focusedId = RND.integers(NUM_COMPS)
  newCompDf = mgr.compDf.loc[focusedId]
  fImg.updateAll(app.mainImg.image, newCompDf)
  assert fImg.image is not None
  assert np.array_equal(fImg.compSer[REQD_TBL_FIELDS.VERTICES],
                        newCompDf[REQD_TBL_FIELDS.VERTICES])

