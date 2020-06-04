from unittest import TestCase

import numpy as np

from appsetup import (CompDfTester, NUM_COMPS, SAMPLE_IMG,
                      SAMPLE_IMG_DIR, RND)
from cdef import FRCdefApp, FR_SINGLETON
from cdef.projectvars import REQD_TBL_FIELDS

# Construct app outside setUp to drastically reduce loading times
app = FRCdefApp(Image=SAMPLE_IMG_DIR)
mgr = app.compMgr
fImg = app.focusedImg
stack = FR_SINGLETON.actionStack

dfTester = CompDfTester(NUM_COMPS)
dfTester.fillRandomVerts(imShape=SAMPLE_IMG.shape)
dfTester.fillRandomClasses()

mgr.addComps(dfTester.compDf)


class FocusedImageTester(TestCase):
  def setUp(self):
    fImg.resetImage()

  def test_update(self):
    assert fImg.image is None
    mgr.addComps(dfTester.compDf.copy())
    focusedId = RND.integers(NUM_COMPS)
    newCompDf = mgr.compDf.loc[focusedId]
    fImg.updateAll(app.mainImg.image, newCompDf)
    assert fImg.image is not None
    assert np.array_equal(fImg.compSer[REQD_TBL_FIELDS.VERTICES],
                          newCompDf[REQD_TBL_FIELDS.VERTICES])

