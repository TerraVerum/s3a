import shutil
import sys

import matplotlib.pyplot as plt
import pytest
from numpy import VisibleDeprecationWarning

from helperclasses import CompDfTester
from s3a import FR_SINGLETON, S3A
from testingconsts import SAMPLE_IMG, SAMPLE_IMG_FNAME, NUM_COMPS, \
  SAMPLE_SMALL_IMG, SAMPLE_SMALL_IMG_FNAME, IMG_DIR

app = S3A(Image=SAMPLE_IMG_FNAME, exceptionsAsDialogs=False)
mgr = app.compMgr
stack = FR_SINGLETON.actionStack

dfTester = CompDfTester(NUM_COMPS)
dfTester.fillRandomVerts(imShape=SAMPLE_IMG.shape)
dfTester.fillRandomClasses()

@pytest.fixture(scope="module")
def sampleComps():
  return dfTester.compDf.copy()

# Each test can request wheter it starts with components, small image, etc.
# After each test, all components are removed from the app
@pytest.fixture(autouse=True)
def resetApp_tester(request):
  if 'smallimage' in request.keywords:
    app.resetMainImg(SAMPLE_SMALL_IMG_FNAME, SAMPLE_SMALL_IMG)
  else:
    app.resetMainImg(SAMPLE_IMG_FNAME, SAMPLE_IMG)
  if 'withcomps' in request.keywords:
    dfTester.fillRandomVerts(app.mainImg.image.shape)
    dfTester.fillRandomClasses()
    mgr.addComps(dfTester.compDf.copy())
  yield
  stack.clear()
  app.clearBoundaries()


class _block_pltShow:
  def __init__(self):
    self.oldShow = None

  def __enter__(self):
    self.oldShow = plt.show
    plt.show = lambda: None

  def __exit__(self, exc_type, exc_val, exc_tb):
    plt.show = self.oldShow