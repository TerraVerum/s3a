import matplotlib.pyplot as plt
import pytest

from helperclasses import CompDfTester, S3ATester
from s3a import FR_SINGLETON
from testingconsts import SAMPLE_IMG, SAMPLE_IMG_FNAME, NUM_COMPS, \
  SAMPLE_SMALL_IMG, SAMPLE_SMALL_IMG_FNAME

app = S3ATester(Image=SAMPLE_IMG_FNAME, exceptionsAsDialogs=False)
mgr = app.compMgr
stack = FR_SINGLETON.actionStack

dfTester = CompDfTester(NUM_COMPS)
dfTester.fillRandomVerts(imShape=SAMPLE_IMG.shape)
dfTester.fillRandomClasses()

@pytest.fixture(scope="module")
def sampleComps():
  return dfTester.compDf.copy()

@pytest.fixture(autouse=True)
def resetApp_tester(request):
  if 'noclear' in request.keywords:
    return
  if 'smallimage' in request.keywords:
    app.resetMainImg(SAMPLE_SMALL_IMG_FNAME, SAMPLE_SMALL_IMG)
  else:
    app.resetMainImg(SAMPLE_IMG_FNAME, SAMPLE_IMG)
  if 'withcomps' in request.keywords:
    mgr.addComps(dfTester.compDf.copy())
    return
  app.clear()
  dfTester.fillRandomVerts(app.mainImg.image.shape)
  dfTester.fillRandomClasses()
  return


class _block_pltShow:
  def __init__(self):
    self.oldShow = None

  def __enter__(self):
    self.oldShow = plt.show
    plt.show = lambda: None

  def __exit__(self, exc_type, exc_val, exc_tb):
    plt.show = self.oldShow