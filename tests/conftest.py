import pytest

from appsetup import CompDfTester, SAMPLE_IMG_DIR, NUM_COMPS, SAMPLE_IMG

# Construct app outside setUp to drastically reduce loading times
from cdef import FRCdefApp, FR_SINGLETON

app = FRCdefApp(Image=SAMPLE_IMG_DIR)
mgr = app.compMgr
stack = FR_SINGLETON.actionStack

dfTester = CompDfTester(NUM_COMPS)
dfTester.fillRandomVerts(imShape=SAMPLE_IMG.shape)
dfTester.fillRandomClasses()


@pytest.fixture(scope="module")
def sampleComps():
  return dfTester.compDf.copy()