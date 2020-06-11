import pytest

from appsetup import CompDfTester, SAMPLE_IMG_FNAME, NUM_COMPS, SAMPLE_IMG

# Construct app outside setUp to drastically reduce loading times
from s3a import S3A, FR_SINGLETON

app = S3A(Image=SAMPLE_IMG_FNAME)
mgr = app.compMgr
stack = FR_SINGLETON.actionStack

dfTester = CompDfTester(NUM_COMPS)
dfTester.fillRandomVerts(imShape=SAMPLE_IMG.shape)
dfTester.fillRandomClasses()


@pytest.fixture(scope="module")
def sampleComps():
  return dfTester.compDf.copy()