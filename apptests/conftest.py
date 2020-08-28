import pytest
import os

from s3a.structures import FRS3AException

os.environ['S3A_PLATFORM'] = 'minimal'
from helperclasses import CompDfTester
from s3a import FR_SINGLETON, FRTableVertsPlugin
from s3a.views.s3agui import S3A
from testingconsts import SAMPLE_IMG, SAMPLE_IMG_FNAME, NUM_COMPS, \
  SAMPLE_SMALL_IMG, SAMPLE_SMALL_IMG_FNAME

app = S3A(Image=SAMPLE_IMG_FNAME, guiMode=False, loadLastState=False, author='testauthor')
mgr = app.compMgr
vertsPlugin = None
for plugin in FR_SINGLETON.tableFieldPlugins:
  if isinstance(plugin, FRTableVertsPlugin):
    vertsPlugin = plugin
    break
if vertsPlugin is None:
  raise FRS3AException('Vertices plugin was not provided. Some tests are guaranteed to fail.')
stack = FR_SINGLETON.actionStack

dfTester = CompDfTester(NUM_COMPS)
dfTester.fillRandomVerts(imShape=SAMPLE_IMG.shape)
dfTester.fillRandomClasses()

# @pytest.fixture(scope='session', autouse=True)
# def cfg_warnings():
#   oldWarnings = warnings.showwarning
#   def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
#
#     log = file if hasattr(file,'write') else sys.stderr
#     traceback.print_stack(file=log)
#     log.write(warnings.formatwarning(message, category, filename, lineno, line))
#
#   warnings.showwarning = warn_with_traceback
#   yield
#   # Set back to normal
#   warnings.showwarning = oldWarnings

@pytest.fixture(scope="module")
def sampleComps():
  return dfTester.compDf.copy()

# Each test can request wheter it starts with components, small image, etc.
# After each test, all components are removed from the app
@pytest.fixture(autouse=True)
def resetApp_tester(request):
  if 'smallimage' in request.keywords:
    app.setMainImg(SAMPLE_SMALL_IMG_FNAME, SAMPLE_SMALL_IMG)
  else:
    app.setMainImg(SAMPLE_IMG_FNAME, SAMPLE_IMG)
  if 'withcomps' in request.keywords:
    dfTester.fillRandomVerts(app.mainImg.image.shape)
    dfTester.fillRandomClasses()
    mgr.addComps(dfTester.compDf.copy())
  yield
  stack.clear()
  app.clearBoundaries()