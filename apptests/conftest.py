import os
from typing import Type

import pytest

os.environ['S3A_PLATFORM'] = 'minimal'
from s3a import constants
from helperclasses import CompDfTester
from s3a import FR_SINGLETON
from s3a.views.s3agui import S3A
from testingconsts import SAMPLE_IMG, SAMPLE_IMG_FNAME, NUM_COMPS, \
  SAMPLE_SMALL_IMG, SAMPLE_SMALL_IMG_FNAME, TEST_AUTHOR
from s3a.plugins.tablefield import VerticesPlugin
from s3a.plugins.file import FilePlugin

stack = FR_SINGLETON.actionStack
FR_SINGLETON.tableData.annAuthor = TEST_AUTHOR

dfTester = CompDfTester(NUM_COMPS)
dfTester.fillRandomVerts(imShape=SAMPLE_IMG.shape)

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

# Assign temporary project directory
@pytest.fixture(scope="session", autouse=True)
def app(tmpdir_factory):
  constants.APP_STATE_DIR = tmpdir_factory.mktemp('settings')
  app_ = S3A(Image=SAMPLE_IMG_FNAME, guiMode=False, loadLastState=False, author=TEST_AUTHOR)
  app_.filePlg.projData.create(name=str(tmpdir_factory.mktemp('proj')), parent=app_.filePlg.projData)
  return app_

@pytest.fixture(scope='session')
def filePlg(app):
  plg: FilePlugin = app.filePlg
  return plg

@pytest.fixture(scope="session")
def mgr(app):
  return app.compMgr

@pytest.fixture(scope='session', autouse=True)
def vertsPlugin(app):
  try:
    plg = FR_SINGLETON.clsToPluginMapping[VerticesPlugin]
  except KeyError:
    raise RuntimeError('Vertices plugin was not provided. Some tests are guaranteed to fail.')

  plg.procCollection.switchActiveProcessor('Basic Shapes')
  proc = plg.curProcessor
  for stage in proc.processor.stages:
    if stage.allowDisable:
      proc.setStageEnabled([stage.name], False)

  return plg

# Each test can request wheter it starts with components, small image, etc.
# After each test, all components are removed from the app
@pytest.fixture(autouse=True)
def resetApp_tester(request, app, filePlg, mgr):
  for img in filePlg.projData.images:
    try:
      if img != app.srcImgFname:
        filePlg.projData.removeImage(img)
    except (FileNotFoundError,):
      pass
  app.mainImg.shapeCollection.forceUnlock()
  if 'smallimage' in request.keywords:
    app.setMainImg(SAMPLE_SMALL_IMG_FNAME, SAMPLE_SMALL_IMG)
  else:
    app.setMainImg(SAMPLE_IMG_FNAME, SAMPLE_IMG)
  if 'withcomps' in request.keywords:
    dfTester.fillRandomVerts(app.mainImg.image.shape)
    mgr.addComps(dfTester.compDf.copy())
  yield
  stack.clear()
  app.clearBoundaries()

def assertExInList(exList, typ: Type[Exception]=Exception):
  assert any(issubclass(ex[0], typ) for ex in exList)