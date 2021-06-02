import shutil

import pytest

from apptests.helperclasses import CompDfTester
from apptests.testingconsts import SAMPLE_SMALL_IMG_FNAME, SAMPLE_SMALL_IMG, SAMPLE_IMG_FNAME, TEST_FILE_DIR
from apptests.conftest import dfTester
from s3a import S3A
from s3a.plugins.file import ProjectData, FilePlugin
from utilitys import fns


@pytest.fixture
def tmpProj(tmp_path):
  return ProjectData.create(name=tmp_path / 'tmpproj')

@pytest.fixture
def prjWithSavedStuff(tmpProj):
  tester = CompDfTester(50)
  tester.fillRandomVerts(SAMPLE_SMALL_IMG.shape[:2])
  tmpProj.addAnnotation(data=tester.compDf, image=SAMPLE_SMALL_IMG_FNAME)
  tmpProj.addAnnotation(data=tester.compDf, image=SAMPLE_IMG_FNAME)
  return tmpProj

def test_create_project(app, sampleComps, tmpProj):
  dummyAnnFile  = SAMPLE_SMALL_IMG_FNAME.with_suffix(f'{SAMPLE_SMALL_IMG_FNAME.suffix}.pkl')
  tmpProj.loadCfg(tmpProj.cfgFname, {'images': [SAMPLE_SMALL_IMG_FNAME], 'annotation-format': 'pkl'},
                  force=True)
  tmpProj.addAnnotation(dummyAnnFile, dfTester.compDf, image=SAMPLE_SMALL_IMG_FNAME.name)

  assert tmpProj.cfgFname.exists()
  assert tmpProj.annotationsDir.exists()
  assert tmpProj.imagesDir.exists()
  tmpProj.addAnnotation(data=sampleComps, image=SAMPLE_SMALL_IMG_FNAME)
  assert len(list(tmpProj.imagesDir.glob(SAMPLE_SMALL_IMG_FNAME.name))) == 1
  assert len(list(tmpProj.annotationsDir.glob(dummyAnnFile.name))) == 1

def test_update_props(filePlg):
  annFmt = lambda: filePlg.projData.cfg['annotation-format']
  oldFmt = annFmt()
  filePlg.updateProjectProperties(annotationFormat='pkl')
  assert annFmt() == 'pkl'
  filePlg.updateProjectProperties(annotationFormat=oldFmt)
  loc = filePlg.projData.location/'newcfg.tblcfg'
  newCfg = {'fields': {'Class': ''}}
  fns.saveToFile(newCfg, loc)
  oldName = filePlg.projData.tableData.cfgFname
  filePlg.updateProjectProperties(tableConfig=loc)
  assert newCfg == filePlg.projData.tableData.cfg
  filePlg.updateProjectProperties(tableConfig=oldName)

@pytest.mark.withcomps
def test_export(prjWithSavedStuff, tmp_path):
  prj = prjWithSavedStuff

  out = tmp_path/'my-project'
  prj.exportProj(out)
  assert out.exists()
  for fileA, fileB in zip(out.rglob('*.*'), prj.location.rglob('*.*')):
    assert fileA.name == fileB.name

@pytest.mark.withcomps
def test_export_anns(prjWithSavedStuff, tmp_path):
  prj = prjWithSavedStuff
  outpath = tmp_path/'export-anns'
  prj.exportAnnotations(outpath, combine=True)
  assert (outpath/'annotations.csv').exists()
  assert (outpath/'images').exists()
  assert len(list((outpath/'images').iterdir())) == 2

  for typ in ['csv', 'pkl']:
    shutil.rmtree(outpath, ignore_errors=True)
    prj.exportAnnotations(outpath, annotationFormat=typ)
    assert (outpath/'annotations').exists()
    assert sorted(f.stem for f in (outpath/'annotations').iterdir()) \
          == sorted(f.stem for f in prj.annotationsDir.iterdir())

  # Make sure self export doesn't break anything
  prj.exportAnnotations(prj.location)

def test_load_startup_img(tmp_path, app, filePlg):
  prjcfg = {'startup': {'image': str(SAMPLE_SMALL_IMG_FNAME)}}
  oldCfg = filePlg.projData.cfgFname, filePlg.projData.cfg
  filePlg.open(tmp_path/'test-startup.s3aprj', prjcfg)
  assert app.srcImgFname == filePlg.projData.imagesDir/SAMPLE_SMALL_IMG_FNAME.name
  filePlg.open(*oldCfg)

def test_load_with_plg(monkeypatch, tmp_path):
  # Make separate win to avoid clobbering existing menus/new projs
  app = S3A(loadLastState=False)
  filePlg = app.filePlg
  with monkeypatch.context() as m:
    m.syspath_prepend(str(TEST_FILE_DIR))
    from files.sample_plg import SamplePlugin
    cfg = {
      'plugin-cfg': {'Test': 'files.sample_plg.SamplePlugin'}
    }
    filePlg.open(tmp_path/'plgprj.s3aprj', cfg)
    assert SamplePlugin in app.clsToPluginMapping
    assert len(filePlg.projData.spawnedPlugins) == 1
    assert filePlg.projData.spawnedPlugins[0].win

  # Remove existing plugin
  cfg = {
    'plugin-cfg': {'New Name': 'nonsense.Plugin'}
  }
  with pytest.raises(ValueError):
    filePlg.open(tmp_path/'plgprj2.s3aprj', cfg)
  # Add nonsense plugin
  cfg['plugin-cfg']['Test'] = 'files.sample_plg.SamplePlugin'
  with pytest.warns(UserWarning):
    filePlg.open(tmp_path/'plgprj2.s3aprj', cfg)

def test_unique_tblcfg(tmp_path, tmpProj):
  tblCfg = {'fields': {'Test': ''}}
  tblName = tmp_path/'tbl.yml'
  fns.saveToFile(tblCfg, tblName)

  cfg = {'table-cfg': str(tblName)}
  tmpProj.loadCfg(tmp_path/'myprj.s3aprj', cfg)
  assert tmpProj.tableData.fieldFromName('Test')

def test_img_ops(tmpProj, tmp_path):
  img = {'data': SAMPLE_SMALL_IMG, 'name': 'my image.png'}
  cfg = {'images': [img]}
  tmpProj.loadCfg(tmp_path/'test.s3aprj', cfg)
  assert len(tmpProj.images) == 1
  assert tmpProj.images[0].name == 'my image.png'

  with pytest.raises(IOError):
    tmpProj.addImage('this image does not exist.png', copyToProj=True)

def test_ann_opts(prjWithSavedStuff, sampleComps):
  img, toRemove = next(iter(prjWithSavedStuff.imgToAnnMapping.items()))
  prjWithSavedStuff.removeAnnotation(toRemove)
  assert img not in prjWithSavedStuff.imgToAnnMapping

  with pytest.raises(IOError):
    prjWithSavedStuff.addAnnotation(data=sampleComps, image='garbage.png')