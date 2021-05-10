import pytest

from apptests.testingconsts import SAMPLE_SMALL_IMG_FNAME
from apptests.conftest import dfTester
from s3a.plugins.file import ProjectData
from utilitys import fns


@pytest.fixture
def tmpProj(tmp_path):
  return ProjectData.create(name=tmp_path / 'tmpproj')


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