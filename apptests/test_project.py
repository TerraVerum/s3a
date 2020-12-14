from pathlib import Path

from apptests.testingconsts import SAMPLE_SMALL_IMG_FNAME
from apptests.conftest import dfTester
from s3a import ComponentIO
from s3a.parameditors.project import ProjectData


def test_create_project(tmp_path, app, sampleComps):
  projName = 'tmpproj'
  dummyAnnFile  = SAMPLE_SMALL_IMG_FNAME.with_suffix(f'{SAMPLE_SMALL_IMG_FNAME.suffix}.pkl')
  proj = ProjectData.create(name=tmp_path / projName, cfg={'images': [SAMPLE_SMALL_IMG_FNAME], 'export-opts': {'annotation-format': 'pkl'}})
  proj.addAnnotation(dummyAnnFile, dfTester.compDf, image=SAMPLE_SMALL_IMG_FNAME.name)

  assert tmp_path.exists()
  assert proj.annotationsDir.exists()
  assert proj.imagesDir.exists()
  proj.addAnnotation(data=sampleComps, image=SAMPLE_SMALL_IMG_FNAME)
  assert len(list(proj.imagesDir.glob(SAMPLE_SMALL_IMG_FNAME.name))) == 1
  assert len(list(proj.annotationsDir.glob(dummyAnnFile.name))) == 1