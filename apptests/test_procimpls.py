import pytest

from conftest import app
from s3a.generalutils import imgCornerVertices
from s3a.processing import FRImageProcess
from s3a.structures import FRVertices

# Use a small image for faster testing
from testingconsts import SAMPLE_SMALL_IMG_FNAME, SAMPLE_SMALL_IMG

mImg = app.mainImg
fImg = app.focusedImg
pc = app.focusedImg.procCollection

allAlgs = pc.nameToProcMapping.keys()

@pytest.mark.smallimage
def test_algs_working():
  mImg.handleShapeFinished(imgCornerVertices(app.mainImg.image))
  for alg in allAlgs:
    pc.switchActiveProcessor(alg)
    fImg.handleShapeFinished(FRVertices())

@pytest.mark.smallimage
def test_disable_top_stages():
  mImg.handleShapeFinished(imgCornerVertices(app.mainImg.image))
  for proc in pc.nameToProcMapping.values():
    for stage in proc.processor.stages:
      if stage.allowDisable and isinstance(stage, FRImageProcess):
        proc.setStageEnabled([stage.name], False)
    pc.switchActiveProcessor(proc)
    mImg.handleShapeFinished(FRVertices())





