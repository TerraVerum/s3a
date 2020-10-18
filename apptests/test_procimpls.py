import pytest

from conftest import app, vertsPlugin
from s3a.generalutils import imgCornerVertices
from s3a.processing import ImageProcess
from s3a.structures import XYVertices

# Use a small image for faster testing
from testingconsts import SAMPLE_SMALL_IMG_FNAME, SAMPLE_SMALL_IMG

mImg = app.mainImg
fImg = app.focusedImg
pc = vertsPlugin.procCollection

allAlgs = pc.nameToProcMapping.keys()

@pytest.mark.smallimage
def test_algs_working():
  mImg.handleShapeFinished(imgCornerVertices(app.mainImg.image))
  for alg in allAlgs:
    pc.switchActiveProcessor(alg)
    fImg.handleShapeFinished(XYVertices())

@pytest.mark.smallimage
def test_disable_top_stages():
  mImg.handleShapeFinished(imgCornerVertices(app.mainImg.image))
  for proc in pc.nameToProcMapping.values():
    for stage in proc.processor.stages:
      if stage.allowDisable and isinstance(stage, ImageProcess):
        proc.setStageEnabled([stage.name], False)
    pc.switchActiveProcessor(proc)
    mImg.handleShapeFinished(XYVertices())





