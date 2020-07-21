import pytest
from imageprocessing.processing import ImageProcess

from conftest import app
from s3a.structures import FRVertices

# Use a small image for faster testing
mImg = app.mainImg
pc = mImg.procCollection
allAlgs = pc.nameToProcMapping.keys()

@pytest.mark.smallimage
def test_algs_working():
  for alg in allAlgs:
    pc.switchActiveProcessor(alg)
    mImg.handleShapeFinished(FRVertices())

@pytest.mark.smallimage
def test_disable_top_stages():
  for proc in pc.nameToProcMapping.values():
    for stage in proc.processor.stages:
      if stage.allowDisable and isinstance(stage, ImageProcess):
        proc.setStageEnabled([stage.name], False)
    pc.switchActiveProcessor(proc)
    mImg.handleShapeFinished(FRVertices())





