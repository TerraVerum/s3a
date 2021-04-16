import warnings

import pytest

from s3a.generalutils import imgCornerVertices
from s3a.processing import ImageProcess
from s3a.structures import XYVertices

@pytest.mark.smallimage
def test_algs_working(app, vertsPlugin):
  mImg = app.mainImg
  pe = vertsPlugin.procEditor
  allAlgs = vertsPlugin.procEditor.clctn.topProcs

  # Some exceptions may occur in the processor, this is fine since behavior might be undefined
  mImg.shapeCollection.sigShapeFinished.emit(imgCornerVertices(app.mainImg.image))
  for alg in allAlgs:
    pe.changeActiveProcessor(alg)
    with warnings.catch_warnings():
      warnings.simplefilter('ignore', UserWarning)
      mImg.shapeCollection.sigShapeFinished.emit(XYVertices())

@pytest.mark.smallimage
def test_disable_top_stages(app, vertsPlugin):
  mImg = app.mainImg
  pe = vertsPlugin.procEditor
  oldProc = pe.curProcessor.algName
  mImg.shapeCollection.sigShapeFinished.emit(imgCornerVertices(app.mainImg.image))
  for name in pe.clctn.topProcs:
    proc = pe.clctn.parseProcName(name)
    pe.changeActiveProcessor(proc, saveBeforeChange=False)
    for stage in proc.stages_flattened:
      if stage.allowDisable and isinstance(stage, ImageProcess):
        pe.curProcessor.setStageEnabled([stage.name], False)
    # Some exceptions may occur in the processor, this is fine since behavior might be undefined
    with warnings.catch_warnings():
      warnings.simplefilter('ignore', UserWarning)
      mImg.shapeCollection.sigShapeFinished.emit(XYVertices())
  # Make sure to put stages back after
  pe.changeActiveProcessor(oldProc, saveBeforeChange=False)





