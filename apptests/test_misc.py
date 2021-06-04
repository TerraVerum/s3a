from s3a import generalutils as gu, ComplexXYVertices, XYVertices
import numpy as np

from s3a.plugins.misc import miscFuncsPluginFactory, MultiPredictionsPlugin


def test_resize_pad():
  img = np.zeros((100, 500), 'uint8')

  for sz in (500,500), (100,500), (1000, 100):
    rp = gu.resize_pad(img, sz)
    assert rp.shape == sz

def test_plg_factory(app):
  count = 0
  def add():
    nonlocal count
    count += 1
  mp = miscFuncsPluginFactory('test', [add])()
  mp.attachWinRef(app)
  assert mp.toolsEditor.procToParamsMapping
  assert mp.name == 'test'
  next(iter(mp.toolsEditor.procToParamsMapping)).run()
  assert count == 1

def test_pred(app):
  predPlg: MultiPredictionsPlugin = app.clsToPluginMapping[MultiPredictionsPlugin]
  # Correctness of algo already tested elsewhere, run to assert no errors
  predPlg.makePrediction(app.exportableDf)
  predPlg.predictFromSelection()

def test_vertices_offset():
  subVerts = [[50, 50]], [[100,100], [200, 200], [300,300]]
  verts = ComplexXYVertices(subVerts, coerceListElements=True)
  vertsCopy = ComplexXYVertices(subVerts, coerceListElements=True)
  out = verts.removeOffset()
  assert out == ([[0,0]], [[50,50], [150,150], [250,250]])
  assert verts == vertsCopy

  verts.removeOffset(inplace=True)
  assert verts != vertsCopy
