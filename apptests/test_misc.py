from s3a import generalutils as gu
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