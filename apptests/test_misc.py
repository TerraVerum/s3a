import numpy as np
import pytest
from skimage import data

from s3a import generalutils as gu, ComplexXYVertices, PRJ_ENUMS
from s3a.generalutils import deprecateKwargs
from s3a.plugins.misc import miscFuncsPluginFactory, MultiPredictionsPlugin

_rots = list(np.linspace(-180, 180, 25)) + [PRJ_ENUMS.ROT_OPTIMAL]


@pytest.mark.parametrize('rot', _rots)
@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('shape', [(100,100), (200,200), None])
@pytest.mark.parametrize('margin', [0, 5, [10, 10]])
def test_sub_image_correctness(rot, transpose, shape, margin):
  vertsBox = np.array(
    [[96, 77],
     [96, 179],
     [356, 179],
     [356, 77]]
  )
  im = data.chelsea()
  sub, stats = gu.subImageFromVerts(
    im, vertsBox, rotationDeg=rot, returnStats=True, allowTranspose=transpose, shape=shape, margin=margin)
  inv = gu.inverseSubImage(sub, stats)
  orig = gu.getCroppedImg(im, vertsBox, returnCoords=False)
  if orig.shape == inv.shape:
    inv = gu.inverseSubImage(sub, stats)
    diff = np.abs(orig.astype(float) - inv.astype(float))
    # The section covered by this rotated rect should be correct, but padding must've been
    # substituted for all otherwise unknown pixels. See if this is the case which will
    # result in a darker image overall. However, a direct comparison like this won't work
    # when the subImage extended beyond image dimensions since the shapes won't match
    assert diff.mean() < inv.mean()

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

def test_vertices_offset():
  subVerts = [[50, 50]], [[100,100], [200, 200], [300,300]]
  verts = ComplexXYVertices(subVerts, coerceListElements=True)
  vertsCopy = ComplexXYVertices(subVerts, coerceListElements=True)
  out = verts.removeOffset()
  assert out == ([[0,0]], [[50,50], [150,150], [250,250]])
  assert verts == vertsCopy

  verts.removeOffset(inplace=True)
  assert verts != vertsCopy

def test_deprecation():
  @deprecateKwargs(b='a')
  def sampleFunc(a=5):
    return a
  with pytest.warns(DeprecationWarning):
    assert sampleFunc(b=10) == 10
