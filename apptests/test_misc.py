import pytest

from skimage import data

from s3a import generalutils as gu, ComplexXYVertices, XYVertices, PRJ_ENUMS
import numpy as np

from s3a.generalutils import deprecateKwargs
from s3a.plugins.misc import miscFuncsPluginFactory, MultiPredictionsPlugin

_rots = list(np.linspace(-360, 360, 25)) + [PRJ_ENUMS.ROT_OPTIMAL]


@pytest.mark.parametrize('size', [(500, 500), (100, 500), (1000, 100)])
@pytest.mark.parametrize('rot', _rots)
def test_sub_image_shape(size, rot):
  img = np.zeros((100, 500), 'uint8')
  imgVerts = np.array([[0, 0], [499, 99]])
  subimg, stats = gu.subImageFromVerts(img, imgVerts, shape=size, returnStats=True, rotationDeg=rot)
  assert subimg.shape == size
  orig = gu.inverseSubImage(subimg, stats)
  assert orig.shape == img.shape

@pytest.mark.parametrize('rot', _rots[:-1:5] + [_rots[-1]])
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
  bbox = gu.coordsToBbox(vertsBox)
  sub, stats = gu.subImageFromVerts(
    im, vertsBox, rotationDeg=rot, returnStats=True, allowTranspose=transpose, shape=shape, margin=margin)
  inv = gu.inverseSubImage(sub, stats, finalBbox=bbox)
  orig = gu.getCroppedImg(im, vertsBox, returnCoords=False)
  diff = np.abs(orig.astype(float) - inv.astype(float))
  assert diff.mean() < 17.5

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

def test_deprecation():
  @deprecateKwargs(b='a')
  def sampleFunc(a=5):
    return a
  with pytest.warns(DeprecationWarning):
    assert sampleFunc(b=10) == 10
