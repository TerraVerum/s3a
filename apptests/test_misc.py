import ast

import numpy as np
import pytest
from qtextras import OptionsDict
from skimage import data

from s3a import PRJ_ENUMS, ComplexXYVertices, generalutils as gu
from s3a.compio.helpers import deserialize
from s3a.generalutils import DirectoryDict, deprecateKwargs
from s3a.plugins.multipred import MultiPredictionsPlugin
from s3a.plugins.tools import functionPluginFactory

_rots = list(np.linspace(-180, 180, 5)) + [PRJ_ENUMS.ROTATION_OPTIMAL]


@pytest.mark.parametrize("rot", _rots)
@pytest.mark.parametrize("transpose", [True, False])
@pytest.mark.parametrize("shape", [(100, 100), (200, 200), None])
@pytest.mark.parametrize("margin", [0, 5, [10, 10]])
def test_sub_image_correctness(rot, transpose, shape, margin):
    vertsBox = np.array([[96, 77], [96, 179], [356, 179], [356, 77]])
    im = data.chelsea()
    sub, stats = gu.subImageFromVertices(
        im,
        vertsBox,
        rotationDegrees=rot,
        returnStats=True,
        allowTranspose=transpose,
        shape=shape,
        margin=margin,
    )
    inv = gu.inverseSubImage(sub, stats)
    orig = gu.getCroppedImage(im, vertsBox, returnBoundingBox=False)
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

    mp = functionPluginFactory([add], name="My Random Tools")()
    mp.attachToWindow(app)
    assert mp.nameFunctionMap
    assert mp.name == "My Random Tools"
    next(iter(mp.nameFunctionMap.values()))()
    assert count == 1


def test_pred(app):
    predPlg: MultiPredictionsPlugin = app.classPluginMap[MultiPredictionsPlugin]
    # Correctness of algo already tested elsewhere, run to assert no errors
    predPlg.makePrediction(app.componentDf)


def test_vertices_offset():
    subVerts = [[50, 50]], [[100, 100], [200, 200], [300, 300]]
    verts = ComplexXYVertices(subVerts, coerceListElements=True)
    vertsCopy = ComplexXYVertices(subVerts, coerceListElements=True)
    out = verts.removeOffset()
    assert out == ([[0, 0]], [[50, 50], [150, 150], [250, 250]])
    assert verts == vertsCopy

    verts.removeOffset(inplace=True)
    assert verts != vertsCopy


@pytest.mark.parametrize("warningType", [DeprecationWarning, FutureWarning])
def test_deprecation(warningType):
    @deprecateKwargs(b="a", warningType=warningType)
    def sampleFunc(a=5):
        return a

    with pytest.warns(warningType):
        assert sampleFunc(b=10) == 10


@pytest.mark.parametrize("type_", ["checklist", "list"])
@pytest.mark.parametrize("fixedLims", [True, False])
@pytest.mark.parametrize("limits", [["a"], ["a", "b"]])
@pytest.mark.parametrize("value", ["['a', 'b']", "['a']"])
def test_list_serdes(type_, fixedLims, value, limits):
    trueValue = ast.literal_eval(value)
    if type_ == "checklist":
        initialValue = limits
    else:
        initialValue = limits[0]
        # 'a' or 'b'
        value = trueValue = trueValue[-1]
    param = OptionsDict(
        "test", initialValue, type_, fixedLimits=fixedLims, limits=limits
    )
    out, errs = deserialize(param, [value])
    if trueValue in limits or not set(trueValue).difference(limits) or not fixedLims:
        assert len(out) == 1 and out[0] == trueValue
    else:
        assert len(errs) == 1 and isinstance(errs[0], ValueError)


def test_directorydict(tmp_path):
    tmp_path.joinpath("testfile.txt").touch()
    numReads = 0

    def reader(filename):
        nonlocal numReads
        numReads += 1
        with open(filename, "r") as ifile:
            return ifile.read()

    dd = DirectoryDict(tmp_path, reader)
    assert dd.get("testfile.txt", None) is not None

    dd.cacheReads = True
    numReads = 0
    dd.clear()
    for ii in range(10):
        # Test that the file isn't read multiple times as a side effect
        # noinspection PyStatementEffect
        dd["testfile.txt"]
    assert numReads == 1


def test_directorydict_equality():
    def reader(file):
        return ""

    dd1 = DirectoryDict("./somefolder", readFunction=reader)
    dd2 = DirectoryDict("./differentfolder", readFunction=reader)

    assert dd1 != dd2
    dd2.folder = dd1.folder
    assert dd1 == dd2

    assert dd1 == {}
