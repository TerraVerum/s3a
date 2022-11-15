import warnings

import numpy as np
import pandas as pd
import pytest
from qtextras import seriesAsFrame
from skimage import util

from apptests.conftest import SAMPLE_SMALL_IMG
from apptests.testingconsts import SAMPLE_IMG
from s3a import REQD_TBL_FIELDS, ComplexXYVertices
from s3a.generalutils import imageCornerVertices
from s3a.processing import PipelineParameter
from s3a.processing.algorithms import (
    imageproc as ip,
    make_grid_components,
    multipred as mulp,
)
from s3a.structures import XYVertices


@pytest.mark.smallimage
def test_algs_working(app, vertsPlugin):
    mImg = app.mainImage
    pe = vertsPlugin.processEditor
    allAlgs = vertsPlugin.processEditor.collection.topProcesses

    # Some exceptions may occur in the processor, this is fine since behavior might be
    # undefined
    mImg.shapeCollection.sigShapeFinished.emit(imageCornerVertices(app.mainImage.image))
    for alg in allAlgs:
        pe.changeActiveProcessor(alg)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            mImg.shapeCollection.sigShapeFinished.emit(XYVertices())


@pytest.mark.smallimage
def test_disable_top_stages(app, vertsPlugin):
    mImg = app.mainImage
    pe = vertsPlugin.processEditor
    mImg.shapeCollection.sigShapeFinished.emit(imageCornerVertices(app.mainImage.image))
    for name in pe.collection.topProcesses:
        proc: PipelineParameter = pe.collection.parseProcessName(name)
        pe.changeActiveProcessor(proc, saveBeforeChange=False)
        for stage in proc:
            stage.setOpts(enabled=False)
        # Some exceptions may occur in the processor, this is fine since behavior might
        # be undefined
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            mImg.shapeCollection.sigShapeFinished.emit(XYVertices())
    # Make sure to put stages back after
    pe.loadParameterValues(pe.stateName)


# -----
# Administrative Image processes (e.g. cropping, formatting, etc. stages that must
# work for every kind of process hierarchy
# -----
def test_crop_image():
    testImg = SAMPLE_SMALL_IMG.copy()
    testShp = testImg.shape[:2]
    testMask = np.zeros(testShp, dtype=bool)
    converters = [
        func for (name, func) in vars(util).items() if name.startswith("img_as_")
    ]
    testSizes = max(testShp) * np.array([0.5, 0.8, 1, 5])
    for converter in converters:
        for size in testSizes:
            # Make sure this works for all image types and downsample sizes
            out = ip.crop_to_local_area(
                image=converter(testImg),
                foregroundVertices=XYVertices(),
                backgroundVertices=XYVertices(),
                componentMask=testMask,
                componentVertices=ComplexXYVertices(),
                viewbox=XYVertices(),
                historyMask=np.zeros_like(testMask),
                reference="image",
                maxSize=size,
            )
            outShape = max(out["image"].shape)
            assert outShape <= size and outShape <= max(
                testImg.shape
            ), f"Failed for type {converter.__name__}"


# -----
# Multi-predictions
# -----
def test_template_dispatch(app):
    x = np.zeros((512, 512), "uint8")
    rr, cc = np.ogrid[0 : x.shape[0], 0 : x.shape[1]]
    rad = 100
    origins = np.array([[rad, rad], [3 * rad, 3 * rad]])
    for o in origins:
        x += (rr - o[0]) ** 2 + (cc - o[1]) ** 2 <= rad**2

    # Two perfectly matching circles should be detected by template matching
    templateSize = 2 * rad + 1
    dummyComp = app.componentIo.tableData.makeComponentDf(1)
    templateVerts = XYVertices([[0, 0], [templateSize, templateSize]])
    dummyComp.at[dummyComp.index[0], REQD_TBL_FIELDS.VERTICES] = ComplexXYVertices(
        [templateVerts], coerceListElements=True
    )
    tm = mulp.cv_template_match
    out = tm(image=x, components=dummyComp, viewbox=np.zeros((2, 2), int), area="image")
    assert len(out["components"]) == 2
    assert np.all(out["scores"] == 1)

    with pytest.raises(ValueError):
        tm(image=x, components=dummyComp, viewbox=np.zeros((2, 2), int), area="viewbox")
    # No matches in this viewbox
    vb = np.array([[2 * rad + 1, x.shape[1]], [0, 2 * rad + 1]])
    out = tm(image=x, components=dummyComp, viewbox=vb, area="viewbox")
    assert len(out["components"]) == 0


def test_focused_dispatch(sampleComps):
    def dummyFunc(component: pd.Series, image=None):
        return dict(components=seriesAsFrame(component), image=image)

    dispatched = mulp.ProcessDispatcher(dummyFunc)
    result = dispatched(image=SAMPLE_SMALL_IMG, components=sampleComps)
    assert sampleComps is not result["components"]
    assert len(result["components"]) == len(sampleComps)


@pytest.mark.parametrize("area", ("viewbox", "image"))
@pytest.mark.parametrize("windowParam", [5, 500])
@pytest.mark.parametrize("winType", ["Row/Col Divisions", "Row Size"])
@pytest.mark.parametrize("maxNumComponents", [10, 100])
def test_grid(sampleComps, area, windowParam, winType, maxNumComponents):
    comps = make_grid_components(
        SAMPLE_IMG,
        sampleComps,
        np.array([[0, 0], [50, 50]]),
        area=area,
        windowParam=windowParam,
        winType=winType,
        maxNumComponents=maxNumComponents,
    )
    assert len(comps) <= maxNumComponents


def test_empty_components(sampleComps):
    assert np.array_equal(
        make_grid_components(
            SAMPLE_SMALL_IMG,
            sampleComps.iloc[0:0],
            np.array([[0, 0], [3, 3]]),
            area="image",
        )["components"].columns,
        sampleComps.columns,
    )
