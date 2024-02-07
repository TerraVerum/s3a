import numpy as np
import pyqtgraph as pg
import pytest

from s3a import REQD_TBL_FIELDS as RTF, XYVertices
from s3a.structures.vertices import ComplexXYVertices
from s3a.tabledata import TableData
from s3a.views.fielddelegates import (
    ComplexXYVerticesDelegate,
    FieldDisplay,
    XYVerticesDelegate,
)


@pytest.fixture
def td():
    cfg = {
        "fields": {
            "class": "test",
            "another": 2,
            "oncemore": True,
            "xyvertices": {"type": "xyvertices", "value": XYVertices()},
            "complex": {"type": "complexxyvertices", "value": ComplexXYVertices()},
        }
    }
    td = TableData("temp", configDict=cfg)
    return td


def test_combined(sampleComps, td):
    comps = td.makeComponentDf(len(sampleComps))
    sampleComps = sampleComps.copy()
    comps.index = sampleComps.index
    comps.update(sampleComps)

    pi = pg.PlotItem()

    disp = FieldDisplay(pi)
    disp.table = td
    disp.showFieldData(comps)
    # No vertices, should be hidden
    assert len(disp.inUseDelegates) == 3
    # Make sure no exceptions here
    disp.callDelegateFunction("show")
    disp.callDelegateFunction("hide")
    assert len(disp.inUseDelegates) == 3
    disp.callDelegateFunction("clear")
    assert not len(disp.inUseDelegates)

    disp.showFieldData(
        comps, fields=np.setdiff1d(td.allFields, td.fieldFromName("complex"))
    )
    assert len(disp.inUseDelegates) == 2


def test_xyvertices(td, sampleComps):
    v1 = XYVertices(np.arange(10).reshape(5, 2))
    v2 = XYVertices(np.arange(6).reshape(3, 2), connected=False)
    v3 = XYVertices(np.arange(4).reshape(2, 2))
    field = td.fieldFromName("xyvertices")
    sampleComps = sampleComps.iloc[:3].copy()
    sampleComps.loc[:, field] = [v1, v2, v3]

    deleg = XYVerticesDelegate()
    deleg.setData(sampleComps, field)
    # Single symbol for poly, individual points for point
    assert len(deleg.polyScatter.getData()[0]) == 1
    assert len(deleg.pointScatter.getData()[0]) == 5


def test_complexxyvertices(td, sampleComps):
    field = td.fieldFromName("complex")
    sampleComps[field] = sampleComps[RTF.VERTICES]
    deleg = ComplexXYVerticesDelegate()
    deleg.setData(sampleComps, field)
    assert len(deleg.region.regionData) == len(sampleComps)
