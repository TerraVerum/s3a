import contextlib
from io import StringIO
from pathlib import Path
from typing import Sequence, Union

import numpy as np
import pytest
from qtextras import fns

from apptests.conftest import dfTester
from apptests.testingconsts import RND
from s3a import REQD_TBL_FIELDS, ComplexXYVertices
from s3a.tabledata import TableData

cfgDict = {
    "fields": {
        "Class": {
            "value": "test",
            "type": "popuplineeditor",
            "limits": ["test", "this", "out"],
        }
    }
}


@pytest.fixture
def td():
    return TableData()


@pytest.fixture
def newConfig(app):
    @contextlib.contextmanager
    def newCfg(name: Union[str, Path], cfg: dict):
        td = app.tableData
        oldCfg = td.config
        oldFname = td.configPath
        td.loadConfig(name, cfg)
        yield
        td.loadConfig(oldFname, oldCfg)

    return newCfg


@pytest.mark.withcomps
def test_no_opt_fields(app, newConfig):
    app.tableData.loadConfig("testcfg", cfgDict, force=True)
    with newConfig("none", {}):
        assert len(app.componentManager.compDf) == 0
        assert app.componentManager.columnTitles == list(map(str, REQD_TBL_FIELDS))
        newComps = app.tableData.makeComponentDf(3).reset_index(drop=True)
        dfTester.fillRandomVerts(compDf=newComps)
        # Just make sure no errors are thrown on adding components
        app.addAndFocusComponents(newComps)
        assert len(app.componentManager.compDf) == 3


def test_params_for_class(newConfig, app):
    with newConfig("testcfg", cfgDict):
        assert "Class" in [f.name for f in app.tableData.allFields]


@pytest.mark.withcomps
def test_no_change(app, newConfig):
    with newConfig(app.tableData.configPath, app.tableData.config):
        assert len(app.componentManager.compDf) > 0


def test_filter():
    # Try a bunch of types
    td = TableData(makeFilter=True)
    mockCfg = """
  fields:
    List:
      - A
      - B
      - C
    Bool: False
    Int: 0
    String: ''
    Complex:
      type: complexxyvertices
      vaue: [[[0,0], [100,100]]]
    Bad:
      type: unrecognizable
  """
    file = StringIO(mockCfg)
    parsed = fns.yamlLoad(file)
    with pytest.warns(UserWarning):
        td.loadConfig("testfilter", parsed, force=True)

    del parsed["fields"]["Bad"]
    td.loadConfig(td.configPath, parsed, force=True)
    for name in parsed["fields"]:
        assert name in td.filter.rootParameter.names
        assert td.fieldFromName(name)

    filterStatus = {"List": {"Active": True, "Options": ["A"]}}
    filtered = apply_assertFilter(td, filterStatus, 2, list("AABBCCC"))
    assert np.array_equal(["A"], np.unique(filtered))

    filterStatus = {"Bool": {"Active": True, "Options": ["Bool"]}}
    vals = RND.integers(0, 1, size=15, endpoint=True, dtype=bool)
    numTrue = np.count_nonzero(vals)
    filtered = apply_assertFilter(td, filterStatus, numTrue, vals)
    assert np.array_equal([True] * numTrue, filtered)

    filterStatus = {"Int": {"Active": True, "min": 5, "max": 100}}
    filtered = apply_assertFilter(td, filterStatus, 5, np.arange(10))
    assert np.array_equal(np.arange(5, 10), filtered)

    filterStatus = {"String": {"Active": True, "Regex Value": "test.*"}}
    vals = ["a", "b", "testthis", "notthis"]
    filtered = apply_assertFilter(td, filterStatus, 1, vals)
    assert filtered.iloc[0] == "testthis"

    filterStatus = {
        "Complex": {"Active": True, "X Bounds": {"min": 5}, "Y Bounds": {"max": 100}}
    }
    v1 = ComplexXYVertices([[[4, 100], [100, 200]]], coerceListElements=True)
    v2 = ComplexXYVertices([[[6, 15], [1000, 10]]], coerceListElements=True)
    v3 = ComplexXYVertices([[[6, 15], [1000, 1000]]], coerceListElements=True)
    filtered = apply_assertFilter(td, filterStatus, 1, [v1, v2, v3])
    assert np.array_equal([v2], filtered.to_list())


def apply_assertFilter(tableData, status: dict, resultLen: int, setVals: Sequence):
    fieldName = next(iter(status))
    param = tableData.fieldFromName(fieldName)
    tableData.filter.loadParameterValues(
        tableData.filter.stateName, status, addDefaults=True
    )
    df = tableData.makeComponentDf(len(setVals))
    df[param] = setVals
    filteredDf = tableData.filter.filterComponentDf(df)
    assert len(filteredDf) == resultLen
    return filteredDf[param]
