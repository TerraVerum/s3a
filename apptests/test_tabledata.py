import contextlib
import warnings
from pathlib import Path
from typing import Union, Sequence
from io import StringIO

import numpy as np
import pytest

from apptests.testingconsts import RND
from conftest import dfTester
from s3a import REQD_TBL_FIELDS, ComplexXYVertices
from utilitys import fns

from s3a.parameditors.table import TableData

cfgDict = {
  'fields': {
    'Class': {
      'value': 'test',
      'pType': 'popuplineeditor',
      'limits': ['test', 'this', 'out']
    }
  }
}

@pytest.fixture
def td():
  return TableData()

@pytest.fixture
def newCfg(app):
  @contextlib.contextmanager
  def newCfg(name: Union[str, Path], cfg: dict):
    td = app.sharedAttrs.tableData
    oldCfg = td.cfg
    oldFname = td.cfgFname
    td.loadCfg(name, cfg)
    yield
    td.loadCfg(oldFname, oldCfg)
  return newCfg

@pytest.mark.withcomps
def test_no_opt_fields(app, newCfg):
  app.sharedAttrs.tableData.loadCfg('testcfg', cfgDict, force=True)
  with newCfg('none', {}):
    assert len(app.compMgr.compDf) == 0
    assert app.compMgr.colTitles == list(map(str, REQD_TBL_FIELDS))
    newComps = app.sharedAttrs.tableData.makeCompDf(3).reset_index(drop=True)
    dfTester.fillRandomVerts(compDf=newComps)
    # Just make sure no errors are thrown on adding comps
    app.add_focusComps(newComps)
    assert len(app.compMgr.compDf) == 3

def test_params_for_class(newCfg, app):
  with newCfg('testcfg', cfgDict):
    assert 'Class' in [f.name for f in app.sharedAttrs.tableData.allFields]

@pytest.mark.withcomps
def test_no_change(app, newCfg):
  with newCfg(app.sharedAttrs.tableData.cfgFname, app.sharedAttrs.tableData.cfg):
    assert len(app.compMgr.compDf) > 0

def test_filter(td, qtbot):
  # Try a bunch of types
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
      pType: complexxyvertices
      vaue: [[[0,0], [100,100]]]
    Bad:
      pType: unrecognizable
  """
  file = StringIO(mockCfg)
  parsed = fns.yamlLoad(file)
  with pytest.warns(UserWarning):
    td.loadCfg('testfilter', parsed, force=True)

  del parsed['fields']['Bad']
  td.loadCfg(td.cfgFname, parsed, force=True)
  for name in parsed['fields']:
    assert name in td.filter.params.names
    assert td.fieldFromName(name)

  filterStatus = {'List': {
    'Active': True, 'A': True, 'B': False, 'C': False
  }}
  filtered = apply_assertFilter(td, filterStatus, 2, list('AABBCCC'))
  assert np.array_equal(['A'], np.unique(filtered))

  filterStatus = {'Bool': {
    'Active': True, 'Bool': True, 'Not Bool': False
  }}
  vals = RND.integers(0, 1, size=15, endpoint=True, dtype=bool)
  numTrue = np.count_nonzero(vals)
  filtered = apply_assertFilter(td, filterStatus, numTrue, vals)
  assert np.array_equal([True]*numTrue, filtered)

  filterStatus = {'Int': {
    'Active': True, 'min': 5, 'max': 100
  }}
  filtered = apply_assertFilter(td, filterStatus, 5, np.arange(10))
  assert np.array_equal(np.arange(5, 10), filtered)

  filterStatus = {'String': {
    'Active': True, 'Regex Value': 'test.*'
  }}
  vals = ['a', 'b', 'testthis', 'notthis']
  filtered = apply_assertFilter(td, filterStatus, 1, vals)
  assert filtered.iloc[0] == 'testthis'

  filterStatus = {'Complex': {
    'Active': True, 'X Bounds': {'min': 5}, 'Y Bounds': {'max': 100}
  }}
  v1 = ComplexXYVertices([[[4, 100], [100,200]]], coerceListElements=True)
  v2 = ComplexXYVertices([[[6, 15], [1000, 10]]], coerceListElements=True)
  v3 = ComplexXYVertices([[[6, 15], [1000, 1000]]], coerceListElements=True)
  filtered = apply_assertFilter(td, filterStatus, 1, [v1, v2, v3])
  assert np.array_equal([v2], filtered.to_list())

def apply_assertFilter(tableData, status: dict, resultLen: int,
                       setVals: Sequence):
  fieldName = next(iter(status))
  param = tableData.fieldFromName(fieldName)
  tableData.filter.loadParamValues(tableData.filter.stateName, status)
  df = tableData.makeCompDf(len(setVals))
  df[param] = setVals
  filteredDf = tableData.filter.filterCompDf(df)
  assert len(filteredDf) == resultLen
  return filteredDf[param]