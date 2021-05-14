import contextlib
from pathlib import Path
from typing import Union
from io import StringIO

import numpy as np
import pytest

from conftest import dfTester
from s3a import REQD_TBL_FIELDS
from utilitys import fns

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
def td(app):
  return app.sharedAttrs.tableData

@pytest.fixture
def newCfg(app, td):
  @contextlib.contextmanager
  def newCfg(name: Union[str, Path], cfg: dict):
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
def test_no_change(app, newCfg, td):
  with newCfg(td.cfgFname, td.cfg):
    assert len(app.compMgr.compDf) > 0

def test_filter(td):
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
    Bad:
      pType: unrecognizable
  """
  file = StringIO(mockCfg)
  parsed = fns.yamlLoad(file)
  with pytest.warns(UserWarning):
    td.loadCfg(td.cfgFname, parsed, force=True)

  del parsed['fields']['Bad']
  td.loadCfg(td.cfgFname, parsed, force=True)
  for name in parsed['fields']:
    assert name in td.filter.params.names
    assert td.fieldFromName(name)
  filterStatus = {'List': {
    'Active': True, 'A': True, 'B': False, 'C': False
  }}
  listParam = td.fieldFromName('List')
  td.filter.loadParamValues(td.filter.stateName, filterStatus)
  tmpdf = td.makeCompDf(7)
  tmpdf[listParam] = list('AABBCCC')
  filteredDf = td.filter.filterCompDf(tmpdf)
  assert len(filteredDf) == 2
  assert np.array_equal(['A'], np.unique(filteredDf[listParam]))

