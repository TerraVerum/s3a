import contextlib
from pathlib import Path
from typing import Union

import numpy as np
import pytest

from s3a import FR_SINGLETON, REQD_TBL_FIELDS
from conftest import dfTester

td = FR_SINGLETON.tableData
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
def newCfg(app):
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
  FR_SINGLETON.tableData.loadCfg('testcfg', cfgDict, force=True)
  with newCfg('none', {}):
    assert len(app.compMgr.compDf) == 0
    assert app.compMgr.colTitles == list(map(str, REQD_TBL_FIELDS))
    newComps = FR_SINGLETON.tableData.makeCompDf(3).reset_index(drop=True)
    dfTester.fillRandomVerts(compDf=newComps)
    # Just make sure no errors are thrown on adding comps
    app.add_focusComps(newComps)
    assert len(app.compMgr.compDf) == 3

def test_params_for_class(newCfg):
  with newCfg('testcfg', cfgDict):
    assert 'Class' in [f.name for f in FR_SINGLETON.tableData.allFields]

@pytest.mark.withcomps
def test_no_change(app, newCfg):
  with newCfg(td.cfgFname, td.cfg):
    assert len(app.compMgr.compDf) > 0