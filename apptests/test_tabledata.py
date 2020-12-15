import contextlib
from pathlib import Path
from typing import Union

import numpy as np
import pytest

from s3a import FR_SINGLETON, REQD_TBL_FIELDS
from conftest import dfTester

td = FR_SINGLETON.tableData

@pytest.fixture
def newCfg(app):
  @contextlib.contextmanager
  def newCfg(name: Union[str, Path], cfg: dict):
    oldCfg = td.cfg
    oldFname = td.cfgFname
    td.loadCfg(name, cfg)
    app.resetTblFields()
    yield
    td.loadCfg(oldFname, oldCfg)
    app.resetTblFields()
  return newCfg

@pytest.mark.withcomps
def test_no_classes_no_opt_fields(app, newCfg):
  with newCfg('none', {}):
    assert len(app.compMgr.compDf) == 0
    assert app.compMgr.colTitles == list(map(str, REQD_TBL_FIELDS))
    assert td.compClasses == [REQD_TBL_FIELDS.COMP_CLASS.value]
    newComps = FR_SINGLETON.tableData.makeCompDf(3).reset_index(drop=True)
    dfTester.fillRandomVerts(compDf=newComps)
    # Just make sure no errors are thrown on adding comps
    app.add_focusComps(newComps)
    assert len(app.compMgr.compDf) == 3

def test_params_for_class(newCfg):
  cfgDict = {'classes': {
    'value': 'test',
    'pType': 'popuplineeditor',
    'limits': ['test', 'this', 'out']
  }}
  with newCfg('testcfg', cfgDict):
    assert FR_SINGLETON.tableData.compClasses == cfgDict['classes']['limits']

@pytest.mark.withcomps
def test_no_change(app, newCfg):
  with newCfg(td.cfgFname, td.cfg):
    assert len(app.compMgr.compDf) > 0