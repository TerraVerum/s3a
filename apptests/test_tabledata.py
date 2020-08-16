import contextlib
from pathlib import Path
from typing import Union

import numpy as np
import pytest

from s3a import FR_SINGLETON, REQD_TBL_FIELDS
from conftest import app, dfTester
from s3a.projectvars import FR_CONSTS

td = FR_SINGLETON.tableData

@contextlib.contextmanager
def newCfg(name: Union[str, Path], cfg: dict):
  oldCfg = td.cfg
  oldFname = td.cfgFname
  td.loadCfg(name, cfg)
  app.resetTblFields()
  yield
  td.loadCfg(oldFname, oldCfg)
  app.resetTblFields()

@pytest.mark.withcomps
def test_no_classes_no_opt_fields():
  with newCfg('none', {}):
    assert len(app.compMgr.compDf) == 0
    assert app.compMgr.colTitles == list(map(str, REQD_TBL_FIELDS))
    assert td.compClasses == [REQD_TBL_FIELDS.COMP_CLASS.value]
    newComps = FR_SINGLETON.tableData.makeCompDf(3).reset_index(drop=True)
    dfTester.fillRandomVerts(compDf=newComps)
    # Just make sure no errors are thrown on adding comps
    app.add_focusComp(newComps)
    assert len(app.compMgr.compDf) == 3

@pytest.mark.withcomps
def test_no_change():
  with newCfg(td.cfgFname, td.cfg):
    assert len(app.compMgr.compDf) > 0