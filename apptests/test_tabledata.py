from pathlib import Path

import numpy as np
import pytest

from s3a import FR_SINGLETON, REQD_TBL_FIELDS
from conftest import app, dfTester

td = FR_SINGLETON.tableData
@pytest.mark.withcomps
def test_no_classes_no_opt_fields():
  oldCfg = td.cfg
  oldFname = td.cfgFname
  cfg = {}
  td.loadCfg('none', cfg)
  app.resetTblFields()
  assert len(app.compMgr.compDf) == 0
  assert app.compMgr.colTitles == list(map(str, REQD_TBL_FIELDS))
  newComps = FR_SINGLETON.tableData.makeCompDf(3).reset_index(drop=True)
  dfTester.fillRandomVerts(compDf=newComps)
  # Just make sure no errors are thrown on adding comps
  app.add_focusComp(newComps)
  assert len(app.compMgr.compDf) == 3
  td.loadCfg(oldFname, oldCfg)
  app.resetTblFields()