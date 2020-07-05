import pytest
from pyqtgraph.Qt import QtCore
from time import sleep
from s3a import appInst

from conftest import app, mgr


# TODO: Figure out why this is throwing windows fatal exceptions...
# @pytest.mark.noclear
# def test_merge_selected_comps(sampleComps):
#   mgr.addComps(sampleComps)
#   app.compTbl.selectAll()
#   app.compDisplay.mergeSelectedComps()
#   QtCore.QTimer.singleShot(0, doMerge)
#   appInst.processEvents()
#   assert len(mgr.compDf) == 1
#   # # Undo swap current comp, undo merge
#   stack.undo()
#   stack.undo()
#   assert len(mgr.compDf) == len(sampleComps)