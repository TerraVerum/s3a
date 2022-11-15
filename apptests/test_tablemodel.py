import numpy as np
import pytest
from pyqtgraph.Qt import QtCore

from apptests.conftest import NUM_COMPS, dfTester
from apptests.testingconsts import RND, TEST_FILE_DIR
from s3a import TableData
from s3a.compio.importers import SerialImporter
from s3a.constants import PRJ_ENUMS, REQD_TBL_FIELDS
from s3a.structures import ComplexXYVertices, XYVertices

oldIds = np.arange(NUM_COMPS, dtype=int)


@pytest.fixture
def sampleComps(app):
    app.clearBoundaries()
    return dfTester.compDf.copy()


def test_normal_add(sampleComps, mgr):
    mgr.removeComponents()
    changeList = mgr.addComponents(sampleComps)
    cmpChangeList(changeList, oldIds)


def test_undo_add(sampleComps, mgr):
    mgr.addComponents(sampleComps)
    mgr.actionStack.undo()
    assert len(mgr.compDf) == 0
    mgr.actionStack.redo()
    assert len(mgr.compDf) > 0


def test_empty_add(mgr):
    changeList = mgr.addComponents(mgr.tableData.makeComponentDf(0))
    cmpChangeList(changeList)


def test_rm_by_empty_vert_add(sampleComps, mgr):
    numDeletions = NUM_COMPS // 3
    perm = RND.permutation(NUM_COMPS)
    deleteIdxs = np.sort(perm[:numDeletions])
    changeIdxs = np.sort(perm[numDeletions:])
    changes = mgr.addComponents(sampleComps)
    sampleComps[REQD_TBL_FIELDS.ID] = changes["ids"]

    # List assignment behaves poorly for list-inherited objs (like frcomplexverts) so
    # use individual assignment
    for idx in deleteIdxs:
        sampleComps.at[idx, REQD_TBL_FIELDS.VERTICES] = ComplexXYVertices()
    changeList = mgr.addComponents(sampleComps, PRJ_ENUMS.COMPONENT_ADD_AS_MERGE)
    cmpChangeList(changeList, deleted=deleteIdxs, changed=changeIdxs)


def test_double_add(sampleComps, mgr):
    changeList = mgr.addComponents(sampleComps, PRJ_ENUMS.COMPONENT_ADD_AS_NEW)
    cmpChangeList(changeList, added=oldIds)

    # Should be new IDs during 'add as new'
    changeList = mgr.addComponents(sampleComps, PRJ_ENUMS.COMPONENT_ADD_AS_NEW)
    cmpChangeList(changeList, added=oldIds + NUM_COMPS)


def test_change_comps(sampleComps, mgr):
    changeList = mgr.addComponents(sampleComps, PRJ_ENUMS.COMPONENT_ADD_AS_NEW)
    cmpChangeList(changeList, added=oldIds)
    sampleComps[REQD_TBL_FIELDS.ID] = changeList["ids"]

    newVerts = dfTester.fillRandomVerts(compDf=sampleComps)
    changeList = mgr.addComponents(sampleComps, PRJ_ENUMS.COMPONENT_ADD_AS_MERGE)
    cmpChangeList(changeList, changed=oldIds)
    assert (
        newVerts == mgr.compDf[REQD_TBL_FIELDS.VERTICES].to_list()
    ), '"Vertices" list doesn\'t match during test_change_comps'


def test_rm_comps(sampleComps, mgr):
    ids = np.arange(NUM_COMPS, dtype=int)

    # Standard remove
    mgr.addComponents(sampleComps)
    changeList = mgr.removeComponents(ids)
    cmpChangeList(changeList, deleted=ids)

    # Remove when ids don't exist
    changeList = mgr.removeComponents(ids)
    cmpChangeList(changeList)

    # Remove all
    for _ in range(10):
        mgr.addComponents(sampleComps)
    prevIds = mgr.compDf[REQD_TBL_FIELDS.ID].values
    changeList = mgr.removeComponents()
    cmpChangeList(changeList, deleted=prevIds)

    # Remove single
    mgr.addComponents(sampleComps)
    mgr.removeComponents(sampleComps.index[0])


def test_rm_undo(sampleComps, mgr):
    ids = np.arange(NUM_COMPS, dtype=int)

    # Standard remove
    mgr.addComponents(sampleComps)
    mgr.removeComponents(ids)
    mgr.actionStack.undo()
    assert np.setdiff1d(ids, mgr.compDf.index).size == 0


def test_merge_comps(sampleComps, mgr):
    mgr.addComponents(sampleComps)
    mgr.mergeById(sampleComps.index)
    assert len(mgr.compDf) == 1
    mgr.actionStack.undo()
    assert len(mgr.compDf) == len(sampleComps)


def test_bad_merge(sampleComps, mgr):
    mgr.addComponents(sampleComps)
    with pytest.warns(UserWarning):
        mgr.mergeById([0])
    with pytest.warns(UserWarning):
        mgr.mergeById([])


def test_table_setdata(sampleComps, app, mgr):
    mgr.addComponents(sampleComps)

    _ = REQD_TBL_FIELDS
    colVals = {
        _.VERTICES: ComplexXYVertices([XYVertices([[1, 2], [3, 4]])]),
        _.IMAGE_FILE: "newfilename",
    }
    intColMapping = {mgr.tableData.allFields.index(k): v for k, v in colVals.items()}

    for col, newVal in intColMapping.items():
        row = RND.integers(NUM_COMPS)
        idx = app.tableView.model().index(row, col)
        oldVal = mgr.data(idx, QtCore.Qt.ItemDataRole.EditRole)
        mgr.setData(idx, newVal)
        # Test with no change
        mgr.setData(idx, newVal)
        assert mgr.compDf.iloc[row, col] == newVal
        mgr.actionStack.undo()
        assert mgr.compDf.iloc[row, col] == oldVal


def test_table_getdata(sampleComps, mgr):
    mgr.addComponents(sampleComps)
    idx = mgr.index(0, list(REQD_TBL_FIELDS).index(REQD_TBL_FIELDS.IMAGE_FILE))
    dataVal = sampleComps.iat[0, idx.column()]
    assert mgr.data(idx, QtCore.Qt.ItemDataRole.EditRole) == dataVal
    assert mgr.data(idx, QtCore.Qt.ItemDataRole.DisplayRole) == str(dataVal)
    assert mgr.data(idx, 854) is None


def test_simplify_diags():
    # Ensure removal of diagonal vertices has no effect on output accuracy
    df = SerialImporter(TableData())(TEST_FILE_DIR / "simplify_verts.csv")
    verts = df.at[0, "Vertices"]
    mask = verts.toMask()
    verts2 = ComplexXYVertices.fromBinaryMask(mask)
    assert verts != verts2
    assert (verts2.toMask() == mask).all()


def test_simplify_triang():
    verts = ComplexXYVertices([[[0, 0], [40, 50], [0, 100]]], coerceListElements=True)
    lotsOfPoints = verts.fromBinaryMask(verts.toMask(), approximation=None)
    assert len(lotsOfPoints) == 1 and len(lotsOfPoints[0]) > 3
    simplified = lotsOfPoints.simplify()
    assert (
        len(simplified) == 1
        and len(simplified[0]) == 3
        and np.isin(simplified[0], verts).all()
    )


def cmpChangeList(
    changeList: dict,
    added: np.ndarray = None,
    deleted: np.ndarray = None,
    changed: np.ndarray = None,
):
    emptyArr = np.array([], int)
    arrs = locals()
    for name in "added", "deleted", "changed":
        if arrs[name] is None:
            arrCmp = emptyArr
        else:
            arrCmp = arrs[name]
        np.testing.assert_equal(
            changeList[name], arrCmp, f'Mismatch in "{name}"' f" of change list"
        )
