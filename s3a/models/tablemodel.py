from typing import Any, Union
from warnings import warn

import numpy as np
import pandas as pd
from packaging.version import Version

from pyqtgraph.Qt import QtCore, QtVersion

from ..constants import PRJ_ENUMS, REQD_TBL_FIELDS as RTF
from ..generalutils import coerceDfTypes, concatAllowEmpty
from ..logger import getAppLogger
from ..structures import ComplexXYVertices, OneDArr
from ..tabledata import TableData

__all__ = ["ComponentManager", "ComponentTableModel"]

from qtextras import DeferredActionStackMixin as DASM, seriesAsFrame

Signal = QtCore.Signal


class ComponentTableModel(DASM, QtCore.QAbstractTableModel):
    # Emits 4-element dict: Deleted comp ids, changed comp ids, added comp ids,
    # renamed indexes. Renaming is useful when the new id for an added component should
    # be propagated. "-1" new index indicates that component was deleted (or never
    # added in the first place, in the case of ADD_TYPE_NEW)
    defaultEmitDict = {
        "deleted": np.array([], int),
        "changed": np.array([], int),
        "added": np.array([], int),
        "ids": np.array([], int),
    }
    sigComponentsChanged = Signal(dict)
    sigFieldsChanged = Signal()

    def __init__(self, tableData: TableData = None):
        super().__init__()
        self.tableData = tableData or TableData()
        # Create component dataframe and remove created row. This is to
        # ensure datatypes are correct
        self.resetFields()

    # ------
    # Functions required to implement table model
    # ------
    def columnCount(self, *args, **kwargs):
        return len(self.columnTitles)

    def rowCount(self, *args, **kwargs):
        return len(self.compDf)

    def headerData(self, section, orientation, role=QtCore.Qt.ItemDataRole.DisplayRole):
        if (
            orientation == QtCore.Qt.Orientation.Horizontal
            and role == QtCore.Qt.ItemDataRole.DisplayRole
        ):
            return self.columnTitles[section]

    # noinspection PyMethodOverriding
    def data(self, index: QtCore.QModelIndex, role: int) -> Any:
        outData = self.compDf.iat[index.row(), index.column()]
        if role == QtCore.Qt.ItemDataRole.DisplayRole:
            return str(outData)
        elif role == QtCore.Qt.ItemDataRole.EditRole:
            return outData
        else:
            return None

    @DASM.undoable("Alter Component Data")
    def setData(self, index, value, role=QtCore.Qt.ItemDataRole.EditRole) -> bool:
        if role != QtCore.Qt.ItemDataRole.EditRole:
            return super().setData(index, value, role)

        row = index.row()
        col = index.column()
        oldVal = self.compDf.iat[row, col]
        # Try-catch for case of numpy arrays
        noChange = oldVal == value
        try:
            if noChange:
                return True
        except ValueError:  # lgtm[py/unreachable-statement]
            # LGTM false positive: ValueError is indeed possible with numpy rich bool
            # logic
            pass
        self.compDf.iat[row, col] = value
        # !!! Serious issue! Using iat sometimes doesn't work and I have no idea why
        # since it is not easy to replicate. See
        # https://github.com/pandas-dev/pandas/issues/22740 Also, pandas iloc
        # unnecessarily coerces to 2D ndarray when setting, so iloc will fail when
        # assigning an array to a single location. Not sure how to prevent this... For
        # now, checking this on export
        cmp = (
            self.compDf.iloc[row, [col, col - 1]].values[0] != self.compDf.iat[row, col]
        )
        try:
            cmp = bool(cmp)
        except ValueError:  # lgtm[py/unreachable-statement]
            # See comment on previous ValueError block
            cmp = np.any(cmp)
        if cmp:
            getAppLogger(__name__).warning(
                "Warning! An error occurred setting this value. Please try again using"
                " a *multi-cell* edit. E.g. do not just set this value, set it along"
                " with at least one other selected cell.",
            )
        toEmit = self.defaultEmitDict.copy()
        toEmit["changed"] = np.array([self.compDf.index[index.row()]])
        self.sigComponentsChanged.emit(toEmit)
        yield True
        self.setData(index, oldVal, role)
        return True

    def flags(self, index: QtCore.QModelIndex) -> QtCore.Qt.ItemFlags:
        flgs = QtCore.Qt.ItemFlag
        if index.column() not in self.noEditColIdxs:
            return flgs.ItemIsEnabled | flgs.ItemIsSelectable | flgs.ItemIsEditable
        else:
            return flgs.ItemIsEnabled | flgs.ItemIsSelectable

    # noinspection PyAttributeOutsideInit
    def resetFields(self):
        self.columnTitles = [f.name for f in self.tableData.allFields]

        self.compDf = self.tableData.makeComponentDf(0)

        noEditParams = [
            f for f in self.tableData.allFields if f.opts.get("readonly", False)
        ]

        self.noEditColIdxs = [self.columnTitles.index(col.name) for col in noEditParams]
        self.editColIdxs = np.setdiff1d(
            np.arange(len(self.columnTitles)), self.noEditColIdxs
        )
        self.sigFieldsChanged.emit()


class ComponentManager(ComponentTableModel):
    _nextComponentId = 0
    compDf: pd.DataFrame

    sigUpdatedFocusedComponent = Signal(object)
    """pd.Series, newly focused component"""

    def __init__(self, tableData: TableData = None):
        super().__init__(tableData)
        self.focusedComponent = self.tableData.makeComponentSeries()

    def resetFields(self):
        super().resetFields()
        self._nextComponentId = 0

    @DASM.undoable("Add/Modify Components")
    def addComponents(
        self,
        componentsDf: pd.DataFrame,
        addType: PRJ_ENUMS = PRJ_ENUMS.COMPONENT_ADD_AS_NEW,
        emitChange=True,
    ):
        toEmit = self.defaultEmitDict.copy()
        existingIds = self.compDf.index
        newIdsForOrigComps = componentsDf.index.to_numpy(dtype=int, copy=True)

        if len(componentsDf) == 0:
            # Nothing to undo
            return toEmit

        # Only allow updates from columns that exist
        componentsDf = componentsDf[[c for c in componentsDf if c in self.compDf]]

        # Delete entries with no vertices, since they make work within the app difficult.
        # It is allowed to merge without vertices present
        if RTF.VERTICES in componentsDf:
            verts = componentsDf[RTF.VERTICES]
            dropLocs = verts.map(ComplexXYVertices.isEmpty).to_numpy(bool)
            dropIds = componentsDf.index[dropLocs]
            componentsDf = componentsDf.loc[~dropLocs].copy()
        elif addType != PRJ_ENUMS.COMPONENT_ADD_AS_MERGE:
            warn(
                "Cannot add new components without vertices. Returning.",
                UserWarning,
                stacklevel=2,
            )
            return
        else:
            dropLocs = np.zeros(len(componentsDf), dtype=bool)
            dropIds = np.array([], dtype=int)
            componentsDf = componentsDf.copy()

        newIdsForOrigComps[dropLocs] = -1

        if RTF.ID in componentsDf:
            # IDs take precedence over native index if present
            # Pandas 1.4 warns FutureWarning without guaranteed int dtype
            # A copy was already made above, so this is potentially redundant
            componentsDf[RTF.ID] = componentsDf[RTF.ID].astype(int, copy=False)
            componentsDf = componentsDf.set_index(RTF.ID, drop=False)

        if addType == PRJ_ENUMS.COMPONENT_ADD_AS_NEW:
            # Treat all components as new -> set their IDs to guaranteed new values
            newIds = np.arange(
                self._nextComponentId,
                self._nextComponentId + len(componentsDf),
                dtype=int,
            )
            componentsDf[RTF.ID] = newIds
            dropIds = np.array([], dtype=int)
        else:
            # Merge may have been performed with new components (id -1) mixed in
            needsUpdatedId = componentsDf.index == RTF.ID.value
            newIds = np.arange(
                self._nextComponentId,
                self._nextComponentId + np.sum(needsUpdatedId),
                dtype=int,
            )
            componentsDf.loc[needsUpdatedId, RTF.ID] = newIds

        componentsDf = componentsDf.set_index(RTF.ID, drop=False)
        newIdsForOrigComps[~dropLocs] = componentsDf.index.to_numpy(int)

        # Track dropped data for undo
        alteredIdxs = np.concatenate([componentsDf.index.values, dropIds])
        alteredDataDf = self.compDf.loc[np.intersect1d(self.compDf.index, alteredIdxs)]

        # Delete entries that were updated to have no vertices
        toEmit.update(self.removeComponents(dropIds, emitChange=False))
        # Now, merge existing IDs and add new ones
        newIds = componentsDf.index
        newChangedIdxs = np.isin(newIds, existingIds, assume_unique=True)
        changedIds = newIds[newChangedIdxs]

        # Signal to table that rows should change
        self.layoutAboutToBeChanged.emit()
        # Ensure indices overlap with the components these are replacing
        updateColumns = [c for c in componentsDf if c in self.compDf]
        updateIdxs = self.compDf.index.intersection(changedIds)
        self.compDf.loc[updateIdxs, updateColumns] = componentsDf.loc[
            updateIdxs, updateColumns
        ]
        toEmit["changed"] = changedIds

        # Record mapping for exterior scopes
        toEmit["ids"] = newIdsForOrigComps

        # Finally, add new components
        compsToAdd = componentsDf.iloc[~newChangedIdxs, :]
        # Make sure all required data is present for new rows
        missingCols = np.setdiff1d(self.compDf.columns, compsToAdd.columns)
        if missingCols.size > 0 and len(compsToAdd) > 0:
            embedInfo = self.tableData.makeComponentDf(len(compsToAdd)).set_index(
                compsToAdd.index
            )
            compsToAdd[missingCols] = embedInfo[missingCols]
        self.compDf = concatAllowEmpty((self.compDf, compsToAdd), sort=False)
        # Retain type information
        coerceDfTypes(self.compDf)

        toEmit["added"] = newIds[~newChangedIdxs]
        self.layoutChanged.emit()

        self._nextComponentId = np.max(self.compDf.index.to_numpy(), initial=-1) + 1

        if emitChange:
            self.sigComponentsChanged.emit(toEmit)

        yield toEmit

        # Undo add by deleting new components and un-updating existing ones
        self.addComponents(alteredDataDf, PRJ_ENUMS.COMPONENT_ADD_AS_MERGE)
        addedCompIdxs = toEmit["added"]
        if len(addedCompIdxs) > 0:
            self.removeComponents(toEmit["added"])

    @DASM.undoable("Remove Components")
    def removeComponents(
        self,
        removeIds: Union[np.ndarray, type(PRJ_ENUMS)] = PRJ_ENUMS.COMPONENT_REMOVE_ALL,
        emitChange=True,
    ) -> dict:
        toEmit = self.defaultEmitDict.copy()
        # Generate ID list
        existingCompIds = self.compDf.index
        if removeIds is PRJ_ENUMS.COMPONENT_REMOVE_ALL:
            removeIds = existingCompIds
        elif not hasattr(removeIds, "__iter__"):
            # single number passed in
            removeIds = [removeIds]
        removeIds = np.array(removeIds)

        # Do nothing for IDs not actually in the existing list
        idsActuallyRemoved = np.isin(removeIds, existingCompIds, assume_unique=True)
        if len(idsActuallyRemoved) == 0:
            return toEmit
        removeIds = removeIds[idsActuallyRemoved]

        # Track for undo purposes
        removedData = self.compDf.loc[removeIds]

        tfKeepIdx = np.isin(existingCompIds, removeIds, assume_unique=True, invert=True)

        # Reset manager's component list
        self.layoutAboutToBeChanged.emit()
        self.compDf: pd.DataFrame = self.compDf.iloc[tfKeepIdx, :]
        self.layoutChanged.emit()

        # Preserve type information after change
        coerceDfTypes(self.compDf)

        # Determine next ID for new components
        self._nextComponentId = 0
        if np.any(tfKeepIdx):
            self._nextComponentId = np.max(existingCompIds[tfKeepIdx].to_numpy()) + 1

        # Reflect these changes to the component list
        toEmit["deleted"] = removeIds
        if emitChange:
            self.sigComponentsChanged.emit(toEmit)
        if len(removeIds) > 0:
            yield toEmit
        else:
            # Nothing to undo
            return toEmit

        # Undo code
        self.addComponents(removedData, PRJ_ENUMS.COMPONENT_ADD_AS_MERGE)

    @DASM.undoable("Merge Components")
    def mergeById(self, mergeIds: OneDArr = None, keepId: int = None):
        """
        Merges the selected components based on spatial overlap

        Parameters
        ----------
        mergeIds
            Ids of components to merge. If *None*, defaults to current user selection.
        keepId
            If provided, the selected component with this ID is used as the merged
            component columns (except for the vertices, of course). Else, this will
            default to the first component in the selection.
        """
        if mergeIds is None or len(mergeIds) < 2:
            warn(
                f'Less than two components are selected, so "merge" is a no-op.',
                UserWarning,
                stacklevel=2,
            )
            return
        mergeComps: pd.DataFrame = self.compDf.loc[mergeIds]
        if keepId is None:
            keepId = mergeIds[0]

        keepInfo = mergeComps.loc[keepId].copy()
        keepInfo[RTF.VERTICES] = mergeComps[RTF.VERTICES].s3averts.merge()

        deleted = self.removeComponents(mergeComps.index, emitChange=False)["deleted"]
        toEmit = self.addComponents(
            keepInfo.to_frame().T, PRJ_ENUMS.COMPONENT_ADD_AS_MERGE, emitChange=False
        )
        toEmit["deleted"] = np.concatenate([toEmit["deleted"], deleted])
        self.sigComponentsChanged.emit(toEmit)

        yield
        self.addComponents(mergeComps, PRJ_ENUMS.COMPONENT_ADD_AS_MERGE)

    @DASM.undoable("Split Components")
    def splitById(self, splitIds: OneDArr):
        """
        Makes a separate component for each distinct boundary in all selected
        components. For instance, if two components are selected, and each has two
        separate circles as vertices, then 4 total components will exist after this
        operation. Each new component will have the table fields of its parent

        Parameters
        ----------
        splitIds
            Ids of components to split up
        """
        splitComps = self.compDf.loc[splitIds]
        splitVerts = splitComps[RTF.VERTICES].s3averts.split()
        newComps = splitComps.loc[splitVerts.index].copy()
        newComps[RTF.VERTICES] = splitVerts
        # Keep track of which components were removed and added by this op
        outDict = self.removeComponents(splitComps.index)
        outDict.update(self.addComponents(newComps))
        yield outDict
        undoDict = self.removeComponents(outDict["ids"])
        undoDict.update(
            self.addComponents(splitComps, PRJ_ENUMS.COMPONENT_ADD_AS_MERGE)
        )
        return undoDict

    def removeOverlapById(self, overlapIds: OneDArr):
        """
        Makes sure all specified components have no overlap. Preference is given in
        order of the given IDs, i.e. the last ID in the list is guaranteed to keep its
        full shape. If an area selection is made, priority is given to larger IDs,
        i.e. the largest ID is guaranteed to keep its full original shape.
        """
        overlapComps = self.compDf.loc[overlapIds].copy()
        overlapComps[RTF.VERTICES] = overlapComps[RTF.VERTICES].s3averts.removeOverlap()
        self.addComponents(overlapComps, PRJ_ENUMS.COMPONENT_ADD_AS_MERGE)

    def updateFocusedComponent(self, component: pd.Series = None):
        """
        Updates focused image and component from provided information. Useful for
        creating a 'zoomed-in' view that allows much faster processing than applying
        image processing algorithms to the entire image each iteration.

        Parameters
        ----------
        component
            New component to edit using various plugins (See :class:`TableFieldPlugin`)
        """
        if component is None:
            component = self.tableData.makeComponentSeries()
        else:
            # Since values INSIDE the dataframe are reset instead of modified, there is no
            # need to go through the trouble of deep copying
            component = component.copy(deep=False)

        self.focusedComponent = component

        self.sigUpdatedFocusedComponent.emit(component)

    @property
    def focusedDataframe(self):
        """Return a dataframe version of focused component with correct dtypes"""
        return coerceDfTypes(seriesAsFrame(self.focusedComponent))
