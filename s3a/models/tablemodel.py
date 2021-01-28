from typing import Union, Any
from warnings import warn

import numpy as np
import pandas as pd
import cv2 as cv
from pandas import DataFrame as df
from pyqtgraph.Qt import QtCore

from s3a import PRJ_SINGLETON
from s3a.generalutils import coerceDfTypes
from s3a.constants import REQD_TBL_FIELDS as RTF, PRJ_ENUMS
from s3a.structures import ComplexXYVertices, OneDArr

__all__ = ['ComponentMgr', 'CompTableModel']

from utilitys.fns import warnLater

Signal = QtCore.Signal

TBL_FIELDS = PRJ_SINGLETON.tableData.allFields

class CompTableModel(QtCore.QAbstractTableModel):
  # Emits 3-element dict: Deleted comp ids, changed comp ids, added comp ids
  defaultEmitDict = {'deleted': np.array([]), 'changed': np.array([]), 'added': np.array([])}
  sigCompsChanged = Signal(dict)
  sigFieldsChanged = Signal()

  # Used for efficient deletion, where deleting non-contiguous rows takes 1 operation
  # Instead of N operations

  def __init__(self):
    super().__init__()
    # Create component dataframe and remove created row. This is to
    # ensure datatypes are correct
    self.resetFields()

  # ------
  # Functions required to implement table model
  # ------
  def columnCount(self, *args, **kwargs):
    return len(self.colTitles)

  def rowCount(self, *args, **kwargs):
    return len(self.compDf)

  def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
    if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
      return self.colTitles[section]

  # noinspection PyMethodOverriding
  def data(self, index: QtCore.QModelIndex, role: int) -> Any:
    outData = self.compDf.iloc[index.row(), index.column()]
    if role == QtCore.Qt.DisplayRole:
      return str(outData)
    elif role == QtCore.Qt.EditRole:
      return outData
    else:
      return None

  @PRJ_SINGLETON.actionStack.undoable('Alter Component Data')
  def setData(self, index, value, role=QtCore.Qt.EditRole) -> bool:
    row = index.row()
    col = index.column()
    oldVal = self.compDf.iat[row, col]
    # Try-catch for case of numpy arrays
    noChange = oldVal == value
    try:
      if noChange:
        return True
    except ValueError:
      # Happens with array comparison
      pass
    self.compDf.iat[row, col] = value
    # !!! Serious issue! Using iat sometimes doesn't work and I have no idea why since it is
    # not easy to replicate. See https://github.com/pandas-dev/pandas/issues/22740
    # Also, pandas iloc unnecessarily coerces to 2D ndarray when setting, so iloc will fail
    # when assigning an array to a single location. Not sure how to prevent this...
    # For now, checking this on export
    cmp = self.compDf.iloc[row, [col, col-1]].values[0] != self.compDf.iat[row, col]
    try:
      cmp = bool(cmp)
    except ValueError:
      # Numpy array-like
      cmp = np.any(cmp)
    if cmp:
      warnLater('Warning! An error occurred setting this value. Please try again using a'
           ' <em>multi-cell</em> edit. E.g. do not just set this value, set it along with'
           ' at least one other selected cell.', UserWarning)
    toEmit = self.defaultEmitDict.copy()
    toEmit['changed'] = np.array([self.compDf.index[index.row()]])
    self.sigCompsChanged.emit(toEmit)
    yield True
    self.setData(index, oldVal, role)
    return True

  def flags(self, index: QtCore.QModelIndex) -> QtCore.Qt.ItemFlags:
    if index.column() not in self.noEditColIdxs:
      return QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable
    else:
      return QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable

  # noinspection PyAttributeOutsideInit
  def resetFields(self):
    self.colTitles = [f.name for f in TBL_FIELDS]

    self.compDf = PRJ_SINGLETON.tableData.makeCompDf(0)

    noEditParams = [f for f in PRJ_SINGLETON.tableData.allFields if f.opts.get('readonly', False)]

    self.noEditColIdxs = [self.colTitles.index(col.name) for col in noEditParams]
    self.editColIdxs = np.setdiff1d(np.arange(len(self.colTitles)), self.noEditColIdxs)
    self.sigFieldsChanged.emit()

class ComponentMgr(CompTableModel):
  _nextCompId = 0
  compDf: pd.DataFrame

  def resetFields(self):
    super().resetFields()
    self._nextCompId = 0

  @PRJ_SINGLETON.actionStack.undoable('Add Components')
  def addComps(self, newCompsDf: df, addtype: PRJ_ENUMS = PRJ_ENUMS.COMP_ADD_AS_NEW, emitChange=True):
    toEmit = self.defaultEmitDict.copy()
    existingIds = self.compDf.index

    if len(newCompsDf) == 0:
      # Nothing to undo
      return toEmit

    # Delete entries with no vertices, since they make work within the app difficult.
    # TODO: Is this the appropriate response?
    verts = newCompsDf[RTF.VERTICES]
    dropIds = newCompsDf.index[verts.map(ComplexXYVertices.isEmpty)]
    newCompsDf.drop(index=dropIds, inplace=True)

    if addtype == PRJ_ENUMS.COMP_ADD_AS_NEW:
      # Treat all comps as new -> set their IDs to guaranteed new values
      newIds = np.arange(self._nextCompId, self._nextCompId + len(newCompsDf), dtype=int)
      newCompsDf[RTF.INST_ID] = newIds
      newCompsDf.set_index(newIds, inplace=True)
      dropIds = np.array([], dtype=int)
    else:
      # Merge may have been performed with new comps (id -1) mixed in
      needsUpdatedId = newCompsDf.index == RTF.INST_ID.value
      newIds = np.arange(self._nextCompId, self._nextCompId + np.sum(needsUpdatedId), dtype=int)
      newCompsDf.loc[needsUpdatedId, RTF.INST_ID] = newIds

    # Track dropped data for undo
    alteredIdxs = np.concatenate([newCompsDf.index.values, dropIds])
    alteredDataDf = self.compDf.loc[self.compDf.index.intersection(alteredIdxs)]

    # Delete entries that were updated to have no vertices
    toEmit.update(self.rmComps(dropIds, emitChange=False))
    # Now, merge existing IDs and add new ones
    newIds = newCompsDf.index
    newChangedIdxs = np.isin(newIds, existingIds, assume_unique=True)
    changedIds = newIds[newChangedIdxs]

    # Signal to table that rows should change
    self.layoutAboutToBeChanged.emit()
    # Ensure indices overlap with the components these are replacing
    self.compDf.update(newCompsDf)
    toEmit['changed'] = changedIds

    # Finally, add new comps
    compsToAdd = newCompsDf.iloc[~newChangedIdxs, :]
    # Make sure all required data is present for new rows
    missingCols = np.setdiff1d(self.compDf.columns, compsToAdd.columns)
    if missingCols.size > 0 and len(compsToAdd) > 0:
      embedInfo = PRJ_SINGLETON.tableData.makeCompDf(len(newCompsDf)).set_index(compsToAdd.index)
      compsToAdd[missingCols] = embedInfo[missingCols]
    self.compDf = pd.concat((self.compDf, compsToAdd), sort=False)
    # Retain type information
    coerceDfTypes(self.compDf)

    toEmit['added'] = newIds[~newChangedIdxs]
    self.layoutChanged.emit()


    self._nextCompId = np.max(self.compDf.index.to_numpy(), initial=-1) + 1

    if emitChange:
      self.sigCompsChanged.emit(toEmit)

    yield toEmit

    # Undo add by deleting new components and un-updating existing ones
    self.addComps(alteredDataDf, PRJ_ENUMS.COMP_ADD_AS_MERGE)
    addedCompIdxs = toEmit['added']
    if len(addedCompIdxs) > 0:
      self.rmComps(toEmit['added'])

  @PRJ_SINGLETON.actionStack.undoable('Remove Components')
  def rmComps(self, idsToRemove: Union[np.ndarray, type(PRJ_ENUMS)] = PRJ_ENUMS.COMP_RM_ALL,
              emitChange=True) -> dict:
    toEmit = self.defaultEmitDict.copy()
    # Generate ID list
    existingCompIds = self.compDf.index
    if idsToRemove is PRJ_ENUMS.COMP_RM_ALL:
      idsToRemove = existingCompIds
    elif not hasattr(idsToRemove, '__iter__'):
      # single number passed in
      idsToRemove = [idsToRemove]
      pass
    idsToRemove = np.array(idsToRemove)

    # Do nothing for IDs not actually in the existing list
    idsActuallyRemoved = np.isin(idsToRemove, existingCompIds, assume_unique=True)
    if len(idsActuallyRemoved) == 0:
      return toEmit
    idsToRemove = idsToRemove[idsActuallyRemoved]

    # Track for undo purposes
    removedData = self.compDf.loc[idsToRemove]

    tfKeepIdx = np.isin(existingCompIds, idsToRemove, assume_unique=True, invert=True)

    # Reset manager's component list
    self.layoutAboutToBeChanged.emit()
    self.compDf: df = self.compDf.iloc[tfKeepIdx,:]
    self.layoutChanged.emit()

    # Preserve type information after change
    coerceDfTypes(self.compDf)

    # Determine next ID for new components
    self._nextCompId = 0
    if np.any(tfKeepIdx):
      self._nextCompId = np.max(existingCompIds[tfKeepIdx].to_numpy()) + 1

    # Reflect these changes to the component list
    toEmit['deleted'] = idsToRemove
    if emitChange:
      self.sigCompsChanged.emit(toEmit)
    if len(idsToRemove) > 0:
      yield toEmit
    else:
      # Nothing to undo
      return toEmit

    # Undo code
    self.addComps(removedData, PRJ_ENUMS.COMP_ADD_AS_MERGE)

  @PRJ_SINGLETON.actionStack.undoable('Merge Components')
  def mergeCompVertsById(self, mergeIds: OneDArr=None, keepId: int=None):
    """
    Merges the selected components

    :param mergeIds: Ids of components to merge. If *None*, defaults to current user
      selection.
    :param keepId: If provided, the selected component with this ID is used as
      the merged component columns (except for the vertices, of course). Else,
      this will default to the first component in the selection.
    """
    if mergeIds is None or len(mergeIds) < 2:
      warn(f'Less than two components are selected, so "merge" is a no-op.', UserWarning)
      return
    mergeComps: df = self.compDf.loc[mergeIds].copy()
    if keepId is None:
      keepId = mergeIds[0]

    keepInfo = mergeComps.loc[keepId].copy()
    allVerts = [v.stack() for v in mergeComps[RTF.VERTICES]]
    maskShape = np.max(np.vstack(allVerts), 0)[::-1]
    mask = np.zeros(maskShape, bool)
    for verts in mergeComps[RTF.VERTICES]: # type: ComplexXYVertices
      mask |= verts.toMask(tuple(maskShape))
    newVerts = ComplexXYVertices.fromBwMask(mask)
    keepInfo[RTF.VERTICES] = newVerts

    self.rmComps(mergeComps.index)
    self.addComps(keepInfo.to_frame().T, PRJ_ENUMS.COMP_ADD_AS_MERGE)
    yield
    self.addComps(mergeComps, PRJ_ENUMS.COMP_ADD_AS_MERGE)

  @PRJ_SINGLETON.actionStack.undoable('Split Components')
  def splitCompVertsById(self, splitIds: OneDArr):
    """
    Makes a separate component for each distinct boundary in all selected components.
    For instance, if two components are selected, and each has two separate circles as
    vertices, then 4 total components will exist after this operation.

    Each new component will have the table fields of its parent

    :param splitIds: Ids of components to split up
    """
    splitComps = self.compDf.loc[splitIds].copy()
    newComps_lst = []
    for _, comp in splitComps.iterrows():
      verts: ComplexXYVertices = comp[RTF.VERTICES]
      tmpMask = verts.toMask(asBool=False).astype('uint8')
      nComps, ccompImg = cv.connectedComponents(tmpMask)
      newVerts = []
      for ii in range(1, nComps):
        newVerts.append(ComplexXYVertices.fromBwMask(ccompImg == ii))
      childComps = pd.concat([comp.to_frame().T]*(nComps-1))
      childComps[RTF.VERTICES] = newVerts
      newComps_lst.append(childComps)
    newComps = pd.concat(newComps_lst)
    # Keep track of which comps were removed and added by this op
    outDict = self.rmComps(splitComps.index)
    outDict.update(self.addComps(newComps))
    yield outDict
    undoDict = self.rmComps(newComps.index)
    undoDict.update(self.addComps(splitComps, PRJ_ENUMS.COMP_ADD_AS_MERGE))
    return undoDict

  def removeOverlapById(self, overlapIds: OneDArr):
    """
    Makes sure all specified components have no overlap. Preference is given
    in order of the given IDs, i.e. the last ID in the list is guaranteed to
    keep its full shape. If an area selection is made, priority is given to larger
    IDs, i.e. the largest ID is guaranteed to keep its full original shape.
    """
    overlapComps = self.compDf.loc[overlapIds].copy()
    allVerts = np.vstack([v.stack() for v in overlapComps[RTF.VERTICES]])
    wholeMask = np.zeros(allVerts.max(0)[::-1], dtype='uint16')
    for ii, (_, comp) in enumerate(overlapComps.iterrows(), 1):
      comp[RTF.VERTICES].toMask(wholeMask, ii, asBool=False)
    for ii, compId in enumerate(overlapIds, 1):
      verts = ComplexXYVertices.fromBwMask(wholeMask == ii)
      overlapComps.at[compId, RTF.VERTICES] = verts
    self.addComps(overlapComps, PRJ_ENUMS.COMP_ADD_AS_MERGE)