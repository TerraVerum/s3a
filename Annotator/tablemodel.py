from __future__ import annotations

import re
import sys
from ast import literal_eval
from enum import Enum
from typing import Union, Any, Optional

import numpy as np
import pandas as pd
from pandas import DataFrame as df
from pyqtgraph.Qt import QtCore
from tqdm import tqdm

from .constants import TEMPLATE_COMP as TC, CompParams, ComponentTypes

Slot = QtCore.pyqtSlot
Signal = QtCore.pyqtSignal

def makeCompDf(numRows=1) -> df:
  """
  Creates a dataframe for the requested number of components.
  This is the recommended method for component instantiation prior to table insertion.
  """
  df_list = []
  for _ in range(numRows):
    # Make sure to construct a separate component instance for
    # each row no objects have the same reference
    df_list.append([field.value for field in CompParams()])
  return df(df_list, columns=TC.paramNames()).set_index(TC.INST_ID.name, drop=False)

class AddTypes(Enum):
  NEW: Enum = 'new'
  MERGE: Enum = 'merge'

class CsvIOError(Exception):
  def __init__(self, message):
    super().__init__(message)

class CompTableModel(QtCore.QAbstractTableModel):
  colTitles = TC.paramNames()

  # Emits 3-element dict: Deleted comp ids, changed comp ids, added comp ids
  defaultEmitDict = {'deleted': np.array([]), 'changed': np.array([]), 'added': np.array([])}
  sigCompsChanged = Signal(dict)

  def __init__(self):
    super().__init__()
    # Create component dataframe and remove created row. This is to
    # ensure datatypes are correct
    self.compDf = makeCompDf()
    self.compDf = self.compDf.drop(index=TC.INST_ID.value)

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

  def data(self, index, role=QtCore.Qt.DisplayRole):
    outData = self.compDf.iloc[index.row(), index.column()]
    if role == QtCore.Qt.DisplayRole:
      return str(outData)
    elif role == QtCore.Qt.EditRole:
      return outData
    else:
      return None

  def setData(self, index, value, role=QtCore.Qt.EditRole):
    self.compDf.iloc[index.row(), index.column()] = value
    toEmit = self.defaultEmitDict.copy()
    toEmit['changed'] = np.array([self.compDf.index[index.row()]])
    self.sigCompsChanged.emit(toEmit)
    return True

  def flags(self, index: QtCore.QModelIndex):
    noEditColIdxs = [self.colTitles.index(col)for col in
                     [TC.INST_ID.name, TC.VERTICES.name]]
    if index.column() not in noEditColIdxs:
      return QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable
    else:
      return QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled

class ComponentMgr(CompTableModel):
  _nextCompId = 0

  def __init__(self):
    super().__init__()

  def addComps(self, newCompsDf: df, addtype: AddTypes = AddTypes.NEW):
    toEmit = self.defaultEmitDict.copy()
    existingIds = self.compDf.index

    # Delete entries with no vertices, since they make work within the app difficult.
    # TODO: Is this the appropriate response?
    dropIds = newCompsDf.index[newCompsDf[TC.VERTICES.name].map(lambda el: len(el) == 0)]
    newCompsDf.drop(index=dropIds, inplace=True)
    # Inform graphics elements of deletion if this ID is already in our dataframe
    toEmit.update(self.rmComps(dropIds, emitChange=False))

    if addtype == AddTypes.NEW:
      # Treat all comps as new -> set their IDs to guaranteed new values
      newIds = np.arange(self._nextCompId, self._nextCompId + len(newCompsDf), dtype=int)
      newCompsDf.loc[:,TC.INST_ID.name] = newIds
      newCompsDf = newCompsDf.set_index(newIds)
    # Now, merge existing IDs and add new ones
    # TODO: Add some metric for merging other than a total override. Currently, even if the existing
    #  component has a value in e.g. vertices while the new component does not, the new, empty value
    #  will override the older. This might often be desirable, but it would still be good to let the
    #  user have the final say on what happens
    newIds = newCompsDf.index
    newChangedIdxs = np.isin(newIds, existingIds, assume_unique=True)

    # Signal to table that rows should change
    self.layoutAboutToBeChanged.emit()
    # Ensure indices overlap with the components these are replacing
    self.compDf.update(newCompsDf)
    toEmit['changed'] = newIds[newChangedIdxs]

    # Finally, add new comps
    compsToAdd = newCompsDf.iloc[~newChangedIdxs, :]
    self.compDf = pd.concat((self.compDf, compsToAdd), sort=False)
    toEmit['added'] = newIds[~newChangedIdxs]
    self.layoutChanged.emit()


    self._nextCompId = np.max(self.compDf.index) + 1
    self.sigCompsChanged.emit(toEmit)

  def rmComps(self, idsToRemove: Union[np.array, str] = 'all', emitChange=True) -> Optional[dict]:
    toEmit = self.defaultEmitDict.copy()
    # Generate ID list
    existingCompIds = self.compDf.index
    if idsToRemove is 'all':
      idsToRemove = existingCompIds
    elif not hasattr(idsToRemove, '__iter__'):
      # single number passed in
      idsToRemove = [idsToRemove]
      pass

    tfKeepIdx = np.isin(existingCompIds, idsToRemove, assume_unique=True, invert=True)

    # Reset manager's component list
    self.layoutAboutToBeChanged.emit()
    self.compDf = self.compDf.iloc[tfKeepIdx,:]
    self.layoutChanged.emit()

    # Determine next ID for new components
    self._nextCompId = 0
    if np.any(tfKeepIdx):
      self._nextCompId = np.max(existingCompIds[tfKeepIdx].to_numpy()) + 1

    # Reflect these changes to the component list
    toEmit['deleted'] = idsToRemove
    if emitChange:
      self.sigCompsChanged.emit(toEmit)
    return toEmit

  def csvExport(self, outFile: str, **pdExportArgs) -> bool:
    """
    Serializes the table data and returns the success or failure of the operation.

    :param outFile: Name of the output file location
    :param pdExportArgs: Dictionary of values passed to underlying pandas export function.
           These will overwrite the default options for :func:`exportToFile
           <ComponentMgr.exportToFile>`
    :return: Success or failure of the operation.
    """
    defaultExportParams = {
      'na_rep': 'NaN',
      'float_format': '{:0.10n}',
      'index': False
    }
    defaultExportParams.update(pdExportArgs)
    # Make sure no rows are truncated
    pd.set_option('display.max_rows', sys.maxsize)
    oldNpOpts=  np.get_printoptions()
    np.set_printoptions(threshold=sys.maxsize)
    success = False
    try:
      # TODO: Currently the additional options are causing errors. Find out why and fix
      #  them, since this may be useful if it can be modified
      # TODO: Add some comment to the top of the CSV or some extra text file output with additional metrics
      #  about the export, like time, who did it, what image it was from, etc.
      self.compDf.to_csv(outFile, index=False)
      success = True
    except IOError:
      # success is already false
      pass
    finally:
      pd.reset_option('display.max_rows')
      # False positive checker warning for some reason
      # noinspection PyTypeChecker
      np.set_printoptions(oldNpOpts)
      return success

  def csvImport(self, inFile: str, loadType=AddTypes.NEW,
                imShape: Optional[tuple]=None) -> Optional[Exception]:
    """
    Deserializes data from a csv file to create a Component :class:`DataFrame`.
    The input .csv should be the same format as one exported by
    :func:`csvImport <ComponentMgr.csvImport>`.

    :param imShape: If included, this ensures all imported components lie within imSize
           boundaries. If any components do not, an error is thrown since this is
           indicative of components that actually came from a different reference image.
    :param inFile: Name of file to import
    :param loadType: Whether new components should be added to the exisitng component list as new
           or if they should be merged by ID with existing entries. Currently, all fields of the
           existing component will be overwritten by the new values, even if they had text/values.
    :return: Exception that occurs if the operation did not succeed. Otherwise,
           this value will be `None`.
    """
    try:
      csvDf = pd.read_csv(inFile, keep_default_na=False)
      # Objects in the original frame are represented as strings, so try to convert these
      # as needed
      stringCols = csvDf.columns[csvDf.dtypes == object]
      valToParamMap = {param.name: param.value for param in TC}
      for col in stringCols:
        paramVal = valToParamMap[col]
        # No need to perform this expensive computation if the values are already strings
        if not isinstance(paramVal, str):
          csvDf[col] = _strSerToParamSer(csvDf[col], valToParamMap[col])
      csvDf = csvDf.set_index(TC.INST_ID.name, drop=False)

      # Image shape from row-col -> x-y
      imShape = np.array(imShape[1::-1])[None,:]
      # Remove components whose vertices go over any image edges
      vertMaxs = [verts.max(0) for verts in csvDf[TC.VERTICES.name] if len(verts) > 0]
      vertMaxs = np.vstack(vertMaxs)
      offendingIds = np.nonzero(np.any(vertMaxs >= imShape, axis=1))[0]
      if len(offendingIds) > 0:
        raise CsvIOError(f'Vertices on some components extend beyond image dimensions. '
                         f'Perhaps this export came from a different image?\n'
                         f'Offending IDs: {offendingIds}')
    except Exception as ex:
      return ex
    # TODO: Apply this function to individual rows instead of the whole dataframe. This will allow malformed
    #  rows to gracefully fall off the dataframe with some sort of warning message
    self.addComps(csvDf, loadType)
    return None

def _strSerToParamSer(strSeries: pd.Series, paramVal: Any) -> Any:
  paramType = type(paramVal)
  funcMap = {
    # Format string to look like a list, use ast to convert that string INTO a list, make a numpy array from the list
    np.ndarray    : lambda strVal: np.array(literal_eval(re.sub(r'(\d|\])\s+', '\\1,', strVal.replace('\n', '')))),
    bool          : lambda strVal: strVal.lower() == 'true',
    ComponentTypes: lambda strVal: ComponentTypes.fromString(strVal)
  }
  defaultFunc = lambda strVal: paramType(strVal)
  funcToUse = funcMap.get(paramType, defaultFunc)
  return strSeries.apply(funcToUse)

if __name__ == '__main__':
  pass
  #t = CompTableModel()
  #tbl = makeCompDf(6)
  #tbl.set_index(np.arange(len(tbl)), inplace=True)

  #tbl2 = makeCompDf(6)
  #tbl2.set_index(np.arange(0, 2*len(tbl), 2), inplace=True)
  #point=1