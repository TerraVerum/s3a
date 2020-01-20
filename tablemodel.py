from __future__ import annotations
from pyqtgraph.Qt import QtCore

from pandas import DataFrame as df
import pandas as pd
import numpy as np

from constants import TEMPLATE_COMP as TC, CompParams

from typing import Union

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

  def addComps(self, newCompsDf: df, addtype='new'):
    toEmit = self.defaultEmitDict.copy()
    idCol = TC.INST_ID.name
    if addtype == 'new':
      # Treat all comps as new -> set their IDs to guaranteed new values
      newIds = np.arange(self._nextCompId, self._nextCompId + len(newCompsDf), dtype=int)
      newCompsDf[idCol] = newIds
      newCompsDf = newCompsDf.set_index(newIds)
    # Now, merge existing IDs and add new ones
    existingIds = self.compDf.index
    newIds = newCompsDf.index
    newChangedIdxs = np.isin(newIds, existingIds, assume_unique=True)

    # Signal to table that rows should change
    self.layoutAboutToBeChanged.emit()
    # Ensure indices overlap with the components these are replacing
    self.compDf.update(newCompsDf)
    self.compDf.update(newCompsDf)
    toEmit['changed'] = newIds[newChangedIdxs]

    # Finally, add new comps
    compsToAdd = newCompsDf.iloc[~newChangedIdxs, :]
    self.compDf = pd.concat((self.compDf, compsToAdd))
    toEmit['added'] = newIds[~newChangedIdxs]
    self.layoutChanged.emit()


    self._nextCompId = np.max(self.compDf[idCol]) + 1
    self.sigCompsChanged.emit(toEmit)

  def rmComps(self, idsToRemove: Union[np.array, str] = 'all'):
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
    self.sigCompsChanged.emit(toEmit)

  def exportToFile(self, outFile: str, **pdExportArgs) -> bool:
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
    try:
      # TODO: Currently the additional options are causing errors. Find out why and fix
      #  them, since this may be useful if it can be modified
      self.compDf.to_csv(outFile, index=False)
      return True
    except FileNotFoundError:
      return False

  def importFromFile(self, inFile: str) -> bool:
    """
    Deserializes data from a csv file to create a Component :class:`DataFrame`.
    The input .csv should be the same format as one exported by
    :func:`exportToFile <ComponentMgr.exportToFile>`.

    :param inFile: Name of file to import
    :return: Success or failure of the operation
    """
    stringDf = pd.read_csv(inFile)
    # TODO: Find out how to convert this into an object df somehow
    compDf = makeCompDf(len(stringDf))
    # compDf = stringDfToCompDf(stringDf)
    self.addComps(compDf)
    return True

if __name__ == '__main__':
  pass
  #t = CompTableModel()
  #tbl = makeCompDf(6)
  #tbl.set_index(np.arange(len(tbl)), inplace=True)

  #tbl2 = makeCompDf(6)
  #tbl2.set_index(np.arange(0, 2*len(tbl), 2), inplace=True)
  #point=1