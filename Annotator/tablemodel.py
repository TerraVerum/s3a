import re
import sys
from ast import literal_eval
from typing import Union, Any, Optional, Sequence

import numpy as np
import pandas as pd
from pandas import DataFrame as df
from pyqtgraph.Qt import QtCore

from .FRGraphics.parameditors import FR_SINGLETON
from .generalutils import coerceDfTypes
from .projectvars import FR_ENUMS, TEMPLATE_COMP as TC, CompParams
from .projectvars.constants import FR_CONSTS
from .structures import FRComplexVertices, FRParam, FRCsvIOError

Slot = QtCore.pyqtSlot
Signal = QtCore.pyqtSignal

def makeCompDf(numRows=1) -> df:
  """
  Creates a dataframe for the requested number of components.
  This is the recommended method for component instantiation prior to table insertion.
  """
  df_list = []
  dropRow = False
  if numRows <= 0:
    # Create one row and drop it, which ensures data types are correct in the empty
    # dataframe
    numRows = 1
    dropRow = True
  for _ in range(numRows):
    # Make sure to construct a separate component instance for
    # each row no objects have the same reference
    df_list.append([field.value for field in CompParams()])
  outDf = df(df_list, columns=TC).set_index(TC.INST_ID, drop=False)
  # Set the metadata for this application run
  outDf[TC.ANN_AUTHOR] = FR_SINGLETON.annotationAuthor
  outDf[TC.ANN_TIMESTAMP] = pd.datetime.utcnow()
  outDf[TC.ANN_FILENAME] = FR_CONSTS.ANN_CUR_FILE_INDICATOR
  if dropRow:
    outDf = outDf.drop(index=TC.INST_ID.value)
  return outDf

class CompTableModel(QtCore.QAbstractTableModel):
  colTitles = TC.paramNames()

  # Emits 3-element dict: Deleted comp ids, changed comp ids, added comp ids
  defaultEmitDict = {'deleted': np.array([]), 'changed': np.array([]), 'added': np.array([])}
  sigCompsChanged = Signal(dict)

  # Used for efficient deletion, where deleting non-contiguous rows takes 1 operation
  # Instead of N operations

  def __init__(self):
    super().__init__()
    # Create component dataframe and remove created row. This is to
    # ensure datatypes are correct
    self.compDf = makeCompDf(0)

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

  def data(self, index: QtCore.QModelIndex, role: int) -> Any:
    outData = self.compDf.iloc[index.row(), index.column()]
    if role == QtCore.Qt.DisplayRole:
      return str(outData)
    elif role == QtCore.Qt.EditRole:
      return outData
    else:
      return None

  def setData(self, index, value, role=QtCore.Qt.EditRole) -> bool:
    self.compDf.iloc[index.row(), index.column()] = value
    toEmit = self.defaultEmitDict.copy()
    toEmit['changed'] = np.array([self.compDf.index[index.row()]])
    self.sigCompsChanged.emit(toEmit)
    return True

  def flags(self, index: QtCore.QModelIndex) -> QtCore.Qt.ItemFlags:
    noEditColIdxs = [self.colTitles.index(col.name)for col in
                     [TC.INST_ID, TC.VERTICES]]
    if index.column() not in noEditColIdxs:
      return QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable
    else:
      return QtCore.Qt.ItemIsEnabled

@FR_SINGLETON.registerClass(FR_CONSTS.CLS_COMP_MGR)
class ComponentMgr(CompTableModel):
  @FR_SINGLETON.generalProps.registerProp(FR_CONSTS.EXP_ONLY_VISIBLE)
  def exportOnlyVis(self): pass

  _nextCompId = 0

  def __init__(self):
    super().__init__()

  def addComps(self, newCompsDf: df, addtype: FR_ENUMS = FR_ENUMS.COMP_ADD_AS_NEW):
    toEmit = self.defaultEmitDict.copy()
    existingIds = self.compDf.index

    # Delete entries with no vertices, since they make work within the app difficult.
    # TODO: Is this the appropriate response?
    verts = newCompsDf[TC.VERTICES]
    dropIds = newCompsDf.index[verts.map(lambda complexVerts: len(complexVerts.stack()) == 0)]
    newCompsDf.drop(index=dropIds, inplace=True)
    # Inform graphics elements of deletion if this ID is already in our dataframe
    toEmit.update(self.rmComps(dropIds, emitChange=False))

    if addtype == FR_ENUMS.COMP_ADD_AS_NEW:
      # Treat all comps as new -> set their IDs to guaranteed new values
      newIds = np.arange(self._nextCompId, self._nextCompId + len(newCompsDf), dtype=int)
      newCompsDf.loc[:,TC.INST_ID] = newIds
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
    # Retain type information
    coerceDfTypes(self.compDf, TC)

    toEmit['added'] = newIds[~newChangedIdxs]
    self.layoutChanged.emit()


    self._nextCompId = np.max(self.compDf.index.to_numpy(), initial=int(-1)) + 1
    self.sigCompsChanged.emit(toEmit)
    return toEmit

  def rmComps(self, idsToRemove: Union[np.array, str] = 'all', emitChange=True) -> dict:
    toEmit = self.defaultEmitDict.copy()
    # Generate ID list
    existingCompIds = self.compDf.index
    if idsToRemove is 'all':
      idsToRemove = existingCompIds
    elif not hasattr(idsToRemove, '__iter__'):
      # single number passed in
      idsToRemove = [idsToRemove]
      pass
    idsToRemove = np.array(idsToRemove)

    # Do nothing for IDs not actually in the existing list
    idsActuallyRemoved = np.isin(idsToRemove, existingCompIds, assume_unique=True)
    idsToRemove = idsToRemove[idsActuallyRemoved]

    tfKeepIdx = np.isin(existingCompIds, idsToRemove, assume_unique=True, invert=True)

    # Reset manager's component list
    self.layoutAboutToBeChanged.emit()
    self.compDf = self.compDf.iloc[tfKeepIdx,:]
    self.layoutChanged.emit()

    # Preserve type information after change
    coerceDfTypes(self.compDf, TC)

    # Determine next ID for new components
    self._nextCompId = 0
    if np.any(tfKeepIdx):
      self._nextCompId = np.max(existingCompIds[tfKeepIdx].to_numpy()) + 1

    # Reflect these changes to the component list
    toEmit['deleted'] = idsToRemove
    if emitChange:
      self.sigCompsChanged.emit(toEmit)
    return toEmit

  def csvExport(self, outFile: str,
                exportIds:Union[FR_ENUMS, Sequence] = FR_ENUMS.COMP_EXPORT_ALL,
                **pdExportArgs) \
      -> bool:
    """
    Serializes the table data and returns the success or failure of the operation.

    :param outFile: Name of the output file location
    :param exportIds: If :var:`FR_ENUMS.EXPORT_ALL`,
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
      if isinstance(exportIds, FR_ENUMS) and exportIds == FR_ENUMS.EXPORT_ALL:
        exportDf = self.compDf
      else:
        exportDf = self.compDf.loc[exportIds,:]
      exportDf: df = exportDf.copy(deep=True)
      # Format special columns appropriately
      for col in exportDf:
        if hasattr(col.value, 'serialize'):
          exportDf[col] = exportDf[col].map(type(col.value).serialize)
      exportDf.to_csv(outFile, index=False)
      success = True
    except IOError:
      # success is already false
      pass
    finally:
      pd.reset_option('display.max_rows')
      # False positive checker warning for some reason
      # noinspection PyTypeChecker
      np.set_printoptions(**oldNpOpts)
    return success

  def csvImport(self, inFile: str, loadType=FR_ENUMS.COMP_ADD_AS_NEW,
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
      csvDf.columns = TC
      csvDf = csvDf.set_index(TC.INST_ID, drop=False)

      if imShape is not None:
        # Image shape from row-col -> x-y
        imShape = np.array(imShape[1::-1])[None,:]
        # Remove components whose vertices go over any image edges
        vertMaxs = [verts.stack().max(0) for verts in csvDf[TC.VERTICES] if len(verts) > 0]
        vertMaxs = np.vstack(vertMaxs)
        offendingIds = np.nonzero(np.any(vertMaxs >= imShape, axis=1))[0]
        if len(offendingIds) > 0:
          raise FRCsvIOError(f'Vertices on some components extend beyond image dimensions. '
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
    np.ndarray        : lambda strVal: np.array(literal_eval(re.sub(r'(\d|\])\s+', '\\1,', strVal.replace('\n', '')))),
    FRComplexVertices : FRComplexVertices.deserialize,
    bool              : lambda strVal: strVal.lower() == 'true',
    FRParam           : lambda strVal: paramVal.group.fromString(strVal)
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