import pickle
import sys
from ast import literal_eval
from pathlib import Path
from stat import S_IREAD, S_IRGRP, S_IROTH
from typing import Union, Any, Optional, List, Tuple

import cv2 as cv
import numpy as np
import pandas as pd
from pandas import DataFrame as df
from pyqtgraph.Qt import QtCore
from skimage import io

from s3a.structures import OneDArr, FRParamGroup, FilePath
from s3a.structures.typeoverloads import TwoDArr, NChanImg
from . import FR_SINGLETON
from .generalutils import coerceDfTypes, augmentException
from .projectvars import FR_ENUMS, REQD_TBL_FIELDS
from .projectvars.constants import FR_CONSTS
from .structures import FRComplexVertices, FRParam, FRAppIOError

Slot = QtCore.pyqtSlot
Signal = QtCore.pyqtSignal

TBL_FIELDS = FR_SINGLETON.tableData.allFields

class FRCompTableModel(QtCore.QAbstractTableModel):
  # Emits 3-element dict: Deleted comp ids, changed comp ids, added comp ids
  defaultEmitDict = {'deleted': np.array([]), 'changed': np.array([]), 'added': np.array([])}
  sigCompsChanged = Signal(dict)

  # Used for efficient deletion, where deleting non-contiguous rows takes 1 operation
  # Instead of N operations

  def __init__(self):
    super().__init__()
    # Create component dataframe and remove created row. This is to
    # ensure datatypes are correct
    self.colTitles =   colTitles = [f.name for f in TBL_FIELDS]

    self.compDf = FR_SINGLETON.tableData.makeCompDf(0)

    noEditParams = set(REQD_TBL_FIELDS) - {REQD_TBL_FIELDS.COMP_CLASS}
    self.noEditColIdxs = [self.colTitles.index(col.name) for col in noEditParams]

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

  @FR_SINGLETON.actionStack.undoable('Alter Component Data')
  def setData(self, index, value, role=QtCore.Qt.EditRole) -> bool:
    oldVal = self.compDf.iloc[index.row(), index.column()]
    # Try-catch for case of numpy arrays
    noChange = oldVal == value
    try:
      if noChange:
        return True
    except ValueError:
      pass
    self.compDf.iloc[index.row(), index.column()] = value
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

@FR_SINGLETON.registerGroup(FR_CONSTS.CLS_COMP_MGR)
class FRComponentMgr(FRCompTableModel):
  _nextCompId = 0

  def __init__(self):
    super().__init__()

  @FR_SINGLETON.actionStack.undoable('Add Components')
  def addComps(self, newCompsDf: df, addtype: FR_ENUMS = FR_ENUMS.COMP_ADD_AS_NEW, emitChange=True):
    toEmit = self.defaultEmitDict.copy()
    existingIds = self.compDf.index

    if len(newCompsDf) == 0:
      # Nothing to undo
      return toEmit

    # Delete entries with no vertices, since they make work within the app difficult.
    # TODO: Is this the appropriate response?
    verts = newCompsDf[REQD_TBL_FIELDS.VERTICES]
    dropIds = newCompsDf.index[verts.map(FRComplexVertices.isEmpty)]
    newCompsDf.drop(index=dropIds, inplace=True)


    if addtype == FR_ENUMS.COMP_ADD_AS_NEW:
      # Treat all comps as new -> set their IDs to guaranteed new values
      newIds = np.arange(self._nextCompId, self._nextCompId + len(newCompsDf), dtype=int)
      newCompsDf.loc[:,REQD_TBL_FIELDS.INST_ID] = newIds
      newCompsDf.set_index(newIds, inplace=True)
      dropIds = np.array([], dtype=int)

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
    self.compDf = pd.concat((self.compDf, compsToAdd), sort=False)
    # Retain type information
    coerceDfTypes(self.compDf, TBL_FIELDS)

    toEmit['added'] = newIds[~newChangedIdxs]
    self.layoutChanged.emit()


    self._nextCompId = np.max(self.compDf.index.to_numpy(), initial=int(-1)) + 1

    if emitChange:
      self.sigCompsChanged.emit(toEmit)

    yield toEmit

    # Undo add by deleting new components and un-updating existing ones
    self.addComps(alteredDataDf, FR_ENUMS.COMP_ADD_AS_MERGE)
    addedCompIdxs = toEmit['added']
    if len(addedCompIdxs) > 0:
      self.rmComps(toEmit['added'])

  @FR_SINGLETON.actionStack.undoable('Remove Components')
  def rmComps(self, idsToRemove: Union[np.ndarray, type(FR_ENUMS)] = FR_ENUMS.COMP_RM_ALL,
              emitChange=True) -> dict:
    toEmit = self.defaultEmitDict.copy()
    # Generate ID list
    existingCompIds = self.compDf.index
    if idsToRemove is FR_ENUMS.COMP_RM_ALL:
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
    coerceDfTypes(self.compDf, TBL_FIELDS)

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
    self.addComps(removedData, FR_ENUMS.COMP_ADD_AS_MERGE)

def _strSerToParamSer(strSeries: pd.Series, paramVal: Any) -> pd.Series:
  paramType = type(paramVal)
  # TODO: Move this to a more obvious place?
  funcMap = {
    # Format string to look like a list, use ast to convert that string INTO a list, make a numpy array from the list
    np.ndarray        : lambda strVal: np.array(literal_eval(strVal)),
    FRComplexVertices : FRComplexVertices.deserialize,
    bool              : lambda strVal: strVal.lower() == 'true',
    FRParam           : lambda strVal: FRParamGroup.fromString(paramVal.group, strVal)
  }
  defaultFunc = lambda strVal: paramType(strVal)
  funcToUse = funcMap.get(paramType, defaultFunc)
  return strSeries.apply(funcToUse)

def _paramSerToStrSer(paramSer: pd.Series, paramVal: Any) -> pd.Series:
  # TODO: Move along with above function?
  paramType = type(paramVal)
  funcMap = {
    # Format string to look like a list, use ast to convert that string INTO a list, make a numpy array from the list
    np.ndarray: lambda param: str(param.tolist()),
    FRComplexVertices: FRComplexVertices.serialize,
  }
  defaultFunc = lambda param: str(param)

  funcToUse = funcMap.get(paramType, defaultFunc)
  return paramSer.apply(funcToUse)


@FR_SINGLETON.registerGroup(FR_CONSTS.CLS_COMP_EXPORTER)
class FRComponentIO:
  """
  Exporter responsible for saving Component information to a file or object.
  Once created, users can extract different representations of components by
  calling exporter.exportCsv, exportPkl, etc. for those objects / files respectively.
  """
  @classmethod
  def __initEditorParams__(cls):
    cls.exportOnlyVis, cls.includeFullSourceImgName = \
      FR_SINGLETON.generalProps.registerProps(cls,
      [FR_CONSTS.EXP_ONLY_VISIBLE, FR_CONSTS.INCLUDE_FNAME_PATH]
    )

  def __init__(self):
    self.compDf: Optional[df] = None

  def prepareDf(self, compDf: df, displayIds: OneDArr=None, srcImgFname: Path=None):
    """
    :param compDf: The component dataframe that came from the component manager
    :param displayIds: If not self.exportOnlyVis, exports every component in the
      dataframe. Otherwise, just exports the requested IDs
    :param srcImgFname: Main image filename. This associates each new annotation
      to this main image where they came from
    """
    if self.exportOnlyVis and displayIds is not None:
      exportIds = displayIds
    else:
      exportIds = compDf.index
    exportDf: df = compDf.loc[exportIds,:].copy()
    if not self.includeFullSourceImgName and srcImgFname is not None:
      # Only use the file name, not the whole path
      srcImgFname = srcImgFname.name
    # Assign correct export name for only new components
    overwriteIdxs = exportDf[REQD_TBL_FIELDS.SRC_IMG_FILENAME] == FR_CONSTS.ANN_CUR_FILE_INDICATOR.value
    # TODO: Maybe the current filename will match the current file indicator. What happens then?
    exportDf.loc[overwriteIdxs, REQD_TBL_FIELDS.SRC_IMG_FILENAME] = srcImgFname
    self.compDf = exportDf

  # -----
  # Export options
  # -----
  def exportCsv(self, outFile: str=None, **pdExportArgs) -> (Any, str):
    """

    :param outFile: Name of the output file location. If *None*, no file is created. However,
      the export object will still be created and returned.
    :param pdExportArgs: Dictionary of values passed to underlying pandas export function.
      These will overwrite the default options for :func:`exportToFile <FRComponentMgr.exportToFile>`
    :return: (Export object, Success or failure of the operation) tuple.
    """
    defaultExportParams = {
      'na_rep': 'NaN',
      'float_format': '{:0.10n}',
      'index': False
    }
    outPath = Path(outFile)
    outPath.parent.mkdir(exist_ok=True)
    defaultExportParams.update(pdExportArgs)
    # Make sure no rows are truncated
    pd.set_option('display.max_rows', sys.maxsize)
    oldNpOpts = np.get_printoptions()
    np.set_printoptions(threshold=sys.maxsize)
    col = None
    try:
      # TODO: Currently the additional options are causing errors. Find out why and fix
      #  them, since this may be useful if it can be modified
      # Format special columns appropriately
      # Since CSV export significantly modifies the df, make a copy before doing all these
      # operations
      exportDf = self.compDf.copy(deep=True)
      for col in exportDf:
        if not isinstance(col.value, str):
          exportDf[col] = _paramSerToStrSer(exportDf[col], col.value)
      if outFile is not None:
        exportDf.to_csv(outFile, index=False)
        outPath.chmod(S_IREAD|S_IRGRP|S_IROTH)
    except Exception as ex:
      errMsg = f'Error on parsing column "{col.name}"\n'
      augmentException(ex, errMsg)
      raise
    finally:
      pd.reset_option('display.max_rows')
      # False positive checker warning for some reason
      # noinspection PyTypeChecker
      np.set_printoptions(**oldNpOpts)
    return exportDf

  def exportPkl(self, outFile=None) -> (Any, str):
    """
    See the function signature for :func:`exportCsv <FRComponentIO.exportCsv>`
    """
    # Since the write-out is a single operation there isn't an intermediate form to return
    pklDf = None
    if outFile is not None:
      pklDf = pickle.dumps(self.compDf)
      self.compDf.to_pickle(outFile)
    return pklDf

  def exportLabeledImg(self,
                       mainImgShape: Tuple[int],
                       outFile: str = None,
                       types: List[FRParam] = None,
                       colorPerType: TwoDArr = None,
                       ) -> (NChanImg, str):
    # Set up input arguments
    if types is None:
      types = [param for param in FR_SINGLETON.tableData.compClasses]
    if colorPerType is None:
      colorPerType = np.arange(1, len(types) + 1, dtype=int)[:,None]
    outShape = mainImgShape[:2]
    if colorPerType.shape[1] > 1:
      outShape += (colorPerType.shape[1],)
    out = np.zeros(outShape, 'uint8')

    # Create label to output mapping
    for curType, color in zip(types, colorPerType):
      outlines = self.compDf.loc[self.compDf[REQD_TBL_FIELDS.COMP_CLASS] == curType, REQD_TBL_FIELDS.VERTICES].values
      cvFillArg = [arr[0] for arr in outlines]
      cvClrArg = tuple([int(val) for val in color])
      cv.fillPoly(out, cvFillArg, cvClrArg)

    if outFile is not None:
      io.imsave(outFile, out, check_contrast=False)

    return out

  # -----
  # Import options
  # -----

  @classmethod
  def buildFromCsv(cls, inFile: FilePath, imShape: Tuple=None) -> df:
    """
    Deserializes data from a csv file to create a Component :class:`DataFrame`.
    The input .csv should be the same format as one exported by
    :func:`csvImport <FRComponentMgr.csvImport>`.

    :param imShape: If included, this ensures all imported components lie within imSize
           boundaries. If any components do not, an error is thrown since this is
           indicative of components that actually came from a different reference image.
    :param inFile: Name of file to import
    :return: Tuple: DF that will be exported if successful extraction
    """
    col = None
    try:
      csvDf = pd.read_csv(inFile, keep_default_na=False)
      # Objects in the original frame are represented as strings, so try to convert these
      # as needed
      csvDf = csvDf[[field.name for field in TBL_FIELDS]]
      stringCols = csvDf.columns[csvDf.dtypes == object]
      valToParamMap = {param.name: param.value for param in TBL_FIELDS}
      for col in stringCols:
        paramVal = valToParamMap[col]
        # No need to perform this expensive computation if the values are already strings
        if not isinstance(paramVal, str):
          csvDf[col] = _strSerToParamSer(csvDf[col], valToParamMap[col])
      csvDf.columns = TBL_FIELDS
      csvDf = csvDf.set_index(REQD_TBL_FIELDS.INST_ID, drop=False)

      cls.checkVertBounds(csvDf[REQD_TBL_FIELDS.VERTICES], imShape)
    except Exception as ex:
      # Rethrow exception with insight about column number
      # Procedure copied from https://stackoverflow.com/a/6062677/9463643
      errMsg = f'Error importing column "{col}":\n'
      augmentException(ex, errMsg)
      raise
    # TODO: Apply this function to individual rows instead of the whole dataframe. This will allow malformed
    #  rows to gracefully fall off the dataframe with some sort of warning message
    return csvDf

  @classmethod
  def buildFromPkl(cls, inFile: FilePath, imShape: Tuple=None) -> df:
    """
    See docstring for :func:`self.buildFromCsv`
    """
    pklDf = None
    pklDf = pd.read_pickle(inFile)
    cls.checkVertBounds(pklDf[REQD_TBL_FIELDS.VERTICES], imShape)
    return pklDf

  @staticmethod
  def checkVertBounds(vertSer: pd.Series, imShape: tuple):
    """
    Checks whether any vertices in the imported dataframe extend past image dimensions. This is an indicator
    they came from the wrong import file.

    :param vertSer: Vertices from incoming component dataframe
    :param imShape: Shape of the main image these vertices are drawn on
    :return: Raises error if offending vertices are present, since this is an indication the component file
      was from a different image
    """
    if imShape is None:
      # Nothing we can do if no shape is given
      return
    # Image shape from row-col -> x-y
    imShape = np.array(imShape[1::-1])[None, :]
    # Remove components whose vertices go over any image edges
    vertMaxs = [verts.stack().max(0) for verts in vertSer if len(verts) > 0]
    vertMaxs = np.vstack(vertMaxs)
    offendingIds = np.nonzero(np.any(vertMaxs >= imShape, axis=1))[0]
    if len(offendingIds) > 0:
      raise FRAppIOError(f'Vertices on some components extend beyond image dimensions. '
                          f'Perhaps this export came from a different image?\n'
                          f'Offending IDs: {offendingIds}')