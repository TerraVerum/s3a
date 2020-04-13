import pickle
import re
import sys
from ast import literal_eval
from typing import Union, Any, Optional, Sequence, List, Tuple, Dict
from pathlib import Path
from stat import S_IRGRP

import numpy as np
import pandas as pd
from datetime import datetime
from pandas import DataFrame as df

import cv2 as cv
from skimage import io
from pyqtgraph.Qt import QtCore

from cdef.projectvars import TEMPLATE_COMP_CLASSES
from cdef.structures.typeoverloads import TwoDArr, NChanImg
from .frgraphics.parameditors import FR_SINGLETON
from .generalutils import coerceDfTypes
from .projectvars import FR_ENUMS, TEMPLATE_COMP as TC, CompParams
from .projectvars.constants import FR_CONSTS
from .structures import FRComplexVertices, FRParam, FRCompIOError

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
  outDf[TC.ANN_TIMESTAMP] = datetime.now()
  outDf[TC.ANN_FILENAME] = FR_CONSTS.ANN_CUR_FILE_INDICATOR.value
  if dropRow:
    outDf = outDf.drop(index=TC.INST_ID.value)
  return outDf

class FRCompTableModel(QtCore.QAbstractTableModel):
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
class FRComponentMgr(FRCompTableModel):
  @FR_SINGLETON.generalProps.registerProp(FR_CONSTS.EXP_ONLY_VISIBLE)
  def exportOnlyVis(self): pass
  @FR_SINGLETON.generalProps.registerProp(FR_CONSTS.INCLUDE_FNAME_PATH)
  def includeFullSourceImgName(self): pass
  

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

@FR_SINGLETON.registerClass(FR_CONSTS.CLS_COMP_EXPORTER)
class FRComponentIO:
  @FR_SINGLETON.generalProps.registerProp(FR_CONSTS.EXP_ONLY_VISIBLE)
  def expOnlyVisible(self): pass
  @FR_SINGLETON.generalProps.registerProp(FR_CONSTS.INCLUDE_FNAME_PATH)
  def includeFullSourceImgName(self): pass

  def __init__(self, compDf: df, mainImgFpath: str,
                exportIds: Union[FR_ENUMS, Sequence] = FR_ENUMS.COMP_EXPORT_ALL):
    """
    Exporter responsible for saving Component information to a file or object.
    Once created, users can extract different representations of components by
    calling exporter.exportCsv, exportPkl, etc. for those objects / files respectively.

    :param compDf: The component dataframe that came from the component manager
    :param mainImgFpath: Name of the image being annotated. This helps associate
      all annotations to their source file.
    :param exportIds: If :var:`FR_ENUMS.EXPORT_ALL`, exports every component in the
      dataframe. Otherwise, just exports the requested IDs
    """
    if isinstance(exportIds, FR_ENUMS) and exportIds == FR_ENUMS.COMP_EXPORT_ALL:
      exportDf = compDf
    else:
      exportDf = compDf.loc[exportIds, :]
    exportDf: df = exportDf.copy(deep=True)
    if not self.includeFullSourceImgName:
      # Only use the file name, not the whole path
      mainImgFpath = Path(mainImgFpath).name
    # Assign correct export name for only new components
    overwriteIdxs = exportDf[TC.ANN_FILENAME] == FR_CONSTS.ANN_CUR_FILE_INDICATOR.value
    # TODO: Maybe the current filename will match the current file indicator. What happens then?
    exportDf.loc[overwriteIdxs, TC.ANN_FILENAME] = mainImgFpath

    self.exportDf = exportDf

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
    defaultExportParams.update(pdExportArgs)
    # Make sure no rows are truncated
    pd.set_option('display.max_rows', sys.maxsize)
    oldNpOpts = np.get_printoptions()
    np.set_printoptions(threshold=sys.maxsize)
    errMsg = None
    exportDf: Optional[df] = None
    try:
      # TODO: Currently the additional options are causing errors. Find out why and fix
      #  them, since this may be useful if it can be modified
      # Format special columns appropriately
      # Since CSV export significantly modifies the df, make a copy before doing all these
      # operations
      exportDf = self.exportDf.copy(deep=True)
      for col in exportDf:
        if hasattr(col.value, 'serialize'):
          exportDf[col] = exportDf[col].map(type(col.value).serialize)
      if outFile is not None:
        exportDf.to_csv(outFile, index=False)
        outPath.chmod(S_IRGRP)
    except Exception as ex:
      errMsg = str(ex)
    finally:
      pd.reset_option('display.max_rows')
      # False positive checker warning for some reason
      # noinspection PyTypeChecker
      np.set_printoptions(**oldNpOpts)
    return exportDf, errMsg

  def exportPkl(self, outFile=None) -> (Any, str):
    """
    See the function signature for :func:`exportCsv <FRComponentIO.exportCsv>`
    """
    errMsg = None
    # Since the write-out is a single operation there isn't an intermediate form to return
    retObj = None
    try:
      if outFile is not None:
        pklDf = pickle.dumps(self.exportDf)

        self.exportDf.to_pickle(outFile)
    except Exception as ex:
      errMsg = str(ex)
    return retObj, errMsg

  def exportLabeledImg(self,
                        mainImgShape: Tuple[int],
                        outFile: str = None,
                        types: List[CompParams] = None,
                        colorPerType: TwoDArr = None,
                      ) -> (NChanImg, str):
    errMsg = None
    # Set up input arguments
    if types is None:
      types = [param for param in TEMPLATE_COMP_CLASSES]
    if colorPerType is None:
      colorPerType = np.arange(1, len(types) + 1, dtype=int)[:,None]
    outShape = mainImgShape[:2]
    if colorPerType.shape[1] > 1:
      outShape += (colorPerType.shape[1],)
    out = np.zeros(outShape, 'uint8')

    # Create label to output mapping
    for curType, color in zip(types, colorPerType):
      outlines = self.exportDf.loc[self.exportDf[TC.COMP_CLASS] == curType, TC.VERTICES].values
      cvFillArg = [arr[0] for arr in outlines]
      cvClrArg = tuple([int(val) for val in color])
      cv.fillPoly(out, cvFillArg, cvClrArg)

    if outFile is not None:
      try:
        io.imsave(outFile, out, check_contrast=False)
      except Exception as ex:
        errMsg = str(ex)

    return out, errMsg

  # -----
  # Import options
  # -----

  @classmethod
  def buildFromCsv(cls, inFile: str, imShape: Optional[Tuple]) -> (df, str):
    """
    Deserializes data from a csv file to create a Component :class:`DataFrame`.
    The input .csv should be the same format as one exported by
    :func:`csvImport <FRComponentMgr.csvImport>`.

    :param imShape: If included, this ensures all imported components lie within imSize
           boundaries. If any components do not, an error is thrown since this is
           indicative of components that actually came from a different reference image.
    :param inFile: Name of file to import
    :return: Tuple: DF if successful extraction + Exception message that occurs if the operation did not succeed.
      Otherwise, this value will be `None`.
    """
    csvDf = None
    errMsg = None
    try:
      csvDf = pd.read_csv(inFile, keep_default_na=False)
      # Objects in the original frame are represented as strings, so try to convert these
      # as needed
      # Organize the fields of the incoming CSV in the order expected by the template component
      csvDf = csvDf[[str(field) for field in TC]]
      stringCols = csvDf.columns[csvDf.dtypes == object]
      valToParamMap = {param.name: param.value for param in TC}
      for col in stringCols:
        paramVal = valToParamMap[col]
        # No need to perform this expensive computation if the values are already strings
        if not isinstance(paramVal, str):
          csvDf[col] = _strSerToParamSer(csvDf[col], valToParamMap[col])
      csvDf.columns = TC
      csvDf = csvDf.set_index(TC.INST_ID, drop=False)

      cls.checkVertBounds(csvDf[TC.VERTICES], imShape)
    except Exception as ex:
      errMsg = str(ex)
    # TODO: Apply this function to individual rows instead of the whole dataframe. This will allow malformed
    #  rows to gracefully fall off the dataframe with some sort of warning message
    return csvDf, errMsg

  @classmethod
  def buildFromPkl(cls, inFile: str, imShape: Optional[Tuple]) -> (df, str):
    """
    See docstring for :func:`self.buildFromCsv`
    """
    errMsg = None
    try:
      pklDf = pd.read_pickle(inFile)
      cls.checkVertBounds(pklDf[TC.VERTICES], imShape)
    except Exception as ex:
      errMsg = str(ex)
    return pklDf, errMsg

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
      raise FRCompIOError(f'Vertices on some components extend beyond image dimensions. '
                          f'Perhaps this export came from a different image?\n'
                          f'Offending IDs: {offendingIds}')