import pickle
import sys
from ast import literal_eval
from pathlib import Path
from stat import S_IREAD, S_IRGRP, S_IROTH
from typing import Union, Any, Optional, List, Tuple
from typing_extensions import Literal
from warnings import warn
import re

import cv2 as cv
import numpy as np
import pandas as pd
from pandas import DataFrame as df
from pyqtgraph.Qt import QtCore
from skimage import io
from skimage.measure import label

from s3a.structures import OneDArr, FRParamGroup, FilePath, FRS3AWarning, GrayImg
from s3a.structures.typeoverloads import TwoDArr, NChanImg
from s3a import FR_SINGLETON
from s3a.generalutils import coerceDfTypes, augmentException
from s3a.projectvars import FR_ENUMS, REQD_TBL_FIELDS
from s3a.projectvars.constants import FR_CONSTS
from s3a.structures import FRComplexVertices, FRParam, FRAppIOError

__all__ = ['FRComponentMgr', 'FRComponentIO', 'FRCompTableModel']

Signal = QtCore.Signal

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

  @FR_SINGLETON.actionStack.undoable('Merge Components')
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
      warn(f'Less than two components are selected, so "merge" is a no-op.', FRS3AWarning)
      return
    mergeComps: df = self.compDf.loc[mergeIds].copy()
    if keepId is None:
      keepId = mergeIds[0]

    keepInfo = mergeComps.loc[keepId].copy()
    allVerts = [v.stack() for v in mergeComps[REQD_TBL_FIELDS.VERTICES]]
    maskShape = np.max(np.vstack(allVerts), 0)[::-1]
    mask = np.zeros(maskShape, bool)
    for verts in mergeComps[REQD_TBL_FIELDS.VERTICES]: # type: FRComplexVertices
      mask |= verts.toMask(tuple(maskShape))
    newVerts = FRComplexVertices.fromBwMask(mask)
    keepInfo[REQD_TBL_FIELDS.VERTICES] = newVerts

    self.rmComps(mergeComps.index)
    self.addComps(keepInfo.to_frame().T, FR_ENUMS.COMP_ADD_AS_MERGE)
    yield
    self.addComps(mergeComps, FR_ENUMS.COMP_ADD_AS_MERGE)

  @FR_SINGLETON.actionStack.undoable('Split Components')
  def splitCompVertsById(self, splitIds: OneDArr):
    """
    Makes a separate component for each distinct boundary in all selected components.
    For instance, if two components are selected, and each has two separate circles as
    vertices, then 4 total components will exist after this operation.

    Each new component will have the table fields of its parent

    :param splitIds: Ids of components to split up
    """
    splitComps = self.compDf.loc[splitIds, :].copy()
    newComps_lst = []
    for _, comp in splitComps.iterrows():
      verts = comp[REQD_TBL_FIELDS.VERTICES]
      childComps = pd.concat([comp.to_frame().T]*len(verts))
      newVerts = [FRComplexVertices([v]) for v in verts]
      childComps.loc[:, REQD_TBL_FIELDS.VERTICES] = newVerts
      newComps_lst.append(childComps)
    newComps = pd.concat(newComps_lst)
    # Keep track of which comps were removed and added by this op
    outDict = self.rmComps(splitComps.index)
    outDict.update(self.addComps(newComps))
    yield outDict
    undoDict = self.rmComps(newComps.index)
    undoDict.update(self.addComps(splitComps, FR_ENUMS.COMP_ADD_AS_MERGE))
    return undoDict

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
  return strSeries.apply(funcToUse).values

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

  @property
  def handledIoTypes(self):
      return {'csv': 'CSV Files', 'pkl': 'Pickle Files', 'id.png': 'ID Grayscale Image',
              'class.png': 'Class Grayscale Image'}

  def handledIoTypes_fileFilter(self, typeFilter='', **extraOpts):
    """
    Helper for creating a file filter out of the handled IO types. The returned list of
    strings is suitable for inserting into a QFileDialog.

    :param typeFilter: type filter for handled io types. For instanece, if typ='png', then
      a file filter list with only 'id.png' and 'class.png' will appear.
    """
    if isinstance(typeFilter, str):
      typeFilter = [typeFilter]
    fileFilters = []
    for typ, info in dict(**self.handledIoTypes, **extraOpts).items():
      if any([t in typ for t in typeFilter]):
        fileFilters.append(f'{info} (*.{typ})')
    return ';;'.join(fileFilters)

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

  def exportByFileType(self, outFile: Union[str, Path], **exportArgs):
    self._ioByFileType(outFile, 'export', **exportArgs)


  def buildByFileType(self, inFile: Union[str, Path], imShape: Tuple[int]=None, **importArgs):
    return self._ioByFileType(inFile, 'buildFrom', imShape=imShape, **importArgs)

  def _ioByFileType(self, fpath: Union[str, Path],
                    buildOrExport=Literal['buildFrom', 'export'], **ioArgs):
    fpath = Path(fpath)
    fname = fpath.name
    cmpTypes = np.array(list(self.handledIoTypes.keys()))
    typIdx = [typ in fname for typ in cmpTypes]
    if not any(typIdx):
      return
    fnNameSuffix = cmpTypes[typIdx][0].title().replace('.', '')
    ioFn = getattr(self, buildOrExport+fnNameSuffix, None)
    return ioFn(fpath, **ioArgs)
  # -----
  # Export options
  # -----
  def exportCsv(self, outFile: Union[str, Path]=None, readOnly=True, **pdExportArgs):
    """
    :param outFile: Name of the output file location. If *None*, no file is created. However,
      the export object will still be created and returned.
    :param pdExportArgs: Dictionary of values passed to underlying pandas export function.
      These will overwrite the default options for :func:`exportToFile <FRComponentMgr.exportToFile>`
    :param readOnly: Whether this export should be read-only
    :return: Export version of the component data.
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
    col = None
    try:
      # TODO: Currently the additional options are causing errors. Find out why and fix
      #  them, since this may be useful if it can be modified
      # Format special columns appropriately
      # Since CSV export significantly modifies the df, make a copy before doing all these
      # operations
      outPath.parent.mkdir(exist_ok=True, parents=True)
      exportDf = self.compDf.copy(deep=True)
      for col in exportDf:
        if not isinstance(col.value, str):
          exportDf[col] = _paramSerToStrSer(exportDf[col], col.value)
      if outFile is not None:
        exportDf.to_csv(outFile, index=False)
        if readOnly:
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

  def exportPkl(self, outFile: Union[str, Path]=None) -> (Any, str):
    """
    See the function signature for :func:`exportCsv <FRComponentIO.exportCsv>`
    """
    # Since the write-out is a single operation there isn't an intermediate form to return
    pklDf = None
    if outFile is not None:
      pklDf = pickle.dumps(self.compDf)
      self.compDf.to_pickle(outFile)
    return pklDf

  def exportClassPng(self, outFile: FilePath = None, imShape: Tuple[int]=None, **kwargs):
    # Create label to output mapping
    classes = FR_SINGLETON.tableData.compClasses
    colors = self.compDf[REQD_TBL_FIELDS.COMP_CLASS].apply(classes.index)+1
    origIdxs = self.compDf.index
    self.compDf.index = colors
    ret = self.exportIdPng(outFile, imShape, **kwargs)
    self.compDf.index = origIdxs

    return ret

  def exportIdPng(self, outFile: FilePath=None,
                  imShape: Tuple[int]=None, **kwargs):
    """
    Creates a 2D grayscale image where each component is colored with its isntance ID + 1.
    *Note* Since Id 0 would end up not coloring the mask, all IDs must be offest by 1.
    :param imShape: The size of this output image
    :param outFile: Where to save the output. If *None*, no export is created.
    :return:
    """
    if imShape is None:
      vertMax = FRComplexVertices.stackedMax(self.compDf[REQD_TBL_FIELDS.VERTICES])
      imShape = tuple(vertMax[::-1] + 1)
    outMask = np.zeros(imShape[:2], 'int32')
    for idx, comp in self.compDf.iterrows():
      verts: FRComplexVertices = comp[REQD_TBL_FIELDS.VERTICES]
      idx: int
      outMask = verts.toMask(outMask, idx+1, False, False)

    if outFile is not None:
      io.imsave(outFile, outMask.astype('uint16'), check_contrast=False)
    return outMask

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
    field = FRParam('None', None)
    try:
      csvDf = pd.read_csv(inFile, keep_default_na=False, dtype=object)
      # Decouple index from instance ID until after transfer from csvDf is complete
      # This was causing very strange behavior without reset_index()...
      outDf = FR_SINGLETON.tableData.makeCompDf(len(csvDf)).reset_index(drop=True)
      # Objects in the original frame are represented as strings, so try to convert these
      # as needed
      for field in TBL_FIELDS:
        if field.name in csvDf:
          if isinstance(field.value, str):
            outDf[field] = csvDf[field.name]
          else:
            outDf[field] = _strSerToParamSer(csvDf[field.name], field.value)
      outDf = outDf.set_index(REQD_TBL_FIELDS.INST_ID, drop=False)

      cls.checkVertBounds(outDf[REQD_TBL_FIELDS.VERTICES], imShape)
    except Exception as ex:
      # Rethrow exception with insight about column number
      # Procedure copied from https://stackoverflow.com/a/6062677/9463643
      errMsg = f'Error importing column "{field.name}":\n'
      augmentException(ex, errMsg)
      raise
    # TODO: Apply this function to individual rows instead of the whole dataframe. This will allow malformed
    #  rows to gracefully fall off the dataframe with some sort of warning message
    return outDf

  @classmethod
  def buildFromPkl(cls, inFile: FilePath, imShape: Tuple=None) -> df:
    """
    See docstring for :func:`self.buildFromCsv`
    """
    pklDf = pd.read_pickle(inFile)
    cls.checkVertBounds(pklDf[REQD_TBL_FIELDS.VERTICES], imShape)
    return pklDf

  @classmethod
  def buildFromIdPng(cls, inFile: FilePath, imShape: Tuple=None) -> df:
    labelImg = io.imread(inFile, as_gray=True)
    outDf = cls._idImgToDf(labelImg)
    cls.checkVertBounds(outDf[REQD_TBL_FIELDS.VERTICES], imShape)
    return outDf

  @classmethod
  def buildFromClassPng(cls, inFile: FilePath, imShape: Tuple=None) -> df:
    outDf = cls.buildFromIdPng(inFile, imShape)
    # Convert what the ID import treated as an inst id into indices for class
    clsArray = np.array(FR_SINGLETON.tableData.compClasses)
    outDf[REQD_TBL_FIELDS.COMP_CLASS] = clsArray[outDf[REQD_TBL_FIELDS.INST_ID]]
    outDf.reset_index(inplace=True, drop=True)
    outDf[REQD_TBL_FIELDS.INST_ID] = outDf.index
    return outDf

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
    if imShape is None or len(vertSer) == 0:
      # Nothing we can do if no shape is given
      return
    # Image shape from row-col -> x-y
    imShape = np.array(imShape[1::-1])[None, :]
    # Remove components whose vertices go over any image edges
    vertMaxs = [verts.stack().max(0) for verts in vertSer if len(verts) > 0]
    vertMaxs = np.vstack(vertMaxs)
    offendingIds = np.nonzero(np.any(vertMaxs >= imShape, axis=1))[0]
    if len(offendingIds) > 0:
      warn(f'Vertices on some components extend beyond image dimensions. '
           f'Perhaps this export came from a different image?\n'
           f'Offending IDs: {offendingIds}', FRS3AWarning)

  @classmethod
  def _idImgToDf(cls, idImg: GrayImg):
    # Skip 0 since it's indicative of background
    regionIds = np.unique(idImg)[1:]
    allVerts = []
    for curId in regionIds:
      verts = FRComplexVertices.fromBwMask(idImg == curId)
      allVerts.append(verts)
    outDf = FR_SINGLETON.tableData.makeCompDf(regionIds.size)
    # Subtract 1 since instance ids are 0-indexed
    outDf[REQD_TBL_FIELDS.INST_ID] = regionIds-1
    outDf[REQD_TBL_FIELDS.VERTICES] = allVerts
    return outDf