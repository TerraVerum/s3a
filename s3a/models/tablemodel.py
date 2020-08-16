import pickle
import sys
from ast import literal_eval
from pathlib import Path
from stat import S_IREAD, S_IRGRP, S_IROTH
from typing import Union, Any, Optional, Tuple
from warnings import warn

import numpy as np
import pandas as pd
from pandas import DataFrame as df
from pyqtgraph.Qt import QtCore, QtWidgets
from skimage import io, measure
from typing_extensions import Literal

from s3a import FR_SINGLETON
from s3a.generalutils import coerceDfTypes, augmentException, getCroppedImg
from s3a.projectvars import FR_ENUMS, REQD_TBL_FIELDS as RTF
from s3a.projectvars.constants import FR_CONSTS
from s3a.structures import FRComplexVertices, FRParam
from s3a.structures import OneDArr, FRParamGroup, FilePath, FRS3AWarning, GrayImg

__all__ = ['FRComponentMgr', 'FRComponentIO', 'FRCompTableModel']

Signal = QtCore.Signal

TBL_FIELDS = FR_SINGLETON.tableData.allFields

FilePathOrDf = Union[FilePath, pd.DataFrame]
class FRCompTableModel(QtCore.QAbstractTableModel):
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
    if self.compDf.iloc[row, [col, col-1]].values[0] != self.compDf.iat[row, col]:
      warn('Warning! An error occurred setting this value. Please try again using a'
           ' <em>multi-cell</em> edit. E.g. do not just set this value, set it along with'
           ' at least one other selected cell.', FRS3AWarning)
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

    self.compDf = FR_SINGLETON.tableData.makeCompDf(0)

    noEditParams = set(RTF) - {RTF.COMP_CLASS}
    self.noEditColIdxs = [self.colTitles.index(col.name) for col in noEditParams]
    self.editColIdxs = np.setdiff1d(np.arange(len(self.colTitles)), self.noEditColIdxs)
    self.sigFieldsChanged.emit()

@FR_SINGLETON.registerGroup(FR_CONSTS.CLS_COMP_MGR)
class FRComponentMgr(FRCompTableModel):
  _nextCompId = 0

  def __init__(self):
    super().__init__()

  def resetFields(self):
    super().resetFields()
    self._nextCompId = 0

  @FR_SINGLETON.actionStack.undoable('Add Components')
  def addComps(self, newCompsDf: df, addtype: FR_ENUMS = FR_ENUMS.COMP_ADD_AS_NEW, emitChange=True):
    toEmit = self.defaultEmitDict.copy()
    existingIds = self.compDf.index

    if len(newCompsDf) == 0:
      # Nothing to undo
      return toEmit

    # Delete entries with no vertices, since they make work within the app difficult.
    # TODO: Is this the appropriate response?
    verts = newCompsDf[RTF.VERTICES]
    dropIds = newCompsDf.index[verts.map(FRComplexVertices.isEmpty)]
    newCompsDf.drop(index=dropIds, inplace=True)


    if addtype == FR_ENUMS.COMP_ADD_AS_NEW:
      # Treat all comps as new -> set their IDs to guaranteed new values
      newIds = np.arange(self._nextCompId, self._nextCompId + len(newCompsDf), dtype=int)
      newCompsDf.loc[:,RTF.INST_ID] = newIds
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
    allVerts = [v.stack() for v in mergeComps[RTF.VERTICES]]
    maskShape = np.max(np.vstack(allVerts), 0)[::-1]
    mask = np.zeros(maskShape, bool)
    for verts in mergeComps[RTF.VERTICES]: # type: FRComplexVertices
      mask |= verts.toMask(tuple(maskShape))
    newVerts = FRComplexVertices.fromBwMask(mask)
    keepInfo[RTF.VERTICES] = newVerts

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
      verts = comp[RTF.VERTICES]
      childComps = pd.concat([comp.to_frame().T]*len(verts))
      newVerts = [FRComplexVertices([v]) for v in verts]
      childComps.loc[:, RTF.VERTICES] = newVerts
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
    self.fullSrcImgFname: Optional[Path] = None

  @property
  def handledIoTypes(self):
    """Returns a dict of <type, description> for the file types this I/O obejct can handle"""
    return {'csv': 'CSV Files', 'pkl': 'Pickle Files',
            'id.png': 'ID Grayscale Image', 'class.png': 'Class Grayscale Image'}

  @property
  def roundTripIoTypes(self):
    """
    Not all IO types can export->import and remain the exact same dataframe afterwards.
    For instance, exporting a labeled image will discard all additional fields.
    This property holds export types which can give back the original dataframe after
    a round trip export->import.
    """
    ioTypes = self.handledIoTypes
    return {k: ioTypes[k] for k in ['csv', 'pkl']}

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
    self.fullSrcImgFname = srcImgFname
    if self.exportOnlyVis and displayIds is not None:
      exportIds = displayIds
    else:
      exportIds = compDf.index
    exportDf: df = compDf.loc[exportIds,:].copy()
    if not self.includeFullSourceImgName and srcImgFname is not None:
      # Only use the file name, not the whole path
      srcImgFname = srcImgFname.name
    elif srcImgFname is not None:
      srcImgFname = str(srcImgFname)
    # Assign correct export name for only new components
    overwriteIdxs = exportDf[RTF.SRC_IMG_FILENAME] == FR_CONSTS.ANN_CUR_FILE_INDICATOR.value
    # TODO: Maybe the current filename will match the current file indicator. What happens then?
    exportDf.loc[overwriteIdxs, RTF.SRC_IMG_FILENAME] = srcImgFname
    self.compDf = exportDf

  def exportByFileType(self, outFile: Union[str, Path], verifyIntegrity=True, **exportArgs):
    outFile = Path(outFile)
    self._ioByFileType(outFile, 'export', **exportArgs)
    if verifyIntegrity and outFile.suffix[1:] in self.roundTripIoTypes:
      matchingCols = np.setdiff1d(self.compDf.columns, [RTF.INST_ID,
                                                                RTF.SRC_IMG_FILENAME])
      loadedDf = self.buildByFileType(outFile)
      dfCmp = loadedDf[matchingCols].values == self.compDf[matchingCols].values
      if not np.all(dfCmp):
        problemCells = np.nonzero(~dfCmp)
        problemIdxs = self.compDf.index[problemCells[0]]
        problemCols = matchingCols[problemCells[1]]
        problemMsg = [f'{idx}: {col}' for idx, col in zip(problemIdxs, problemCols)]
        problemMsg = '\n'.join(problemMsg)
        # Try to fix the problem with an iloc write
        warn('<b>Warning!</b> Saved components do not match current component'
             ' state. This can occur when pandas incorrectly caches some'
             ' table values. To rectify this, a multi-cell overwrite was performed'
             ' for the following cells (shown as <id>: <column>):\n'
             + f'{problemMsg}\n'
               f'Please try exporting again to confirm the cleanup was successful.', FRS3AWarning)


  def buildByFileType(self, inFile: Union[str, Path], imShape: Tuple[int]=None,
                      strColumns=False, **importArgs):
    outDf = self._ioByFileType(inFile, 'buildFrom', imShape=imShape, **importArgs)
    if strColumns:
      outDf.columns = list(map(str, outDf.columns))
    return outDf

  def _ioByFileType(self, fpath: Union[str, Path],
                    buildOrExport=Literal['buildFrom', 'export'], **ioArgs):
    fpath = Path(fpath)
    fname = fpath.name
    cmpTypes = np.array(list(self.handledIoTypes.keys()))
    typIdx = [typ in fname for typ in cmpTypes]
    if not any(typIdx):
      return
    fnNameSuffix = cmpTypes[typIdx][-1].title().replace('.', '')
    ioFn = getattr(self, buildOrExport+fnNameSuffix, None)
    return ioFn(fpath, **ioArgs)

  @staticmethod
  def _strToNpArray(array_string: str, **opts):
    # Adapted from https://stackoverflow.com/a/42756309/9463643
    array_string = ','.join(array_string.replace('[ ', '[').split())
    return np.array(literal_eval(array_string), **opts)

  def _pandasCsvExport(self, exportDf: pd.DataFrame, outFile: Union[str, Path]=None,
                       readOnly=True, **pdExportArgs):
    if outFile is None:
      return

    defaultExportParams = {
      'na_rep': 'NaN',
      'float_format': '{:0.10n}',
      'index': False,
    }
    outPath = Path(outFile)
    outPath.parent.mkdir(exist_ok=True, parents=True)

    defaultExportParams.update(pdExportArgs)
    with np.printoptions(threshold=sys.maxsize):
      exportDf.to_csv(outFile, index=False)
    if readOnly:
      outPath.chmod(S_IREAD|S_IRGRP|S_IROTH)

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
    # Make sure no rows are truncated
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
      self._pandasCsvExport(exportDf, outFile, readOnly, **pdExportArgs)
    except Exception as ex:
      errMsg = f'Error on parsing column "{col.name}"\n'
      augmentException(ex, errMsg)
      raise
    return exportDf

  def exportCompimgsDf(self, outFile: Union[str, Path]=None, imgDir: FilePath=None, margin=0):
    """
    Creates a dataframe consisting of extracted images around each component
    :param outFile: Where to save the result, if it should be saved. Caution -- this
      is currently a time-consuming process!
    :param imgDir: Where images corresponding to this dataframe are kept. Source image
      filenames are interpreted relative to this directory if they are not absolute.
    :param margin: How much padding to give around each component
    :return: Dataframe with the following keys:
      - img: The (MxNxC) image corresponding to the component vertices, where MxN are
        the padded row sizes and C is the number of image channels
      - semanticMask: Binary mask representing the component vertices
      - bboxMask: Square box representing (min)->(max) component vertices. This is useful
        for excluding the margin when a semantic mask is not desired and the margin was > 0.
      - instId: The component's Instance ID
      - offset: Image (x,y) coordinate of the min component vertex.
    """
    _imgCache = {}
    compDf = self.compDf
    if imgDir is None and self.fullSrcImgFname is not None:
      imgDir = self.fullSrcImgFname.parent
    elif imgDir is None:
      imgDir = Path('.')
    else:
      imgDir = Path(imgDir)
    uniqueImgs = np.unique(compDf[RTF.SRC_IMG_FILENAME])
    for imgName in uniqueImgs:
      imgName = Path(imgName)
      if not imgName.is_absolute():
        imgName = imgDir/imgName
      if imgName not in _imgCache:
        _imgCache[imgName] = io.imread(imgName)
    dfGroupingsByImg = []
    for imgName in uniqueImgs:
      dfGroupingsByImg.append(compDf[compDf[RTF.SRC_IMG_FILENAME] == imgName])
    outDf = dict(img=[], semanticMask=[], bboxMask=[], compClass=[], instId=[], offset=[])
    for miniDf, imgName in zip(dfGroupingsByImg, uniqueImgs):
      imgName = imgDir/imgName
      img = _imgCache[imgName]
      for idx, row in miniDf.iterrows():
        allVerts = row[RTF.VERTICES].stack()
        compImg, bounds = getCroppedImg(img, allVerts, margin, coordsAsSlices=False)
        outDf['img'].append(compImg)
        outDf['compClass'].append(str(row[RTF.COMP_CLASS]))
        maskVerts: FRComplexVertices = row[RTF.VERTICES].copy()
        for verts in maskVerts:
          verts -= bounds[0,:]
        allVerts = maskVerts.stack()
        mask = maskVerts.toMask(compImg.shape[:2])
        outDf['semanticMask'].append(mask)
        bboxMask = np.zeros_like(mask)
        bboxBounds = np.r_[allVerts.min(0, keepdims=True), allVerts.max(0, keepdims=True)]
        bboxMask[bboxBounds[0,1]:bboxBounds[1,1], bboxBounds[0,0]:bboxBounds[1,0]] = True
        outDf['bboxMask'].append(bboxMask)
        outDf['instId'].append(row.name)
        outDf['offset'].append(bounds[0,:])
    outDf = pd.DataFrame(outDf)
    if outFile is not None:
      self.compDf.to_pickle(outFile)
    return outDf

  def exportPkl(self, outFile: Union[str, Path]=None, **exportArgs) -> (Any, str):
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
    colors = self.compDf[RTF.COMP_CLASS].apply(classes.index)
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
      vertMax = FRComplexVertices.stackedMax(self.compDf[RTF.VERTICES])
      imShape = tuple(vertMax[::-1] + 1)
    outMask = np.zeros(imShape[:2], 'int32')
    for idx, comp in self.compDf.iterrows():
      verts: FRComplexVertices = comp[RTF.VERTICES]
      idx: int
      outMask = verts.toMask(outMask, idx+1, False, False)

    if outFile is not None:
      io.imsave(outFile, outMask.astype('uint16'), check_contrast=False)
    return outMask

  # -----
  # Import options
  # -----

  @classmethod
  def buildFromCsv(cls, inFileOrDf: FilePathOrDf, imShape: Tuple=None) -> df:
    """
    Deserializes data from a csv file to create a Component :class:`DataFrame`.
    The input .csv should be the same format as one exported by
    :func:`csvImport <FRComponentMgr.csvImport>`.

    :param imShape: If included, this ensures all imported components lie within imSize
           boundaries. If any components do not, an error is thrown since this is
           indicative of components that actually came from a different reference image.
    :param inFileOrDf: Name of file to import, or dataframe if it was already read from this
      file type. Useful if several csv's were concatenated into one dataframe and *that* is
      being imported.
    :return: Tuple: DF that will be exported if successful extraction
    """
    field = FRParam('None', None)
    try:
      if isinstance(inFileOrDf, df):
        csvDf = inFileOrDf
      else:
        csvDf = pd.read_csv(inFileOrDf, keep_default_na=False, dtype=object)
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
      outDf = outDf.set_index(RTF.INST_ID, drop=False)

      cls.checkVertBounds(outDf[RTF.VERTICES], imShape)
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
  def buildFromCompimgsDf(cls, inFile: FilePath, imShape: Tuple=None):
    inDf = pd.read_pickle(inFile)
    outDf = FR_SINGLETON.tableData.makeCompDf(len(inDf))
    outDf[RTF.INST_ID] = inDf['instId']
    allVerts = []

    for idx, row in inDf.iterrows():
      mask = cls._strToNpArray(row.semanticMask, dtype=bool)
      verts = FRComplexVertices.fromBwMask(mask)
      offset = cls._strToNpArray(row.offset)
      for v in verts: v += offset
      allVerts.append(verts)
    outDf[RTF.VERTICES] = allVerts
    outDf[RTF.COMP_CLASS] = inDf.compClass
    cls.checkVertBounds(outDf[RTF.VERTICES], imShape)
    return outDf

  @classmethod
  def buildFromPkl(cls, inFile: FilePath, imShape: Tuple=None) -> df:
    """
    See docstring for :func:`self.buildFromCsv`
    """
    pklDf = pd.read_pickle(inFile)
    cls.checkVertBounds(pklDf[RTF.VERTICES], imShape)
    return pklDf

  @classmethod
  def buildFromIdPng(cls, inFileOrImg: Union[FilePath, GrayImg], imShape: Tuple=None) -> df:
    if isinstance(inFileOrImg, GrayImg):
      labelImg = inFileOrImg
    else:
      labelImg = io.imread(inFileOrImg, as_gray=True)
    outDf = cls._idImgToDf(labelImg)
    cls.checkVertBounds(outDf[RTF.VERTICES], imShape)
    return outDf

  @classmethod
  def buildFromClassPng(cls, inFileOrImg: Union[FilePath, GrayImg], imShape: Tuple=None) -> df:
    if isinstance(inFileOrImg, GrayImg):
      clsImg = inFileOrImg
    else:
      clsImg = io.imread(inFileOrImg)

    clsArray = np.array(FR_SINGLETON.tableData.compClasses)
    idImg = measure.label(clsImg)
    outDf = cls.buildFromIdPng(idImg, imShape)
    outClasses = []
    for curId in outDf[RTF.INST_ID]+1:
      # All ID pixels should be the same class, so any representative will do
      curCls = clsImg[idImg == curId][0]
      outClasses.append(clsArray[curCls-1])
    outDf[RTF.COMP_CLASS] = outClasses
    outDf.reset_index(inplace=True, drop=True)
    outDf[RTF.INST_ID] = outDf.index
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
    regionIds = np.unique(idImg)
    regionIds = regionIds[regionIds != 0]
    allVerts = []
    for curId in regionIds:
      verts = FRComplexVertices.fromBwMask(idImg == curId)
      allVerts.append(verts)
    outDf = FR_SINGLETON.tableData.makeCompDf(regionIds.size)
    # Subtract 1 since instance ids are 0-indexed
    outDf[RTF.INST_ID] = regionIds-1
    outDf[RTF.VERTICES] = allVerts
    return outDf