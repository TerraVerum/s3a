import contextlib
import pickle
import sys
from ast import literal_eval
from pathlib import Path
from stat import S_IREAD, S_IRGRP, S_IROTH
from typing import Any, Optional, Union, Tuple, Callable, Sequence
from warnings import warn

import numpy as np
import pandas as pd
from pandas import DataFrame as df
from skimage import io, measure
from skimage.exposure import rescale_intensity
from typing_extensions import Literal

from s3a.constants import REQD_TBL_FIELDS as RTF
from s3a.generalutils import augmentException, getCroppedImg, resize_pad
from s3a.parameditors.table import TableData
from s3a.structures import FRParamGroup, S3AWarning, S3AIOError, FilePath, GrayImg, \
  ComplexXYVertices, FRParam

FilePathOrDf = Union[FilePath, pd.DataFrame]

def _strSerToParamSer(strSeries: pd.Series, paramVal: Any) -> pd.Series:
  paramType = type(paramVal)
  # TODO: Move this to a more obvious place?
  funcMap = {
    # Format string to look like a list, use ast to convert that string INTO a list, make a numpy array from the list
    np.ndarray        : lambda strVal: np.array(literal_eval(strVal)),
    ComplexXYVertices : ComplexXYVertices.deserialize,
    bool              : lambda strVal: strVal.lower() == 'true',
    FRParam           : lambda strVal: FRParamGroup.fieldFromParam(paramVal.group, strVal)
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
    ComplexXYVertices: ComplexXYVertices.serialize,
  }
  defaultFunc = lambda param: str(param)

  funcToUse = funcMap.get(paramType, defaultFunc)
  return paramSer.apply(funcToUse)

class ComponentIO:
  """
  Exporter responsible for saving Component information to a file or object.
  Once created, users can extract different representations of components by
  calling exporter.exportCsv, exportPkl, etc. for those objects / files respectively.
  """
  handledIoTypes = {'csv': 'CSV Files', 'pkl': 'Pickle Files',
                    'id.png': 'ID Grayscale Image', 'class.png': 'Class Grayscale Image',
                    'compimgs.folder': 'Component Images Folder'}
  """Dict of <type, description> for the file types this I/O obejct can handle"""

  # Dictionary comprehension doesn't work in a class scope
  roundTripIoTypes = {}
  """
  Not all IO types can export->import and remain the exact same dataframe afterwards.
  For instance, exporting a labeled image will discard all additional fields.
  This property holds export types which can give back the original dataframe after
  a round trip export->import.
  """
  for k in ['csv', 'pkl']:
    roundTripIoTypes[k] = handledIoTypes[k]

  tableData = TableData()
  """Table to use for import/export cross checks. This is how class and table field information is derived."""

  @classmethod
  def handledIoTypes_fileFilter(cls, typeFilter='', **extraOpts):
    """
    Helper for creating a file filter out of the handled IO types. The returned list of
    strings is suitable for inserting into a QFileDialog.

    :param typeFilter: type filter for handled io types. For instanece, if typ='png', then
      a file filter list with only 'id.png' and 'class.png' will appear.
    """
    if isinstance(typeFilter, str):
      typeFilter = [typeFilter]
    fileFilters = []
    for typ, info in dict(**cls.handledIoTypes, **extraOpts).items():
      if any([t in typ for t in typeFilter]):
        fileFilters.append(f'{info} (*.{typ})')
    return ';;'.join(fileFilters)

  @classmethod
  def exportByFileType(cls, compDf: df, outFile: Union[str, Path], verifyIntegrity=True, **exportArgs):
    outFile = Path(outFile)
    outFn = cls._ioFnFromFileType(outFile, 'export')
    ret = outFn(compDf, outFile, **exportArgs)
    if verifyIntegrity and outFile.suffix[1:] in cls.roundTripIoTypes:
      matchingCols = np.setdiff1d(compDf.columns, [RTF.INST_ID,
                                                   RTF.SRC_IMG_FILENAME])
      loadedDf = cls.buildByFileType(outFile)
      dfCmp = loadedDf[matchingCols].values == compDf[matchingCols].values
      if not np.all(dfCmp):
        problemCells = np.nonzero(~dfCmp)
        problemIdxs = compDf.index[problemCells[0]]
        problemCols = matchingCols[problemCells[1]]
        problemMsg = [f'{idx}: {col}' for idx, col in zip(problemIdxs, problemCols)]
        problemMsg = '\n'.join(problemMsg)
        # Try to fix the problem with an iloc write
        warn('<b>Warning!</b> Saved components do not match current component'
             ' state. This can occur when pandas incorrectly caches some'
             ' table values. To rectify this, a multi-cell overwrite was performed'
             ' for the following cells (shown as <id>: <column>):\n'
             + f'{problemMsg}\n'
               f'Please try exporting again to confirm the cleanup was successful.', S3AWarning)
    return ret

  @classmethod
  def buildByFileType(cls, inFile: Union[str, Path], imShape: Tuple[int]=None,
                      strColumns=False, **importArgs):
    buildFn = cls._ioFnFromFileType(inFile, 'buildFrom')
    outDf = buildFn(inFile, imShape=imShape, **importArgs)
    if strColumns:
      outDf.columns = list(map(str, outDf.columns))
    return outDf

  @classmethod
  def _ioFnFromFileType(cls, fpath: Union[str, Path],
                        buildOrExport=Literal['buildFrom', 'export']) -> Optional[Callable]:
    fpath = Path(fpath)
    fname = fpath.name
    cmpTypes = np.array(list(cls.handledIoTypes.keys()))
    typIdx = [typ in fname for typ in cmpTypes]
    if not any(typIdx):
      raise S3AIOError(f'Not sure how to handle file {fpath.stem}')
    fnNameSuffix = cmpTypes[typIdx][-1].title().replace('.', '')
    return getattr(cls, buildOrExport+fnNameSuffix, None)

  @staticmethod
  def _strToNpArray(array_string: str, **opts):
    # Adapted from https://stackoverflow.com/a/42756309/9463643
    array_string = ','.join(array_string.replace('[ ', '[').split())
    return np.array(literal_eval(array_string), **opts)

  @classmethod
  def _pandasCsvExport(cls, exportDf: pd.DataFrame, outFile: Union[str, Path]=None,
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
  @classmethod
  def exportCsv(cls, compDf: df, outFile: Union[str, Path]=None, readOnly=True, **pdExportArgs):
    """
    :param compDf: Dataframe to export
    :param outFile: Name of the output file location. If *None*, no file is created. However,
      the export object will still be created and returned.
    :param pdExportArgs: Dictionary of values passed to underlying pandas export function.
      These will overwrite the default options for :func:`exportToFile <ComponentMgr.exportToFile>`
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
      exportDf = compDf.copy(deep=True)
      for col in exportDf:
        if not isinstance(col.value, str):
          exportDf[col] = _paramSerToStrSer(exportDf[col], col.value)
      cls._pandasCsvExport(exportDf, outFile, readOnly, **pdExportArgs)
    except Exception as ex:
      errMsg = f'Error on parsing column "{col.name}"\n'
      augmentException(ex, errMsg)
      raise
    return exportDf

  @classmethod
  def exportCompimgsDf(cls, compDf: df, outFile: Union[str, Path]=None,
                       imgDir: FilePath=None, margin=0, marginAsPct=False,
                       includeCols=('instId', 'img', 'semanticMask', 'bboxMask', 'lbl', 'offset'),
                       lblField='Instance ID', **kwargs):
    """
    Creates a dataframe consisting of extracted images around each component
    :param compDf: Dataframe to export
    :param outFile: Where to save the result, if it should be saved. Caution -- this
      is currently a time-consuming process!
    :param imgDir: Where images corresponding to this dataframe are kept. Source image
      filenames are interpreted relative to this directory if they are not absolute.
    :param margin: How much padding to give around each component
    :param marginAsPct: Whether the margin should be a percentage of the component size or
      a raw pixel value.
    :param includeCols: Which columns to include in the export list
    :param lblField: See ComponentIO.exportLblPng. This label is provided in the output dataframe
      as well, if specified.
    :return: Dataframe with the following keys:
      - instId: The component's Instance ID
      - img: The (MxNxC) image corresponding to the component vertices, where MxN are
        the padded row sizes and C is the number of image channels
      - semanticMask: Binary mask representing the component vertices
      - bboxMask: Square box representing (min)->(max) component vertices. This is useful
        for excluding the margin when a semantic mask is not desired and the margin was > 0.
      - label: Field value of the component for the field specified by `lblField`
      - offset: Image (x,y) coordinate of the min component vertex.
    :param kwargs: Passed to ComponentIO.exportLblPng
    """
    _imgCache = {}
    if imgDir is None:
      imgDir = Path('.')
    else:
      imgDir = Path(imgDir)
    uniqueImgs = np.unique(compDf[RTF.SRC_IMG_FILENAME])
    dfGroupingsByImg = []
    for imgName in uniqueImgs:
      imgName = Path(imgName)
      if not imgName.is_absolute():
        imgName = imgDir/imgName
      if imgName not in _imgCache:
        _imgCache[imgName] = io.imread(imgName)
      dfGroupingsByImg.append(compDf[compDf[RTF.SRC_IMG_FILENAME] == imgName])

    useKeys = set(includeCols)
    outDf = {k: [] for k in useKeys}
    lblField = cls.tableData.fieldFromName(lblField)
    bboxFills = lblField.toNumeric(compDf[lblField])

    for miniDf, imgName in zip(dfGroupingsByImg, uniqueImgs):
      imgName = imgDir/imgName
      img = _imgCache[imgName]
      lblImg = cls.exportLblPng(miniDf, imShape=img.shape[:2], lblField=lblField, **kwargs)

      for ii, (idx, row) in enumerate(miniDf.iterrows()):
        allVerts = row[RTF.VERTICES].stack()
        if marginAsPct:
          compImgSz = allVerts.max(0) - allVerts.min(0)
          marginToUse = (compImgSz*(margin/100)).astype(int)
        else:
          marginToUse = margin
        compImg, bounds = getCroppedImg(img, allVerts, marginToUse, coordsAsSlices=False)

        if 'img' in useKeys:
          outDf['img'].append(compImg)
        lbl = row[lblField]

        if 'label' in useKeys:
          outDf['label'].append(lbl)

        indexer = tuple(slice(*b) for b in bounds[:, ::-1].T)
        if lblImg.ndim > 2:
          indexer += (...,)
        mask = lblImg[indexer]
        if 'semanticMask' in useKeys:
          # x-y to row-col, transpose to get min-max in axis 0 and row-col in axis 1
          outDf['semanticMask'].append(mask)

        if 'bboxMask' in useKeys:
          bboxMask = np.zeros(compImg.shape[:2], dtype=lblImg.dtype)
          bboxBounds = np.r_[allVerts.min(0, keepdims=True)+marginToUse, allVerts.max(0, keepdims=True)-marginToUse]
          bboxMask[bboxBounds[0,1]:bboxBounds[1,1], bboxBounds[0,0]:bboxBounds[1,0]] = bboxFills[ii]
          outDf['bboxMask'].append(bboxMask)

        if 'instId' in useKeys:
          outDf['instId'].append(idx)

        if 'offset' in useKeys:
          outDf['offset'].append(bounds[0,:])
    outDf = pd.DataFrame(outDf)
    if outFile is not None:
      outDf.to_pickle(outFile)
    return outDf

  @classmethod
  def exportPkl(cls, compDf: df, outFile: Union[str, Path]=None, **exportArgs) -> (Any, str):
    """
    See the function signature for :func:`exportCsv <ComponentIO.exportCsv>`
    """
    # Since the write-out is a single operation there isn't an intermediate form to return
    pklDf = None
    if outFile is not None:
      pklDf = pickle.dumps(compDf)
      compDf.to_pickle(outFile)
    return pklDf

  @classmethod
  def exportLblPng(cls, compDf: df, outFile: FilePath=None, imShape: Tuple[int]=None,
                   lblField: Union[FRParam, str]='Instance ID', bgColor=0, allowOffset=None,
                   rescaleOutput=False, returnLblMapping=False,
                   **kwargs):
    """
    :param compDf: Dataframe to export
    :param outFile: Filename to save, leave *None* to avoid saving to a file
    :param imShape: MxN shape of image containing these annotations
    :param lblField: Data field to use as an index label. E.g. "Class" will use the 'class'
      column, but any other column can be specified. The output ground truth masks
      will be colored according to this field.  See :meth:`FRParam.toNumeric` for details.
      If `lblField` is *None*, the foreground mask will be boolean instead of integer-colored.
    :param bgColor: Color of the mask background. Must be an integer.
    :param allowOffset: Some label fields may have 0-based starting values. In cases
      where `bgColor` is also 0, an offset is required to prevent those field values
      from being considered background. `allowOffset` determines whether this change
      can be made to the data. See `FRParam.toNumeric` for detailed description.
    :param rescaleOutput: For images designed for human use, it is helpful to have
      outputs rescaled to the entire intensity range. Otherwise, they usually end
      up looking pure black and it is difficult to see components in the image.
      When `rescaleOutput` is *True*, all numbers are scaled to the 'uint16' range.
    :param returnLblMapping: Whether to return a pd.Series matching original index values
      to their numeric counterparts
    :param kwargs:
    :return:
    """

    lblField = cls.tableData.fieldFromName(lblField)

    if lblField not in cls.tableData.allFields:
      raise S3AIOError(f'Specified label field {lblField} does not exist in the table'
                       f' fields. Must be one of:\n'
                       f'{[f.name for f in cls.tableData.allFields]}')
    if bgColor < 0:
      raise S3AIOError(f'Background color must be >= 0, was {bgColor}')

    labels = compDf[lblField]
    labels_numeric = lblField.toNumeric(labels, allowOffset)
    asBool = np.issubdtype(labels_numeric.dtype, np.bool)

    if rescaleOutput:
      labels_numeric = rescale_intensity(labels_numeric, out_range='uint16')

    if imShape is None:
      vertMax = ComplexXYVertices.stackedMax(compDf[RTF.VERTICES])
      imShape = tuple(vertMax[::-1] + 1)
    maskType = 'uint16' if np.min(bgColor) >= 0 else 'int32'
    outMask = np.full(imShape[:2], bgColor, maskType)
    for fillClr, (_, comp) in zip(labels_numeric, compDf.iterrows()):
      verts: ComplexXYVertices = comp[RTF.VERTICES]
      outMask = verts.toMask(outMask, int(fillClr), asBool, False)

    if outFile is not None:
      io.imsave(outFile, outMask.astype('uint16'), check_contrast=False)
    if returnLblMapping:
      mapping = pd.Series(data=labels_numeric, index=labels, name=lblField)
      return outMask, mapping
    return outMask

  @classmethod
  def exportClassPng(cls, compDf: df, outFile: FilePath = None, imShape: Tuple[int]=None, **kwargs):
    # Create label to output mapping
    return cls.exportLblPng(compDf, outFile, imShape, 'Class', **kwargs)

  @classmethod
  def exportIdPng(cls, compDf: df, outFile: FilePath=None,
                  imShape: Tuple[int]=None, **kwargs):
    """
    Creates a 2D grayscale image where each component is colored with its isntance ID + 1.
    *Note* Since Id 0 would end up not coloring the mask, all IDs must be offest by 1.
    :param compDf: Dataframe to export
    :param imShape: The size of this output image
    :param outFile: Where to save the output. If *None*, no export is created.
    :return:
    """
    return cls.exportLblPng(compDf, outFile, imShape, **kwargs)

  @classmethod
  def exportCompimgsFolders(cls, compDf: df, imgDir: FilePath=None, margin=0, marginAsPct=False,
                            colorMaskByClass=True, outDir: FilePath=None, dataDir='data',
                            semanticDir='masks_semantic', bboxDir: str=None,
                            resizeShape: Sequence[int]=None, **kwargs):
    """
    From a component dataframe, creates output directories for component images and masks.
    This is useful for many neural networks etc. to read individual component images.

    :param compDf: Dataframe to export
    :param imgDir: Passed to `exportCompimgsDf`
    :param margin: Passed to `exportCompimgsDf`
    :param marginAsPct: Passed to `exportCompimgsDf`
    :param colorMaskByClass: Passed to `exportCompimgsDf`
    :param outDir: Where to make the output directories. If `None`, defaults to current
      directory>compimgs_<margin>_margin
    :param dataDir: Where to export the component images
    :param semanticDir: Where to export semantic masks. If `None`, no semantic masks
      are exported.
    :param bboxDir: Where to export bounding box masks. If `None`, no bounding box masks
      are exported.
    :param resizeShape: If provided, it is the shape that all images will be resized to before
      being saved. This is useful for neural networks with a fixed input size which forces all
      inputs to be e.g. 100x100 pixels.
    """
    if outDir is None:
      outDir = Path('.')/f'compimgs_{margin}_margin'
    (outDir/dataDir).mkdir(exist_ok=True, parents=True)
    excludeCols = []
    if semanticDir is None:
      excludeCols.append('semanticMask')
    else:
      (outDir/semanticDir).mkdir(exist_ok=True)
    if bboxDir is None:
      excludeCols.append('bboxMask')
    else:
      (outDir/bboxDir).mkdir(exist_ok=True)

    saveFn = lambda fname, img: io.imsave(fname, img, check_contrast=False)
    if resizeShape is not None:
      saveFn = lambda fname, img: io.imsave(fname, resize_pad(img, resizeShape),
                                            check_contrast=False)

    extractedImgs = cls.exportCompimgsDf(compDf, None, imgDir, margin, colorMaskByClass,
                                         excludeCols, **kwargs)
    for idx, row in extractedImgs.iterrows():
      saveName = f'{row.instId}.png'
      saveFn(outDir/dataDir/saveName, row.img)
      if semanticDir is not None:
        saveFn(outDir/semanticDir/saveName, row.semanticMask)
      if bboxDir is not None:
        saveFn(outDir/bboxDir/saveName, row.bboxMask)

  # -----
  # Import options
  # -----

  @classmethod
  def convert(cls, fromData: FilePathOrDf, toFile: FilePath, doExport=True, importArgs: dict=None,
              exportArgs: dict=None):
    if importArgs is None:
      importArgs = {}
    if exportArgs is None:
      exportArgs = {}
    if not isinstance(fromData, df):
      fromData = cls.buildByFileType(fromData, **importArgs)
    exportFn = cls._ioFnFromFileType(toFile, 'export')
    if not doExport:
      toFile = None
    return exportFn(fromData, toFile, **exportArgs)

  @classmethod
  def buildFromCsv(cls, inFileOrDf: FilePathOrDf, imShape: Tuple=None,
                   reindex=False) -> df:
    """
    Deserializes data from a csv file to create a Component :class:`DataFrame`.
    The input .csv should be the same format as one exported by
    :func:`csvImport <ComponentMgr.csvImport>`.

    :param imShape: If included, this ensures all imported components lie within imSize
           boundaries. If any components do not, an error is thrown since this is
           indicative of components that actually came from a different reference image.
    :param inFileOrDf: Name of file to import, or dataframe if it was already read from this
      file type. Useful if several csv's were concatenated into one dataframe and *that* is
      being imported.
    :param reindex: Whether to disregard the index of the incoming dataframe or file.
      This is useful when *inFileOrDf* is actually a conacatenated df of multiple files, and
      the index doesn't need to be retained.
    :return: Tuple: DF that will be exported if successful extraction
    """
    field = FRParam('None', None)
    try:
      if isinstance(inFileOrDf, df):
        csvDf = inFileOrDf
      else:
        csvDf = pd.read_csv(inFileOrDf, keep_default_na=False)
      if reindex:
        csvDf[RTF.INST_ID.name] = np.arange(len(csvDf), dtype=int)
        csvDf = csvDf.set_index(RTF.INST_ID.name, drop=False)
      # Decouple index from instance ID until after transfer from csvDf is complete
      # This was causing very strange behavior without reset_index()...
      outDf = cls.tableData.makeCompDf(len(csvDf)).reset_index(drop=True)
      # Objects in the original frame are represented as strings, so try to convert these
      # as needed
      for field in cls.tableData.allFields:
        if field.name in csvDf:
          matchingCol = csvDf[field.name]
          # 'Object' type results in false positives
          if matchingCol.dtype != object and type(field.value) == matchingCol.dtype:
            outDf[field] = csvDf[field.name]
          else:
            # Parsing functions only know how to convert from strings to themselves.
            # So, assume the exting types can first convert themselves to strings
            with np.printoptions(threshold=sys.maxsize):
              matchingCol = matchingCol.apply(str)
            outDf[field] = _strSerToParamSer(matchingCol, field.value)
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
  def buildFromCompimgsDf(cls, inFile: FilePath, imShape: Tuple=None,
                          lblField='Class', **importArgs):
    lblField = cls.tableData.fieldFromName(lblField)
    inDf = pd.read_pickle(inFile)
    outDf = cls.tableData.makeCompDf(len(inDf))
    outDf[RTF.INST_ID] = inDf['instId']
    allVerts = []

    for idx, row in inDf.iterrows():
      mask = cls._strToNpArray(row.semanticMask, dtype=bool)
      verts = ComplexXYVertices.fromBwMask(mask)
      offset = cls._strToNpArray(row.offset)
      for v in verts: v += offset
      allVerts.append(verts)
    outDf[RTF.VERTICES] = allVerts
    outDf[lblField] = inDf[lblField]
    cls.checkVertBounds(outDf[RTF.VERTICES], imShape)
    return outDf

  @classmethod
  def buildFromPkl(cls, inFile: FilePath, imShape: Tuple=None, **importArgs) -> df:
    """
    See docstring for :func:`self.buildFromCsv`
    """
    pklDf = pd.read_pickle(inFile)
    cls.checkVertBounds(pklDf[RTF.VERTICES], imShape)
    return pklDf

  @classmethod
  def buildFromLblPng(cls, inFileOrImg: Union[FilePath, GrayImg],
                      labelMapping: pd.Series=None,
                      useDistinctRegions=False,
                      **importArgs) -> df:
    if isinstance(inFileOrImg, GrayImg):
      labelImg = inFileOrImg
    else:
      labelImg = io.imread(inFileOrImg, as_gray=True)
    allVerts = []
    lblField_out = []
    for origVal, numericLbl in labelMapping.iteritems():
      verts = ComplexXYVertices.fromBwMask(labelImg == numericLbl)
      if useDistinctRegions:
        allVerts.extend(verts)
        orig = np.tile(origVal, len(verts))
      else:
        allVerts.append(verts)
        orig = [origVal]
      lblField_out.extend(orig)
    outDf = cls.tableData.makeCompDf(len(allVerts))
    outDf[labelMapping.name] = lblField_out
    outDf[RTF.VERTICES] = allVerts
    return outDf


  @classmethod
  def buildFromIdPng(cls, inFileOrImg: Union[FilePath, GrayImg], imShape: Tuple=None, **importArgs) -> df:
    if isinstance(inFileOrImg, GrayImg):
      labelImg = inFileOrImg
    else:
      labelImg = io.imread(inFileOrImg, as_gray=True)
    outDf = cls._idImgToDf(labelImg)
    cls.checkVertBounds(outDf[RTF.VERTICES], imShape)
    return outDf

  @classmethod
  def buildFromClassPng(cls, inFileOrImg: Union[FilePath, GrayImg], imShape: Tuple=None, **importArgs) -> df:
    if isinstance(inFileOrImg, GrayImg):
      clsImg = inFileOrImg
    else:
      clsImg = io.imread(inFileOrImg)
    clsParam = cls.tableData.fieldFromName('Class')
    clsArray = np.array(clsParam.opts['limits'])
    idImg = measure.label(clsImg)
    outDf = cls.buildFromIdPng(idImg, imShape)
    outClasses = []
    for curId in outDf[RTF.INST_ID]+1:
      # All ID pixels should be the same class, so any representative will do
      curCls = clsImg[idImg == curId][0]
      outClasses.append(clsArray[curCls-1])
    outDf[cls.tableData.fieldFromName('Class')] = outClasses
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
           f'Offending IDs: {offendingIds}', S3AWarning)

  @classmethod
  def _idImgToDf(cls, idImg: GrayImg):
    # Skip 0 since it's indicative of background
    regionIds = np.unique(idImg)
    regionIds = regionIds[regionIds != 0]
    allVerts = []
    for curId in regionIds:
      verts = ComplexXYVertices.fromBwMask(idImg == curId)
      allVerts.append(verts)
    outDf = cls.tableData.makeCompDf(regionIds.size)
    # Subtract 1 since instance ids are 0-indexed
    outDf[RTF.INST_ID] = regionIds-1
    outDf[RTF.VERTICES] = allVerts
    return outDf