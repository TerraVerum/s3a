from __future__ import annotations

import inspect
import json
import pickle
import sys
import tempfile
from ast import literal_eval
from collections import defaultdict
from contextlib import ExitStack
from functools import wraps
from pathlib import Path
from stat import S_IREAD, S_IRGRP, S_IROTH
from typing import Any, Optional, Union, Tuple, Callable
from zipfile import ZipFile

import cv2 as cv
import numpy as np
import pandas as pd
from skimage import io, draw
from skimage.exposure import rescale_intensity
from typing_extensions import Literal
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from s3a.constants import REQD_TBL_FIELDS as RTF, PRJ_ENUMS
from s3a.generalutils import augmentException, getCroppedImg, resize_pad, cvImsave_rgb, orderContourPts, classproperty, \
  cvImread_rgb
from s3a.parameditors.table import TableData
from s3a.structures import PrjParamGroup, FilePath, GrayImg, \
  ComplexXYVertices, PrjParam, XYVertices, AnnParseError
from utilitys import fns
from utilitys.fns import warnLater

FilePathOrDf = Union[FilePath, pd.DataFrame]
# Values are strings
# noinspection PyTypeHints
_litLst = Literal[PRJ_ENUMS.IO_IMPORT, PRJ_ENUMS.IO_EXPORT]

def _attrNameFmt(buildOrExport: _litLst, obj):
  def membership(el):
    for exclude in ['FileType', 'Opts', 'Types', 'Serialized']:
      if el.endswith(exclude):
        return False
    return el.startswith(buildOrExport)
  attrs = [attr.replace(buildOrExport, '') for attr in vars(obj) if membership(attr)]
  out = {}
  for attr in attrs:
    descrFmt = fns.pascalCaseToTitle(attr)
    fileFmt = descrFmt.replace(' ', '.').lower()
    out[fileFmt] = descrFmt + ' Files'
  return out

def _getImg_mapping_offset(inFileOrImg: Union[FilePath, GrayImg]):
  if isinstance(inFileOrImg, np.ndarray):
    return inFileOrImg, None, None
  try:
    image: Image.Image = Image.open(inFileOrImg)
    info = image.info
    image = np.array(image)
  except TypeError:
    # E.g. float image
    return cvImread_rgb(inFileOrImg, mode=cv.IMREAD_UNCHANGED), None, None
  # "Offset" present for numeric data, "mapping" present for textual data
  if 'mapping' in info:
    mapping = pd.Series(json.loads(info['mapping']), name=info.get('field', None))
    mapping.index = mapping.index.astype(int)
    return image, mapping, None
  if 'offset' in info:
    return image, None, int(info['offset'])
  # No extra metadata
  return image, None, None

def _writeImge_meta(outImg: np.ndarray, saveName: FilePath, mapping: pd.Series=None, offset: int=None):
  outImg = Image.fromarray(outImg)
  info = PngInfo()
  if mapping is not None:
    info.add_text('mapping', json.dumps(mapping.to_dict()))
    info.add_text('field', str(mapping.name))
  if offset is not None:
    info.add_text('offset', str(offset))
  outImg.save(saveName, pnginfo=info)

def _getPdExporters():
  members = inspect.getmembers(
    pd.DataFrame, lambda meth: inspect.isfunction(meth) and meth.__name__.startswith('to_'))
  return [mem[0].replace('to_', '') for mem in members]
def _getPdImporters():
  members = inspect.getmembers(
    pd.DataFrame,lambda meth: inspect.isfunction(meth) and meth.__name__.startswith('read_'))
  return [mem[0].replace('read_', '') for mem in members]

def _strSerToParamSer(strSeries: pd.Series, paramVal: Any) -> pd.Series:
  paramType = type(paramVal)
  # TODO: Move this to a more obvious place?
  funcMap = {
    # Format string to look like a list, use ast to convert that string INTO a list, make a numpy array from the list
    np.ndarray        : lambda strVal: np.array(literal_eval(strVal)),
    ComplexXYVertices : ComplexXYVertices.deserialize,
    XYVertices        : XYVertices.deserialize,
    bool              : lambda strVal: strVal.lower() == 'true',
    PrjParam          : lambda strVal: PrjParamGroup.fieldFromParam(paramVal.group, strVal)
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
    XYVertices: XYVertices.serialize,
    ComplexXYVertices: ComplexXYVertices.serialize,
  }
  defaultFunc = lambda param: str(param)

  funcToUse = funcMap.get(paramType, defaultFunc)
  return paramSer.apply(funcToUse)

# Credit: https://stackoverflow.com/a/28238047/9463643
# class classorinstancemethod(classmethod):
#   def __get__(self, instance, type_):
#     descr_get = super().__get__ if instance is None else self.__func__.__get__
#     return descr_get(instance, type_)

class ComponentIO:
  """
  Exporter responsible for saving Component information to a file or object.
  Once created, users can extract different representations of components by
  calling exporter.exportCsv, exportPkl, etc. for those objects / files respectively.
  """
  @classproperty
  def roundTripTypes(self):
    """
    Not all IO types can export->import and remain the exact same dataframe afterwards.
    For instance, exporting a labeled image will discard all additional fields.
    This property holds export types which can give back the original dataframe after
    a round trip export->import.
    """
    # Since both import and export should have these keys, can use either dict
    return {k: self.importTypes[k] for k in ['csv', 'pkl']}

  tableData = TableData()
  """Table to use for import/export cross checks. This is how class and table field information is derived."""
  importOpts = {}
  """
  Propagated to every importByFileType call to provide user-specified defaults as desired
  """
  exportOpts = {}
  """
  Propagated to every exportByFileType call to provide user-specified defaults as desired
  """

  def __init__(self):
    # Propagate custom defaults to each desired function
    for typeDict, fnType in zip([self.importTypes, self.exportTypes], [PRJ_ENUMS.IO_IMPORT, PRJ_ENUMS.IO_EXPORT]):
      for fileExt in typeDict:
        func = self._ioFnFromFileType(fileExt, fnType)
        setattr(self, func.__name__, self._ioWrapper(func))

  @classproperty
  def exportTypes(cls):
    """
    File types this class can export. {file type: descriptoin} useful for adding to
    file picker dialog
    """
    return _attrNameFmt(PRJ_ENUMS.IO_EXPORT, cls)

  @classproperty
  def importTypes(cls):
    """
    Types this class can import. {file type: descriptoin} useful for adding to
    file picker dialog
    """
    return _attrNameFmt(PRJ_ENUMS.IO_IMPORT, cls)

  @classmethod
  def ioFileFilter(cls, which=PRJ_ENUMS.IO_ROUND_TRIP, typeFilter='', **extraOpts):
    """
    Helper for creating a file filter out of the handled IO types. The returned list of
    strings is suitable for inserting into a QFileDialog.

    :param which: Whether to generate filters for build types, export types, or round trip
    :param typeFilter: type filter for handled io types. For instanece, if typ='png', then
      a file filter list with only 'id.png' and 'class.png' will appear.
    :param extraOpts; Extra file types to include in the filter
    """
    ioDict = {
      PRJ_ENUMS.IO_ROUND_TRIP: cls.roundTripTypes,
      PRJ_ENUMS.IO_IMPORT: cls.importTypes,
      PRJ_ENUMS.IO_EXPORT: cls.exportTypes
    }[which]
    if isinstance(typeFilter, str):
      typeFilter = [typeFilter]
    fileFilters = []
    for typ, info in dict(**ioDict, **extraOpts).items():
      if any([t in typ for t in typeFilter]):
        fileFilters.append(f'{info} (*.{typ})')
    return ';;'.join(fileFilters)

  def _ioWrapper(self, func: Callable):
    """Wraps build and export functions to provide defaults specified by build/exportOpts before the function call"""
    which = PRJ_ENUMS.IO_IMPORT if func.__name__.startswith(PRJ_ENUMS.IO_IMPORT) else PRJ_ENUMS.IO_EXPORT
    @wraps(func)
    def wrapper(*args, **kwargs):
      useOpts = self.importOpts if which is PRJ_ENUMS.IO_IMPORT else self.exportOpts
      # Turn args into kwargs
      fnKwargs = {**useOpts, **kwargs}
      return func(*args, **fnKwargs)

    return wrapper

  def exportByFileType(self, compDf: pd.DataFrame, outFile: Union[str, Path], verifyIntegrity=True, **exportArgs):
    outFile = Path(outFile)
    outFn = self._ioFnFromFileType(outFile, PRJ_ENUMS.IO_EXPORT)

    ret = outFn(compDf, outFile, **exportArgs)
    if verifyIntegrity and outFile.suffix[1:] in self.roundTripTypes:
      matchingCols = np.setdiff1d(compDf.columns, [RTF.INST_ID,
                                                   RTF.SRC_IMG_FILENAME])
      loadedDf = self.importByFileType(outFile)
      # For some reason, there are cases in which numy comparison spots false errors and cases where
      # pandas spots false errors. Only a problem if both see problms
      dfCmp = np.array_equal(loadedDf[matchingCols], compDf[matchingCols]) \
              or loadedDf[matchingCols].equals(compDf[matchingCols])
      problemCells = defaultdict(list)

      if not dfCmp:
        dfA = loadedDf[matchingCols]
        dfB = compDf[matchingCols]
        for ii in range(len(dfA)):
          for jj in range(len(dfA.columns)):
            if not np.array_equal(dfA.iat[ii, jj], dfB.iat[ii,jj]):
              problemCells[compDf.at[dfB.index[ii], RTF.INST_ID]].append(str(matchingCols[jj]))
        # The only way to prevent "truth value of array is ambiguous" is cell-by-cell iteration
        problemMsg = [f'{idx}: {cols}' for idx, cols in problemCells.items()]
        problemMsg = '\n'.join(problemMsg)
        # Try to fix the problem with an iloc write
        warnLater('<b>Warning!</b> Saved components do not match current component'
             ' state. This can occur when pandas incorrectly caches some'
             ' table values. Problem cells (shown as [id]: [columns]):\n'
             + f'{problemMsg}\n'
               f'Please try manually altering these values before exporting again.', UserWarning)
    return ret

  def importByFileType(self, inFile: Union[str, Path], imShape: Tuple[int]=None,
                      strColumns=False, **importArgs):
    buildFn = self._ioFnFromFileType(inFile, PRJ_ENUMS.IO_IMPORT)
    outDf = buildFn(inFile, imShape=imShape, **importArgs)
    if strColumns:
      outDf.columns = list(map(str, outDf.columns))
    return outDf

  def _ioFnFromFileType(self, fpath: Union[str, Path],
                        buildOrExport=_litLst,
                        missingOk=False) -> Optional[Callable]:
    fpath = Path(fpath)
    fname = fpath.name
    cmpTypes = np.array(list(_attrNameFmt(buildOrExport, type(self))))
    typIdx = [fname.endswith(typ) for typ in cmpTypes]
    if not any(typIdx):
      raise IOError(f'Not sure how to handle file {fpath.name}')
    fnNameSuffix = cmpTypes[typIdx][-1].title().replace('.', '')
    outFn: Callable =  getattr(self, buildOrExport + fnNameSuffix, None)
    if outFn is None and not missingOk:
      raise ValueError(f'Full I/O specification missing for type {fnNameSuffix}')
    return outFn

  def _pandasSerialImport(self, inFileOrDf: Union[str, Path, pd.DataFrame],
                          **pdImportArgs):
    if isinstance(inFileOrDf, pd.DataFrame):
      serialDf = inFileOrDf
    else:
      fType = Path(inFileOrDf).suffix.lower().replace('.', '')
      importFn = getattr(pd, f'read_{fType}', None)
      if importFn is None:
        raise ValueError(f'File type {fType} cannot be handled by the serial importer.'
                         f' Must be one of {",".join(_getPdImporters())}')
      # Special case: csv imports need to avoid interpreting nan results
      pdImportArgs.update(keep_default_na=False)
      acceptedArgs = inspect.signature(importFn).parameters
      useArgs = pdImportArgs.keys() & acceptedArgs
      serialDf = importFn(inFileOrDf, **{k: pdImportArgs[k] for k in useArgs})
    return serialDf

  def _pandasSerialExport(self, exportDf: pd.DataFrame, outFile: Union[str, Path]=None,
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
    exporter = outPath.suffix.lower().replace('.', '')

    defaultExportParams.update(pdExportArgs)
    exportFn = getattr(exportDf, f'to_{exporter}', None)
    if exportFn is None:
      raise ValueError(f'Exporter {exporter} not recognized. Acceptable options:\n'
                       + ', '.join(_getPdExporters()))

    with np.printoptions(threshold=sys.maxsize):
      exportFn(outFile, index=False)
    if readOnly:
      outPath.chmod(S_IREAD|S_IRGRP|S_IROTH)

  # -----
  # Export options
  # -----
  def exportSerialized(self, compDf: pd.DataFrame, outFile: Union[str, Path]=None,
                       readOnly=True, **pdExportArgs):
    """
    Converts dataframe into a string-serialized version and uses pandas to write it to disk.

    :param compDf: Dataframe to export
    :param outFile: Name of the output file location. If *None*, no file is created. However,
      the export object (string dtype dataframe) will still be created and returned.
      The file suffix can be any option supported by a pandas exporter. This can be
      csv, json, feather, etc.
      Note: pickle is a special case. In some cases, it is significantly more benficial
      to export the raw dataframe compared to a serialized version. In these cases, use
      ComponentIO.exportPkl. Otherwise, `pickle` is still a valid option here for a serialized
      format. For a full list of export options, see
      `the documentation`https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html`.
    :param pdExportArgs: Dictionary of values passed to underlying pandas export function.
      These will overwrite the default options for :func:`exportToFile <ComponentMgr.exportByFileType>`
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
      self._pandasSerialExport(exportDf, outFile, readOnly, **pdExportArgs)
    except Exception as ex:
      errMsg = f'Error on parsing column "{col.name}"\n'
      augmentException(ex, errMsg)
      raise
    return exportDf

  @wraps(exportSerialized, assigned=('__doc__', '__annotations__'))
  def exportCsv(self, *args, **kwargs):
    """Exposed format from the more general exportSerialized"""
    return self.exportSerialized(*args, **kwargs)

  def exportCompImgsDf(self, compDf: pd.DataFrame, outFile: Union[str, Path]=None,
                       imgDir: FilePath=None, margin=0, marginAsPct=False,
                       includeCols=('instId', 'img', 'labelMask', 'label', 'offset'),
                       lblField='Instance ID', asIndiv=False,
                       returnLblMapping=False,
                       missingOk=False, **kwargs):
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
    :param asIndiv: Whether components should be exported as individual units (i.e.
      neighbors are guaranteed never to show up in the ground truth label mask) or whether
      a cropped area around the component should be used as the label mask. This will
      include mask values of neighbors if they are within mask range. Note: This is
      performed with preference toward higher ids, i.e. if a high ID is on top of a low ID,
      the low ID will still be covered in its export mask
    :param missingOk: Whether a missing image is acceptable. When no source image is found
      for an annotation, this will simpy the 'image' output property
    :param returnLblMapping: Whether to return the mapping of label numeric values to table field values
    :return: Dataframe with the following keys:
      - instId: The component's Instance ID
      - img: The (MxNxC) image corresponding to the component vertices, where MxN are
        the padded row sizes and C is the number of image channels
      - labelMask: Binary mask representing the component vertices
      - label: Field value of the component for the field specified by `lblField`
      - offset: Image (x,y) coordinate of the min component vertex.
    :param kwargs: Passed to ComponentIO.exportLblPng
    """
    bgColor = kwargs.get('bgColor', 0)
    _imgCache = {}
    if imgDir is None:
      imgDir = Path('.')
    else:
      imgDir = Path(imgDir)
    uniqueImgs = np.unique(compDf[RTF.SRC_IMG_FILENAME])
    dfGroupingsByImg = []
    for imgName in uniqueImgs:
      fullImgName = Path(imgName)
      if not fullImgName.is_absolute():
        fullImgName = imgDir / fullImgName
      if fullImgName not in _imgCache:
        if not fullImgName.exists() and missingOk:
          _imgCache[fullImgName] = None
        else:
          _imgCache[fullImgName] = io.imread(fullImgName)
      dfGroupingsByImg.append(compDf[compDf[RTF.SRC_IMG_FILENAME] == imgName])

    useKeys = set(includeCols)
    outDf = {k: [] for k in useKeys}
    lblField = self.tableData.fieldFromName(lblField)
    # imshape is automatically inferred by the exporter
    kwargs.pop('imShape', None)
    mappings = {}
    for miniDf, fullImgName in zip(dfGroupingsByImg, uniqueImgs):
      fullImgName = imgDir / fullImgName
      img = _imgCache[fullImgName]
      shape = img if img is None else img.shape[:2]
      lblImg, mapping = self.exportLblPng(miniDf,
                                          imShape=shape,
                                          lblField=lblField,
                                          returnLblMapping=True,
                                          **kwargs)
      mappings[fullImgName.name] = mapping
      if img is None:
        img = np.zeros_like(lblImg)
      if asIndiv:
        # Also need an ID mask
        if lblField == RTF.INST_ID:
          idImg = lblImg.copy()
        else:
          idImg, mapping = self.exportLblPng(miniDf, imShape=img.shape[:2],
                                             lblField=RTF.INST_ID,
                                             returnLblMapping=True)
        invertedMap = pd.Series(mapping.index, mapping)


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

        # x-y to row-col, transpose to get min-max in axis 0 and row-col in axis 1
        indexer = tuple(slice(*b) for b in bounds[:, ::-1].T)
        xyOffset = bounds[0,:]
        if lblImg.ndim > 2:
          indexer += (...,)
        if 'labelMask' in useKeys:
          mask = lblImg[indexer]
          if asIndiv:
            # Need to black out every pixel not in the current component, copy to avoid
            # in-place modification
            mask = mask.copy()
            idMask = idImg[indexer]
            mask[idMask != invertedMap[idx]] = bgColor

          outDf['labelMask'].append(mask)

        if 'instId' in useKeys:
          outDf['instId'].append(idx)

        if 'offset' in useKeys:
          outDf['offset'].append(xyOffset)
    outDf = pd.DataFrame(outDf)
    if len(mappings) == 1:
      # Common case where annotations for just one image were converted
      mappings = next(iter(mappings.values()))
    outDf.attrs['mapping'] = mappings
    if outFile is not None:
      outDf.to_pickle(outFile)
    if returnLblMapping:
      return outDf, mappings
    return outDf

  def exportPkl(self, compDf: pd.DataFrame, outFile: Union[str, Path]=None, **exportArgs) -> (Any, str):
    """
    See the function signature for :func:`exportCsv <ComponentIO.exportCsv>`
    """
    # Since the write-out is a single operation there isn't an intermediate form to return
    pklDf = None
    if outFile is not None:
      pklDf = pickle.dumps(compDf)
      compDf.to_pickle(outFile)
    return pklDf

  def exportLblPng(self, compDf: pd.DataFrame, outFile: FilePath=None, imShape: Tuple[int]=None,
                   lblField: Union[PrjParam, str]='Instance ID', bgColor=0,
                   rescaleOutput=False, returnLblMapping=False,
                   writeMeta=True,
                   **kwargs):
    """
    :param compDf: Dataframe to export
    :param outFile: Filename to save, leave *None* to avoid saving to a file
    :param imShape: MxN shape of image containing these annotations
    :param lblField: Data field to use as an index label. E.g. "Class" will use the 'class'
      column, but any other column can be specified. The output ground truth masks
      will be colored according to this field.  See :meth:`PrjParam.toNumeric` for details.
      If `lblField` is *None*, the foreground mask will be boolean instead of integer-colored.
    :param bgColor: Color of the mask background. Must be an integer.
    :param rescaleOutput: For images designed for human use, it is helpful to have
      outputs rescaled to the entire intensity range. Otherwise, they usually end
      up looking pure black and it is difficult to see components in the image.
      When `rescaleOutput` is *True*, all numbers are scaled to the 'uint16' range.
    :param returnLblMapping: Whether to return a pd.Series matching original index values
      to their numeric counterparts. Note: this is important in cases where an offset must be applied to the underlying
      data. If the background color is 0 and a valid numeric value is also 0, it will be impossible to detect this
      object in the labeled output. So, an offset must be applied in these cases (background - min(data) + 1). This
      mapping records the relevant information to import original values back during `importLblPng`.
    :param writeMeta: Whether to write the field mapping/offset to the output image file as png metadata.
      Useful to preserve label information when re-importing.
    :param kwargs:
    :return:
    """

    lblField = self.tableData.fieldFromName(lblField)

    if bgColor < 0:
      raise ValueError(f'Background color must be >= 0, was {bgColor}')

    readMapping = returnLblMapping or (writeMeta and outFile is not None)
    labels = compDf[lblField]
    labels_numeric = lblField.toNumeric(labels, returnMapping=readMapping)
    # Make sure numeric labels aren't the same as background, otherwise they will be forever lost
    mapping = None
    if readMapping:
      labels_numeric, mapping = labels_numeric
    diff = max(bgColor - np.min(labels_numeric, initial=0) + 1, 0)
    if readMapping:
      mapping.index += diff
    labels_numeric += diff
    asBool = np.issubdtype(labels_numeric.dtype, np.bool_)

    if rescaleOutput:
      if mapping is not None:
        max_ = np.max(np.asarray(mapping.index), initial=bgColor)
        mapping.index = rescale_intensity(mapping.index, in_range=(bgColor, max_), out_range='uint16')
      else:
        max_ = np.max(labels_numeric, initial=bgColor)
      labels_numeric = rescale_intensity(labels_numeric, in_range=(bgColor, max_), out_range='uint16')

    if imShape is None:
      # Without any components the image is non-existant
      if len(compDf) == 0:
        raise ValueError('imShape cannot be *None* if no components are present')
      vertMax = ComplexXYVertices.stackedMax(compDf[RTF.VERTICES])
      imShape = tuple(vertMax[::-1] + 1)
    maskType = 'uint16' if np.min(bgColor) >= 0 else 'int32'
    outMask = np.full(imShape[:2], bgColor, dtype=maskType)
    for fillClr, (_, comp) in zip(labels_numeric, compDf.iterrows()):
      verts: ComplexXYVertices = comp[RTF.VERTICES]
      outMask = verts.toMask(outMask, int(fillClr), False, False)
    if asBool:
      outMask = outMask > 0

    if outFile is not None:
      if writeMeta:
        _writeImge_meta(outMask, outFile, mapping)
      else:
        cvImsave_rgb(outFile, outMask)
    if returnLblMapping:
      return outMask, mapping
    return outMask

  # def exportClassPng(self, compDf: pd.DataFrame, outFile: FilePath = None, imShape: Tuple[int]=None, **kwargs):
  #   # Create label to output mapping
  #   return self.exportLblPng(compDf, outFile, imShape, 'Class', **kwargs)

  def exportCompImgsZip(self, compDf: pd.DataFrame,
                        outDir:FilePath='s3a-export',
                        resizeShape: Tuple[int, int]=None,
                        archive=False,
                        **kwargs):
    """
    From a component dataframe, creates output directories for component images and masks.
    This is useful for many neural networks etc. to read individual component images.

    :param compDf: Dataframe to export
    :param outDir:
      helpText: "Where to make the output directories. If `None`, defaults to current
      directory>compimgs_<margin>_margin"
      pType: filepicker
    :param resizeShape: If provided, it is the shape that all images will be resized to before
      being saved. This is useful for neural networks with a fixed input size which forces all
      inputs to be e.g. 100x100 pixels.
    :param archive: Whether to compress into a zip archive instead of directly outputting a folder
    :param kwargs: Passed directly to :meth:`ComponentIO.exportCompImgsDf`
    """
    outDir = Path(outDir)
    useDir = outDir

    saveFn = lambda fname, img: cvImsave_rgb(fname, img)
    if resizeShape is not None:
      saveFn = lambda fname, img: cvImsave_rgb(fname, resize_pad(img, resizeShape))

    with ExitStack() as stack:
      if archive:
        useDir = Path(stack.enter_context(tempfile.TemporaryDirectory()))
      dataDir = useDir/'data'
      labelsDir= useDir/'labels'
      dataDir.mkdir(exist_ok=True, parents=True)
      labelsDir.mkdir(exist_ok=True, parents=True)

      extractedImgs = self.exportCompImgsDf(compDf, None, **kwargs)
      for idx, row in extractedImgs.iterrows():
        saveName = f'{row.instId}.png'
        saveFn(dataDir/saveName, row.img)
        saveFn(labelsDir/saveName, row.labelMask)
      if archive:
        if outDir.suffix != '.zip':
          outDir = outDir.with_suffix(outDir.suffix + '.zip')
        with ZipFile(outDir, 'w') as ozip:
          for dir_ in labelsDir, dataDir:
            for file in dir_.iterdir():
              ozip.write(file, f'{dir_.name}/{file.name}')

  # -----
  # Import options
  # -----
  def convert(self, fromData: FilePathOrDf, toFile: FilePath, doExport=True, importArgs: dict=None,
              exportArgs: dict=None):
    if importArgs is None:
      importArgs = {}
    if exportArgs is None:
      exportArgs = {}
    if not isinstance(fromData, pd.DataFrame):
      fromData = self.importByFileType(fromData, **importArgs)
    exportFn = self._ioFnFromFileType(toFile, PRJ_ENUMS.IO_EXPORT)
    if not doExport:
      toFile = None
    return exportFn(fromData, toFile, **exportArgs)

  @fns.dynamicDocstring(availImporters=_getPdImporters())
  def importSerialized(self, inFileOrDf: FilePathOrDf, imShape: Tuple=None,
                          reindex=False, **importArgs):
    """
    Deserializes data from a file or string dataframe to create a S3A Component
    :class:`DataFrame`.
    The input file or dataframe should be the same format as one exported by
    :func:`exportSerialized <ComponentMgr.exportSerialized>`.

    :param imShape: If included, this ensures all imported components lie within imSize
           boundaries. If any components do not, an error is thrown since this is
           indicative of components that actually came from a different reference image.
    :param inFileOrDf: Name of file to import, or dataframe itself if it was already read
    from this file. Note: avoid passing in a dataframe directly unless you know what
    you're doing since null values and some converter types can produce misleading
    values. Most file types supported by pandas (pandas.read_*) are supported here.
    :param reindex: Whether to disregard the index of the incoming dataframe or file.
      This is useful when *inFileOrDf* is actually a conacatenated df of multiple files, and
      the index doesn't need to be retained.
    :return: Tuple: pd.DataFrame that will be exported if successful extraction
    """
    field = PrjParam('None', None)
    serialDf = self._pandasSerialImport(inFileOrDf, **importArgs)
    if reindex:
      serialDf[RTF.INST_ID.name] = np.arange(len(serialDf), dtype=int)
      serialDf = serialDf.set_index(RTF.INST_ID.name, drop=False)
    try:
      # Decouple index from instance ID until after transfer from csvDf is complete
      # This was causing very strange behavior without reset_index()...
      outDf = self.tableData.makeCompDf(len(serialDf)).reset_index(drop=True)
      # Objects in the original frame are repre sented as strings, so try to convert these
      # as needed
      for field in self.tableData.allFields:
        if field.name in serialDf:
          matchingCol = serialDf[field.name]
          # 'Object' type results in false positives
          if matchingCol.dtype != object and type(field.value) == matchingCol.dtype:
            outDf[field] = serialDf[field.name]
          else:
            # Parsing functions only know how to convert from strings to themselves.
            # So, assume the exting types can first convert themselves to strings
            with np.printoptions(threshold=sys.maxsize):
              matchingCol = matchingCol.apply(str)
            outDf[field] = _strSerToParamSer(matchingCol, field.value)
      outDf = outDf.set_index(RTF.INST_ID, drop=False)

      self.checkVertBounds(outDf[RTF.VERTICES], imShape)
    except Exception as ex:
      # Rethrow exception with insight about column number
      # Procedure copied from https://stackoverflow.com/a/6062677/9463643
      errMsg = f'Error importing column "{field.name}":\n'
      augmentException(ex, errMsg)
      raise
    # TODO: Apply this function to individual rows instead of the whole dataframe. This will allow malformed
    #  rows to gracefully fall off the dataframe with some sort of warning message
    return outDf

  @wraps(importSerialized, assigned=('__doc__', '__annotations__'))
  def importCsv(self, *args, **kwargs):
    """Exposed format from the more general importSerialized"""
    return self.importSerialized(*args, **kwargs)

  def importGeojson(self, inFileOrDict: Union[FilePath, dict], parseErrorOk=False, **importArgs):
    fileName = None
    if not isinstance(inFileOrDict, dict):
      fileName = Path(inFileOrDict)
      with open(inFileOrDict, 'r') as ifile:
        inFileOrDict = json.load(ifile)
    verts = []
    badInsts = []
    for ann in inFileOrDict['features']:
      geo = ann['geometry']
      if geo['type'] == 'Polygon':
        verts.append(ComplexXYVertices(geo['coordinates'], coerceListElements=True))
      else:
        badInsts.append(ann)
    if badInsts and not parseErrorOk:
      raise AnnParseError(f'Currently, S3A only supports polygon geojson types',
                          fileName=fileName, instances=badInsts)
    outDf = self.tableData.makeCompDf(len(verts))
    outDf[RTF.VERTICES] = verts
    return outDf

  def importSuperannotateJson(self, inFileOrDict: Union[FilePath, dict], parseErrorOk=False, **importArgs):
    fileName = None
    if not isinstance(inFileOrDict, dict):
      fileName = Path(inFileOrDict)
      with open(inFileOrDict, 'r') as ifile:
        inFileOrDict = json.load(ifile)
    instances = inFileOrDict['instances']
    parsePts = []
    invalidInsts = []
    for inst in instances:
      typ = inst['type']
      if typ in ['polygon', 'bbox']:
        pts = inst['points']
        if typ == 'bbox':
          pts = list(pts.values())
        parsePts.append(np.column_stack([pts[::2], pts[1::2]]))
      elif typ == 'ellipse':
        vals = inst['cy'], inst['cx'], inst['ry'], inst['rx'], inst['angle']
        pts = draw.ellipse_perimeter(*(int(v) for v in vals))
        pts = np.column_stack(pts[::-1])
        parsePts.append(orderContourPts(pts))
      else:
        invalidInsts.append(inst)
    if invalidInsts and not parseErrorOk:
      raise AnnParseError('Currently, S3A only supports polygon annotations from SuperAnnotate',
                          fileName=fileName, instances=invalidInsts)
    outDf = self.tableData.makeCompDf(len(parsePts))
    outDf[RTF.VERTICES] = [ComplexXYVertices([pts], coerceListElements=True) for pts in parsePts]
    outDf[RTF.SRC_IMG_FILENAME] = inFileOrDict['metadata']['name']
    imShape = (inFileOrDict['metadata']['height'], inFileOrDict['metadata']['width'])
    self.checkVertBounds(outDf[RTF.VERTICES], imShape)
    return outDf

  def importCompImgsDf(self, inFile: FilePath, imShape: Tuple=None,
                          lblField='Instance ID', **importArgs):
    lblField = self.tableData.fieldFromName(lblField)
    inDf = pd.read_pickle(inFile)
    outDf = self.tableData.makeCompDf(len(inDf))
    outDf[RTF.INST_ID] = inDf['instId']
    allVerts = []

    for idx, row in inDf.iterrows():
      mask = row.labelMask
      verts = ComplexXYVertices.fromBwMask(mask)
      offset = row.offset
      for v in verts: v += offset
      allVerts.append(verts)
    outDf[RTF.VERTICES] = allVerts
    outDf[lblField] = inDf['label']
    self.checkVertBounds(outDf[RTF.VERTICES], imShape)
    return outDf

  def importPkl(self, inFile: FilePath, imShape: Tuple=None, **importArgs) -> pd.DataFrame:
    """
    See docstring for :func:`self.importCsv`
    """
    pklDf = pd.read_pickle(inFile)
    self.checkVertBounds(pklDf[RTF.VERTICES], imShape)
    templateDf = self.tableData.makeCompDf(len(pklDf)).set_index(pklDf.index)
    templateDf.update(pklDf)
    return templateDf

  def importLblPng(self, inFileOrImg: Union[FilePath, GrayImg],
                      lblField: Union[str, PrjParam]=None,
                      lblMapping: pd.Series=None,
                      distinctRegions=True,
                      offset=0,
                      **importArgs) -> pd.DataFrame:
    """
    Build dataframe from label image

    :param inFileOrImg: Image or image file to parse
    :param lblField: label field to associate with this image. Pixels values within the image
      correspond to values from this field in the table data. If *None*, this is inferred by the mapping read
      from the image file (see `lblMapping` description)
    :param lblMapping: For parameters that aren't numeric and don't have limits (e.g. arbitrary string values),
      this mapping determines how numeric values should be turned into field values. See `PrjParam.toNumeric` for
      details, since this is the mapping expected. If not provided, first the image metadata tags are searched for
      a 'lblMapping' text attribute (this is often added to label images saved by S3A). Note that metadata can only be
      read from the file if a file path is provided, of course. If this check fails, it is inferred based on the
      allowed options of `lblField` (`lblField.opts['limits']`). Finally, if this is not present, it is assumed the
      raw image values can be used directly as field values.
    :param offset: When `lblMapping` is not provided and field values are directly inferred from label values, this
      determines whether (and how much if not *None*) to offset numeric labels during import. I.e. if the png label
      is 1, but offset is 1, the corresponding *field* value will be 0 (1 - offset = 0).
    :param distinctRegions: Whether separate regions with the same ID should be separate IDs, or
      one ID with a group of polygons
    """
    if lblMapping is None:
      # Ignoring offset for now
      labelImg, lblMapping, _ = _getImg_mapping_offset(inFileOrImg)
    else:
      labelImg = cvImread_rgb(str(inFileOrImg), mode=cv.IMREAD_UNCHANGED)
    lblField = self.tableData.fieldFromName(lblField or lblMapping.name)
    if lblMapping is None:
      vals = lblField.opts.get('limits', None) or np.unique(labelImg)
      _, lblMapping = lblField.toNumeric(vals, returnMapping=True)
      lblMapping.index += offset
    allVerts = []
    lblField_out = []
    for numericLbl, origVal in lblMapping.iteritems():
      origVal: np.number # silence warning
      verts = ComplexXYVertices.fromBwMask(labelImg == numericLbl)
      if distinctRegions:
        newRegions = [ComplexXYVertices([v]) for v in verts]
        allVerts.extend(newRegions)
        orig = np.tile(origVal, len(verts))
      else:
        allVerts.append(verts)
        orig = [origVal]
      lblField_out.extend(orig)
    outDf = self.tableData.makeCompDf(len(allVerts))
    outDf[lblField] = lblField_out
    outDf[RTF.VERTICES] = allVerts
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
    offendingIds = np.nonzero(np.any(vertMaxs > imShape, axis=1))[0]
    if len(offendingIds) > 0:
      warnLater(f'Vertices on some components extend beyond image dimensions. '
           f'Perhaps this export came from a different image?\n'
           f'Offending IDs: {offendingIds}', UserWarning)

  def _idImgToDf(self, idImg: GrayImg):
    # Skip 0 since it's indicative of background
    regionIds = np.unique(idImg)
    regionIds = regionIds[regionIds != 0]
    allVerts = []
    for curId in regionIds:
      verts = ComplexXYVertices.fromBwMask(idImg == curId)
      allVerts.append(verts)
    outDf = self.tableData.makeCompDf(regionIds.size)
    # Subtract 1 since instance ids are 0-indexed
    outDf[RTF.INST_ID] = regionIds-1
    outDf[RTF.VERTICES] = allVerts
    return outDf


defaultIo = ComponentIO()