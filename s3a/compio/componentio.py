from __future__ import annotations

import errno
import inspect
import json
import os
import pickle
import sys
import tempfile
import warnings
from collections import defaultdict
from contextlib import ExitStack
from functools import wraps
from pathlib import Path
from stat import S_IREAD, S_IRGRP, S_IROTH
from typing import Any, Optional, Union, Tuple, Callable
from zipfile import ZipFile

import numpy as np
import pandas as pd
from pandas.core.dtypes.missing import array_equivalent
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from s3a.constants import REQD_TBL_FIELDS as RTF, PRJ_ENUMS
from s3a.generalutils import augmentException, getCroppedImg, resize_pad, cvImsave_rgb, classproperty, \
  cvImread_rgb, imgPathtoHtml, deprecateKwargs, DirectoryDict
from s3a.parameditors.table import TableData
from s3a.shims import typing_extensions
from s3a.structures import FilePath, ComplexXYVertices, PrjParam
from skimage.exposure import rescale_intensity
from utilitys import fns

from .helpers import serialize, _getPdExporters, _getPdImporters
from .importers import *

FilePathOrDf = Union[FilePath, pd.DataFrame]
# Values are strings
# noinspection PyTypeHints
_litLst = typing_extensions.Literal[PRJ_ENUMS.IO_IMPORT, PRJ_ENUMS.IO_EXPORT]
_maybeCallable = Optional[Callable]


def _attrNameFmt(buildOrExport: _litLst, obj):
  def membership(el):
    for exclude in ['FileType', 'Opts', 'Types', 'Serialized']:
      if el.endswith(exclude):
        return False
    return el.startswith(buildOrExport)

  attrs = [attr.replace(buildOrExport, '') for attr in list(vars(obj)) + list(vars(type(obj))) if membership(attr)]
  out = {}
  for attr in attrs:
    descrFmt = fns.pascalCaseToTitle(attr)
    fileFmt = descrFmt.replace(' ', '.').lower()
    out[fileFmt] = descrFmt + ' Files'
  return out


def _writeImge_meta(
    outImg: np.ndarray,
    saveName: FilePath,
    mapping: pd.Series = None,
    offset: int = None,
):
  outImg = Image.fromarray(outImg)
  info = PngInfo()
  if mapping is not None:
    info.add_text('mapping', json.dumps(mapping.to_dict()))
    info.add_text('field', str(mapping.name))
  if offset is not None:
    info.add_text('offset', str(offset))
  outImg.save(saveName, pnginfo=info)

class ComponentIO:
  """
  Exporter responsible for saving Component information to a file or object.
  Once created, users can extract different representations of components by
  calling exporter.exportCsv, exportPkl, etc. for those objects / files respectively.
  """
  @property
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
    td = self.tableData
    self.importCsv = CsvImporter(td)
    self.importSuperannotateJson = SuperannotateJsonImporter(td)
    self.importGeojson = GeojsonImporter(td)
    self.importSerialized = SerialImporter(td)
    self.importLblPng = LblPngImporter(td)
    self.importPkl = PklImporter(td)
    self.importCompImgsDf = CompImgsDfImporter(td)

    # Propagate custom defaults to each desired function
    for typeDict, fnType in zip([self.importTypes, self.exportTypes],
                                [PRJ_ENUMS.IO_IMPORT, PRJ_ENUMS.IO_EXPORT]):
      for fileExt in typeDict:
        func, name = self._ioFnFromFileType(fileExt, fnType, returnAttrName=True)
        setattr(self, name, self._ioWrapper(func, name))

  @property
  def exportTypes(self):
    """
    File types this class can export. {file type: descriptoin} useful for adding to
    file picker dialog
    """
    return _attrNameFmt(PRJ_ENUMS.IO_EXPORT, self)

  @property
  def importTypes(self):
    """
    Types this class can import. {file type: descriptoin} useful for adding to
    file picker dialog
    """
    return _attrNameFmt(PRJ_ENUMS.IO_IMPORT, self)

  def ioFileFilter(
      self,
      which=PRJ_ENUMS.IO_ROUND_TRIP,
      typeFilter='',
      allFilesOpt=True,
      **extraOpts,
  ):
    """
    Helper for creating a file filter out of the handled IO types. The returned list of
    strings is suitable for inserting into a QFileDialog.

    :param which: Whether to generate filters for build types, export types, or round trip
    :param typeFilter: type filter for handled io types. For instanece, if typ='png', then
      a file filter list with only 'id.png' and 'class.png' will appear.
    :param allFilesOpt: Whether to add an "All Files (*.*) option to the dropdown menu
    :param extraOpts; Extra file types to include in the filter
    """
    ioDict = {
        PRJ_ENUMS.IO_ROUND_TRIP: self.roundTripTypes,
        PRJ_ENUMS.IO_IMPORT: self.importTypes,
        PRJ_ENUMS.IO_EXPORT: self.exportTypes
    }[which]
    if isinstance(typeFilter, str):
      typeFilter = [typeFilter]
    fileFilters = []
    for typ, info in dict(**ioDict, **extraOpts).items():
      if any([t in typ for t in typeFilter]):
        fileFilters.append(f'{info} (*.{typ})')
    if allFilesOpt:
      fileFilters.append('All Files (*.*)')
    return ';;'.join(fileFilters)

  def _ioWrapper(self, func: Callable, overrideName=None):
    """Wraps build and export functions to provide defaults specified by build/exportOpts before the function call"""
    checkName = overrideName or func.__name__
    which = PRJ_ENUMS.IO_IMPORT if checkName.startswith(
        PRJ_ENUMS.IO_IMPORT
    ) else PRJ_ENUMS.IO_EXPORT

    @wraps(func)
    def wrapper(*args, **kwargs):
      useOpts = self.importOpts if which is PRJ_ENUMS.IO_IMPORT else self.exportOpts
      # Turn args into kwargs
      fnKwargs = {**useOpts, **kwargs}
      return func(*args, **fnKwargs)

    return wrapper

  def exportByFileType(
      self,
      compDf: pd.DataFrame,
      outFile: Union[str, Path],
      verifyIntegrity=True,
      **exportArgs,
  ):
    outFile = Path(outFile)
    outFn = self._ioFnFromFileType(outFile, PRJ_ENUMS.IO_EXPORT)

    ret = outFn(compDf, outFile, **exportArgs)
    if verifyIntegrity and outFile.suffix[1:] in self.roundTripTypes:
      matchingCols = np.setdiff1d(compDf.columns, [RTF.INST_ID, RTF.SRC_IMG_FILENAME])
      loadedDf = self.importByFileType(outFile)
      # For some reason, there are cases in which all values truly are equal but np.array_equal,
      # x.equals(y), x.eq(y), etc. all fail. Something to do with block ordering?
      # https://github.com/pandas-dev/pandas/issues/9330 indicates it should be fixed, but the error still occasionally
      # happens for me. array_equivalent is not affected by this, in testing so far
      dfCmp = array_equivalent(
        loadedDf[matchingCols].values, compDf[matchingCols].values
      )
      problemCells = defaultdict(list)

      if not dfCmp:
        dfA = loadedDf[matchingCols]
        dfB = compDf[matchingCols]
        for ii in range(len(dfA)):
          for jj in range(len(dfA.columns)):
            if not np.array_equal(dfA.iat[ii, jj], dfB.iat[ii, jj]):
              problemCells[compDf.at[dfB.index[ii], RTF.INST_ID]].append(str(matchingCols[jj]))
        # The only way to prevent "truth value of array is ambiguous" is cell-by-cell iteration
        problemMsg = [f'{idx}: {cols}' for idx, cols in problemCells.items()]
        problemMsg = '\n'.join(problemMsg)
        # Try to fix the problem with an iloc write
        warnings.warn(
            '<b>Warning!</b> Saved components do not match current component'
            ' state. This can occur when pandas incorrectly caches some'
            ' table values. Problem cells (shown as [id]: [columns]):\n' + f'{problemMsg}\n'
            f'Please try manually altering these values before exporting again.', UserWarning
        )
    return ret

  def importByFileType(
      self, inFile: Union[str, Path], imShape: Tuple[int] = None, strColumns=False, **importArgs
  ):
    buildFn = self._ioFnFromFileType(inFile, PRJ_ENUMS.IO_IMPORT)
    outDf = buildFn(inFile, imShape=imShape, **importArgs)
    if strColumns:
      outDf.columns = list(map(str, outDf.columns))
    return outDf

  def _ioFnFromFileType(self,
                        fpath: Union[str, Path],
                        buildOrExport=_litLst,
                        returnAttrName=False,
                        missingOk=False) -> Union[_maybeCallable, Tuple[_maybeCallable, str]]:
    fpath = Path(fpath)
    fname = fpath.name
    cmpTypes = np.array(list(_attrNameFmt(buildOrExport, self)))
    typIdx = [fname.endswith(typ) for typ in cmpTypes]
    if not any(typIdx):
      raise IOError(f'Not sure how to handle file {fpath.name}')
    fnNameSuffix = cmpTypes[typIdx][-1].title().replace('.', '')
    attrName = buildOrExport + fnNameSuffix
    outFn: Callable = getattr(self, attrName, None)
    if outFn is None and not missingOk:
      raise ValueError(f'Full I/O specification missing for type {fnNameSuffix}')
    if returnAttrName:
      return outFn, attrName
    return outFn

  def _pandasSerialExport(
      self,
      exportDf: pd.DataFrame,
      outFile: Union[str, Path] = None,
      readOnly=True,
      **pdExportArgs
  ):
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
      raise ValueError(
          f'Exporter {exporter} not recognized. Acceptable options:\n' +
          ', '.join(_getPdExporters())
      )

    with np.printoptions(threshold=sys.maxsize):
      exportFn(outFile, index=False)
    if readOnly:
      outPath.chmod(S_IREAD | S_IRGRP | S_IROTH)

  # -----
  # Export options
  # -----
  def exportSerialized(
      self, compDf: pd.DataFrame, outFile: Union[str, Path] = None, readOnly=True, **pdExportArgs
  ):
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
          serial, errs = serialize(col, exportDf[col])
          exportDf[col] = serial.to_numpy()
          if len(errs):
            raise ValueError(f'Encountered errors on the following rows:\n{errs.to_string()}')
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

  @deprecateKwargs(imgDir='srcDir')
  def exportCompImgsDf(
      self,
      compDf: pd.DataFrame,
      outFile: Union[str, Path] = None,
      *,
      srcDir: Union[FilePath, dict, DirectoryDict] = None,
      margin=0,
      marginAsPct=False,
      includeCols=('instId', 'img', 'labelMask', 'label', 'offset'),
      lblField='Instance ID',
      asIndiv=False,
      returnLblMapping=False,
      missingOk=False,
      **kwargs
  ):
    """
    Creates a dataframe consisting of extracted images around each component
    :param compDf: Dataframe to export
    :param outFile: Where to save the result, if it should be saved. Caution -- this
      is currently a time-consuming process!
    :param srcDir: Where images corresponding to this dataframe are kept. Source image
      filenames are interpreted relative to this directory if they are not absolute. Alternatively, can be a dict
      of {name: np.ndarray} image mappings
    :param margin: How much padding to give around each component
    :param marginAsPct: Whether the margin should be a percentage of the component size or
      a raw pixel value.
    :param includeCols: Which columns to include in the export list
    :param lblField: See ComponentIO.exportLblPng. This label is provided in the output dataframe
      as well, if specified.
    :param asIndiv: Whether components should be exported as individual units (i.e.
      neighbors are guaranteed never to show up in the ground truth label mask) or whether
      a cropped area around the component should be used as the label mask. This will
      include mask values of neighbors if they are within mask range.
      Note: When false, the mask is created with preference toward higher ids, i.e. if a high ID is on
      top of a low ID, the low ID will still be covered in its export mask
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
    if srcDir is None:
      srcDir = Path('.')
    srcDir = DirectoryDict(srcDir, allowAbsolute=True, readFunc=cvImread_rgb)
    uniqueImgs = np.unique(compDf[RTF.SRC_IMG_FILENAME])
    dfGroupingsByImg = []
    for imgName in uniqueImgs:
      dfGroupingsByImg.append(compDf[compDf[RTF.SRC_IMG_FILENAME] == imgName])

    useKeys = set(includeCols)
    outDf = {k: [] for k in useKeys}
    lblField = self.tableData.fieldFromName(lblField)
    # imshape is automatically inferred by the exporter
    kwargs.pop('imShape', None)
    mappings = {}
    for miniDf, fullImgName in zip(dfGroupingsByImg, uniqueImgs):
      img = srcDir.get(fullImgName)
      if img is None and not missingOk:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), fullImgName)
      shape = img if img is None else img.shape[:2]
      lblImg, mapping = self.exportLblPng(
          miniDf, imShape=shape, lblField=lblField, returnLblMapping=True, **kwargs
      )
      mappings[Path(fullImgName).name] = mapping
      if img is None:
        img = np.zeros_like(lblImg)
      invertedMap = pd.Series(mapping.index, mapping)

      for ii, (idx, row) in enumerate(miniDf.iterrows()):
        allVerts = row[RTF.VERTICES].stack()
        if marginAsPct:
          compImgSz = allVerts.max(0) - allVerts.min(0)
          marginToUse = (compImgSz * (margin/100)).astype(int)
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
        xyOffset = bounds[0, :]
        if lblImg.ndim > 2:
          indexer += (..., )
        if 'labelMask' in useKeys:
          mask = lblImg[indexer]
          if asIndiv:
            # Only color in this id's region. Since it might've been covered over by a different ID, regenerate the
            # mask from vertices, taking margin into account
            colorVerts = row[RTF.VERTICES].removeOffset(bounds[0, :])
            colorVerts.hierarchy = row[RTF.VERTICES].hierarchy
            mask = np.full_like(mask, bgColor)
            # 'Float' works for int and float orig values, required since opencv complains about some numpy
            # dtypes
            colorVerts.toMask(mask, float(invertedMap[lbl]), asBool=False)

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

  def exportPkl(
      self,
      compDf: pd.DataFrame,
      outFile: Union[str, Path] = None,
      **exportArgs,
  ) -> (Any, str):
    """
    See the function signature for :func:`exportCsv <ComponentIO.exportCsv>`
    """
    # Since the write-out is a single operation there isn't an intermediate form to return
    pklDf = None
    if outFile is not None:
      pklDf = pickle.dumps(compDf)
      compDf.to_pickle(outFile)
    return pklDf

  def exportLblPng(
      self,
      compDf: pd.DataFrame,
      outFile: FilePath = None,
      imShape: Tuple[int] = None,
      lblField: Union[PrjParam, str] = 'Instance ID',
      bgColor=0,
      rescaleOutput=False,
      returnLblMapping=False,
      writeMeta=True,
      **kwargs
  ):
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
        mapping.index = rescale_intensity(
            mapping.index, in_range=(bgColor, max_), out_range='uint16'
        )
      else:
        max_ = np.max(labels_numeric, initial=bgColor)
      labels_numeric = rescale_intensity(
          labels_numeric, in_range=(bgColor, max_), out_range='uint16'
      )

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

  def exportCompImgsZip(
      self,
      compDf: pd.DataFrame,
      outDir: FilePath = 's3a-export',
      resizeShape: Tuple[int, int] = None,
      archive=False,
      makeSummary=False,
      **kwargs
  ):
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
    :param makeSummary: Whether to include an html table showing each component from the dataframe along with
      its image and mask representations
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
      dataDir = useDir / 'data'
      labelsDir = useDir / 'labels'
      dataDir.mkdir(exist_ok=True, parents=True)
      labelsDir.mkdir(exist_ok=True, parents=True)

      summaryName = useDir / 'summary.html'

      extractedImgs = self.exportCompImgsDf(compDf, None, **kwargs)
      for idx, row in extractedImgs.iterrows():
        saveName = f'{row.instId}.png'
        if 'img' in row.index:
          saveFn(dataDir / saveName, row.img)
        if 'labelMask' in row.index:
          saveFn(labelsDir / saveName, row.labelMask)

      if makeSummary:
        extractedImgs = extractedImgs.rename({'instId': RTF.INST_ID.name}, axis=1)
        # Prevent merge error by renaming index
        # INST_ID.name has to be used instead of raw INST_ID due to strange pandas issue
        # throwing a TypeError: keywords must be a string
        outDf: pd.DataFrame = compDf.drop([RTF.VERTICES], axis=1).rename(str, axis=1)
        outDf = outDf.merge(
          extractedImgs, on=RTF.INST_ID.name
        )
        for colName, imgDir in zip(['labelMask', 'img'], [labelsDir, dataDir]):
          if colName not in extractedImgs:
            continue
          relDir = imgDir.relative_to(useDir)
          outDf[colName] = outDf[RTF.INST_ID.name].apply(
              lambda el: imgPathtoHtml((relDir / str(el)).with_suffix('.png').as_posix())
          )
        outDf.columns = list(map(str, outDf.columns))
        outDf.to_html(summaryName, escape=False, index=False)

      if archive:
        if outDir.suffix != '.zip':
          outDir = outDir.with_suffix(outDir.suffix + '.zip')
        with ZipFile(outDir, 'w') as ozip:
          for dir_ in labelsDir, dataDir:
            if not dir_.exists():
              continue
            for file in dir_.iterdir():
              ozip.write(file, f'{dir_.name}/{file.name}')
          if makeSummary:
            ozip.write(summaryName, file.name)

  # -----
  # Import options
  # -----
  def convert(
      self,
      fromData: FilePathOrDf,
      toFile: FilePath,
      doExport=True,
      importArgs: dict = None,
      exportArgs: dict = None
  ):
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

defaultIo = ComponentIO()