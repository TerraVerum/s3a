from __future__ import annotations

import json
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Optional, Union, Tuple, Callable

import numpy as np
import pandas as pd
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from pandas.core.dtypes.missing import array_equivalent

from s3a.constants import REQD_TBL_FIELDS as RTF, PRJ_ENUMS
from s3a.shims import typing_extensions
from s3a.structures import FilePath
from utilitys import fns
from .base import AnnotationIOBase
from .exporters import *
from .helpers import compareDataframes
from .importers import *
from ..parameditors.table.data import TableData

FilePathOrDf = Union[FilePath, pd.DataFrame]
# Values are strings
# noinspection PyTypeHints
_litLst = typing_extensions.Literal[PRJ_ENUMS.IO_IMPORT, PRJ_ENUMS.IO_EXPORT]
_maybeCallable = Optional[Union[Callable, AnnotationIOBase]]


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

_defaultResizeOpts = dict(
  shape=None,
  keepAspectRatio=True,
  padVal=np.nan,
  allowReorient=True
)

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

  def __init__(self, tableData: TableData=None):
    if tableData is not None:
      self.tableData = tableData
    td = self
    self.importCsv = CsvImporter(td)
    self.importSerialized = SerialImporter(td)
    self.importLblPng = LblPngImporter(td)
    self.importPkl = PklImporter(td)
    self.importCompImgsDf = CompImgsDfImporter(td)

    self.importSuperannotateJson = SuperannotateJsonImporter(td)
    self.importGeojson = GeojsonImporter(td)
    self.importViaCsv = VGGImageAnnotatorImporter(td)

    self.exportCsv = CsvExporter()
    self.exportLblPng = LblPngExporter()
    self.exportCompImgsZip = CompImgsZipExporter()
    self.exportCompImgsDf = CompImgsDfExporter()
    self.exportPkl = PklExporter()
    self.exportSerialized = SerialExporter()

  def updateOpts(self, importOrExport: _litLst, **opts):
    # Propagate custom defaults to each desired function
    if importOrExport == PRJ_ENUMS.IO_EXPORT:
      typeDict = self.exportTypes
    else:
      typeDict = self.importTypes
    for fileExt in typeDict:
      ioFunc = self._ioFnFromFileType(fileExt, importOrExport)
      try:
        ioFunc.opts.update(**opts)
      except AttributeError:
        # Can't set opts on a regular function
        continue

  @property
  def exportTypes(self):
    """
    File types this class can export. {file type: description} useful for adding to
    file picker dialog
    """
    return _attrNameFmt(PRJ_ENUMS.IO_EXPORT, self)

  @property
  def importTypes(self):
    """
    Types this class can import. {file type: description} useful for adding to
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

  def exportByFileType(
      self,
      compDf: pd.DataFrame,
      file: Union[str, Path],
      verifyIntegrity=True,
      **exportArgs,
  ):
    file = Path(file)
    outFn = self._ioFnFromFileType(file, PRJ_ENUMS.IO_EXPORT)

    ret = outFn(compDf, file, **exportArgs)
    if verifyIntegrity and file.suffix[1:] in self.roundTripTypes:
      loadedDf = self.importByFileType(file)
      compareDataframes(compDf, loadedDf)
    return ret

  def importByFileType(
      self, inFile: Union[str, Path], imageShape: Tuple[int] = None, strColumns=False, **importArgs
  ) -> pd.DataFrame:
    buildFn = self._ioFnFromFileType(inFile, PRJ_ENUMS.IO_IMPORT)
    outDf = buildFn(inFile, imageShape=imageShape, **importArgs)
    if strColumns:
      outDf.columns = list(map(str, outDf.columns))
    return outDf

  def _ioFnFromFileType(self,
                        fpath: Union[str, Path],
                        buildOrExport=_litLst,
                        returnAttrName=False,
                        missingOk=False) -> _maybeCallable | Tuple[_maybeCallable, str]:
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
