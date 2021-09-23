from __future__ import annotations

import copy
import inspect
import typing as t
from pathlib import Path

import numpy as np
import pandas as pd

from utilitys import PrjParam, fns, RunOpts
from utilitys.typeoverloads import FilePath
from .helpers import serialize, deserialize, checkVertBounds
from ..constants import REQD_TBL_FIELDS as RTF
from ..generalutils import pd_iterdict
from ..parameditors.table import TableData
from ..parameditors.table.data import getFieldAliases
from ..parameditors.table.templatemgr import IOTemplateManager
from ..shims import typing_extensions
from ..structures import AnnInstanceError, AnnParseError


class TableContainer:
  """Dummy component io in case a raw tableData is directly given to an importer/exporter"""
  def __init__(self, tableData=None):
    self.tableData = tableData

class TblContainer_T(typing_extensions.Protocol):
  tableData: TableData

class _GenericExportProtocol(typing_extensions.Protocol):
  def __call__(self, compDf: pd.DataFrame, exportObj, **kwargs) -> (t.Any, pd.DataFrame):
    return exportObj, NO_ERRORS

class _UpdateExportObjProtocol(typing_extensions.Protocol):
  def __call__(self, inst: dict, exportObj, **kwargs) -> t.Any:
    return exportObj

# Alias just for better readability
NO_ERRORS = pd.DataFrame()

_exportCallable = t.Callable[[pd.DataFrame, t.Any], t.Tuple[t.Any, pd.DataFrame]]

class AnnotationIOBase:
    class UNSET_IO_TYPE:
        pass

    __name__: t.Optional[str] = None
    ioType: t.Optional[str] = None
    """Type indicating what required fields from the IOTemplateManager should be applied"""

    def __init__(self, ioType=UNSET_IO_TYPE, options=None):
      """
      Provides access to a modularized version of the common import structure:

        * read a file
        * parse bulk columns, where applicable (one to one column mapping)
        * parse individual instances, where applicable (one to many or many to one column mapping)
        * apply formatting

      This is all viewable under the `__call__` function.

      :param ioType: Determines which config template's required fields are necessary for this input. That way,
        required fields don't have to be explicitly enumerated in a project's table configuration
      :param options: Dict-like metadata for this importer/exporter. If *None*, defaults to empty option set.
        This will be updated with kwargs from being called.
      """
      # Compatibility with function analysis done in ComponentIO
      clsName = type(self).__name__
      prefix = 'import' if 'Importer' in clsName else 'export'
      fmtName = type(self).__name__.replace("Importer", "").replace("Exporter", "")
      self.__name__ = self.__name__ or f'{prefix}{fmtName}'

      useType = ioType
      if useType is self.UNSET_IO_TYPE:
          useType = self.ioType or fmtName.lower()
      self.ioType = useType

      if options is None:
        options = {}
      self.opts = options

    def populateMetadata(self, **kwargs):
      return self._forwardMetadata(**kwargs)

    @classmethod
    def optsMetadata(cls):
      """Get all metadata descriptions from self and any base class `populateMetadata`."""
      metadata = {}
      classes = [curcls for curcls in inspect.getmro(cls)
                 if issubclass(curcls, AnnotationIOBase)
                 ]
      # Reverse so most current class is last to override options
      for subcls in reversed(classes):
        parsed = fns.funcToParamDict(subcls.populateMetadata, title=fns.nameFormatter)
        curMeta = {ch['name']: ch for ch in parsed['children']
               if not ch.get('ignore', False)
                   and ch.get('value') is not RunOpts.PARAM_UNSET}
        metadata.update(curMeta)
      for kk in list(metadata.keys()):
        if kk.startswith('_'):
          del metadata[kk]
      return metadata

    def _forwardMetadata(self, locals_=None, *keys, **kwargs):
      """
      Convenience function to update __call__ kwargs from some locals and extra keywords, since this is a common
      paradigm in `populateMetadata`
      """
      if locals_ is None:
        locals_ = {}
      keySource = kwargs.copy()
      keySource.update(locals_)

      useKeys = set(kwargs).union(self.optsMetadata()).union(keys)
      # Can only populate requested keys if they exist in the keysource
      return {kk: keySource[kk] for kk in useKeys.intersection(keySource)}

class AnnotationExporter(AnnotationIOBase):
  exportObj: t.Any
  compDf: pd.DataFrame

  bulkExport: _GenericExportProtocol = None
  """
  Can be defined if bulk-exporting (whole dataframe at once) is possible. Must have the signature
  def bulkExport(self, compDf: pd.DataFrame, exportObj, **kwargs) -> exportObj, error dataframe
  """

  updateExportObj: _UpdateExportObjProtocol = None
  """
  Can be defined if individual importing (row-by-row) is possible. This is fed the current dataframe row as a dict
  of cell values and is expected to output the updated export object:
  def updateExportObj(self, inst: dict, exportObj, **kwargs) -> exportObj
  """

  class ERROR_COL: pass
  """Sentinel class to add errors to an explanatory message during export"""

  def writeFile(self, file: FilePath, exportObj, **kwargs):
    raise NotImplementedError

  def createExportObj(self, **kwargs):
    raise NotImplementedError

  def individualExport(self, compDf: pd.DataFrame, exportObj, **kwargs):
    """Returns an export object + dataframe of row + errors, if any occurred for some rows"""
    if self.updateExportObj is None:
      # Can't do anything, don't modify the object and save time not iterating over rows
      return exportObj, NO_ERRORS
    errs = {}
    for row in pd_iterdict(compDf):
      try:
        exportObj = self.updateExportObj(row, exportObj)
      except Exception as err:
        errs.update(row)
        errs[self.ERROR_COL] = err
    return exportObj, pd.DataFrame(errs)

  def formatReturnObj(self, exportObj, **kwargs):
    # If metadata options change return behavior, that can be resolved here.
    return exportObj

  def __call__(self,
               compDf: pd.DataFrame,
               file: FilePath=None,
               errorOk=False,
               **kwargs):
    file = Path(file) if isinstance(file, FilePath.__args__) else None
    self.compDf = compDf

    kwargs.update(file=file)
    activeOpts = {**self.opts, **kwargs}
    meta = self.populateMetadata(**activeOpts)
    kwargs.update(**meta)

    exportObj = self.createExportObj(**kwargs)
    for func in self.bulkExport, self.individualExport: # type: _GenericExportProtocol
      if func is None:
        continue
      exportObj, errs = func(compDf, exportObj, **kwargs)
      if len(errs) and not errorOk:
        raise ValueError('Encountered problems exporting the following annotations:\n'
                         + errs.to_string())
    self.exportObj = exportObj
    if file is not None:
      self.writeFile(kwargs.pop('file'), exportObj, **kwargs)
    toReturn = self.formatReturnObj(exportObj, **kwargs)
    return toReturn

class AnnotationImporter(AnnotationIOBase):
  importObj: t.Any

  formatSingleInstance = None
  """
  Can be defined to cause row-by-row instance parsing. If defined, must have the signature:
  ``def formatSingleInstance(self, inst, **kwargs) -> dict``
  """

  bulkImport = None
  """
  Can be defined to parse multiple traits from the imported object into a component dataframe all at once. Must have
  the signature:
  ``def bulkImport(self, importObj, **kwargs) -> pd.DataFrame``
  Note that in some cases, a direct conversion of instances to a dataframe is convenient, so ``defaultBulkImport``
  is provided for these cases. Simply set bulkImport = ``AnnotationImporter.defaultBulkImport`` if you wish.
  """

  def __init__(self, tableData: TableData | TblContainer_T = None, ioType=AnnotationIOBase.UNSET_IO_TYPE):
    """
    Provides access to a modularized version of the common import structure:

      * read a file
      * parse bulk columns, where applicable (one to one column mapping)
      * parse individual instances, where applicable (one to many or many to one column mapping)
      * apply formatting

    This is all viewable under the `__call__` function.

    :param tableData: Table configuration for fields in the input file. If a container, ``container.tableData`` leads to
      the table data. This allows references to be reassigned in e.g. an outer ComponentIO without losing connection
      to this importer
    :param ioType: Determines which config template's required fields are necessary for this input. That way,
      required fields don't have to be explicitly enumerated in a project's table configuration
    """

    # Make a copy to allow for internal changes such as adding extra required fields, aliasing, etc.
    # 'and' avoids asking for 'cfg' of 'none' table
    super().__init__(ioType=ioType)
    if isinstance(tableData, TableData):
      container = TableContainer(tableData)
    else:
      container = tableData
    self.container = container
    self.tableData = TableData()
    self.destTableMapping = self.container.tableData
    self.refreshTableData()

  def refreshTableData(self):
      self.destTableMapping = tableData = self.container.tableData
      requiredCfg = IOTemplateManager.getTableCfg(self.ioType)
      if tableData is not None:
          # Make sure not to incorporate fields that only exist to provide logistics for the other table setup
          optionalFields = {key: val for key, val in tableData.cfg['fields'].items() if
                            key not in tableData.template['fields']}
          optionalCfg = {'fields': optionalFields}
      else:
          optionalCfg = None
      self.tableData.template = requiredCfg
      self.tableData.loadCfg(cfgDict=optionalCfg)

  def readFile(self, file: FilePath, **kwargs):
    raise NotImplementedError

  def getInstances(self, importObj, **kwargs):
    raise NotImplementedError

  @staticmethod
  def _findSrcFieldForDest(destField, allSourceFields):
    """
    Helper function during ``finalizeImport`` to find a match between a yet-to-serialize dataframe and
    destination tableData. Basically, a more primitive version of ``resolveFieldAliases``
    Returns *None* if no sensible mapping could be found, and errs if multiple sources alias to the same destination
    """
    # Check for destination aliases primitively (one-way mappings). A full (two-way) check will occur
    # later (see __call__ -> resolveFieldAliases)
    match = tuple(getFieldAliases(destField) & allSourceFields)
    if not match:
      return
    if len(match) == 1:
      srcField = match[0]
    else:
      # Make sure there aren't multiple aliases, since this is not easily resolvable
      # The only exception is that direct matches trump alias matches, so check for this directly
      if destField.name in match:
        srcField = destField.name
      else:
        raise IndexError(f'Multiple aliases to "{destField}": {match}\n'
                         f'Cannot determine appropriate column matchup.')
    return srcField

  def finalizeImport(self, compDf, **kwargs):
      """Deserializes any columns that are still strings"""

      # Objects in the original frame may be represented as strings, so try to convert these
      # as needed
      outDf = pd.DataFrame()
      # Preserve / transcribe fields that are already PrjParams
      for destField in [f for f in compDf.columns if isinstance(f, PrjParam)]:
        outDf[destField] = compDf[destField]

      # Need to serialize / convert string names since they indicate yet-to-serialize columns
      toConvert = set(compDf.columns)
      for destField in self.tableData.allFields:
        srcField = self._findSrcFieldForDest(destField, toConvert)
        if not srcField:
          # No match
          continue
        dfVals = compDf[srcField]
        # Parsing functions only know how to convert from strings to themselves.
        # So, assume the exting types can first convert themselves to strings
        serializedDfVals = serialize(destField, dfVals, returnErrs=False)
        parsedDfVals, errs = deserialize(destField, serializedDfVals)
        # Turn problematic cells into instance errors for detecting problems in the outer scope
        errs = errs.apply(AnnInstanceError)
        parsedDfVals = parsedDfVals.append(errs)
        outDf[destField] = parsedDfVals
      # All recognized output fields should now be deserialied; make sure required fields exist
      return outDf

  def __call__(self,
               inFileOrObj: t.Union[FilePath, t.Any],
               parseErrorOk=False,
               reindex=False,
               **kwargs):
    self.refreshTableData()

    file = Path(inFileOrObj) if isinstance(inFileOrObj, FilePath.__args__) else None
    if file is not None:
      inFileOrObj = self.readFile(inFileOrObj, **kwargs)
    self.importObj = inFileOrObj

    kwargs.update(file=file, reindex=reindex)
    activeOpts = {**self.opts, **kwargs}
    meta = self.populateMetadata(**activeOpts)
    kwargs.update(meta)

    parsedDfs = []
    for func in self.individualImport, self.bulkImport: # type: t.Callable[[t.Any, ...], pd.DataFrame]
      if func is None:
        parsedDfs.append(pd.DataFrame())
      else:
        parsedDfs.append(func(inFileOrObj, **kwargs))

    indivParsedDf, bulkParsedDf = parsedDfs
    for col in indivParsedDf:
      # Overwrite bulk-parsed information with individual if needed, or add to it
      bulkParsedDf[col] = indivParsedDf[col]
    # Some cols could be deserialized, others could be serialized still. Handle the still serialized cases
    parsedDf = self.finalizeImport(bulkParsedDf, **kwargs)
    parsedDf = self.validInstances(parsedDf, parseErrorOk)

    # Determine any destination mappings
    if self.destTableMapping:
      parsedDf.columns = self.destTableMapping.resolveFieldAliases(parsedDf.columns, kwargs.get('mapping', {}))

    # Make sure IDs are present
    if reindex or RTF.INST_ID not in parsedDf:
      parsedDf[RTF.INST_ID] = np.arange(len(parsedDf), dtype=int)
    parsedDf = parsedDf.set_index(RTF.INST_ID, drop=False)

    checkVertBounds(parsedDf[RTF.VERTICES], kwargs.get('imageShape'))
    return parsedDf

  @classmethod
  def validInstances(cls, parsedDf: pd.DataFrame, parseErrorOk=False):
    errIdxs = parsedDf.apply(lambda row: any(isinstance(vv, AnnInstanceError) for vv in row),
                             axis=1)
    errIdxs = errIdxs.to_numpy()
    if not np.any(errIdxs):
      return parsedDf
    validInsts = parsedDf[~errIdxs]
    invalidInsts = parsedDf[errIdxs]
    if not parseErrorOk:
      raise AnnParseError(
        f'Encountered problems on annotation import:\n{invalidInsts.to_string()}')
    return validInsts

  def defaultBulkImport(self, importObj, **kwargs) -> pd.DataFrame:
    return pd.DataFrame(self.getInstances(importObj, **kwargs))

  def individualImport(self, importObj, **kwargs):
    parsed = []
    if self.formatSingleInstance is None:
      return pd.DataFrame()
    for inst in self.getInstances(importObj, **kwargs):
      parsedInst = self.formatSingleInstance(inst, **kwargs)
      parsed.append(parsedInst)

    indivParsedDf = pd.DataFrame(parsed)
    return indivParsedDf
