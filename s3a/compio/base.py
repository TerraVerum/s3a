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
      self._initialOpts = copy.copy(options)
      self.opts = options

    def setInitialOpts(self, **opts):
      self._initialOpts = opts

    def resetOpts(self, clear=False):
      """
      Reset initially provided option keys to initial state without reassigning object reference.
      :param clear: If *True*, options will be completely cleared instead of reset to initial options.
      """
      if clear:
        self.opts.clear()
      else:
        for kk in self._initialOpts.keys():
          self.opts[kk] = self._initialOpts[kk]

    def populateMetadata(self, **kwargs):
      self._updateOpts(**kwargs)

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

    def _updateOpts(self, locals_=None, *keys, **kwargs):
      """
      Convenience function to update self.opts from some locals and extra keywords, since this is a common
      paradigm in `populateMetadata`
      """
      if locals_ is None:
        locals_ = {}
      keySource = kwargs.copy()
      keySource.update(locals_)

      useKeys = set(kwargs).union(self.optsMetadata()).union(keys)
      # Can only populate requested keys if they exist in the keysource
      for kk in useKeys.intersection(keySource):
        self.opts[kk] = keySource[kk]

class AnnotationExporter(AnnotationIOBase):
  exportObj: t.Any
  compDf: pd.DataFrame

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
    # Use dict combo to allow duplicate keys
    activeOpts = {**self._initialOpts, **kwargs}
    self.compDf = compDf
    self.populateMetadata(**activeOpts)
    # Add new opts to kwargs
    kwargs.update(self.opts, file=file)

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
  _canBulkImport = True

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

  def getInstances(self, importObj):
    return []

  def formatSingleInstance(self, inst, **kwargs) -> dict:
    return {}

  def finalizeImport(self, compDf, **kwargs):
      """Deserializes any columns that are still strings"""

      # Objects in the original frame may be represented as strings, so try to convert these
      # as needed
      outDf = pd.DataFrame()
      strNames = [f.name for f in self.tableData.allFields]
      for field in compDf:
        # Only handle string names that match stringified versions of existing table data names.
        if field in strNames:
          dfVals = compDf[field]
          field = self.tableData.fieldFromName(field)
          # Parsing functions only know how to convert from strings to themselves.
          # So, assume the exting types can first convert themselves to strings
          serializedDfVals = serialize(field, dfVals, returnErrs=False)
          parsedDfVals, errs = deserialize(field, serializedDfVals)
          # Turn problematic cells into instance errors for detecting problems in the outer scope
          errs = errs.apply(AnnInstanceError)
          parsedDfVals = parsedDfVals.append(errs)
          outDf[field] = parsedDfVals
        elif isinstance(field, PrjParam):
          # All other data is transcribed without modification if it's a PrjParam
          outDf[field] = compDf[field]
      # All recognized output fields should now be deserialied; make sure required fields exist
      return outDf

  def __call__(self,
               inFileOrObj: t.Union[FilePath, t.Any],
               parseErrorOk=False,
               **kwargs):
    self.refreshTableData()
    self.resetOpts()

    file = Path(inFileOrObj) if isinstance(inFileOrObj, FilePath.__args__) else None
    if file is not None:
      inFileOrObj = self.readFile(inFileOrObj, **kwargs)
    self.importObj = inFileOrObj

    self.populateMetadata(file=file, **kwargs)
    # Add new opts to kwargs
    kwargs.update(self.opts)

    indivParsedDf = self.individualImport(inFileOrObj, **kwargs)
    bulkParsedDf = self.bulkImport(inFileOrObj, **kwargs)

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
    if kwargs.get('reindex') or RTF.INST_ID not in parsedDf:
      parsedDf[RTF.INST_ID] = np.arange(len(parsedDf), dtype=int)
    parsedDf = parsedDf.set_index(RTF.INST_ID, drop=False)

    checkVertBounds(parsedDf[RTF.VERTICES], kwargs.get('imShape'))
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

  def bulkImport(self, importObj, **kwargs):
    if self._canBulkImport:
      return pd.DataFrame(self.getInstances(importObj))
    return pd.DataFrame()

  def individualImport(self, importObj, **kwargs):
    parsed = []
    for inst in self.getInstances(importObj):
      parsedInst = self.formatSingleInstance(inst, **kwargs)
      parsed.append(parsedInst)

    indivParsedDf = pd.DataFrame(parsed)
    return indivParsedDf
