import typing as t
from pathlib import Path

import numpy as np
import pandas as pd

from utilitys import PrjParam
from utilitys.typeoverloads import FilePath
from .helpers import serialize, deserialize, checkVertBounds, IOTemplateManager
from ..constants import REQD_TBL_FIELDS as RTF
from ..parameditors.table import TableData
from ..structures import AnnInstanceError, AnnParseError

class AnnotationExporter:
  __name__ = None
  opts = {}

  def __init__(self, tableData: TableData=None):
    self.tableData = tableData
    # Compatibility with function analysis done in ComponentIO
    self.__name__ = f'export{type(self).__name__.replace("Exporter", "")}'

  def writeFile(self, filename: FilePath, exportObj):
    raise NotImplementedError

  def updateExportObj(self, inst: t.Any, exportObj) -> (t.Any, bool):
    try:
      exportObj.append(inst)
    except:
      return exportObj, False
    return exportObj, True

  def createExportObj(self):
    return []

  def individualExport(self, compDf: pd.DataFrame, exportObj, **kwargs):
    errs = []
    for _, row in compDf.iterrows():
      exportObj, success = self.updateExportObj(row, exportObj)
      if not success:
        errs.append(row)
    return exportObj, pd.DataFrame(errs)

  def bulkExport(self, compDf, exportObj, **kwargs):
    return exportObj

  def populateMetadata(self, **kwargs):
    self.opts.update(kwargs)

  def finalizeExport(self, exportObj, errorOk=False, **kwargs):
    raise NotImplementedError

  def __call__(self, compDf: pd.DataFrame,
               outFile: FilePath=None,
               errorOk=False,
               **kwargs):
    self.opts = {}
    self.populateMetadata(**kwargs)
    # Add new opts to kwargs
    for kk in set(self.opts).difference(kwargs):
      kwargs[kk] = self.opts[kk]

    exportObj = self.createExportObj()
    for func in self.bulkExport, self.individualExport:
      # False positive
      # noinspection PyArgumentList
      exportObj, errs = func(compDf, exportObj)
      if errs and not errorOk:
        raise ValueError('Encountered problems exporting the following annotations:\n'
                         + errs.to_string())
    exportObj = self.finalizeExport(exportObj, errorOk)
    if outFile is not None:
      self.writeFile(outFile, exportObj)
    return exportObj

class AnnotationImporter:
  __name__ = None

  importObj: t.Any
  opts = {}
  _canBulkImport = True

  def __init__(self, tableData: TableData=None, destTableMapping: TableData=None):
    """
    Provides access to a modularized version of the common import structure:

      * read a file
      * parse bulk columns, where applicable (one to many or many to one column mapping)
      * parse individual instances, where applicable (one to one column mapping)
      * apply formatting

    This is all viewable under the `__call__` function
    """
    # Compatibility with function analysis done in ComponentIO
    self.__name__ = self.__name__ or f'import{type(self).__name__.replace("Importer", "")}'
    if tableData is None:
      tableData = TableData(requiredFields=[], template=IOTemplateManager.getTableData(self.ioType))
    if destTableMapping is None:
      destTableMapping = TableData()
    self.destTableMapping = destTableMapping

    self.tableData = tableData

  @property
  def ioType(self):
    out = self.__name__.lower()
    if out.startswith('import') or out.startswith('export'):
      # Clip off the starting phrase
      out = out[6:]
    return out

  def readFile(self, filename: FilePath, **kwargs):
    raise NotImplementedError

  def populateMetadata(self, **kwargs):
    self.opts.update(kwargs)

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
          field = self.tableData.allFields[strNames.index(field)]
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
    self.opts = {}

    filename = Path(inFileOrObj) if isinstance(inFileOrObj, FilePath.__args__) else None
    if filename is not None:
      inFileOrObj = self.readFile(inFileOrObj, **kwargs)
    self.importObj = inFileOrObj

    self.populateMetadata(filename=filename, **kwargs)
    # Add new opts to kwargs
    for kk in set(self.opts).difference(kwargs):
      kwargs[kk] = self.opts[kk]

    indivParsedDf = self.individualImport(inFileOrObj, **kwargs)
    bulkParsedDf = self.bulkImport(inFileOrObj, **kwargs)

    for col in indivParsedDf:
      # Overwrite bulk-parsed information with individual if needed, or add to it
      bulkParsedDf[col] = indivParsedDf[col]
    # Some cols could be deserialized, others could be serialized still. Handle the still serialized cases
    parsedDf = self.finalizeImport(bulkParsedDf, **kwargs)
    parsedDf = self.validInstances(parsedDf, parseErrorOk)

    # Determine any destination mappings
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