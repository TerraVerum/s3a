import functools
import typing as t
from functools import wraps

import numpy as np
import pandas as pd

from utilitys import PrjParam
from utilitys.typeoverloads import FilePath
from .helpers import serialize, deserialize, checkVertBounds
from ..constants import REQD_TBL_FIELDS as RTF, IO_TEMPLATES_DIR
from ..parameditors.table import TableData
from ..structures import AnnInstanceError, AnnParseError
from pathlib import  Path

# Turn mapping into a dict that knows how to find either strings or fields
class _FancyMap(dict):
  def __getitem__(self, item):
    assert isinstance(item, PrjParam)
    for key in item, item.name:
      if super().__contains__(key):
        return self[key]
    return None


def _destNameForSrcName(srcField, destTbl, mapping: _FancyMap):
  potentialNames = lambda _field: set([_field.name] + _field.opts.get('aliases', []))

  # Mapping takes priority, if it exists
  outCol = mapping[srcField]
  if outCol is not None:
    return destTbl.fieldFromName(outCol)
  elif srcField in destTbl.allFields:
    # Exact match, don't rename
    return srcField
  else:
    # Not in mapping, no exact match. TODO: what if multiple dest cols have a matching alias?
    # Otherwise, a 'break' can be added
    curOutName = srcField
    for destField in destTbl.allFields:
      if potentialNames(srcField) & potentialNames(destField):
        # Match between source field's aliases and dest field aliases
        # Make sure it didn't match multiple names that weren't itself with the assert statement
        assert curOutName == srcField
        curOutName = destField
  return curOutName


def renameDfColsFromAliases(compDf: pd.DataFrame,
                            destTbl: TableData,
                            mapping: dict=None
                            ):
    """
    Several forms of imports / exports handle data that may not be compatible with the current table data.
    In these cases, it is beneficial to determine a mapping between names to allow greater compatibility between
    I/O formats. Mapping is also extended in both directions by parameter name aliases (param.opts['aliases']),
    which are a list of strings of common mappings for that parameter (e.g. [Class, Label] are often used
    interchangeably)

    :param compDf: Dataframe with maybe foreign fields
    :param destTbl: Destination table from the ComponentIO importer
    :param mapping: Foreign to local field name mapping
    """

    mapping = _FancyMap(mapping or {})

    outCols = []
    for srcField in compDf.columns:
      outCols.append(_destNameForSrcName(srcField, destTbl, mapping))
    compDf.columns = outCols


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

  imageShape: tuple = None
  tableData = TableData()
  importObj: t.Any
  opts = {}
  _canBulkImport = True

  def __init__(self, destTableData: TableData=None):
    self.destTableData = destTableData
    # Compatibility with function analysis done in ComponentIO
    self.__name__ = self.__name__ or f'import{type(self).__name__.replace("Importer", "")}'

  def readFile(self, filename: FilePath, **kwargs):
    raise NotImplementedError

  def populateMetadata(self, **kwargs):
    self.opts.update(kwargs)

  def getInstances(self, importObj):
    return []

  def formatSingleInstance(self, inst, **kwargs) -> dict:
    return {}

  @classmethod
  def getForeignTableData(cls, ioFuncName: str, templateCfg=None):
    """Returns a TableData object responsible for importing / exporting data of this format"""
    name = ioFuncName.lower().replace('import', '').replace('export', '')
    cfgName = IO_TEMPLATES_DIR / (name + '.tblcfg')
    if not cfgName.exists():
      return None
    out = TableData(cfgName, template=templateCfg)
    return out

  def finalizeImport(self, compDf, reindex=False, **kwargs):
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
      # All recognized output fields should now be deserialied
      if reindex or RTF.INST_ID not in outDf:
        outDf[RTF.INST_ID] = np.arange(len(outDf), dtype=int)
      return outDf.set_index(RTF.INST_ID, drop=False)

  def __call__(self,
               inFileOrObj: t.Union[FilePath, t.Any],
               parseErrorOk=False,
               **kwargs):
    destTableData = self.destTableData
    if destTableData is None:
      destTableData = TableData()
    self.opts = {}
    srcTableData = self.getForeignTableData(self.__name__, destTableData.cfg)
    if srcTableData is not None:
      self.tableData = srcTableData
    else:
      self.tableData = destTableData

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

    # Finally, deal with any name aliases
    renameDfColsFromAliases(parsedDf, destTableData, kwargs.get('mapping'))

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
    for ii, inst in enumerate(self.getInstances(importObj)):
      parsedInst = self.formatSingleInstance(inst, **kwargs)
      parsedInst[RTF.INST_ID] = ii
      parsed.append(parsedInst)

    indivParsedDf = pd.DataFrame(parsed)
    return indivParsedDf