import copy
from pathlib import Path
from typing import Union, Dict, Callable, Any, Optional, List, Sequence

import numpy as np
import pyqtgraph as pg
from pandas import DataFrame as df
from pyqtgraph import QtCore

from utilitys import PrjParam, fns
from utilitys.fns import hierarchicalUpdate
from utilitys.typeoverloads import FilePath
from .filter import TableFilterEditor
from .templatemgr import IOTemplateManager
from .yamlparser import YamlParser
from ...constants import REQD_TBL_FIELDS as RTF
from ...structures import PrjParamGroup


def getFieldAliases(field: PrjParam):
  """
  Returns the set of all potential aliases to a given field
  """
  return set([field.name] + field.opts.get('aliases', []))


def aliasesToRequired(field: PrjParam):
  """
  Returns true or false depending on whether this field shares aliases with required
  fields. This is useful when an alternative (incoming) representation of e.g. Vertices
  must be suppressed on import, but still used on export
  """
  requiredAliases = set()
  for reqdField in RTF:
    requiredAliases.update(getFieldAliases(reqdField))
  srcAliases = getFieldAliases(field)
  return srcAliases & requiredAliases


class TableData(QtCore.QObject):
  sigCfgUpdated = QtCore.Signal(object)
  """dict (self.cfg) during update"""

  def __init__(self,
               cfgFname: FilePath=None,
               cfgDict: dict=None,
               template: Union[FilePath, dict]=None,
               makeFilter=False):
    super().__init__()
    if template is None:
      template = IOTemplateManager.getTableCfg('s3a')
    self.template = template

    self.factories: Dict[PrjParam, Callable[[], Any]] = {}

    if makeFilter:
      self.filter = TableFilterEditor()
    else:
      self.filter = None
    self.paramParser: Optional[YamlParser] = None

    self.cfgFname: Optional[Path] = None
    self.cfg: Optional[dict] = {}

    self.allFields: List[PrjParam] = []
    self.resetLists()

    cfgFname = cfgFname or None
    self.loadCfg(cfgFname, cfgDict, force=True)

  def makeCompDf(self, numRows=1, sequentialIds=False) -> df:
    """
    Creates a dataframe for the requested number of components.
    This is the recommended method for component instantiation prior to table insertion.
    """
    df_list = []
    dropRow = False
    if numRows <= 0:
      # Create one row and drop it, which ensures data types are correct in the empty
      # dataframe
      numRows = 1
      dropRow = True
    populators = []
    for f in self.allFields:
      if f in self.factories:
        val = self.factories[f]()
      else:
        val = f.value
      populators.append(val)

    for _ in range(numRows):
      # Make sure to construct a separate component instance for
      # each row no objects have the same reference
      df_list.append(copy.copy(populators))
    outDf = df(df_list, columns=self.allFields)
    if RTF.INST_ID in self.allFields:
      if sequentialIds:
        outDf[RTF.INST_ID] = np.arange(len(outDf))
      outDf = outDf.set_index(RTF.INST_ID, drop=False)
    if dropRow:
      outDf = outDf.iloc[0:0]
    return outDf

  def addFieldFactory(self, fieldLbl: PrjParam, factory: Callable[[], Any]):
    """
    For fields that are simple functions (i.e. don't require input from the user), a
    factory can be used to create default values when instantiating new table rows.

    :param fieldLbl: WHich field this factory is used for instead of just the default value
    :param factory: Callable to use instead of field value. This is called with no parameters.
    """
    self.factories[fieldLbl] = factory

  def addField(self, field: PrjParam):
    """
    Adds a new field to the table. If the field already exists in the current table, no action is performed.
    Returns *True* if a field really was added, *False* if this field is already in the table list or aliases to
    an existing field
    """

    # Problems occur when fields alias to already existing ones. When this is the case, ignore the extra fields.
    # Not only does this solve the many-to-one alias issue, but also allows table datas with different required
    # fields to seamlessly share and swap fields with eachother while avoiding vestigial table columns
    if field in self.allFields or self._findMatchingField(field) is not field:
      return False
    field.group = self.allFields
    self.allFields.append(field)
    return True

  def makeCompSer(self):
    return self.makeCompDf().squeeze()

  def loadCfg(self, cfgFname: FilePath=None,
              cfgDict: dict=None,
              force=False):
    """
    Lodas the specified table configuration file for S3A. Alternatively, a name
    and dict pair can be supplied instead.
    :param cfgFname: If *cfgDict* is *None*, this is treated as the file containaing
      a YAML-compatible table configuration dictionary. Otherwise, this is the
      configuration name assiciated with the given dictionary.
    :param cfgDict: If not *None*, this is the config data used instad of
      reading *cfgFname* as a file.
    :param force: If *True*, the new config will be loaded even if it is the same name as the
    current config
    """
    baseCfgDict = copy.deepcopy(self.template)
    if cfgFname is not None:
      cfgFname, cfgDict = fns.resolveYamlDict(cfgFname, cfgDict)
      cfgFname = cfgFname.resolve()
    # Often, a table config can be wrapped in a project config; look for this case first
    if cfgDict is not None and 'table-cfg' in cfgDict:
      cfgDict = cfgDict['table-cfg']

    hierarchicalUpdate(baseCfgDict, cfgDict, uniqueListElements=True)
    cfg = baseCfgDict
    if not force and self.cfgFname == cfgFname and pg.eq(cfg, self.cfg):
      return None

    self.cfgFname = cfgFname or self.cfgFname
    self.cfg = cfg
    self.paramParser = YamlParser(cfg)
    self.resetLists()
    for field in cfg.get('fields', {}):
      param = self.paramParser['fields', field]
      self.addField(param)

    if self.filter:
      self.filter.updateParamList(self.allFields)
    self.sigCfgUpdated.emit(self.cfg)

  def clear(self):
    self.loadCfg(cfgDict={})

  def resetLists(self):
    self.allFields.clear()

  def fieldFromName(self, name: Union[str, PrjParam], default=None):
    """
    Helper function to retrieve the PrjParam corresponding to the field with this name
    """
    return PrjParamGroup.fieldFromParam(self.allFields, name, default)

  def resolveFieldAliases(self,
                          fields: Sequence[PrjParam],
                          mapping: dict = None
                          ):
      """
      Several forms of imports / exports handle data that may not be compatible with the current table data.
      In these cases, it is beneficial to determine a mapping between names to allow greater compatibility between
      I/O formats. Mapping is also extended in both directions by parameter name aliases (param.opts['aliases']),
      which are a list of strings of common mappings for that parameter (e.g. [Class, Label] are often used
      interchangeably)

      :param fields: Dataframe with maybe foreign fields
      :param mapping: Foreign to local field name mapping
      """

      outFields = []
      for srcField in fields:
        outFields.append(self._findMatchingField(srcField, mapping))
      return outFields

  def _findMatchingField(self, srcField, mapping: dict = None):
    # Mapping takes priority, if it exists
    if mapping is None:
      mapping = {}
    potentialSrcNames = getFieldAliases(srcField)
    for key in srcField, srcField.name:
      # Mapping can either be by string or PrjParam, so account for either case
      outCol = mapping.get(key)
      if outCol:
        break

    if outCol is not None:
      # A mapping was explicitly provided for this field, use that
      return self.fieldFromName(outCol)
    elif srcField in self.allFields:
      return srcField
    else:
      # Not in mapping, no exact match. TODO: what if multiple dest cols have a matching alias?
      # Otherwise, a 'break' can be added
      curOutName = srcField
      for destField in self.allFields:
        if potentialSrcNames & getFieldAliases(destField):
          # Match between source field's aliases and dest field aliases
          # Make sure it didn't match multiple names that weren't itself with the assert statement
          # In other words, if multiple dest fields have the same alias, this assert will fail
          assert curOutName == srcField
          curOutName = destField
    return curOutName

  def __reduce__(self):
    return TableData, (self.cfgFname, self.cfg, self.template, self.filter is not None)
