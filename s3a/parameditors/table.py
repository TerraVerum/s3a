import copy
import sys
import warnings
from functools import lru_cache
from pathlib import Path
from typing import List, Union, Tuple, Any, Optional, Callable, Dict, Sequence

import numpy as np
import pyqtgraph as pg
from pandas import DataFrame as df
from pyqtgraph.Qt import QtCore
from pyqtgraph.parametertree import Parameter

from s3a.constants import TABLE_DIR, REQD_TBL_FIELDS as RTF, TBL_BASE_TEMPLATE
from s3a.generalutils import hierarchicalUpdate
from s3a.structures import PrjParamGroup, FilePath
from utilitys import ParamEditor, PrjParam
from utilitys import fns


def genParamList(nameIter, paramType, defaultVal, defaultParam='value'):
  """Helper for generating children elements"""
  return [{'name': name, 'type': paramType, defaultParam: defaultVal} for name in nameIter]

def _filterForParam(param: PrjParam):
  """Constructs a filter for the parameter based on its type"""
  children = []
  pType = param.pType.lower()
  paramWithChildren = {'name': param.name, 'type': 'group', 'children': children}
  children.append(dict(name='Active', type='bool', value=False))
  if pType in ['int', 'float']:
    retVal = genParamList(['min', 'max'], pType, 0)
    retVal[0]['value'] = -sys.maxsize
    retVal[1]['value'] = sys.maxsize
    children.extend(retVal)
  elif pType in ['prjparam', 'enum', 'list', 'popuplineeditor', 'bool']:
    if pType == 'prjparam':
      iterGroup = [param.name for param in param.value.group]
    elif pType == 'enum':
      iterGroup = [param for param in param.value]
    elif pType == 'bool':
      iterGroup = [f'{param.name}', f'Not {param.name}']
    else: # pType == 'list' or 'popuplineeditor'
      iterGroup = param.opts['limits']
    optsParam = Parameter.create(name='Options', type='checklist', limits=iterGroup, value=iterGroup)
    paramWithChildren = Parameter.create(**paramWithChildren)
    paramWithChildren.addChild(optsParam)
  elif 'xyvertices' in pType:
    minMax = _filterForParam(PrjParam('', 5))
    minMax.removeChild(minMax.childs[0])
    minMax = minMax.saveState()['children']
    xyVerts = genParamList(['X Bounds', 'Y Bounds'], 'group', minMax, 'children')
    children.extend(xyVerts)
  elif pType in ['str', 'text']:
    # Assumes string
    children.append(dict(name='Regex Value', type='str', value=''))
  else:
    # Don't know how to handle the parameter
    return None

  if isinstance(paramWithChildren, dict):
    paramWithChildren = Parameter.create(**paramWithChildren)
  return paramWithChildren

def filterParamCol(compDf: df, column: PrjParam, filterOpts: dict):
  # TODO: Each type should probably know how to filter itself. That is,
  #  find some way of keeping this logic from just being an if/else tree...
  pType = column.pType
  # idx 0 = value, 1 = children
  dfAtParam = compDf.loc[:, column]

  if pType in ['int', 'float']:
    curmin, curmax = [filterOpts[name] for name in ['min', 'max']]

    compDf = compDf.loc[(dfAtParam >= curmin) & (dfAtParam <= curmax)]
  elif pType == 'bool':
    filterOpts = filterOpts['Options']
    allowTrue, allowFalse = [filterOpts[name] for name in
                             [f'{column.name}', f'Not {column.name}']]

    validList = np.array(dfAtParam, dtype=bool)
    if not allowTrue:
      compDf = compDf.loc[~validList]
    if not allowFalse:
      compDf = compDf.loc[validList]
  elif pType in ['prjparam', 'list', 'popuplineeditor']:
    existingParams = np.array(dfAtParam)
    allowedParams = []
    filterOpts = filterOpts['Options']
    if pType == 'prjparam':
      groupSubParams = [p.name for p in column.value.group]
    else:
      groupSubParams = column.opts['limits']
    for groupSubParam in groupSubParams:
      isAllowed = filterOpts[groupSubParam]
      if isAllowed:
        allowedParams.append(groupSubParam)
    compDf = compDf.loc[np.isin(existingParams, allowedParams)]
  elif pType in ['str', 'text']:
    allowedRegex = filterOpts['Regex Value']
    isCompAllowed = dfAtParam.str.contains(allowedRegex, regex=True, case=False)
    compDf = compDf.loc[isCompAllowed]
  elif pType in ['complexxyvertices', 'xyvertices']:
    vertsAllowed = np.ones(len(dfAtParam), dtype=bool)

    xParam = filterOpts['X Bounds']
    yParam = filterOpts['Y Bounds']
    xmin, xmax, ymin, ymax = [param[val] for param in (xParam, yParam) for val in ['min', 'max']]

    for vertIdx, verts in enumerate(dfAtParam):
      if pType == 'complexxyvertices':
        stackedVerts = verts.stack()
      else:
        stackedVerts = verts
      xVerts, yVerts = stackedVerts.x, stackedVerts.y
      isAllowed = np.all((xVerts >= xmin) & (xVerts <= xmax)) & \
                  np.all((yVerts >= ymin) & (yVerts <= ymax))
      vertsAllowed[vertIdx] = isAllowed
    compDf = compDf.loc[vertsAllowed]
  else:
    warnings.warn('No filter type exists for parameters of type ' f'{pType}.'
              f' Did not filter column {column.name}.',
              UserWarning)
  return compDf


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

class TableFilterEditor(ParamEditor):
  def __init__(self, paramList: List[PrjParam]=None, parent=None):
    if paramList is None:
      paramList = []
    filterParams = [fil for fil in map(_filterForParam, paramList) if fil is not None]
    super().__init__(parent, paramList=filterParams, saveDir=TABLE_DIR,
                     fileType='filter', name='Component Table Filter')

  def updateParamList(self, paramList: List[PrjParam]):
    newParams = []
    badCols = []
    for param in paramList:
      try:
        curFilter = _filterForParam(param)
      except KeyError:
        curFilter = None
      if curFilter is None:
        badCols.append(param)
      else:
        newParams.append(curFilter)
    self.params.clearChildren()
    self.params.addChildren(newParams)
    if len(badCols) > 0:
      colNames = [f'"{col}"' for col in badCols]
      colTypes = np.unique([f'"{col.pType}"' for col in badCols])
      warnings.warn(f'The table does not know how to create a filter for fields'
            f' {", ".join(colNames)}'
            f' since types {", ".join(colTypes)} do not have corresponding filters', UserWarning)
    self.saveParamValues(blockWrite=True)
    self.saveCurStateAsDefault()

  @property
  def activeFilters(self):
    filters = {}
    for child in self.params.childs:
      if child['Active']:
        cState = next(iter(fns.paramValues(child, includeDefaults=True).values()))
        cState.pop('Active')
        filters[child.name()] = cState
    return filters

  def filterCompDf(self, compDf: df):
    strNames = [str(f) for f in compDf.columns]
    for fieldName, opts in self.activeFilters.items():
      try:
        matchIdx = strNames.index(fieldName)
      except IndexError:
        # This filter can be used on dataframes that didn't have to come from S3A,
        # so silently ignore mismatched filter requests
        continue
      col = compDf.columns[matchIdx]
      compDf = filterParamCol(compDf, col, opts)
    return compDf

class TableData(QtCore.QObject):
  sigCfgUpdated = QtCore.Signal(object)
  """dict (self.cfg) during update"""

  def __init__(self,
               cfgFname: FilePath=None,
               cfgDict: dict=None,
               template: Union[FilePath, dict]=None,
               requiredFields: Sequence[PrjParam]=None):
    super().__init__()
    if requiredFields is None:
      requiredFields = RTF
    self._requiredFields = requiredFields
    if template is None:
      template = fns.attemptFileLoad(TBL_BASE_TEMPLATE)
    self._template = template

    self.factories: Dict[PrjParam, Callable[[], Any]] = {}

    self.filter = TableFilterEditor()
    self.paramParser: Optional[YamlParser] = None

    self.cfgFname: Optional[Path] = None
    self.cfg: Optional[dict] = fns.attemptFileLoad(TBL_BASE_TEMPLATE)

    self.allFields: List[PrjParam] = []
    self.resetLists()

    cfgFname = cfgFname or None
    self.loadCfg(cfgFname, cfgDict, template)

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
              template: Union[FilePath, dict]=None,
              force=False):
    """
    Lodas the specified table configuration file for S3A. Alternatively, a name
    and dict pair can be supplied instead.
    :param cfgFname: If *cfgDict* is *None*, this is treated as the file containaing
      a YAML-compatible table configuration dictionary. Otherwise, this is the
      configuration name assiciated with the given dictionary.
    :param cfgDict: If not *None*, this is the config data used instad of
      reading *cfgFname* as a file.
    :param template: Template file or dict whose fields must exist to ensure updates happen properly.
      This can also be used to specify the base of a compositional configs
    :param force: If *True*, the new config will be loaded even if it is the same name as the
    current config
    """
    if template is None:
      template = self._template
    if isinstance(template, dict):
      baseCfgDict = template.copy()
    else:
      baseCfgDict = fns.attemptFileLoad(template)
    if cfgFname is not None:
      cfgFname, cfgDict = fns.resolveYamlDict(cfgFname, cfgDict)
      cfgFname = cfgFname.resolve()
    # Often, a table config can be wrapped in a project config; look for this case first
    if cfgDict is not None and 'table-cfg' in cfgDict:
      cfgDict = cfgDict['table-cfg']

    hierarchicalUpdate(baseCfgDict, cfgDict)
    cfg = baseCfgDict
    if not force and self.cfgFname == cfgFname and pg.eq(cfg, self.cfg):
      return None

    self.cfgFname = cfgFname
    self.cfg = cfg
    self.paramParser = YamlParser(cfg)
    self.resetLists()
    for field in cfg.get('fields', {}):
      param = self.paramParser['fields', field]
      self.addField(param)

    self.filter.updateParamList(self.allFields)
    self.sigCfgUpdated.emit(self.cfg)

  def clear(self):
    self.loadCfg(cfgDict={})

  def resetLists(self):
    self.allFields.clear()
    self.allFields.extend(self._requiredFields)

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
    outCol = None
    for key in srcField, srcField.name:
      # Mapping can either be by string or PrjParam, so account for either case
      outCol = outCol or mapping.get(key)
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
          assert curOutName == srcField
          curOutName = destField
    return curOutName

NestedIndexer = Union[str, Tuple[Union[str,int],...]]
class YamlParser:
  def __init__(self, cfg: dict):
    self.cfg = cfg

  @lru_cache(maxsize=None)
  def __getitem__(self, paramName: NestedIndexer):
    value = self.getNestedCfgName(paramName)
    if not isinstance(paramName, tuple):
      paramName = (paramName,)
    leafName = paramName[-1]
    # Assume leaf until proven otherwise since most mechanics are still applicable
    if isinstance(value, PrjParam):
      # Can happen with programmatically generated cfgs. Make a copy to
      # ensure no funky business
      parsedParam = copy.copy(value)
    elif not isinstance(value, dict):
      parsedParam = self.parseLeaf(leafName, value)
    else:
      value = value.copy()
      # Format nicely for PrjParam creation
      nameArgs = {'value': value.pop('value', None),
                  'pType': value.pop('pType', None),
                  'helpText': value.pop('helpText', '')}
      # Forward additional args if they exist
      parsedParam = PrjParam(leafName, **nameArgs, **value)
    return parsedParam

  def parseLeaf(self, paramName: str, value: Any):
    leafParam = PrjParam(paramName, value)
    value = leafParam.value
    if isinstance(value, bool):
      pass
      # Keeps 'int' from triggering
    elif isinstance(value, float):
      leafParam.pType = 'float'
    elif isinstance(value, int):
      leafParam.pType = 'int'

    elif isinstance(value, list):
      leafParam.pType = 'list'
      testVal = value[0]
      if isinstance(testVal, dict):
        # Value is on the other side of the mapping
        testVal = next(iter(testVal.values()))
      # list of simple values, implied these are the limits. Since no default
      # is specified, it'll be the first in the list
      leafParam.opts['limits'] = value
      leafParam.value = testVal
    return leafParam

  def getNestedCfgName(self, namePath: NestedIndexer):
    if isinstance(namePath, str):
      namePath = (namePath,)
    out = self.cfg
    while len(namePath) > 0:
      out = out[namePath[0]]
      namePath = namePath[1:]
    return out
