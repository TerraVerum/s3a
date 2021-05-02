import copy
import sys
from functools import lru_cache
from pathlib import Path
from typing import List, Union, Tuple, Any, Optional, Callable, Dict

import numpy as np
from pandas import DataFrame as df
from pyqtgraph.Qt import QtCore
from pyqtgraph.parametertree import Parameter

from s3a.constants import TABLE_DIR, REQD_TBL_FIELDS, PRJ_CONSTS, TBL_BASE_TEMPLATE
from s3a.generalutils import hierarchicalUpdate
from s3a.structures import PrjParamGroup, FilePath
from utilitys import ParamEditor, PrjParam
from utilitys import fns
from utilitys.fns import warnLater

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
    optsParam = Parameter.create(name='Options', type='group', children=genParamList(iterGroup, 'bool', True))
    def changeOpts(allowed: bool):
      for param in optsParam.childs:
        param.setValue(allowed)
    paramWithChildren = Parameter.create(**paramWithChildren)
    actions = [Parameter.create(name=name, type='action') for name in ('Select All', 'Clear All')]
    actions[0].sigActivated.connect(lambda: changeOpts(True))
    actions[1].sigActivated.connect(lambda: changeOpts(False))
    paramWithChildren.addChildren(actions)
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
    curmin, curmax = [filterOpts[name]['value'] for name in ['min', 'max']]

    compDf = compDf.loc[(dfAtParam >= curmin) & (dfAtParam <= curmax)]
  elif pType == 'bool':
    filterOpts = filterOpts['Options']['children']
    allowTrue, allowFalse = [filterOpts[name]['value'] for name in
                             [f'{column.name}', f'Not {column.name}']]

    validList = np.array(dfAtParam, dtype=bool)
    if not allowTrue:
      compDf = compDf.loc[~validList]
    if not allowFalse:
      compDf = compDf.loc[validList]
  elif pType in ['prjparam', 'list', 'popuplineeditor']:
    existingParams = np.array(dfAtParam)
    allowedParams = []
    filterOpts = filterOpts['Options']['children']
    if pType == 'prjparam':
      groupSubParams = [p.name for p in column.value.group]
    else:
      groupSubParams = column.opts['limits']
    for groupSubParam in groupSubParams:
      isAllowed = filterOpts[groupSubParam]['value']
      if isAllowed:
        allowedParams.append(groupSubParam)
    compDf = compDf.loc[np.isin(existingParams, allowedParams)]
  elif pType in ['str', 'text']:
    allowedRegex = filterOpts['Regex Value']['value']
    isCompAllowed = dfAtParam.str.contains(allowedRegex, regex=True, case=False)
    compDf = compDf.loc[isCompAllowed]
  elif pType in ['complexxyvertices', 'xyvertices']:
    vertsAllowed = np.ones(len(dfAtParam), dtype=bool)

    xParam = filterOpts['X Bounds']['children']
    yParam = filterOpts['Y Bounds']['children']
    xmin, xmax, ymin, ymax = [param[val]['value'] for param in (xParam, yParam) for val in ['min', 'max']]

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
    warnLater('No filter type exists for parameters of type ' f'{pType}.'
              f' Did not filter column {column.name}.',
              UserWarning)
  return compDf

class TableFilterEditor(ParamEditor):
  def __init__(self, paramList: List[PrjParam]=None, parent=None):
    if paramList is None:
      paramList = []
    _FILTER_PARAMS = [
      _filterForParam(param) for param in paramList
    ]
    super().__init__(parent, paramList=_FILTER_PARAMS, saveDir=TABLE_DIR,
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
      warnLater(f'The table does not know how to create a filter for fields'
            f' {", ".join(colNames)}'
            f' since types {", ".join(colTypes)} do not have corresponding filters', UserWarning)
    self.saveParamValues(blockWrite=True)

  @property
  def activeFilters(self):
    filters = {}
    for child in self.params.childs:
      if child['Active']:
        cState = child.saveState('user')
        keepChildren = cState['children']
        keepChildren.pop('Active')
        filters[child.name()] = keepChildren
    return filters

  def filterCompDf(self, compDf: df):
    strNames = [str(f) for f in compDf.columns]
    for fieldName, opts in self.activeFilters.items():
      try:
        matchIdx = strNames.index(fieldName)
      except IndexError:
        continue
      col = compDf.columns[matchIdx]
      compDf = filterParamCol(compDf, col, opts)
    return compDf

class TableData(QtCore.QObject):
  sigCfgUpdated = QtCore.Signal(object)
  """dict (self.cfg) during update"""

  def __init__(self):
    super().__init__()
    self._factories: Dict[PrjParam, Callable[[], Any]] = {}

    self.filter = TableFilterEditor()
    self.paramParser: Optional[YamlParser] = None

    self.cfgFname: Optional[Path] = None
    self.cfg: Optional[dict] = None

    self.allFields: List[PrjParam] = []
    self.resetLists()

  def makeCompDf(self, numRows=1) -> df:
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
      if f in self._factories:
        val = self._factories[f]()
      else:
        val = f.value
      populators.append(val)

    for _ in range(numRows):
      # Make sure to construct a separate component instance for
      # each row no objects have the same reference
      df_list.append(copy.copy(populators))
    outDf = df(df_list, columns=self.allFields).set_index(REQD_TBL_FIELDS.INST_ID, drop=False)
    # Set the metadata for this application run
    outDf[REQD_TBL_FIELDS.SRC_IMG_FILENAME] = PRJ_CONSTS.ANN_CUR_FILE_INDICATOR.value
    if dropRow:
      outDf = outDf.drop(index=REQD_TBL_FIELDS.INST_ID.value)
    return outDf

  def addFieldFactory(self, fieldLbl: PrjParam, factory: Callable[[], Any]):
    """
    For fields that are simple functions (i.e. don't require input from the user), a
    factory can be used to create default values when instantiating new table rows.

    :param fieldLbl: WHich field this factory is used for instead of just the default value
    :param factory: Callable to use instead of field value. This is called with no parameters.
    """
    self._factories[fieldLbl] = factory

  def makeCompSer(self):
    return self.makeCompDf().squeeze()

  def loadCfg(self, cfgFname: FilePath=None, cfgDict: dict=None, force=False):
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

    _, baseCfgDict = fns.resolveYamlDict(TBL_BASE_TEMPLATE)
    cfgFname, cfgDict = fns.resolveYamlDict(cfgFname, cfgDict)
    cfgFname = cfgFname.resolve()
    if not force and self.cfgFname == cfgFname:
      return None
    hierarchicalUpdate(baseCfgDict, cfgDict)

    cfg = baseCfgDict
    if not force and cfg == self.cfg:
      # No need to update things
      return

    self.cfgFname = cfgFname
    self.cfg = cfg
    self.paramParser = YamlParser(cfg)
    self.resetLists()
    for field in cfg.get('fields', {}):
      param = self.paramParser['fields', field]
      if param in self.allFields:
        continue
      param.group = self.allFields
      self.allFields.append(param)

    self.filter.updateParamList(self.allFields)
    self.sigCfgUpdated.emit(self.cfg)

  def clear(self):
    self.loadCfg(cfgDict={})

  def resetLists(self):
    self.allFields.clear()
    self.allFields.extend(REQD_TBL_FIELDS)

  def fieldFromName(self, name: Union[str, PrjParam]):
    """
    Helper function to retrieve the PrjParam corresponding to the field with this name
    """
    return PrjParamGroup.fieldFromParam(self.allFields, name)

NestedIndexer = Union[str, Tuple[Union[str,int],...]]
class YamlParser:
  def __init__(self, cfg: dict):
    self.cfg = cfg

  def parseParamList(self, listName: NestedIndexer, groupOwner: Union[List, PrjParamGroup]=None):
    """
    A simple list is only a list of strings. A complex list is a dict, where each
    (key, val) pair is a list element. See the structural setup in pg.ListParameter.
    """
    if not isinstance(listName, tuple):
      listName = (listName,)
    paramList = self.getNestedCfgName(listName)
    outList = []
    if groupOwner is None:
      groupOwner = outList
    for ii in range(len(paramList)):
      # Default to list of string values
      accessor = listName + (ii,)
      curParam = self[accessor]
      curParam.group = groupOwner
      outList.append(curParam)
    return outList

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
      # If inner values are dicts, it could be a complex list. Otherwise,
      # it is a primitive list
      testVal = value[0]
      if isinstance(testVal, dict):
        leafParam.value = self.parseParamList(leafParam.name)[0]
      else:
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