import copy
import sys
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import List, Union, Tuple, Any, Optional
from warnings import warn

import numpy as np
from pandas import DataFrame as df
from ruamel.yaml import YAML
from pyqtgraph.parametertree import Parameter
from pyqtgraph.Qt import QtWidgets

from s3a.graphicsutils import raiseErrorLater
from s3a.generalutils import attemptFileLoad, resolveYamlDict
from s3a.constants import TABLE_DIR, REQD_TBL_FIELDS, DATE_FORMAT, \
  FR_CONSTS
from s3a.structures import FRParam, FilePath, FRParamGroup, ParamEditorError, \
  S3AException, S3AWarning
from s3a.parameditors import ParamEditor

yaml = YAML()

def genParamList(nameIter, paramType, defaultVal, defaultParam='value'):
  """Helper for generating children elements"""
  return [{'name': name, 'type': paramType, defaultParam: defaultVal} for name in nameIter]

def _filterForParam(param: FRParam):
  """Constructs a filter for the parameter based on its type"""
  children = []
  pType = param.pType
  paramWithChildren = {'name': param.name, 'type': 'group', 'children': children}
  children.append(dict(name='Active', type='bool', value=False))
  if pType in ['int', 'float']:
    retVal = genParamList(['min', 'max'], pType, 0)
    retVal[0]['value'] = -sys.maxsize
    retVal[1]['value'] = sys.maxsize
    children.extend(retVal)
  elif pType in ['FRParam', 'Enum', 'list', 'popuplineeditor', 'bool']:
    if pType == 'FRParam':
      iterGroup = [param.name for param in param.value.group]
    elif pType == 'Enum':
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
  elif pType == 'ComplexXYVertices':
    minMax = _filterForParam(FRParam('', 5))
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

class TableFilterEditor(ParamEditor):
  def __init__(self, paramList: List[FRParam]=None, parent=None):
    if paramList is None:
      paramList = []
    _FILTER_PARAMS = [
      _filterForParam(param) for param in paramList
    ]
    super().__init__(parent, paramList=_FILTER_PARAMS, saveDir=TABLE_DIR,
                     fileType='filter', name='&Component Table Filter')

  def updateParamList(self, paramList: List[FRParam]):
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
      warn(f'The table does not know how to create a filter for fields {", ".join(colNames)}'
            f' since types {", ".join(colTypes)} do not have corresponding filters', S3AWarning)
    self.applyChanges()

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


class TableData:

  def __init__(self, annAuthor: str=None):
    self.filter = TableFilterEditor()
    self.paramParser: Optional[YamlParser] = None

    self.annAuthor = annAuthor
    self.cfgFname: Optional[Path] = None
    self.cfg: Optional[dict] = None

    self.allFields: List[FRParam] = []
    self.compClasses: List[str] = []
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
    for _ in range(numRows):
      # Make sure to construct a separate component instance for
      # each row no objects have the same reference
      df_list.append([field.value for field in copy.copy(self.allFields)])
    outDf = df(df_list, columns=self.allFields).set_index(REQD_TBL_FIELDS.INST_ID, drop=False)
    # Set the metadata for this application run
    outDf[REQD_TBL_FIELDS.ANN_AUTHOR] = self.annAuthor
    outDf[REQD_TBL_FIELDS.ANN_TIMESTAMP] = datetime.now().strftime(DATE_FORMAT)
    outDf[REQD_TBL_FIELDS.SRC_IMG_FILENAME] = FR_CONSTS.ANN_CUR_FILE_INDICATOR.value
    if dropRow:
      outDf = outDf.drop(index=REQD_TBL_FIELDS.INST_ID.value)
    return outDf

  def makeCompSer(self):
    return self.makeCompDf().squeeze()

  def loadCfg(self, cfgFname: FilePath=None, cfgDict: dict=None):
    """
    Lodas the specified table configuration file for S3A. Alternatively, a name
    and dict pair can be supplied instead.
    :param cfgFname: If *cfgDict* is *None*, this is treated as the file containaing
      a YAML-compatible table configuration dictionary. Otherwise, this is the
      configuration name assiciated with the given dictionary.
    :param cfgDict: If not *None*, this is the config data used instad of
      reading *cfgFname* as a file.
    """

    if cfgFname is None:
      cfgFname = 'default.yml'
      cfgDict = {}
    self.cfgFname, cfg = resolveYamlDict(cfgFname, cfgDict)
    if cfg == self.cfg:
      # No need to update things
      return
    self.cfg = cfg
    self.paramParser = YamlParser(cfg)
    newClasses = []
    if 'classes' in cfg:
      classParam = self.paramParser['classes']
      if isinstance(classParam, FRParam):
        if classParam.value is not None:
          REQD_TBL_FIELDS.COMP_CLASS.value = classParam.value
        if classParam.pType is not None:
          REQD_TBL_FIELDS.COMP_CLASS.pType = classParam.pType
        classParam = classParam.opts['limits']
      else:
        REQD_TBL_FIELDS.COMP_CLASS.pType = 'list'
      newClasses.extend(classParam)
    if REQD_TBL_FIELDS.COMP_CLASS.value not in newClasses:
      newClasses.append(REQD_TBL_FIELDS.COMP_CLASS.value)
    REQD_TBL_FIELDS.COMP_CLASS.opts['limits'] = newClasses.copy()
    # for compCls in cfg.get('classes', []):
    #   newParam = FRParam(compCls, group=self.compClasses)
    #   self.compClasses.append(newParam)
    self.resetLists()
    for field in cfg.get('opt-tbl-fields', {}):
      param = self.paramParser['opt-tbl-fields', field]
      param.group = self.allFields
      self.allFields.append(param)

    for field in cfg.get('hidden-cols', []):
      try:
        FRParamGroup.fromString(self.allFields, field).opts['colHidden'] = True
      except S3AException:
        # Specified field to hide isn't in all table fields
        pass

    self.filter.updateParamList(self.allFields)

  def clear(self):
    self.loadCfg(cfgDict={})

  def resetLists(self):
    for lst in self.allFields, self.compClasses:
      lst.clear()
    self.allFields.extend(list(REQD_TBL_FIELDS))
    self.compClasses.extend(REQD_TBL_FIELDS.COMP_CLASS.opts['limits'])

  def fieldFromName(self, name: str):
    """
    Helper function to retrieve the FRParam corresponding to the field with this name
    """
    return FRParamGroup.fromString(self.allFields, name)



NestedIndexer = Union[str, Tuple[Union[str,int],...]]
class YamlParser:
  def __init__(self, cfg: dict):
    self.cfg = cfg

  def parseParamList(self, listName: NestedIndexer, groupOwner: Union[List, FRParamGroup]=None):
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
    if not isinstance(value, dict):
      parsedParam = self.parseLeaf(leafName, value)
    else:
      value = value.copy()
      # Format nicely for FRParam creation
      nameArgs = {'value': value.pop('value', None),
                  'pType': value.pop('pType', 'NoneType'),
                  'helpText': value.pop('helpText', '')}
      # Forward additional args if they exist
      parsedParam = FRParam(leafName, **nameArgs, **value)
    return parsedParam

  def parseLeaf(self, paramName: str, value: Any):
    leafParam = FRParam(paramName, value)
    value = leafParam.value
    if isinstance(value, bool):
      pass
      # Keeps 'int' from triggering
    elif isinstance(value, float):
      leafParam.pType = 'float'
    elif isinstance(value, int):
      leafParam.pType = 'int'

    elif isinstance(value, list):
      # If inner values are dicts, it could be a complex list. Otherwise,
      # it is a primitive list
      testVal = value[0]
      if isinstance(testVal, dict):
        leafParam = self.parseParamList(leafParam.name)[0]
      else:
        # List of primitive values
        leafParam = value
    return leafParam

  def getNestedCfgName(self, namePath: NestedIndexer):
    if isinstance(namePath, str):
      namePath = (namePath,)
    out = self.cfg
    while len(namePath) > 0:
      out = out[namePath[0]]
      namePath = namePath[1:]
    return out