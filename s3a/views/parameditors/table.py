import copy
import sys
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import List, Union, Tuple, Any, Optional

import numpy as np
from pandas import DataFrame as df
from ruamel.yaml import YAML

from s3a.graphicsutils import raiseErrorLater
from s3a.projectvars import TABLE_DIR, REQD_TBL_FIELDS, DATE_FORMAT, \
  FR_CONSTS
from s3a.structures import FRParam, FilePath, FRParamGroup, FRParamEditorError, \
  FRS3AException
from s3a.views.parameditors import FRParamEditor

yaml = YAML()

def genParamList(nameIter, paramType, defaultVal, defaultParam='value'):
  """Helper for generating children elements"""
  return [{'name': name, 'type': paramType, defaultParam: defaultVal} for name in nameIter]

def _filterForParam(param: FRParam):
  """Constructs a filter for the parameter based on its type"""
  children = []
  pType = param.pType
  paramWithChildren = {'name': param.name, 'type': 'group', 'children': children}
  paramWithoutChild = {'name': param.name, 'type': pType, 'value': None}
  if pType in ['int', 'float']:
    retVal = genParamList(['min', 'max'], pType, 0)
    retVal[1]['value'] = sys.maxsize
    children.extend(retVal)
    return paramWithChildren
  elif pType in ['FRParam', 'Enum', 'list', 'popuplineeditor']:
    if pType == 'FRParam':
      iterGroup = [param.name for param in param.value.group]
    elif pType == 'Enum':
      iterGroup = [param for param in param.value]
    else: # pType == 'list' or 'popuplineeditor'
      iterGroup = param.opts['limits']
    children.extend(genParamList(iterGroup, 'bool', True))
    return paramWithChildren
  elif pType == 'FRComplexVertices':
    minMax = _filterForParam(FRParam('', 5))['children']
    xyVerts = genParamList(['X Bounds', 'Y Bounds'], 'group', minMax, 'children')
    children.extend(xyVerts)
    return paramWithChildren
  elif pType == 'bool':
    children.extend(genParamList([f'{param.name}', f'Not {param.name}'], pType, True))
    return paramWithChildren
  else:
    # Assumes string
    paramWithoutChild['value'] = ''
    paramWithoutChild['type'] = 'str'
    return paramWithoutChild

class FRTableFilterEditor(FRParamEditor):
  def __init__(self, paramList: List[FRParam]=None, parent=None):
    if paramList is None:
      paramList = []
    _FILTER_PARAMS = [
      _filterForParam(param) for param in paramList
    ]
    super().__init__(parent, paramList=_FILTER_PARAMS, saveDir=TABLE_DIR, fileType='filter',
                     name='Component Table Filter')

  def updateParamList(self, paramList: List[FRParam]):
    newParams = [
      _filterForParam(param) for param in paramList
    ]
    self.params.clearChildren()
    badCols = []
    for ii, child in enumerate(newParams):
      try:
        self.params.addChild(child)
      except KeyError:
        badCols.append(paramList[ii])
    if len(badCols) > 0:
      colNames = [f'"{col}"' for col in badCols]
      colTypes = np.unique([f'"{col.pType}"' for col in badCols])
      raiseErrorLater(FRParamEditorError(f'The table does not know how to create a'
                                        f' filter for fields {", ".join(colNames)}'
                                        f' since types {", ".join(colTypes)} do not'
                                        f' have corresponding filters'))
    self.applyChanges()


class FRTableData:

  def __init__(self, annAuthor: str=None):
    self.filter = FRTableFilterEditor()

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

  def loadCfg(self, cfgFname: FilePath, cfgDict: dict=None):
    """
    Lodas the specified table configuration file for S3A. Alternatively, a name
    and dict pair can be supplied instead.
    :param cfgFname: If *cfgDict* is *None*, this is treated as the file containaing
      a YAML-compatible table configuration dictionary. Otherwise, this is the
      configuration name assiciated with the given dictionary.
    :param cfgDict: If not *None*, this is the config data used instad of
      reading *cfgFname* as a file.
    """
    if cfgDict is not None:
      cfg = cfgDict
    else:
      with open(cfgFname, 'r') as ifile:
        cfg: dict = yaml.load(ifile)
    self.cfgFname = Path(cfgFname)
    if cfg == self.cfg:
      # No need to update things
      return
    self.cfg = cfg
    paramParser = FRYamlParser(cfg)

    self.resetLists()
    if 'classes' in cfg:
      classParam = paramParser['classes']
      if isinstance(classParam, FRParam):
        if classParam.value is not None:
          REQD_TBL_FIELDS.COMP_CLASS.value = classParam.value
        if classParam.pType is not None:
          REQD_TBL_FIELDS.COMP_CLASS.pType = classParam.pType
        classParam = classParam.opts['limits']
      else:
        REQD_TBL_FIELDS.COMP_CLASS.pType = 'list'
      if REQD_TBL_FIELDS.COMP_CLASS.value in classParam:
        classParam.remove(REQD_TBL_FIELDS.COMP_CLASS.value)
      self.compClasses.extend(classParam)
      REQD_TBL_FIELDS.COMP_CLASS.opts['limits'] = self.compClasses.copy()
    # for compCls in cfg.get('classes', []):
    #   newParam = FRParam(compCls, group=self.compClasses)
    #   self.compClasses.append(newParam)
    for field in cfg.get('opt-tbl-fields', {}):
      param = paramParser['opt-tbl-fields', field]
      param.group = self.allFields
      self.allFields.append(param)

    for field in cfg.get('hidden-cols', []):
      try:
        FRParamGroup.fromString(self.allFields, field).opts['colHidden'] = True
      except FRS3AException:
        # Specified field to hide isn't in all table fields
        pass

    self.filter.updateParamList(self.allFields)

  def resetLists(self):
    for lst in self.allFields, self.compClasses:
      lst.clear()
    self.allFields.extend(list(REQD_TBL_FIELDS))
    self.compClasses.append(REQD_TBL_FIELDS.COMP_CLASS.value)

  def fieldFromName(self, name: str):
    """
    Helper function to retrieve the FRParam corresponding to the field with this name
    """
    return FRParamGroup.fromString(self.allFields, name)



NestedIndexer = Union[str, Tuple[Union[str,int],...]]
class FRYamlParser:
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
      if len(value) > 0:
        nameArgs['opts'] = value
      parsedParam = FRParam(leafName, **nameArgs)
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