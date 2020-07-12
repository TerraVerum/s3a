import copy
import sys
from datetime import datetime
from functools import lru_cache
from typing import List, Union, Tuple

from pandas import DataFrame as df
from ruamel.yaml import YAML
import numpy as np

from s3a.graphicsutils import raiseErrorLater
from s3a.views.parameditors import FRParamEditor
from s3a.projectvars import TABLE_DIR, REQD_TBL_FIELDS, COMP_CLASS_NA, DATE_FORMAT, \
  FR_CONSTS
from s3a.structures import FRParam, FilePath, FRParamGroup, FRParamEditorError

yaml = YAML()

def _genList(nameIter, paramType, defaultVal, defaultParam='value'):
  """Helper for generating children elements"""
  return [{'name': name, 'type': paramType, defaultParam: defaultVal} for name in nameIter]

def _filterForParam(param: FRParam):
  """Constructs a filter for the parameter based on its type"""
  children = []
  valType = param.valType
  paramWithChildren = {'name': param.name, 'type': 'group', 'children': children}
  paramWithoutChild = {'name': param.name, 'type': valType, 'value': None}
  if valType in ['int', 'float']:
    retVal = _genList(['min', 'max'], valType, 0)
    retVal[1]['value'] = sys.maxsize
    children.extend(retVal)
    return paramWithChildren
  elif valType in ['FRParam', 'Enum']:
    if valType == 'FRParam':
      iterGroup = [param.name for param in param.value.group]
    else:
      iterGroup = [param for param in param.value]
    children.extend(_genList(iterGroup, 'bool', True))
    return paramWithChildren
  elif valType == 'FRComplexVertices':
    minMax = _filterForParam(FRParam('', 5))['children']
    xyVerts = _genList(['X Bounds', 'Y Bounds'], 'group', minMax, 'children')
    children.extend(xyVerts)
    return paramWithChildren
  elif valType == 'bool':
    children.extend(_genList([f'{param.name}', f'Not {param.name}'], valType, True))
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
    super().__init__(parent, paramList=_FILTER_PARAMS, saveDir=TABLE_DIR, fileType='filter')

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
      colTypes = np.unique([f'"{col.valType}"' for col in badCols])
      raiseErrorLater(FRParamEditorError(f'The table does not know how to create a'
                                        f' filter for fields {", ".join(colNames)}'
                                        f' since types {", ".join(colTypes)} do not'
                                        f' have corresponding filters'))


class FRTableData:

  def __init__(self, annAuthor: str=None):
    self.filter = FRTableFilterEditor()

    self.annAuthor = annAuthor

    self.allFields: List[FRParam] = list(REQD_TBL_FIELDS)
    self.compClasses = [COMP_CLASS_NA]
    COMP_CLASS_NA.group = self.compClasses

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

  def loadCfg(self, cfgFname: FilePath):
    with open(cfgFname, 'r') as ifile:
      cfg: dict = yaml.load(ifile)
    paramParser = FRYamlParser(cfg)

    self.resetLists()
    if 'classes' in cfg:
      self.compClasses.extend(paramParser.parseList('classes',self.compClasses))
    # for compCls in cfg.get('classes', []):
    #   newParam = FRParam(compCls, group=self.compClasses)
    #   self.compClasses.append(newParam)
    for field in cfg.get('opt-tbl-fields', {}):
      param = paramParser['opt-tbl-fields', field]
      param.group = self.allFields
      self.allFields.append(param)

    self.filter.updateParamList(self.allFields)

  def resetLists(self):
    for lst in self.allFields, self.compClasses:
      lst.clear()
    self.allFields.extend(list(REQD_TBL_FIELDS))
    self.compClasses.append(COMP_CLASS_NA)


NestedIndexer = Union[str, Tuple[str,...]]
class FRYamlParser:
  def __init__(self, cfg: dict):
    self.cfg = cfg

  def parseList(self, listName: NestedIndexer, groupOwner: Union[List, FRParamGroup]=None):
    paramList = self.getNestedDict(listName)
    outList = []
    if groupOwner is None:
      groupOwner = outList
    for paramName in paramList:
      curParam = FRParam(paramName)
      curParam.group = groupOwner
      outList.append(curParam)
    return outList

  @lru_cache(maxsize=None)
  def __getitem__(self, paramName: NestedIndexer):
    value = self.getNestedDict(paramName)
    if isinstance(paramName, tuple):
      paramName = paramName[-1]
    parsedParam = FRParam(paramName, value)
    # Handle parameters that are changed by ruamel into SimpleParameter
    if isinstance(value, bool):
      pass
      # Keeps 'int' from triggering
    elif isinstance(value, float):
      parsedParam.valType = 'float'
    elif isinstance(value, int):
      parsedParam.valType = 'int'

    elif isinstance(value, list):
      parsedParam = self.parseList(paramName[-1])[0]

    # User-specified cases
    elif isinstance(value, dict):
      trueType = value.setdefault('valType', 'NoneType')
      trueValue = value['value']
      if trueType in self.cfg:
        # Just handle lists as the non-general case for now
        # TODO: Extend?
        lst = self.parseList(trueType)
        parsedParam.value = FRParamGroup.fromString(lst, trueValue)
        parsedParam.valType = 'FRParam'
      else:
        parsedParam = FRParam(paramName, **value)
    return parsedParam

  def getNestedDict(self, namePath: NestedIndexer):
    if isinstance(namePath, str):
      namePath = (namePath,)
    out = self.cfg
    while len(namePath) > 0:
      out = out[namePath[0]]
      namePath = namePath[1:]
    return out