import copy
import sys
from datetime import datetime
from typing import List

from pandas import DataFrame as df
from ruamel.yaml import YAML

import s3a.frgraphics
from s3a.frgraphics.parameditors import FRParamEditor
from s3a.projectvars import FILTERS_DIR, REQD_TBL_FIELDS, COMP_CLASS_NA, DATE_FORMAT, \
  FR_CONSTS
from s3a.structures import FRParam, FilePath

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
    return paramWithoutChild

class FRTableFilterEditor(FRParamEditor):
  def __init__(self, paramList: List[FRParam]=None, parent=None):
    if paramList is None:
      paramList = []
    _FILTER_PARAMS = [
      _filterForParam(param) for param in paramList
    ]
    super().__init__(parent, paramList=_FILTER_PARAMS, saveDir=FILTERS_DIR, fileType='filter')

  def updateParamList(self, paramList: List[FRParam]):
    newParams = [
      _filterForParam(param) for param in paramList
    ]
    self.params.clearChildren()
    self.params.addChildren(newParams)


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
    self.resetLists()
    for compCls in cfg.get('classes', []):
      newParam = FRParam(compCls, group=self.compClasses)
      self.compClasses.append(newParam)
    for field, values in cfg.get('opt-tbl-fields', {}).items():
      if isinstance(values, dict):
        param = FRParam(field, **values)
      else:
        param = FRParam(field, values)
      param.group = self.allFields
      self.allFields.append(param)
    self.filter.updateParamList(self.allFields)

  def resetLists(self):
    for lst in self.allFields, self.compClasses:
      lst.clear()
    self.allFields.extend(list(REQD_TBL_FIELDS))
    self.compClasses.append(COMP_CLASS_NA)