import sys
import warnings
from typing import List

import numpy as np
from pandas import DataFrame as df
from pyqtgraph.parametertree import Parameter

from ...constants import TABLE_DIR
from utilitys import PrjParam, ParamEditor, fns


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
