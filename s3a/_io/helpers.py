import inspect
from ast import literal_eval
import typing as t

import numpy as np
import pandas as pd
from s3a.structures import AnnParseError

from ..structures import PrjParamGroup, ComplexXYVertices, XYVertices
from utilitys import PrjParam

_serFunc = t.Callable
_deserFunc = t.Callable
_valsType: t.Sequence[t.Any]

def _newHandlerTemplate():
  return {
    'serialize': None,
    'deserialize': None,
    'takesParam': False
  }

_serdesHandlers = pd.DataFrame(columns=list(_newHandlerTemplate()))

def registerIoHandler(pType: str, force=False, **kwargs):
  global _serdesHandlers
  if pType not in _serdesHandlers.index:
    _serdesHandlers = _serdesHandlers.append(pd.Series(_newHandlerTemplate(), name=pType))
  for which in {'serialize', 'deserialize'} & set(kwargs):
    if _serdesHandlers.loc[pType, which] is not None and not force:
      raise KeyError(f'Already have {which} handler for "{pType}"')
  _serdesHandlers.loc[pType, list(kwargs)] = list(kwargs.values())

def _runFunc(param: PrjParam, values, which: str, default: t.Callable, returnErrs=True):
  out = {}
  errs = {}
  parseType = param.opts.get('parser', param.pType)
  if parseType not in _serdesHandlers.index:
    handlerRow = _newHandlerTemplate()
  else:
    handlerRow = _serdesHandlers.loc[parseType]
  _handler = handlerRow[which] or default
  takesParam = handlerRow['takesParam']
  for ii, val in enumerate(values):
    try:
      out[ii] = _handler(param, val) if takesParam else _handler(val)
    except Exception as ex:
      errs[ii] = ex
  out = pd.Series(out, name=param, dtype=object)
  if returnErrs:
    return out, pd.Series(errs, name=param, dtype=object)
  return out

def serialize(param: PrjParam, values: t.Sequence[t.Any], returnErrs=True):
  default = lambda val: str(val)
  return _runFunc(param, values, 'serialize', default, returnErrs)

def deserialize(param: PrjParam, values: t.Sequence[str], returnErrs=True):
  paramType = type(param.value)
  default = lambda val: paramType(val)
  return _runFunc(param, values, 'deserialize', default, returnErrs)