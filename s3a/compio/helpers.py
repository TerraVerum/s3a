import inspect
import typing as t
import warnings

import numpy as np
import pandas as pd

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
  if returnErrs:
    retCnt = 2
    retIdx = slice(None)
  else:
    retCnt = 1
    retIdx = 0
  ret = [pd.Series(name=param, dtype=str).copy() for _ in range(retCnt)]
  if not len(values):
    # Nothing to do
    return ret[retIdx]
  # Calling 'serialize' on already serialized data is a no-op
  # TODO: handle heterogeneous arrays?
  if isinstance(values[0], str):
    ret[0] = pd.Series(values, name=param)
    return ret[retIdx]
  default = lambda val: str(val)
  return _runFunc(param, values, 'serialize', default, returnErrs)

def deserialize(param: PrjParam, values: t.Sequence[str], returnErrs=True):
  # Calling 'deserialize' on a stringified data is a no-op
  # TODO: heterogeneous arrays?
  # Unlike serialize, dtype could be different depending on 'param', so leave empty creation to the handler
  if len(values) and not isinstance(values[0], str):
    return pd.Series(values, name=param)
  paramType = type(param.value)
  default = lambda val: paramType(val)
  return _runFunc(param, values, 'deserialize', default, returnErrs)


def _getPdExporters():
  members = inspect.getmembers(
      pd.DataFrame, lambda meth: inspect.isfunction(meth) and meth.__name__.startswith('to_')
  )
  return [mem[0].replace('to_', '') for mem in members]


def _getPdImporters():
  members = inspect.getmembers(
      pd.DataFrame, lambda meth: inspect.isfunction(meth) and meth.__name__.startswith('read_')
  )
  return [mem[0].replace('read_', '') for mem in members]


def checkVertBounds(vertSer: pd.Series, imShape: tuple):
  """
  Checks whether any vertices in the imported dataframe extend past image dimensions. This is an indicator
  they came from the wrong import file.

  :param vertSer: Vertices from incoming component dataframe
  :param imShape: Shape of the main image these vertices are drawn on
  :return: Raises error if offending vertices are present, since this is an indication the component file
    was from a different image
  """
  if imShape is None or len(vertSer) == 0:
    # Nothing we can do if no shape is given
    return
  # Image shape from row-col -> x-y
  imShape = np.array(imShape[1::-1])[None, :]
  # Remove components whose vertices go over any image edges
  vertMaxs = [verts.stack().max(0) for verts in vertSer if len(verts) > 0]
  vertMaxs = np.vstack(vertMaxs)
  offendingIds = np.nonzero(np.any(vertMaxs > imShape, axis=1))[0]
  if len(offendingIds) > 0:
    warnings.warn(
        f'Vertices on some components extend beyond image dimensions. '
        f'Perhaps this export came from a different image?\n'
        f'Offending IDs: {offendingIds}', UserWarning
    )