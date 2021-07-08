from ast import literal_eval
from datetime import datetime

import numpy as np

from ..structures import ComplexXYVertices, XYVertices, PrjParamGroup

from .componentio import ComponentIO, defaultIo
from .helpers import registerIoHandler

registerIoHandler(
  'ndarray',
  deserialize = lambda strVal: np.array(literal_eval(strVal)),
  serialize = lambda val: str(val.tolist()),
)

registerIoHandler(
  'complexxyvertices',
  deserialize = ComplexXYVertices.deserialize,
  serialize = ComplexXYVertices.serialize,
)

registerIoHandler(
  'xyvertices',
  deserialize = XYVertices.deserialize,
  serialize = XYVertices.serialize,
)

registerIoHandler(
  'bool',
  deserialize = lambda strVal: strVal.lower() == 'true',
)

registerIoHandler(
  'prjparam',
  deserialize = lambda param, strVal: PrjParamGroup.fieldFromParam(param.group, strVal),
  serialize = lambda param, val: str(val),
  takesParam=True
)

def listDeser(param, strVal):
  val = type(param.value)(strVal)
  fixedLims = param.opts.get('fixedLimits', False)
  lims = param.opts['limits']
  if fixedLims and val not in lims:
    raise ValueError(f'Limits cannot grow and "{val} is not in limits')
  elif val not in lims:
    lims.append(val)
  return val
registerIoHandler(
  'list',
  deserialize = listDeser,
  serialize = lambda param, val: str(val),
  takesParam=True
)

def superAnnTime(val):
  parsedTime = datetime.strptime(val, '%Y-%m-%dT%H:%M:%S.%fZ')
  return str(parsedTime)

registerIoHandler(
  'superanntime',
  deserialize = superAnnTime,
)