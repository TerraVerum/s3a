import ast
from ast import literal_eval

import numpy as np

from .helpers import registerIoHandler
from ..structures import ComplexXYVertices, XYVertices, PrjParamGroup
# Wait to import compio until after registry is populated

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
  # Slight adjustment to value interpretation for checklist
  fixedLims = param.opts.get('fixedLimits', False)
  lims = param.opts['limits']
  if param.pType == 'checklist':
    if param.opts.get('exclusive'):
      val = [strVal]
    else:
      val = ast.literal_eval(strVal)
    newLims = [v for v in val if v not in lims]
  else:
    val = type(param.value)(strVal)
    newLims = [val] if val not in lims else []

  if fixedLims and len(newLims):
    raise ValueError(f'Limits cannot grow and "{newLims} not in limits')
  lims.extend(newLims)
  return val
for typ in 'list', 'checklist':
  registerIoHandler(
    typ,
    deserialize=listDeser,
    takesParam=True
  )

from .componentio import ComponentIO, defaultIo
