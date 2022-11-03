import ast

import numpy as np

from .helpers import registerIoHandler
from ..structures import ComplexXYVertices, OptionsDictGroup, XYVertices

# Wait to import compio until after registry is populated

registerIoHandler(
    "ndarray",
    deserialize=lambda strVal: np.array(ast.literal_eval(strVal)),
    serialize=lambda val: str(val.tolist()),
)

registerIoHandler(
    "complexxyvertices",
    deserialize=ComplexXYVertices.deserialize,
    serialize=ComplexXYVertices.serialize,
)

registerIoHandler(
    "xyvertices",
    deserialize=XYVertices.deserialize,
    serialize=XYVertices.serialize,
)

registerIoHandler(
    "bool",
    deserialize=lambda strVal: strVal.lower() == "true",
)


def listDeserializer(param, strVal):
    # Slight adjustment to value interpretation for checklist
    fixedLims = param.opts.get("fixedLimits", False)
    lims = param.opts["limits"]
    if param.type == "checklist":
        if param.opts.get("exclusive"):
            val = [strVal]
        else:
            val = ast.literal_eval(strVal)
        newLims = [v for v in val if v not in lims]
    else:
        val = type(param.value)(strVal)
        newLims = [val] if val not in lims else []

    if fixedLims and len(newLims):
        raise ValueError(f"Limits cannot grow and '{newLims}' not in limits")
    lims.extend(newLims)
    return val


for typ in "list", "checklist":
    registerIoHandler(typ, deserialize=listDeserializer, takesParam=True)

from .componentio import ComponentIO, defaultIo
from .exporters import *
from .importers import *
