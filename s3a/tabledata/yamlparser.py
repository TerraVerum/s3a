import copy
from functools import lru_cache
from typing import Any, Tuple, Union

from qtextras import OptionsDict
from qtextras.fns import loader

from ..compio.helpers import deserialize

NestedIndexer = Union[str, Tuple[Union[str, int], ...]]

_yamlReps = [
    typ.__name__.lower() if isinstance(typ, type) else type(typ).__name__.lower()
    for typ in loader.representer.yaml_representers
]


class YamlParser:
    def __init__(self, config: dict):
        self.config = config

    @lru_cache(maxsize=None)
    def __getitem__(self, parameterName: NestedIndexer):
        value = self.getNestedConfigName(parameterName)
        if not isinstance(parameterName, tuple):
            parameterName = (parameterName,)
        leafName = parameterName[-1]
        # Assume leaf until proven otherwise since most mechanics are still applicable
        if isinstance(value, OptionsDict):
            # Can happen with programmatically generated cfgs. Make a copy to
            # ensure no funky business
            parsedParam = copy.copy(value)
        elif not isinstance(value, dict):
            parsedParam = self.parseLeaf(leafName, value)
        else:
            value = value.copy()
            # Format nicely for OptionsDict creation
            pVal = value.pop("value", None)
            # Ignore name, since it comes from the leaf name
            value.pop("name", None)
            nameArgs = {
                "value": pVal,
                "type": value.pop("type", type(pVal).__name__.lower()),
                "helpText": value.pop("helpText", ""),
            }
            # Forward additional args if they exist
            parsedParam = OptionsDict(leafName, **nameArgs, **value)
            # Make sure value is formatted according to its type if needed
            if parsedParam.type.lower() not in _yamlReps:
                deserialized, err = deserialize(parsedParam, [str(parsedParam.value)])
                if len(deserialized):
                    # Success
                    parsedParam.value = deserialized[0]
                else:
                    # error, don't keep the new value
                    pass
        return parsedParam

    def parseLeaf(self, parameterName: str, value: Any):
        leafParam = OptionsDict(parameterName, value)
        value = leafParam.value
        if isinstance(value, bool):
            pass
            # Keeps 'int' from triggering
        elif isinstance(value, float):
            leafParam.type = "float"
        elif isinstance(value, int):
            leafParam.type = "int"

        elif isinstance(value, list):
            leafParam.type = "list"
            testVal = value[0]
            if isinstance(testVal, dict):
                # Value is on the other side of the mapping
                testVal = next(iter(testVal.values()))
            # list of simple values, implied these are the limits. Since no default
            # is specified, it'll be the first in the list
            leafParam.opts["limits"] = value
            leafParam.value = testVal
        return leafParam

    def getNestedConfigName(self, namePath: NestedIndexer):
        if isinstance(namePath, str):
            namePath = (namePath,)
        out = self.config
        while len(namePath) > 0:
            out = out[namePath[0]]
            namePath = namePath[1:]
        return out
