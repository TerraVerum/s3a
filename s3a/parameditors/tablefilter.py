import sys
import warnings
from typing import List

import numpy as np
import pandas as pd
from pyqtgraph.parametertree import Parameter
from pyqtgraph.Qt import QtWidgets
from qtextras import OptionsDict, ParameterEditor, fns


def generateParameterList(nameIter, paramType, defaultValue, defaultParam="value"):
    """Helper for generating children elements"""
    return [
        {"name": name, "type": paramType, defaultParam: defaultValue}
        for name in nameIter
    ]


def _filterForParameter(parameter: OptionsDict):
    """Constructs a filter for the parameter based on its type"""
    children = []
    parameterType = parameter.type.lower()
    paramWithChildren = {"name": parameter.name, "type": "group", "children": children}
    children.append(dict(name="Active", type="bool", value=False))
    if parameterType in ["int", "float"]:
        retVal = generateParameterList(["min", "max"], parameterType, 0)
        retVal[0]["value"] = -sys.maxsize
        retVal[1]["value"] = sys.maxsize
        children.extend(retVal)
    elif parameterType in ["enum", "list", "popuplineeditor", "bool"]:
        if parameterType == "enum":
            iterGroup = [param for param in parameter.value]
        elif parameterType == "bool":
            iterGroup = [f"{parameter.name}", f"Not {parameter.name}"]
        else:  # parameterType == 'list' or 'popuplineeditor'
            iterGroup = parameter.opts["limits"]
        optsParam = Parameter.create(
            name="Options", type="checklist", limits=iterGroup, value=iterGroup
        )
        paramWithChildren = Parameter.create(**paramWithChildren)
        paramWithChildren.addChild(optsParam)
    elif "xyvertices" in parameterType:
        minMax = _filterForParameter(OptionsDict("", 5))
        minMax.removeChild(minMax.childs[0])
        minMax = minMax.saveState()["children"]
        xyVerts = generateParameterList(
            ["X Bounds", "Y Bounds"], "group", minMax, "children"
        )
        children.extend(xyVerts)
    elif parameterType in ["str", "text"]:
        # Assumes string
        children.append(dict(name="Regex Value", type="str", value=""))
    else:
        # Don't know how to handle the parameter
        return None

    if isinstance(paramWithChildren, dict):
        paramWithChildren = Parameter.create(**paramWithChildren)
    return paramWithChildren


def filterParameterColumn(compDf: pd.DataFrame, column: OptionsDict, filterOpts: dict):
    # TODO: Each type should probably know how to filter itself. That is,
    #  find some way of keeping this logic from just being an if/else tree...
    parameterType = column.type
    # idx 0 = value, 1 = children
    dfAtParam = compDf.loc[:, column]

    if parameterType in ["int", "float"]:
        curmin, curmax = [filterOpts[name] for name in ["min", "max"]]

        compDf = compDf.loc[(dfAtParam >= curmin) & (dfAtParam <= curmax)]
    elif parameterType == "bool":
        filterOpts = filterOpts["Options"]
        allowTrue, allowFalse = [
            name in filterOpts for name in [f"{column.name}", f"Not {column.name}"]
        ]
        validList = np.array(dfAtParam, dtype=bool)
        if not allowTrue:
            compDf = compDf.loc[~validList]
        if not allowFalse:
            compDf = compDf.loc[validList]
    elif parameterType in ["list", "popuplineeditor"]:
        existingParams = np.array(dfAtParam)
        allowedParams = filterOpts["Options"]
        compDf = compDf.loc[np.isin(existingParams, allowedParams)]
    elif parameterType in ["str", "text"]:
        allowedRegex = filterOpts["Regex Value"]
        isCompAllowed = dfAtParam.str.contains(allowedRegex, regex=True, case=False)
        compDf = compDf.loc[isCompAllowed]
    elif parameterType in ["complexxyvertices", "xyvertices"]:
        vertsAllowed = np.ones(len(dfAtParam), dtype=bool)

        xParam = filterOpts["X Bounds"]
        yParam = filterOpts["Y Bounds"]
        xmin, xmax, ymin, ymax = [
            param[val] for param in (xParam, yParam) for val in ["min", "max"]
        ]

        for vertIdx, verts in enumerate(dfAtParam):
            if parameterType == "complexxyvertices":
                stackedVerts = verts.stack()
            else:
                stackedVerts = verts
            xVerts, yVerts = stackedVerts.x, stackedVerts.y
            isAllowed = np.all((xVerts >= xmin) & (xVerts <= xmax)) & np.all(
                (yVerts >= ymin) & (yVerts <= ymax)
            )
            vertsAllowed[vertIdx] = isAllowed
        compDf = compDf.loc[vertsAllowed]
    else:
        warnings.warn(
            "No filter type exists for parameters of type "
            f"{parameterType}."
            f" Did not filter column {column.name}.",
            UserWarning,
            stacklevel=3,
        )
    return compDf


class TableFilterEditor(ParameterEditor):
    def __init__(
        self,
        parameterList: List[OptionsDict] = None,
        name="Component Table Filter",
        directory=None,
        suffix=".filter",
    ):
        if parameterList is None:
            parameterList = []
        filterParams = [
            fil for fil in map(_filterForParameter, parameterList) if fil is not None
        ]
        super().__init__(name=name, directory=directory, suffix=suffix)
        self.rootParameter.addChildren(filterParams)

    def _buildGui(self, **kwargs):
        self.applyButton = QtWidgets.QPushButton("Apply Filter")
        return super()._buildGui(**kwargs)

    def _guiChildren(self) -> list:
        return super()._guiChildren() + [self.applyButton]

    def updateParameterList(self, paramList: List[OptionsDict]):
        newParams = []
        badCols = []
        for param in paramList:
            try:
                curFilter = _filterForParameter(param)
            except KeyError:
                curFilter = None
            if curFilter is None:
                badCols.append(param)
            else:
                newParams.append(curFilter)
        self.rootParameter.clearChildren()
        self.rootParameter.addChildren(newParams)
        if len(badCols) > 0:
            colNames = [f'"{col}"' for col in badCols]
            colTypes = np.unique([f'"{col.type}"' for col in badCols])
            warnings.warn(
                f"The table does not know how to create a filter for fields"
                f' {", ".join(colNames)}'
                f' since types {", ".join(colTypes)} do not have corresponding filters',
                UserWarning,
                stacklevel=2,
            )

    @property
    def activeFilters(self):
        filters = {}
        for child in self.rootParameter.childs:
            if child["Active"]:
                cState = fns.parameterValues(child)
                cState.pop("Active")
                filters[child.name()] = cState
        return filters

    def filterComponentDf(self, compDf: pd.DataFrame):
        strNames = [str(f) for f in compDf.columns]
        for fieldName, opts in self.activeFilters.items():
            try:
                matchIdx = strNames.index(fieldName)
            except IndexError:
                # This filter can be used on dataframes that didn't have to come from S3A,
                # so silently ignore mismatched filter requests
                continue
            col = compDf.columns[matchIdx]
            compDf = filterParameterColumn(compDf, col, opts)
        return compDf
