from __future__ import annotations

import typing as t

import numpy as np
import pyqtgraph as pg
from pyqtgraph.parametertree import InteractiveFunction, Parameter
from pyqtgraph.parametertree.parameterTypes import (
    ActionGroupParameter,
    ActionGroupParameterItem,
)
from pyqtgraph.Qt import QtCore, QtGui
from qtextras import ParameterContainer, ParameterEditor, fns
from qtextras._funcparse import FROM_PREV_IO

from ..generalutils import simpleCache

__all__ = [
    "ActionGroupParameter",
    "ImagePipeline",
    "maybeGetFunction",
    "PipelineFunction",
    "PipelineParameter",
    "PipelineParameterItem",
    "PipelineStageType",
]


class PipelineException(Exception):
    def __init__(self, stage: PipelineFunction):
        super().__init__("Exception during processor run")
        stages = []
        while stage:
            stages.append(stage)
            if isinstance(stage, PipelineStageType.__args__):
                if isinstance(stage, PipelineParameter):
                    stage = stage.parent()
                else:
                    stage = stage.parent
            else:
                stage = None
        # Reverse the list so that the first stage is at the beginning
        self.stages = stages[::-1]

    def __str__(self):
        # Convert stage info into a more readable format
        stages = [a.title() for a in self.stages]
        stagePath = " > ".join(stages)
        stageMsg = f"Stage: {stagePath}\n"
        initialMsg = ": ".join([super().__str__(), str(self.__cause__ or "")])
        return stageMsg + initialMsg


def maybeGetFunction(parameter: Parameter) -> PipelineFunction | None:
    """Check if a parameter is a PipelineFunction"""
    return parameter.opts.get("function")


class PipelineFunction(InteractiveFunction):
    defaultInput: dict[str, t.Any] = {}
    infoKey = "info"

    def __init__(self, function, name: str = None, **kwargs):
        super().__init__(function, **kwargs)
        if name:
            self.__name__ = name

        self.defaultInput = dict(self.input)
        self.parent: PipelineParameter | None = None

        self.result = None

    @classmethod
    def fromInteractive(cls, function: InteractiveFunction, title: str = None):
        obj = cls(function.function, title)
        obj.__dict__.update(function.__dict__)
        # `parameters` and their cache should be copied specifically to avoid clearing
        # them from the reference InteractiveFunction during clears/etc.
        obj.parameters, obj.parameterCache = {}, {}
        obj.hookupParameters(function.parameters, clearOld=False)
        obj.defaultInput = dict(obj.input)
        return obj

    def hookupParameters(self, params=None, clearOld=True):
        super().hookupParameters(params, clearOld)

    @property
    def input(self):
        return ParameterContainer(self.parameters, self.extra)

    def __call__(self, **kwargs):
        try:
            self.result = super().__call__(**kwargs)
        except Exception as ex:
            raise PipelineException(self) from ex
        return self.result

    def stageInfo(self) -> dict | list | None:
        if not isinstance(self.result, dict):
            return None
        return self.result.get(self.infoKey, self.result)

    def __str__(self):
        return super().__str__().replace("InteractiveFunction", type(self).__name__)

    def title(self):
        """
        If ``self`` has parameters, try to find the name of the containing stage,
        otherwise default to a formatted version of the function name
        """
        parent = None
        if self.parameters:
            parent = next(iter(self.parameters.values())).parent()
        if parent:
            return parent.title()
        return fns.nameFormatter(self.__name__)

    def updateInput(self, setDefaults=False, **kwargs):
        toUpdate = {kk: kwargs[kk] for kk in self.input.keys() & kwargs}
        if setDefaults:
            self.defaultInput.update(toUpdate)
        self.input.update(toUpdate)

        if setDefaults and (paramKeys := self.parameters.keys() & toUpdate):
            for kk in paramKeys:
                self.parameters[kk].setDefault(toUpdate[kk])


class PipelineParameterItem(ActionGroupParameterItem):
    def __init__(self, param, depth):
        self.enabledFontMap = None
        super().__init__(param, depth)
        if param.opts["enabled"]:
            # Starts out unchecked, adjust at the start
            self.setCheckState(0, QtCore.Qt.CheckState.Checked)

    def _mkFontMap(self):
        if self.enabledFontMap:
            return
        enabledFont = self.font(0)
        disableFont = QtGui.QFont()
        disableFont.setStrikeOut(True)
        self.enabledFontMap = {True: enabledFont, False: disableFont}

    def optsChanged(self, param, opts):
        super().optsChanged(param, opts)
        if "enabled" in opts:
            enabled = opts["enabled"]
            cs = QtCore.Qt.CheckState
            role = cs.Checked if enabled else cs.Unchecked
            # Bypass subclass to prevent early short-circuit
            self.setCheckState(0, role)
            # This gets called before constructor can finish, so add enabled font map here
            self._mkFontMap()
            self.setFont(0, self.enabledFontMap[enabled])

    def updateFlags(self):
        # It's a shame super() doesn't return flags...
        super().updateFlags()
        flags = self.flags()
        flags |= QtCore.Qt.ItemFlag.ItemIsUserCheckable & (
            ~QtCore.Qt.ItemFlag.ItemIsAutoTristate
        )
        self.setFlags(flags)

    def setData(self, column, role, value):
        if role != QtCore.Qt.ItemDataRole.CheckStateRole:
            return super().setData(column, role, value)
        cs = QtCore.Qt.CheckState
        newEnabled = value == cs.Checked
        if newEnabled == self.param.opts["enabled"]:
            # Ensure no mismatch between param enabled and item checkstate
            super().setData(column, role, value)
        else:
            # `optsChanged` above will handle check state
            self.param.setOpts(enabled=newEnabled)
        return True


class PipelineParameter(ActionGroupParameter):
    itemClass = PipelineParameterItem

    metaKeys = ["enabled"]

    def __init__(self, **opts):
        opts.setdefault("type", "pipelinegroup")
        opts.setdefault("title", fns.nameFormatter(opts.get("name", "")))
        super().__init__(**opts)
        self.sigOptionsChanged.connect(self.optsChanged)

    def addStage(
        self,
        stage: InteractiveFunction | PipelineStageType | t.Callable,
        cache=True,
        stageInputOptions: dict = None,
        **metaOptions,
    ):
        if isinstance(stage, PipelineParameter):
            return self.addChild(stage)
        if isinstance(stage, InteractiveFunction):
            # If already a PipelineFunction, the copy created is still useful
            # in case caching is needed or other modifications are made
            stage = PipelineFunction.fromInteractive(stage)
        elif callable(stage):
            stage = PipelineFunction(stage)
        else:
            raise TypeError("Stage must be callable")

        stage: PipelineFunction
        stage.parent = self
        if cache:
            stage.function = simpleCache(stage.function)

        registered = ParameterEditor.defaultInteractor(
            stage, parent=self, runOptions=[], **(stageInputOptions or {})
        )
        # Override item class to allow checkboxes on stages
        registered.itemClass = PipelineParameterItem
        registered.setOpts(title=stage.title(), function=stage, **metaOptions)
        return registered

    def activate(self, **kwargs):
        super().activate()
        if not self.opts["enabled"]:
            return kwargs
        for child in self.children():  # type: ActionGroupParameter
            if isinstance(child, PipelineParameter):
                kwargs.update(child.activate(**kwargs))
            if not child.opts["enabled"] or not (function := maybeGetFunction(child)):
                continue
            useKwargs = {k: v for k, v in kwargs.items() if k in function.input}
            output = function(**useKwargs)
            if isinstance(output, dict):
                kwargs.update(output)
        return kwargs

    def optsChanged(self, _param, opts):
        if "enabled" not in opts:
            return
        enabled = opts["enabled"]
        for child in self:
            if isinstance(child, PipelineParameter) or maybeGetFunction(child):
                child.setOpts(enabled=enabled)

    def flattenedFunctions(self) -> list[PipelineFunction]:
        """
        Return a list of all stages comprising this pipeline without any hierarchical
        structure.
        """
        stages = []
        for child in self.children():
            if function := maybeGetFunction(child):
                stages.append(function)
            elif isinstance(child, PipelineParameter):
                stages.extend(child.flattenedFunctions())
        return stages

    def saveState(self, filter=("meta",), recurse=True) -> dict[str, t.Any]:
        children = []
        includeDefaults = "defaults" in filter
        includeMeta = "meta" in filter
        stateTitle, meta = self.getTitleAndMaybeMetadata(includeMeta)
        for child in self:
            if maybeGetFunction(child):
                childState = self._actionGroupSaveState(
                    child,
                    includeDefaults,
                    includeMeta,
                )
            elif isinstance(child, PipelineParameter):
                if recurse:
                    childState = child.saveState(filter)
                else:
                    childState = child._saveStateWithoutChildren(filter)
            else:
                curFilter = "user" if "user" in filter else None
                childState = child.saveState(curFilter)
            children.append(childState)
        return {stateTitle: children, **meta}

    @classmethod
    def resolveStateTitle(cls, parameter: Parameter, metaDict: dict | None = None):
        title = parameter.title()
        if fns.nameFormatter(parameter.name()) != title:
            if metaDict is not None:
                metaDict["title"] = title
            title = parameter.name()
        return title

    def getTitleAndMaybeMetadata(self, includeMeta=True):
        meta = {}
        stateTitle = self.resolveStateTitle(self, meta)
        if not includeMeta:
            return stateTitle, {}
        if not self.opts["enabled"]:
            meta["enabled"] = False
        return stateTitle, meta

    def _saveStateWithoutChildren(
        self, filter=("meta",), flatten=False
    ) -> dict[str, t.Any] | str:
        stateTitle, meta = self.getTitleAndMaybeMetadata("meta" in filter)
        stateKwargs = self.getFunctionKwargs(
            flatten=flatten, includeDefaults="defaults" in filter
        )
        if not meta and not stateKwargs:
            return stateTitle
        else:
            return {stateTitle: stateKwargs, **meta}

    @classmethod
    def _actionGroupSaveState(
        cls, parameter: ActionGroupParameter, includeDefaults=False, includeMeta=False
    ):
        """
        Action group is defined and used in other contexts, so have a special override
        for the context of pipeline usage
        """
        assert (function := maybeGetFunction(parameter))
        sentinal = object()
        changedValues = {}
        meta = {}
        for kk, vv in function.input.items():
            if vv is not FROM_PREV_IO and (
                includeDefaults
                or not pg.eq(vv, function.defaultInput.get(kk, sentinal))
            ):
                changedValues[kk] = vv
        if includeMeta and not parameter.opts["enabled"]:
            meta["enabled"] = False
        stateTitle = cls.resolveStateTitle(
            parameter, metaDict=meta if includeMeta else None
        )

        if not meta and not changedValues:
            # No need to save anything other than the name
            return stateTitle
        return {stateTitle: changedValues, **meta}

    def updateInput(self, setDefaults=False, **kwargs):
        for func in self.flattenedFunctions():
            func.updateInput(setDefaults=setDefaults, **kwargs)

    def getFunctionKwargs(self, flatten=False, includeDefaults=False):
        """
        Returns all keywords of all functions in this pipeline

        Parameters
        ----------
        flatten
            If ``True``, returns kwargs for all functions in recursively nested
            pipelines as a single dictionary. Otherwise, only returns keywords
            for ``PipelineFunction`` objects in this pipeline.
        includeDefaults
            If ``True``, includes default values for all parameters in the returned
            dictionary. Otherwise, only returns values that have been changed from
            the default.
        """
        if flatten:
            stages = self.flattenedFunctions()
        else:
            stages = list(filter(None, map(maybeGetFunction, self)))

        def valueFilter(function, key):
            value = function.input[key]
            return value is not FROM_PREV_IO and (
                includeDefaults
                or not pg.eq(value, function.defaultInput.get(key, object()))
            )

        return {
            k: v
            for func in stages
            for k, v in func.input.items()
            if valueFilter(func, k)
        }


class ImagePipeline(PipelineParameter):
    keepKeys = ["image"]

    def __init__(self, **opts):
        super().__init__(**opts)
        self._winRefs = []

    def extractUniqueInfos(self):
        # Flatten any list-of-lists encountered
        stages = self.flattenedFunctions()
        flattened = []
        lastInfo = None
        for stage in stages:
            infos = stage.stageInfo() or {}
            if not isinstance(infos, list):
                infos = [infos]
            for info in infos:
                if not isinstance(info, dict):
                    raise ValueError("Stage info must be a dict or list of dicts")
                info = {k: v for k, v in info.items() if k in self.keepKeys}
                if info is None or pg.eq(info, lastInfo):
                    continue
                lastInfo = info.copy()
                info["title"] = fns.nameFormatter(stage.__name__)
                flattened.append(info)
        return flattened

    def _stageSummaryWidget(self):
        flattened = self.extractUniqueInfos()
        numStages = len(flattened)
        nrows = np.sqrt(numStages).astype(int)
        ncols = np.ceil(numStages / nrows)
        outGrid = pg.GraphicsLayoutWidget()
        sizeToAxMapping: t.Dict[tuple, pg.PlotItem] = {}
        for ii, info in enumerate(flattened):
            if info is None:
                continue
            pltItem: pg.PlotItem = outGrid.addPlot(title=info.get("title", None))
            vb = pltItem.getViewBox()
            vb.invertY(True)
            vb.setAspectLocked(True)
            npImg = info["image"]
            imShp = npImg.shape[:2]
            margin = np.array(imShp) * 2
            lim = max(margin + imShp)
            vb.setLimits(maxXRange=lim, maxYRange=lim)
            sameSizePlt = sizeToAxMapping.get(imShp, None)
            if sameSizePlt is not None:
                pltItem.setXLink(sameSizePlt)
                pltItem.setYLink(sameSizePlt)
            sizeToAxMapping[imShp] = pltItem
            imageItem = pg.ImageItem(npImg)
            pltItem.addItem(imageItem)

            if ii % ncols == ncols - 1:
                outGrid.nextRow()
        # See https://github.com/pyqtgraph/pyqtgraph/issues/1348. strange zooming occurs
        # if aspect is locked on all figures
        # for ax in sizeToAxMapping.values():
        #   ax.getViewBox().setAspectLocked(True)
        oldClose = outGrid.closeEvent

        def newClose(ev):
            self._winRefs.remove(outGrid)
            oldClose(ev)

        oldResize = outGrid.resizeEvent

        def newResize(ev):
            oldResize(ev)
            for ax in sizeToAxMapping.values():
                ax.getViewBox().autoRange()

        # Windows that go out of scope get garbage collected. Prevent that here
        self._winRefs.append(outGrid)
        outGrid.closeEvent = newClose
        outGrid.resizeEvent = newResize

        return outGrid

    def stageSummaryGui(self):
        if self.flattenedFunctions()[-1].result is None:
            raise RuntimeError(
                "Analytics can only be shown after the algorithm was run."
            )
        outGrid = self._stageSummaryWidget()
        outGrid.showMaximized()

        def fixedShow():
            for item in outGrid.ci.items:
                item.getViewBox().autoRange()

        QtCore.QTimer.singleShot(0, fixedShow)
        return outGrid


PipelineStageType = t.Union[PipelineFunction, PipelineParameter]
