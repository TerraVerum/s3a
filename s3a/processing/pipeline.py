from __future__ import annotations

import copy
from inspect import Parameter as IParameter
import typing as t

import numpy as np
import pyqtgraph as pg
from pyqtgraph.parametertree import InteractiveFunction, Interactor, Parameter
from pyqtgraph.Qt import QtCore
from qtextras import (
    FROM_PREV_IO,
    ChainedActionGroupParameter,
    ParameterContainer,
    ParameterEditor,
    fns,
)
from qtextras.shims import ActionGroupParameter

from ..generalutils import augmentException, simpleCache

__all__ = [
    "ActionGroupParameter",
    "ImagePipeline",
    "maybeGetFunction",
    "PipelineFunction",
    "PipelineParameter",
    "PipelineStageType",
    "StageAncestryPrinter",
]


class StageAncestryPrinter:
    """
    Used to format exceptions raised during pipeline processing so they show the full
    path to the stage that caused the error
    """

    def __init__(self, stage: PipelineFunction):
        stages = []
        while stage and isinstance(stage, PipelineStageType.__args__):
            stages.append(stage)
            stage = stage.parent()
        # Reverse the list so that the first stage is at the beginning
        self.stages = stages[::-1]

    def __str__(self):
        # Convert stage info into a more readable format
        stages = [a.title() for a in self.stages]
        return " > ".join(stages)


def maybeGetFunction(parameter: Parameter) -> PipelineFunction | None:
    """Check if a parameter is a PipelineFunction"""
    return parameter.opts.get("function")


class PipelineFunction(InteractiveFunction):
    defaultInput: dict[str, t.Any] = {}
    infoKey = "info"
    cacheable = True

    def __init__(self, function, name: str = None, **kwargs):
        super().__init__(function, **kwargs)
        if name:
            self.__name__ = name

        self.defaultInput = dict(self.input)
        self._parent: PipelineParameter | None = None

        self.result = None

    @classmethod
    def fromInteractive(
        cls, interactive: InteractiveFunction, name: str = None
    ) -> t.Self:
        if isinstance(interactive, cls):
            obj = copy.copy(interactive)
        else:
            obj = cls(interactive.function, name=name)
            obj.__dict__.update(interactive.__dict__)
        func = obj.function
        # Special case: interactives whose function is bound to themselves need to be
        # rebound to `obj` so that the `self` argument is correct
        if (
            hasattr(func, "__self__")
            and func.__self__ is interactive
            and hasattr(obj, func.__name__)
        ):
            obj.function = getattr(obj, obj.function.__name__)

        # `parameters` and their cache should be copied specifically to avoid clearing
        # them from the reference InteractiveFunction during clears/etc.
        obj.parameters, obj.parameterCache = {}, {}
        obj.hookupParameters(interactive.parameters.values(), clearOld=False)
        obj: cls
        return obj

    def hookupParameters(self, params=None, clearOld=True):
        ret = super().hookupParameters(params, clearOld)
        self.defaultInput = self.extra.copy()
        for param in self.parameters.values():
            self.defaultInput[param.name()] = param.defaultValue()
        # TODO: connect `defaultInput` values to signals when a parameter's default
        #   changes

        return ret

    def parent(self):
        """
        Normally, this would conform to standards and be an attribute, but
        to be compatible with a ``Parameter`` parent, it must be a method
        """
        return self._parent

    def setParent(self, parent):
        self._parent = parent

    @property
    def input(self):
        return ParameterContainer(self.parameters, self.extra)

    def __call__(self, **kwargs):
        try:
            self.result = super().__call__(**kwargs)
        except Exception as ex:
            augmentException(ex, f"{StageAncestryPrinter(self)}\n")
            raise
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
        currentInput = self.input
        toUpdate = {kk: kwargs[kk] for kk in currentInput.keys() & kwargs}
        for name, value in toUpdate.items():
            inspectParam = IParameter(
                name, IParameter.POSITIONAL_OR_KEYWORD, default=currentInput[name]
            )
            # Allows setting value, parameter options, and more through "updateInput"
            sanitized = ParameterEditor.defaultInteractor.createFunctionParameter(
                name, inspectParam, value
            )
            # keys added by default should not be changeable by this function
            for kk in ("name", "type", "title"):
                sanitized.pop(kk, None)
            if setDefaults:
                sanitized.setdefault("default", sanitized["value"])
                self.defaultInput[name] = sanitized["default"]
            if name in self.parameters:
                self.parameters[name].setOpts(**sanitized)
            else:
                self.extra[name] = sanitized["value"]


class PipelineParameter(ChainedActionGroupParameter):
    metaKeys = ["enabled"]

    def __init__(self, **opts):
        opts.setdefault("type", "pipelinegroup")
        opts.setdefault("title", fns.nameFormatter(opts.get("name", "")))
        super().__init__(**opts)

    def addStage(
        self,
        stage: InteractiveFunction | PipelineStageType | t.Callable,
        *,
        interactor: Interactor = None,
        cache=True,
        stageInputOptions: dict = None,
        **metaOptions,
    ):
        if interactor is None:
            interactor = ParameterEditor.defaultInteractor
        stage = self._resolveStage(stage, cache)
        stage = super().addStage(
            stage,
            interactor=interactor,
            stageInputOptions=stageInputOptions,
            **metaOptions,
        )
        if isinstance(stage, PipelineParameter) and stageInputOptions:
            stage.updateInput(**stageInputOptions)
        return stage

    def _resolveStage(self, stage, cache=True):
        if isinstance(stage, InteractiveFunction):
            stage = PipelineFunction.fromInteractive(stage)
            stage.setParent(self)
        elif callable(stage) and not isinstance(stage, ChainedActionGroupParameter):
            stage = PipelineFunction(stage)
            stage.setParent(self)
        if cache and isinstance(stage, PipelineFunction) and stage.cacheable:
            stage.function = simpleCache(stage.function)

        return stage

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

    @property
    def result(self):
        stages = self.flattenedFunctions()
        if not stages:
            return None
        return stages[-1].result

    def saveState(self, filter=("meta",), recurse=True) -> dict[str, t.Any]:
        children = []
        addDefaults = "defaults" in filter
        includeMeta = "meta" in filter
        stateTitle, meta = self.getTitleAndMaybeMetadata(includeMeta)
        for child in self:
            if maybeGetFunction(child):
                childState = self._actionGroupSaveState(
                    child,
                    addDefaults,
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
            flatten=flatten, addDefaults="defaults" in filter
        )
        if not meta and not stateKwargs:
            return stateTitle
        else:
            return {stateTitle: stateKwargs, **meta}

    @classmethod
    def _actionGroupSaveState(
        cls, parameter: ActionGroupParameter, addDefaults=False, includeMeta=False
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
                addDefaults or not pg.eq(vv, function.defaultInput.get(kk, sentinal))
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

    def getFunctionKwargs(self, flatten=False, addDefaults=False):
        """
        Returns all keywords of all functions in this pipeline

        Parameters
        ----------
        flatten
            If ``True``, returns kwargs for all functions in recursively nested
            pipelines as a single dictionary. Otherwise, only returns keywords
            for ``PipelineFunction`` objects in this pipeline.
        addDefaults
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
                addDefaults
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
            if info is None or "image" not in info:
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
        if self.result is None:
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
