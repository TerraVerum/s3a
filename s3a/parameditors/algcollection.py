from __future__ import annotations

import importlib
import inspect
import pydoc
import types
import typing as t
import webbrowser
from collections import defaultdict
from pathlib import Path

from pyqtgraph.Qt import QtCore
from qtextras import (
    ParameterContainer,
    ParameterEditor,
    RunOptions,
    bindInteractorOptions as bind,
    fns,
)
from qtextras.typeoverloads import FilePath

from . import MetaTreeParameterEditor
from ..constants import PRJ_ENUMS
from ..processing.pipeline import (
    PipelineFunction,
    PipelineParameter,
    PipelineStageType,
    maybeGetFunction,
)

Signal = QtCore.Signal
_topDictType = t.Dict[str, t.Union[t.List[str], PipelineParameter]]
_primitiveDictType = t.Dict[
    str, t.Union[PipelineFunction, t.List[str], PipelineParameter]
]


class _CollectionDict(t.TypedDict):
    top: _topDictType
    primitive: _primitiveDictType
    modules: t.List[str]


def _peekFirst(iterable):
    return next(iter(iterable))


def onlyFirstKeyValue(dict_):
    key, value = _peekFirst(dict_.items())
    return {key: value}


class AlgorithmEditor(MetaTreeParameterEditor):
    sigProcessorChanged = QtCore.Signal(str)
    """Name of newly selected process"""

    def __init__(self, collection: "AlgorithmCollection" = None, **kwargs):
        super().__init__(**kwargs)
        if collection is None:
            collection = AlgorithmCollection()
        self.collection = collection

        processName = _peekFirst(self.collection.topProcesses)
        proc = collection.parseProcessName(processName)
        # Set to None first to force switch, init states
        self.currentProcessor = proc

        self.props = ParameterContainer()
        self.registerFunction(
            self.changeActiveProcessor,
            runOptions=RunOptions.ON_CHANGED,
            parent=self._metaParameter,
            process=processName,
            container=self.props,
        )
        fns.setParametersExpanded(self._metaTree)
        procSelector = self.props.parameters["process"]
        self.collection.stateManager.signals.updated.connect(
            lambda: procSelector.setLimits(list(self.collection.topProcesses))
        )

        def onChange(name):
            with fns.makeDummySignal(procSelector, "sigValueChanged"):
                procSelector.setValue(name)
                # Manually set item labels since valueChange was forcefully disconnected
                for item in procSelector.items:
                    item.valueChanged(procSelector, name)

        self.sigProcessorChanged.connect(onChange)
        self.changeActiveProcessor(proc)

    def saveParameterValues(
        self,
        saveName: str = None,
        parameterState: dict = None,
        *,
        includeDefaults=False,
        **kwargs,
    ):
        """
        The algorithm editor also needs to store information about the selected
        process, so lump this in with the other parameter information before calling
        default save.
        """
        proc = self.currentProcessor
        # Make sure any newly added stages are accounted for
        if parameterState is None:
            # Since inner nested processes are already recorded, flatten here to just
            # save updated parameter values for the outermost stage
            parameterState = self.unnestedProcessState(proc, filter=("meta",))
        self.collection.loadParameterValues(
            self.collection.stateManager.stateName, parameterState
        )
        clctnState = self.collection.saveParameterValues(saveName, blockWrite=True)
        parameterState = {"active": self.currentProcessor.title(), **clctnState}
        return super().saveParameterValues(
            saveName, parameterState, includeDefaults=includeDefaults, **kwargs
        )

    def loadParameterValues(
        self, stateName: FilePath = None, stateDict: dict = None, **kwargs
    ):
        if stateDict is None:
            stateDict = fns.attemptFileLoad(self.stateManager.formatFileName(stateName))
        processName = stateDict.pop("active", None)

        self.collection.loadParameterValues(stateName, stateDict, **kwargs)
        if processName:
            self.changeActiveProcessor(processName, saveBeforeChange=False)
        return super().loadParameterValues(
            stateName, {}, candidateParameters=[], **kwargs
        )

    @bind(process=dict(type="popuplineeditor", limits=[], title="Algorithm"))
    def changeActiveProcessor(
        self, process: str | PipelineParameter, saveBeforeChange=True
    ):
        """
        Changes which processor is active.

        Parameters
        ----------
        process
            Processor to load
        saveBeforeChange
            Whether to propagate current algorithm settings to the processor collection
            before changing
        """
        # TODO: Maybe there's a better way of doing this? Ensures process label is updated
        #  for programmatic calls
        if saveBeforeChange:
            self.saveParameterValues(self.stateManager.stateName, blockWrite=True)
        process = self._resolveProccessor(process)
        if process is None:
            return
        self.rootParameter.clearChildren()
        self.currentProcessor = process
        fns.setParametersExpanded(self.tree)
        self.sigProcessorChanged.emit(process.title())

    def _resolveProccessor(self, processor):
        if isinstance(processor, str):
            processor = self.collection.parseProcessName(processor)
        if processor == self.currentProcessor:
            return None
        return processor

    def editParameterValuesGui(self):
        webbrowser.open(self.stateManager.formatFileName())

    def unnestedProcessState(self, process: PipelineParameter, **kwargs):
        outState = dict(top={}, primitive={}, modules=self.collection.includeModules)

        # Make sure to visit the most deeply nested stages first, so that stages
        # can be accurately ignored if they are already included in a parent stage
        visit = [process]
        depths = defaultdict(int, {process: 0})
        while visit:
            pipe = visit.pop()
            for child in filter(lambda el: isinstance(el, PipelineParameter), pipe):
                depths[child] = max(depths[child], depths[pipe] + 1)
                visit.append(child)

        for pipe in sorted(depths, key=depths.get, reverse=True):
            # Don't record meta changes for top process since it breaks
            # logic for loading from a collection.
            # Do this by only keeping the first key (non-meta information)
            dest = "top" if pipe is process else "primitive"
            outState[dest].update(
                onlyFirstKeyValue(pipe.saveState(recurse=False, **kwargs))
            )
        return outState


class AlgorithmCollection(ParameterEditor):
    def __init__(
        self,
        processType=PipelineParameter,
        suffix=".alg",
        template: FilePath = None,
        **kwargs,
    ):
        super().__init__(suffix=suffix, **kwargs)
        self.processType = processType
        self.primitiveProcesses: _primitiveDictType = {}
        self.topProcesses: _topDictType = {}
        self.includeModules: list[str] = []

        if template is not None:
            self.loadParameterValues(template)

    def saveStagesByReference(
        self,
        process: PipelineParameter,
        **kwargs,
    ):
        """
        To prevent duplication of stages that are already present in ``primitive``,
        replace ``primitive`` stages with their names in the top-level state.
        """
        processState = process.saveState(**kwargs)
        procTitle, allChildStates = _peekFirst(processState.items())

        for child, childState in zip(process, allChildStates):
            if not isinstance(child, PipelineParameter):
                continue
            childTitle = _peekFirst(childState)
            if childTitle in self.primitiveProcesses:
                # Set to blank; the presence of the child name will indicate
                # fetch should be from `primitive` dict
                childState[childTitle] = {}

        # Simplify dict of {chname: {}} to just chname
        for ii, childState in enumerate(allChildStates):
            if len(childState) == 1:
                childState[ii] = _peekFirst(childState)
        return processState

    def addProcess(self, process: PipelineStageType, top=False, force=False):
        addDict = self.topProcesses if top else self.primitiveProcesses
        isFunction = isinstance(process, PipelineFunction)
        title = process.title()
        saveObj = {title: process}

        if force or title not in addDict or type(addDict[title]) != type(process):
            addDict.update(saveObj)
        if isFunction:
            return title

        for stage in process:
            # Don't recurse 'top', since it should only hold the directly passed process
            if function := maybeGetFunction(stage) or isinstance(
                stage, PipelineParameter
            ):
                self.addProcess(function or stage, top=False, force=force)
        return process.name()

    def addAllModuleProcesses(self, module: str | types.ModuleType, force=False):
        if isinstance(module, str):
            module = importlib.import_module(module)

        added = []
        for name, process in inspect.getmembers(module):
            if isinstance(process, PipelineStageType.__args__):
                added.append(self.addProcess(process, force=force))
                continue
            if (
                not hasattr(process, "__module__")
                or process.__module__ != module.__name__
            ):
                continue
            if inspect.isclass(process):
                try:
                    process = process()
                except TypeError:
                    # Needs arguments
                    continue
                added.append(self.addProcess(process, force=force))
            elif callable(process):
                added.append(self.addFunction(process, force=force))
        return added

    def addFunction(self, func: t.Callable, top=False, **kwargs):
        """
        Helper function to wrap a function in a pipeline process and add it as a
        stage
        """
        return self.addProcess(PipelineFunction(func, **kwargs), top)

    def parseProcessName(
        self,
        processName: str,
        topFirst=True,
        **kwargs,
    ):
        """
        From a list of search locations (ranging from most primitive to topmost),
        find the first processor matching the specified name. If 'topFirst' is chosen,
        the search locations are parsed in reverse order.
        """
        proc = self.fetchProcess(
            processName, topFirst=topFirst, **kwargs
        ) or self.parseProcessQualname(processName, **kwargs)

        if proc is None:
            raise ValueError(f"Process `{processName}` not recognized")
        if not isinstance(proc, PipelineStageType.__args__):
            raise ValueError(
                f"Parsed `{processName}`, but got non-pipelinable result: {proc}"
            )

        return proc

    def fetchProcess(self, processName: str, searchDicts=None, topFirst=True, **kwargs):
        if searchDicts is None:
            searchDicts = [self.topProcesses, self.primitiveProcesses]
        if topFirst:
            searchDicts = searchDicts[::-1]
        proc = searchDicts[0].get(processName, searchDicts[1].get(processName))
        if isinstance(proc, (type(None), *PipelineStageType.__args__)):
            return proc
        elif isinstance(proc, list):
            return self.pipelineFromStages(proc, name=processName, **kwargs)
        else:
            raise ValueError(f"Unknown process type: {type(proc)}")

    def pipelineFromStages(
        self,
        stages: t.Sequence[dict | str],
        name: str = None,
        add=PRJ_ENUMS.PROCESS_NO_ADD,
        allowOverwrite=False,
    ):
        """
        Creates a :class:`PipelineParameter` from a sequence of process stages and
        optional name

        Parameters
        ----------
        stages
            Stages to parse
        name
            Pipeline name, defaults to ``:function:fns.nameFormatter(<unnamed>)`
        add
            Whether to add this new pipeline to the current collection's top or
            primitive process blocks, or to not add at all (if ``NO_ADD``)
        allowOverwrite
            If `add` is *True*, this determines whether the new process can overwite an
            already existing proess. If ``add=False``, this value is ignored.
        """
        out = self.processType(name=name)
        for stageName in stages:
            if isinstance(stageName, dict):
                stage = self.parseProcessDict(stageName)
            else:
                stage = self.parseProcessName(stageName, topFirst=False)
            out.addStage(stage)
        exists = out.name in self.topProcesses
        if add is not PRJ_ENUMS.PROCESS_NO_ADD and (not exists or allowOverwrite):
            self.addProcess(
                out, top=add == PRJ_ENUMS.PROCESS_ADD_TOP, force=allowOverwrite
            )
        return out

    def parseProcessDict(self, processDict: dict, topFirst=False):
        # 1. First key is always the process name, values are new inputs for any
        # matching process
        # 2. Second key is whether the process is disabled
        processDict = processDict.copy()
        processName, updateKwargs = _peekFirst(processDict.items())
        processDict.pop(processName)
        proc = self.parseProcessName(processName, topFirst=topFirst)
        if isinstance(updateKwargs, list):
            # TODO: Determine policy for loading nested procs, outer should already
            #  know about inner so it shouldn't _need_ to occur, given outer would've
            #  been saved previously
            raise ValueError("Parsing deep nested processes is currently undefined")
        elif updateKwargs:
            # TODO: Determine when it makes sense to override defaults when kwargs
            #  are passed
            proc.updateInput(updateKwargs)
        # Check for process-level traits
        # For PipelineFunctions, "enabled" status lives with the ActionGroupParameter
        # so query the parent PipelineParameter for its ActionGroup child and set
        # *those* options
        optsSetter = (
            proc.parent.child(proc.__name__)
            if isinstance(proc, PipelineFunction)
            else proc
        )
        optsSetter.setOpts(**processDict)
        return proc
        # TODO: Add recursion, if it's something that will be done. For now, assume
        #  only 1-depth nesting. Otherwise, it's hard to distinguish between actual
        #  input optiosn and a nested process

    def parseProcessQualname(self, processName: str, **kwargs):

        # Special case: Qualname-loaded procs should be added under their qualname
        # otherwise they won't be rediscoverable after saving->restarting S3A
        for prefix in ["", *self.includeModules]:
            fullModuleName = ".".join([prefix, processName])
            proc: t.Any = pydoc.locate(fullModuleName)
            if proc is not None:
                break

        success = True
        if inspect.isclass(proc) and issubclass(proc, PipelineStageType.__args__):
            # False positive assuming only `object` return type
            # noinspection PyCallingNonCallable
            proc: PipelineStageType = proc(**kwargs)
        elif callable(proc) and not isinstance(proc, PipelineFunction):
            proc = PipelineFunction(proc, **kwargs)
        else:
            success = False
        if success:
            if isinstance(proc, PipelineFunction):
                proc.__name__ = processName
            else:
                proc.setOpts(name=processName)
            return proc
        # else
        return None

    def saveParameterValues(
        self, saveName: str = None, parameterState: dict = None, **kwargs
    ):
        def converter(procDict):
            return {
                name: self.saveStagesByReference(stage)[name]
                if isinstance(stage, PipelineParameter)
                else stage
                for name, stage in procDict.items()
                if not isinstance(stage, PipelineFunction)
            }

        if parameterState is None:
            parameterState = {
                "top": converter(self.topProcesses),
                "primitive": converter(self.primitiveProcesses),
                "modules": self.includeModules,
            }
        return super().saveParameterValues(saveName, parameterState, **kwargs)

    def loadParameterValues(
        self,
        stateName: t.Union[str, Path] = None,
        stateDict: _CollectionDict = None,
        **kwargs,
    ):
        if stateDict is None:
            stateDict = self.stateManager.loadState(stateName)
        top, primitive = stateDict.get("top", {}), stateDict.get("primitive", {})
        modules = stateDict.get("modules", [])
        self.includeModules = modules
        self.topProcesses.update(top)
        self.primitiveProcesses.update(primitive)
        stateDict = self.saveParameterValues(blockWrite=True)
        super().loadParameterValues(stateName, stateDict, candidateParameters=[])
        return stateDict
