from __future__ import annotations

import copy
import inspect
import pydoc
import typing as t
from pathlib import Path
from typing import Dict, List, Callable, Union, Type

import numpy as np
from pyqtgraph.Qt import QtCore, QtWidgets
from pyqtgraph.parametertree import Parameter
from typing_extensions import TypedDict

from s3a.constants import MENU_OPTS_DIR
from utilitys import NestedProcess, RunOpts
from utilitys import ParamEditor, NestedProcWrapper, fns, ProcessStage, AtomicProcess
from utilitys.fns import nameFormatter
from utilitys.widgets import EasyWidget

Signal = QtCore.Signal
_procDict = Dict[str, List[str]]

class _AlgClctnDict(TypedDict):
  top: List[Union[str, dict]]
  primitive: List[Union[str, dict]]


class AlgCollection(ParamEditor):
  # sigProcessorCreated = Signal(object) # Signal(AlgCollectionEditor)
  def __init__(self, procWrapType=NestedProcWrapper, procType=NestedProcess, parent=None, fileType='alg',
               **kwargs):
    super().__init__(parent, fileType=fileType, **kwargs)
    self.procWrapType = procWrapType
    self.procType = procType
    self.primitiveProcs: Dict[str, Union[ProcessStage, List[str]]] = {}
    self.topProcs: _procDict = {}
    self.includeModules: List[str] = []

  def createProcessorEditor(self, saveDir: Union[str, Type], editorName='Processor') -> AlgParamEditor:
    """
    Creates a processor editor capable of dynamically loading, saving, and editing collection processes

    :param saveDir: Directory for saved states. If a class type is provided, the __name__ of this class is used.
      Note: Either way, the resulting name is lowercased before being applied.
    :param editorName: The name of the spawned editor
    """
    if not isinstance(saveDir, str):
      saveDir = saveDir.__name__
    formattedClsName = fns.pascalCaseToTitle(saveDir)
    editorDir = MENU_OPTS_DIR/formattedClsName.lower()
    return AlgParamEditor(self, saveDir=editorDir, fileType=self.fileType, name=editorName)

  def addProcess(self, proc: ProcessStage, top=False, force=False):

    addDict = self.topProcs if top else self.primitiveProcs
    saveObj = {proc.name: proc} if isinstance(proc, AtomicProcess) else proc.saveState_flattened()
    if force or proc.name not in addDict:
      addDict.update(saveObj)
    for stage in proc:
      # Don't recurse 'top', since it should only hold the directly passed process
      self.addProcess(stage, False, force)
    return proc.name

  def addFunction(self, func: Callable, top=False, **kwargs):
    """Helper function to wrap a function in an atomic process and add it as a stage"""
    return self.addProcess(AtomicProcess(func, **kwargs), top)

  def parseProcStages(self, stages: t.Sequence[Union[dict, str]], name:str=None, add=False, allowOverwrite=False):
    """
    Creates a nested process from a sequence of process stages and optional name
    :param stages: Stages to parse
    :param name: Name of the nested process
    :param add: Whether to add this new process to the current collection
    :param allowOverwrite: If `add` is *True*, this determines whether the new process can overwite an already existing
      proess. If `add` is *False*, this value is ignored.
    """
    out = self.procType(name)
    for stageName in stages:
      if isinstance(stageName, dict):
        stage = self.parseProcDict(stageName)
      else:
        stage = self.parseProcName(stageName, topFirst=False)
      out.addProcess(stage)
    exists = out.name in self.topProcs
    if add and (not exists or allowOverwrite):
      self.addProcess(out, allowOverwrite)
    return out

  def parseProcName(self, procName: str, topFirst=True):
    procDicts = [self.primitiveProcs, self.topProcs]
    if topFirst:
      procDicts = procDicts[::-1]
    proc = procDicts[0].get(procName, procDicts[1].get(procName))
    # It could still be in an include module, cache if found
    if not proc:
      for module in self.includeModules:
        proc = self.parseProcModule(module, procName)
        if proc:
          # Success, make sure to cache this in processes
          # Top processes must be nested
          self.addProcess(proc, topFirst and isinstance(proc, NestedProcess))
          break
    if proc is None:
      raise ValueError(f'Process "{procName}" not recognized')
    if not isinstance(proc, ProcessStage):
      proc = self.parseProcStages(proc, procName)
    else:
      proc = copy.deepcopy(proc)
      # Default to disableale stages. For non-disablable, use parseDict
      proc.allowDisable = True
    return proc

  def parseProcDict(self, procDict: dict, topFirst=False):
    # 1. First key is always the process name, values are new inputs for any matching process
    # 2. Second key is whether the process is disabled
    keys = list(procDict)
    vals = list(procDict.values())
    procName = keys[0]
    proc = self.parseProcName(procName, topFirst=topFirst)
    updateArgs = vals[0]
    if isinstance(updateArgs, list):
      # TODO: Determine policy for loading nested procs, outer should already know about inner so it shouldn't
      #   _need_ to occur, given outer would've been saved previously
      raise ValueError('Parsing deep nested processes is currently undefined')
    elif updateArgs:
      proc.updateInput(**updateArgs, graceful=True)
    # Check for disables at the nested level
    proc.disabled = procDict.get('disabled', proc.disabled)
    proc.allowDisable = procDict.get('allowDisable', proc.allowDisable)
    proc.name = procDict.get('name', proc.name)
    return proc
    # TODO: Add recursion, if it's something that will be done. For now, assume only 1-depth nesting. Otherwise,
    #   it's hard to distinguish between actual input optiosn and a nested process

  def _addFromModuleName(self, fullModuleName: str, primitive=True):
    """
    Adds all processes defined in a module to this collection. From a full module name (import.path.module). Rules:
      - All functions defined in that file *not* beginning with an underscore (_) will be added, except for
        the rule(s) below
      - All functions ending with 'factory' will be assumed ProcessStage factories, where their return value is exactly
        one ProcessStage. These are expected to take no arguments. Note that if `primitive` is *False* and an
        AtomicProcess is returned, errors will occur. So, it is implicitly forced to be a primitive process if this
        occurs

    :param fullModuleName: Module name to parse. Should be in a format expected by pydoc
    :param primitive: Whether the returned values should be considered top or primitive processes
    """
    if fullModuleName in self.includeModules:
      return
    module = pydoc.locate(fullModuleName)
    if not module:
      raise ValueError(f'Module "{fullModuleName}" not recognized')
    for name, func in inspect.getmembers(module,
                                         lambda el: inspect.isfunction(el)
                                                    and el.__module__ == module.__name__
                                                    and not el.__name__.startswith('_')):
      if name.lower().endswith('factory'):
        obj = func()
        self.addProcess(obj, top=not primitive and not isinstance(obj, AtomicProcess))
      else:
        self.addFunction(func)

  @classmethod
  def parseProcModule(cls, moduleName: str, procName: str, formatter=nameFormatter):
    module = pydoc.locate(moduleName)
    if not module:
      raise ValueError(f'Module "{module}" not recognized')
    # TODO: Depending on search time, maybe quickly search without formatting?
    attr = None
    for name, modAttr in vars(module).items():
      name = nameFormatter(name.split('.')[-1])
      if name == procName:
        attr = modAttr
        break
      elif procName in name and name.lower().endswith('factory'):
        attr = modAttr()
        break

    # Change behavior based on obj type
    if inspect.isfunction(attr):
      return AtomicProcess(attr)
    if isinstance(attr, ProcessStage):
      return attr
    return None

  def saveParamValues(self, saveName: str=None, paramState: dict=None, **kwargs):
    def procFilter(procDict):
      return {k: v for k, v in procDict.items() if not isinstance(v, ProcessStage)}
    if paramState is None:
      paramState = {'top': procFilter(self.topProcs), 'primitive': procFilter(self.primitiveProcs)}
      if self.includeModules:
        paramState['modules'] = self.includeModules
    return super().saveParamValues(saveName, paramState, **kwargs)

  def loadParamValues(self, stateName: t.Union[str, Path],
                      stateDict: _AlgClctnDict=None,
                      **kwargs):
    stateDict = self._parseStateDict(stateName, stateDict)
    top, primitive = stateDict.get('top', {}), stateDict.get('primitive', {})
    modules = stateDict.get('modules', [])
    for mod in modules:
      self._addFromModuleName(mod)
    self.includeModules = modules
    self.topProcs.update(top)
    self.primitiveProcs.update(primitive)

    return super().loadParamValues(stateName, stateDict, candidateParams=[])

class AlgParamEditor(ParamEditor):
  sigProcessorChanged = QtCore.Signal(str)
  """Name of newly selected process"""

  def __init__(self, clctn: AlgCollection=None, **kwargs):
    super().__init__(**kwargs)
    if clctn is None:
      clctn = AlgCollection()
    self.clctn = clctn
    self.treeBtnsWidget.hide()
    self._unflatProc: t.Optional[NestedProcess] = None
    """Retained for saving state on swap without flattening the source processor"""

    noneProc = self.clctn.procType('None')
    clctn.addProcess(noneProc, top=not clctn.topProcs)

    procName = next(iter(self.clctn.topProcs))
    # Set to None first to force switch, init states
    self.curProcessor = self.clctn.procWrapType(noneProc, self.params)
    _, self.changeProcParam = self.registerFunc(self.changeActiveProcessor, runOpts=RunOpts.ON_CHANGED, returnParam=True,
                                                overrideBasePath=(), parentParam=self._metaParamGrp,
                                                proc=procName)
    fns.setParamsExpanded(self._metaTree)
    procSelector = self.changeProcParam.child('proc')
    self.clctn.sigChangesApplied.connect(lambda: procSelector.setLimits(list(self.clctn.topProcs)))
    self.clctn.sigChangesApplied.emit({})
    def onChange(name):
      self.changeProcParam['proc'] = name
    self.sigProcessorChanged.connect(onChange)
    self.changeActiveProcessor(procName)

  @classmethod
  def _unnestedProcState(cls, proc: NestedProcess, _state=None, **kwargs):
    """
    Updates processes without hierarchy so separate stages are unnested. The outermost process is considered a
    'top' process, while all subprocesses are considered 'primitive'.

    :param proc: Process to record the unnested state
    :param _state: Internally used, do not provide in the function call. It will be returned at the end with 'top' and
      'primitive' keys
    :return: Mock Collection state from just the provided nested process. The passed process will be the only 'top'
      value, while all substages are entries (and unnested substages, etc.) are entries in the 'primitive' key
    """
    kwargs.update(includeMeta=True, disabled=False, allowDisable=True)
    first = _state is None
    if first:
      _state = {'top': {}, 'primitive': {}}
    stageVals = []
    for stage in proc:
      if isinstance(stage, NestedProcess):
        cls._unnestedProcState(stage, _state, **kwargs)
        stageVals.append(stage.addMetaProps(stage.name, **kwargs))
      else:
        stageVals.append(stage.saveState(**kwargs))
    entryPt = 'top' if first else 'primitive'
    _state[entryPt][proc.name] = stageVals
    return _state

  def saveParamValues(self, saveName: str=None, paramState: dict=None, *, includeDefaults=False, **kwargs):
    """
    The algorithm editor also needs to store information about the selected algorithm, so lump
    this in with the other parameter information before calling default save.
    """
    proc = self.curProcessor.processor
    # Make sure any newly added stages are accounted for
    if paramState is None:
      # Since inner nested processes are already recorded, flatten here to just save updated parameter values for the
      # outermost stage
      paramState = self._unnestedProcState(proc, includeMeta=True)
    self.clctn.loadParamValues(self.clctn.lastAppliedName, paramState)
    clctnState = self.clctn.saveParamValues(saveName, blockWrite=True)
    paramState = {'Selected Algorithm': self.curProcessor.algName, 'Parameters': clctnState}
    return super().saveParamValues(saveName, paramState, includeDefaults=includeDefaults, **kwargs)

  def loadParamValues(self, stateName: Union[str, Path],
                      stateDict: dict=None, **kwargs):
    stateDict = self._parseStateDict(stateName, stateDict)
    procName = stateDict.get('Selected Algorithm')
    if not procName:
      procName = next(iter(self.clctn.topProcs))
    clctnState = stateDict['Parameters']

    self.clctn.loadParamValues(stateName, clctnState, **kwargs)
    self.changeActiveProcessor(procName, flatten=self.changeProcParam['flatten'], saveBeforeChange=False)

  def changeActiveProcessor(self, proc: Union[str, NestedProcess], flatten=False, saveBeforeChange=True):
    """
    Changes which processor is active.

    :param proc:
      helpText: Processor to load
      pType: popuplineeditor
      limits: []
      title: Algorithm
    :param flatten: Whether to flatten the processor by ignoring all nested hierarchies except the topmost level
    :param saveBeforeChange: Whether to propagate current algorithm settings to the processor collection before changing
    """
    # TODO: Maybe there's a better way of doing this? Ensures proc label is updated for programmatic calls
    if saveBeforeChange and self._unflatProc:
      self.saveParamValues(self.lastAppliedName, self._unnestedProcState(self._unflatProc, includeMeta=True),
                           blockWrite=True)
    if isinstance(proc, str):
      proc = self.clctn.parseProcName(proc)
    unflatProc = proc
    if flatten:
      proc = proc.flatten()
    if proc == self.curProcessor.processor:
      return
    self.curProcessor.clear()
    self.params.clearChildren()
    self.curProcessor = self.clctn.procWrapType(proc, self.params)
    self._unflatProc = unflatProc
    fns.setParamsExpanded(self.tree)
    self.sigProcessorChanged.emit(proc.name)