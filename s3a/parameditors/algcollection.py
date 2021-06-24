from __future__ import annotations

import copy
import inspect
import pydoc
import re
import types
import typing as t
import warnings
import webbrowser
from pathlib import Path

from pyqtgraph.Qt import QtCore

from s3a import PRJ_ENUMS
from s3a.constants import MENU_OPTS_DIR
from s3a.shims import typing_extensions as t_e
from utilitys import NestedProcess, RunOpts, ProcessIO
from utilitys import ParamEditor, NestedProcWrapper, fns, ProcessStage, AtomicProcess
from utilitys.typeoverloads import FilePath
from utilitys.widgets import makeDummySignal

Signal = QtCore.Signal
_procDict = t.Dict[str, t.List[t.Union[str, dict]]]

class _AlgClctnDict(t_e.TypedDict):
  top: _procDict
  primitive: _procDict
  modules: t.List[str]

_underscoreMatcher = re.compile(r'([A-Za-z])_+([A-Za-z])')
def _fmt(name: str):
  # Don't replace first/last underscore to avoid publicizing private / utility functions
  name = name.lower().replace(' ', '')
  return re.sub(_underscoreMatcher, r'\1\2', name)

algoNameFormatter = fns.NameFormatter(_fmt)
"""Strips spaces and underscores from the provided name, and turns everything to 
lowercase"""

class AlgCollection(ParamEditor):
  # sigProcessorCreated = Signal(object) # Signal(AlgCollectionEditor)
  def __init__(self, procWrapType=NestedProcWrapper, procType=NestedProcess, parent=None, fileType='alg',
               template: FilePath=None, **kwargs):
    super().__init__(parent, fileType=fileType, **kwargs)
    self.procWrapType = procWrapType
    self.procType = procType
    self.primitiveProcs: t.Dict[str, t.Union[ProcessStage, t.List[str]]] = {}
    self.topProcs: _procDict = {}
    self.includeModules: t.List[str] = []

    if template is not None:
      self.loadParamValues(template)

    if not self.topProcs:
      # Ensure at least one top-level processor exists
      self.addProcess(self.procType('None'), top=True)

    # Make sure fallthroughs are possible, i.e. for really simple image processes
    self.addProcess(AtomicProcess(lambda **_kwargs: ProcessIO(**_kwargs), name='Fallthrough'))

  def createProcessorEditor(self, saveDir: t.Union[str, t.Type], editorName='Processor') -> AlgParamEditor:
    """
    Creates a processor editor capable of dynamically loading, saving, and editing collection processes

    :param saveDir: Directory for saved states. If a class type is provided, the __name__ of this class is used.
      Note: Either way, the resulting name is lowercased before being applied.
    :param editorName: The name of the spawned editor
    """
    if saveDir is not None:
      if not isinstance(saveDir, str):
        saveDir = saveDir.__name__
      formattedClsName = fns.pascalCaseToTitle(saveDir)
      editorDir = MENU_OPTS_DIR/formattedClsName.lower()
    else:
      editorDir = saveDir
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

  def addFunction(self, func: t.Callable, top=False, **kwargs):
    """Helper function to wrap a function in an atomic process and add it as a stage"""
    return self.addProcess(AtomicProcess(func, **kwargs), top)

  def parseProcStages(self, stages: t.Sequence[t.Union[dict, str]], name:str=None, add=PRJ_ENUMS.PROC_NO_ADD,
                      allowOverwrite=False):
    """
    Creates a nested process from a sequence of process stages and optional name
    :param stages: Stages to parse
    :param name: Name of the nested process
    :param add: Whether to add this new process to the current collection's top or primitive process blocks, or
      to not add at all (if NO_ADD)
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
    if add is not PRJ_ENUMS.PROC_NO_ADD and (not exists or allowOverwrite):
      self.addProcess(out, top=add is PRJ_ENUMS.PROC_ADD_TOP, force=allowOverwrite)
    return out

  def parseProcName(self, procName: str, topFirst=True, searchDicts: t.Sequence[dict]=None, **kwargs):
    """
    From a list of search locations (ranging from most primitive to topmost), find the first processor matching the
    specified name. If 'topFirst' is chosen, the search locations are parsed in reverse order.
    """
    if searchDicts is None:
      searchDicts = [self.primitiveProcs, self.topProcs]
    if topFirst:
      searchDicts = searchDicts[::-1]
    proc = searchDicts[0].get(procName, searchDicts[1].get(procName))
    # It could still be in an include module, cache if found
    if not proc:
      for module in self.includeModules:
        proc = self.parseProcModule(module, procName, **kwargs)
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
    procName, updateArgs = next(iter(procDict.items()))
    procDict.pop(procName)
    proc = self.parseProcName(procName, topFirst=topFirst)
    if isinstance(updateArgs, list):
      # TODO: Determine policy for loading nested procs, outer should already know about inner so it shouldn't
      #   _need_ to occur, given outer would've been saved previously
      raise ValueError('Parsing deep nested processes is currently undefined')
    elif updateArgs:
      proc.updateInput(**updateArgs, allowExtra=True, graceful=True)
      # Set the defaults, too
      # if isinstance(proc, AtomicProcess):
      #   proc.defaultInput.update(**updateArgs)
    # Check for process-level traits
    for kk, vv in procDict.items():
      if hasattr(proc, kk):
        setattr(proc, kk, vv)
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
  def parseProcModule(cls, moduleName: t.Union[str, types.ModuleType], procName: str,
                                       formatter=algoNameFormatter, **factoryArgs):
    if isinstance(moduleName, types.ModuleType):
      module = moduleName
    else:
      module = pydoc.locate(moduleName)
    if not module:
      raise ValueError(f'Module "{moduleName}" not recognized')
    # TODO: Depending on search time, maybe quickly search without formatting?
    procName = formatter(procName)
    # It is possible after name formatting for multiple matches to exist for procName.
    # The first match that is a function or process stage will be retained.
    for name, attr in vars(module).items():
      name = formatter(name.split('.')[-1])
      # Evaluate factories first
      if name.lower().endswith('factory'):
        try:
          attr: ProcessStage = attr(**factoryArgs)
          if not isinstance(attr, ProcessStage):
            continue
          name = formatter(attr.name)
        except:
          continue
      if name == procName:
        if isinstance(attr, ProcessStage):
          return attr
        if callable(attr):
          return AtomicProcess(attr, **factoryArgs)
    return None

  def saveParamValues(self, saveName: str=None, paramState: dict=None, **kwargs):
    def procFilter(procDict):
      return {k: v for k, v in procDict.items() if not isinstance(v, ProcessStage)}
    if paramState is None:
      paramState = {'top': procFilter(self.topProcs), 'primitive': procFilter(self.primitiveProcs),
                    'modules': self.includeModules}
    return super().saveParamValues(saveName, paramState, **kwargs)

  def loadParamValues(self, stateName: t.Union[str, Path],
                      stateDict: _AlgClctnDict=None,
                      **kwargs):
    stateDict = self._parseStateDict(stateName, stateDict)
    top, primitive = stateDict.get('top', {}), stateDict.get('primitive', {})
    modules = stateDict.get('modules', [])
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

    procName = next(iter(self.clctn.topProcs))
    proc = clctn.parseProcName(procName)
    # Set to None first to force switch, init states
    self.curProcessor = self.clctn.procWrapType(proc, self.params)
    _, self.changeProcParam = self.registerFunc(self.changeActiveProcessor, runOpts=RunOpts.ON_CHANGED, returnParam=True,
                                                overrideBasePath=(), parentParam=self._metaParamGrp,
                                                proc=procName)
    fns.setParamsExpanded(self._metaTree)
    procSelector = self.changeProcParam.child('proc')
    self.clctn.sigChangesApplied.connect(lambda: procSelector.setLimits(list(self.clctn.topProcs)))
    self.clctn.sigChangesApplied.emit({})
    def onChange(name):
      with makeDummySignal(procSelector, 'sigValueChanged'):
        procSelector.setValue(name)
        # Manually set item labels since valueChange was forcefully disconnected
        for item in procSelector.items:
          item.valueChanged(procSelector, name)
    self.sigProcessorChanged.connect(onChange)
    self.changeActiveProcessor(procName)

  def _unnestedProcState(self, proc: NestedProcess, _state=None, **kwargs) -> _AlgClctnDict:
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
      _state = {'modules': self.clctn.includeModules, 'primitive': {}, 'top': {}}
    stageVals = []
    for stage in proc:
      if isinstance(stage, NestedProcess):
        self._unnestedProcState(stage, _state, **kwargs)
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
      paramState['modules'] = self.clctn.includeModules
    self.clctn.loadParamValues(self.clctn.stateName, paramState)
    clctnState = self.clctn.saveParamValues(saveName, blockWrite=True)
    paramState = {'Selected Algorithm': self.curProcessor.algName, **clctnState}
    return super().saveParamValues(saveName, paramState, includeDefaults=includeDefaults, **kwargs)

  def loadParamValues(self, stateName: t.Union[str, Path],
                      stateDict: dict=None, **kwargs):
    stateDict = self._parseStateDict(stateName, stateDict)
    procName = stateDict.pop('Selected Algorithm', None)
    if 'Parameters' in stateDict:
      warnings.warn('"Parameters" is deprecated for a loaded state. In the future, set "top", "primitive", etc. at the'
                    ' top dictionary level along with "Selected Algorithm"', DeprecationWarning)
      stateDict = stateDict['Parameters']
    if not procName:
      procName = next(iter(self.clctn.topProcs))

    self.clctn.loadParamValues(stateName, stateDict, **kwargs)
    self.changeActiveProcessor(procName, saveBeforeChange=False)
    return super().loadParamValues(stateName, {}, candidateParams=[], **kwargs)

  def changeActiveProcessor(self, proc: t.Union[str, NestedProcess], saveBeforeChange=True):
    """
    Changes which processor is active.

    :param proc:
      helpText: Processor to load
      pType: popuplineeditor
      limits: []
      title: Algorithm
    :param saveBeforeChange: Whether to propagate current algorithm settings to the processor collection before changing
    """
    # TODO: Maybe there's a better way of doing this? Ensures proc label is updated for programmatic calls
    if saveBeforeChange:
      self.saveParamValues(self.stateName, self._unnestedProcState(self.curProcessor.processor, includeMeta=True),
                           blockWrite=True)
    if isinstance(proc, str):
      proc = self.clctn.parseProcName(proc)
    if proc == self.curProcessor.processor:
      return
    self.curProcessor.clear()
    self.params.clearChildren()
    self.curProcessor = self.clctn.procWrapType(proc, self.params)
    fns.setParamsExpanded(self.tree)
    self.sigProcessorChanged.emit(proc.name)

  def editParamValues_gui(self):
    webbrowser.open(self.formatFileName(self.stateName))