from __future__ import annotations

from abc import ABC
from copy import deepcopy
from functools import wraps
from io import StringIO
from typing import Tuple, List, Sequence
from warnings import warn

import numpy as np
from ..processing.processing import *
from pyqtgraph.parametertree import Parameter
import docstring_parser as dp
from ruamel.yaml import YAML
yaml = YAML()

from s3a.generalutils import augmentException
from s3a.processing.algorithms import crop_to_local_area, apply_process_result, basicOpsCombo, \
  return_to_full_size, format_vertices
from s3a.structures import FRParam, FRComplexVertices, FRAlgProcessorError, FRVertices, \
  FRS3AWarning
from s3a.parameditors.pgregistered import FRCustomMenuParameter
from s3a.models import editorbase

__all__ = ['FRImgProcWrapper', 'FRGeneralProcWrapper']

def docParser(docstring: str):
  """
  From a function docstring, extracts relevant information for how to create smarter
    parameter boxes.

  :param docstring: Function docstring
  """
  parsed = dp.parse(docstring)
  descr = ''
  for parseDescr in parsed.short_description, parsed.long_description:
    if parseDescr is not None:
      descr += parseDescr
  out = {}
  for param in parsed.params:
    stream = StringIO(param.description)
    paramDoc = yaml.load(stream)
    if isinstance(paramDoc, str):
      paramDoc = {'helpText': paramDoc}
    if paramDoc is None:
      continue
    out[param.arg_name] = FRParam(name=param.arg_name, **paramDoc)
    if 'pType' not in paramDoc:
      out[param.arg_name].pType = None
  out['top-descr'] = descr
  return out



def atomicRunWrapper(proc: FRAtomicProcess, names: Sequence[str], params: Sequence[Parameter]):
  oldRun = proc.run
  @wraps(oldRun)
  def newRun(io: FRProcessIO = None, disable=False) -> FRProcessIO:
    newIo = {name: param.value() for name, param in zip(names, params)}
    proc.input.update(**newIo)
    return oldRun(io, disable)
  return newRun

def procRunWrapper(proc: FRGeneralProcess, groupParam: Parameter):
  oldRun = proc.run
  @wraps(oldRun)
  def newRun(io: FRProcessIO = None, disable=False):
    proc.disabled = not groupParam.opts['enabled']
    return oldRun(io, disable=disable)
  return newRun

class FRGeneralProcWrapper(ABC):
  def __init__(self, processor: FRProcessStage, editor: editorbase.FRParamEditorBase,
               paramPath: Tuple[str, ...]=()):
    self.processor = processor
    self.algName = processor.name
    self.algParam = FRParam(self.algName)
    self.output = np.zeros((0,0), bool)

    self.editor = editor
    editor.registerGroup(self.algParam, nameFromParam=True, forceCreate=True, parentPath=paramPath)
    self.unpackStages(self.processor, paramPath)

  def unpackStages(self, stage: FRProcessStage, paramParent: Tuple[str, ...]=()):
    if isinstance(stage, FRAtomicProcess):
      docParams = docParser(stage.func.__doc__)
      params: List[Parameter] = []
      for key in stage.input.hyperParamKeys:
        val = stage.input[key]
        curParam = docParams.get(key, None)
        if curParam is None:
          curParam = FRParam(name=key, value=val)
        else:
          # Default value should be overridden by func signature
          curParam.value = val
          if curParam.pType is None:
            curParam.pType = type(val).__name__
        pgParam = self.editor.registerProp(self.algParam, curParam, paramParent, asProperty=False)
        params.append(pgParam)
      stage.run = atomicRunWrapper(stage, stage.input.hyperParamKeys, params)
      return
    # else: # Process
    stage: FRGeneralProcess
    curGroup = self.editor.params.child(self.algName, *paramParent)
    stage.run = procRunWrapper(stage, curGroup)
    # Special case of a process comprised of just one atomic function
    if len(stage.stages) == 1 and isinstance(stage.stages[0], FRAtomicProcess):
      self.unpackStages(stage.stages[0], paramParent=paramParent)
      return
    for childStage in stage.stages:
      pType = 'atomicgroup'
      if childStage.allowDisable:
        pType = 'procgroup'
      curGroup = FRParam(name=childStage.name, pType=pType, value=[],)
      self.editor.registerProp(self.algParam, curGroup, paramParent, asProperty=False,
                               enabled=not childStage.disabled)
      self.unpackStages(childStage, paramParent=paramParent + (childStage.name,))

  def setStageEnabled(self, stageIdx: Sequence[str], enabled: bool):
    paramForStage: FRCustomMenuParameter = self.editor.params.child(self.algName, *stageIdx)
    prevEnabled = paramForStage.opts['enabled']
    if prevEnabled != enabled:
      paramForStage.menuActTriggered('Toggle Enable')

  def run(self, **kwargs):
    raise NotImplementedError

  def __repr__(self) -> str:
    selfCls = type(self)
    oldName: str = super().__repr__()
    # Remove module name for brevity
    oldName = oldName.replace(f'{selfCls.__module__}.{selfCls.__name__}',
                              f'{selfCls.__name__} \'{self.algName}\'')
    return oldName

  @classmethod
  def getNestedName(cls, curProc: FRProcessStage, nestedName: List[str]):
    if len(nestedName) == 0 or isinstance(curProc, FRAtomicProcess):
      return curProc
    # noinspection PyUnresolvedReferences
    for stage in curProc.stages:
      if stage.name == nestedName[0]:
        if len(nestedName) == 1:
          return stage
        else:
          return cls.getNestedName(stage, nestedName[1:])

def _prependFuncs():
  formatStage = FRImageProcess.fromFunction(format_vertices)
  formatStage.allowDisable = False
  cropStage = FRImageProcess.fromFunction(crop_to_local_area)
  return [formatStage, cropStage]

def _appendFuncs():
  applyStage = FRImageProcess.fromFunction(apply_process_result)
  applyStage.allowDisable = False
  resizeStage = FRImageProcess.fromFunction(return_to_full_size)
  resizeStage.allowDisable = False

  return [applyStage, basicOpsCombo(), resizeStage]

class FRImgProcWrapper(FRGeneralProcWrapper):
  prependProcs = _prependFuncs()
  appendedProcs = _appendFuncs()

  def __init__(self, processor: FRImageProcess, editor: editorbase.FRParamEditorBase,
               excludedStages: List[List[str]]=None, disabledStages: List[List[str]]=None):
    # Each processor is encapsulated in processes that crop the image to the region of
    # interest specified by the user, and re-expand the area after processing

    preStages = [*map(deepcopy, self.prependProcs)]
    finalStages = [*map(deepcopy, self.appendedProcs)]
    processor.stages = preStages + processor.stages + finalStages

    if disabledStages is None:
      disabledStages = []
    if hasattr(processor, 'disabledStages'):
      disabledStages.extend(processor.disabledStages)

    if excludedStages is None:
      excludedStages = []
    if hasattr(processor, 'excludedStages'):
      excludedStages.extend(processor.excludedStages)

    for namePath in disabledStages: # type: List[str]
      proc = self.getNestedName(processor, namePath)
      proc.disabled = True
    for namePath in excludedStages: # type: List[str]
      parentProc = self.getNestedName(processor, namePath[:-1])
      parentProc.stages.remove(self.getNestedName(parentProc, [namePath[-1]]))
    super().__init__(processor, editor)

  def run(self, **kwargs):
    newIo = self._ioDictFromRunKwargs(kwargs)

    try:
      result = self.processor.run(newIo)
    except Exception as ex:
      augmentException(ex, 'Exception during processor run:\n')
      result = FRProcessIO(image=kwargs['prevCompMask'])
      warn(str(ex), FRS3AWarning)

    outImg = result['image'].astype(bool)
    if outImg.ndim > 2:
      outImg = np.bitwise_or.reduce(outImg, 2)
    self.output = outImg
    return outImg

  def resultAsVerts(self, localEstimate=True):
    initialList = FRComplexVertices.fromBwMask(self.output)
    if len(initialList) == 0:
      return initialList
    if not localEstimate:
      return [FRComplexVertices([lst]) for lst in initialList]
    # else, all vertices belong to the same component
    else:
      return [initialList]

  @staticmethod
  def _ioDictFromRunKwargs(runKwargs):
    image = runKwargs.get('image', None)
    if image is None:
      raise FRAlgProcessorError('Cannot run processor without an image')

    runKwargs.setdefault('firstRun', True)
    for name in 'fgVerts', 'bgVerts':
      runKwargs.setdefault(name, FRVertices())

    if runKwargs.get('prevCompMask', None) is None:
      noPrevMask = True
      runKwargs['prevCompMask'] = np.zeros(image.shape[:2], bool)
    else:
      noPrevMask = False

    return FRProcessIO(**runKwargs, noPrevMask=noPrevMask)