from __future__ import annotations

from abc import ABC
from functools import wraps
from typing import Tuple, List, Sequence
from warnings import warn

import numpy as np
from ..processing.processing import *
from pyqtgraph.parametertree import Parameter

from s3a.generalutils import augmentException
from s3a.processing.algorithms import crop_to_local_area, apply_process_result, basicOpsCombo, \
  return_to_full_size, format_vertices
from s3a.structures import FRParam, FRComplexVertices, FRAlgProcessorError, FRVertices, \
  FRS3AWarning
from s3a.parameditors.pgregistered import FRCustomMenuParameter
from s3a.parameditors import genericeditor

__all__ = ['FRImgProcWrapper', 'FRGeneralProcWrapper']

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
  def __init__(self, processor: FRGeneralProcess, editor: genericeditor.FRParamEditor):
    self.processor = processor
    self.algName = processor.name
    self.algParam = FRParam(self.algName)
    self.output = np.zeros((0,0), bool)

    self.editor = editor
    editor.registerGroup(self.algParam, nameFromParam=True, forceCreate=True)
    self.unpackStages(self.processor)

  def unpackStages(self, stage: FRProcessStage, paramParent: Tuple[str, ...]=()):
    if isinstance(stage, FRAtomicProcess):
      params: List[Parameter] = []
      for key in stage.input.hyperParamKeys:
        val = stage.input[key]
        curParam = FRParam(name=key, value=val)
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
      if isinstance(childStage, FRGeneralProcess) and childStage.allowDisable:
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

class FRImgProcWrapper(FRGeneralProcWrapper):
  def __init__(self, processor: FRImageProcess, editor: genericeditor.FRParamEditor,
               excludedStages: List[List[str]]=None, disabledStages: List[List[str]]=None):
    # Each processor is encapsulated in processes that crop the image to the region of
    # interest specified by the user, and re-expand the area after processing
    formatStage = FRImageProcess()
    formatStage.addFunction(format_vertices)
    formatStage.allowDisable = False
    cropStage = FRImageProcess.fromFunction(crop_to_local_area)

    applyStage = FRImageProcess.fromFunction(apply_process_result)
    applyStage.allowDisable = False
    resizeStage = FRImageProcess.fromFunction(return_to_full_size)
    resizeStage.allowDisable = False

    finalStages = [applyStage, basicOpsCombo(), resizeStage]
    processor.stages = [formatStage, cropStage] + processor.stages + finalStages
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