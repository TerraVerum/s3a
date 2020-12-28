from __future__ import annotations

from copy import deepcopy
from functools import singledispatch
from functools import wraps
from io import StringIO
from typing import List, Sequence, Union, Callable
from warnings import warn

import docstring_parser as dp
import numpy as np
from pyqtgraph.parametertree import Parameter
from ruamel import yaml

from s3a.generalutils import augmentException, getParamChild
from s3a.parameditors.pgregistered import CustomMenuParameter
from s3a.processing.algorithms import crop_to_local_area, apply_process_result, \
  basicOpsCombo, return_to_full_size, format_vertices
from s3a.structures import FRParam, ComplexXYVertices, AlgProcessorError, XYVertices, \
  S3AWarning, CompositionMixin
from ..processing.processing import *

__all__ = ['ImgProcWrapper', 'GeneralProcWrapper']

def docParser(docstring: str):
  """
  From a function docstring, extracts relevant information for how to create smarter
    parameter boxes.

  :param docstring: Function docstring
  """
  parsed = dp.parse(docstring)
  descrPieces = [p for p in (parsed.short_description, parsed.long_description) if p is not None]
  descr = ' '.join(descrPieces)
  out = {}
  for param in parsed.params:
    stream = StringIO(param.description)
    paramDoc = yaml.safe_load(stream)
    if isinstance(paramDoc, str):
      paramDoc = {'helpText': paramDoc}
    if paramDoc is None:
      continue
    out[param.arg_name] = FRParam(name=param.arg_name, **paramDoc)
    if 'pType' not in paramDoc:
      out[param.arg_name].pType = None
  out['top-descr'] = descr
  return out

def _attemptCreateChild(parent: Parameter, child: Union[Parameter, dict]) -> Parameter:
  try:
    cname = child.opts['name']
  except AttributeError:
    cname = child['name']
  if cname not in parent.names:
    parent.addChild(child)
  return parent.child(cname)

def atomicRunWrapper(proc: AtomicProcess, names: Sequence[str], params: Sequence[Parameter]):
  oldRun = proc.run
  @wraps(oldRun)
  def newRun(io: ProcessIO = None, disable=False, **runKwargs) -> ProcessIO:
    newIo = {name: param.value() for name, param in zip(names, params)}
    proc.input.update(**newIo)
    return oldRun(io, disable, **runKwargs)
  return newRun

def procRunWrapper(proc: GeneralProcess, groupParam: Parameter):
  oldRun = proc.run
  @wraps(oldRun)
  def newRun(io: ProcessIO = None, disable=False, **runKwargs):
    proc.disabled = not groupParam.opts['enabled']
    return oldRun(io, disable=disable, **runKwargs)
  return newRun

@singledispatch
def addStageToParam(stage: ProcessStage, parentParam: Parameter, **kwargs):
  pass

@addStageToParam.register
def addAtomicToParam(stage: AtomicProcess, parentParam: Parameter,
                 argNameFormat: Callable[[str], str]=None, **kwargs):
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
    paramDict = curParam.toPgDict()
    if argNameFormat is not None and 'title' not in paramDict:
      paramDict['title'] = argNameFormat(key)
    pgParam = _attemptCreateChild(parentParam, paramDict)
    params.append(pgParam)
  stage.run = atomicRunWrapper(stage, stage.input.hyperParamKeys, params)
  return stage

@addStageToParam.register
def addGeneralToParam(stage: GeneralProcess, parentParam: Parameter, nestHyperparams=True,
                      argNameFormat: Callable[[str], str]=None, treatAsAtomic=False, **kwargs):
  if treatAsAtomic:
    collapsed = AtomicProcess(stage.run, stage.name, mainResultKeys=stage.mainResultKeys,
                              mainInputKeys=stage.mainInputKeys)
    collapsed.input = stage.input
    addAtomicToParam(collapsed, parentParam, argNameFormat)
    return
  stage.run = procRunWrapper(stage, parentParam)
  # Special case of a process comprised of just one atomic function
  if len(stage.stages) == 1 and isinstance(stage.stages[0], AtomicProcess):
    # isinstance ensures the type will be correct
    # noinspection PyTypeChecker
    addAtomicToParam(stage.stages[0], parentParam)
    return
  outerParent = parentParam
  for childStage in stage.stages:
    pType = 'atomicgroup'
    if childStage.allowDisable:
      pType = 'procgroup'
    if nestHyperparams:
      paramDict = FRParam(name=childStage.name, pType=pType, value=[],
                          enabled=not childStage.disabled).toPgDict()
      parentParam = _attemptCreateChild(outerParent, paramDict)
    else:
      parentParam = outerParent
    addStageToParam(childStage, parentParam)

class GeneralProcWrapper(CompositionMixin):
  def __init__(self, processor: ProcessStage, parentParam: Parameter=None,
               argNameFormat: Callable[[str], str] = None, treatAsAtomic=False, nestHyperparams=True):
    self.processor = self.exposes(processor)
    self.algName = processor.name
    self.argNameFormat = argNameFormat
    self.treatAsAtomic = treatAsAtomic
    self.nestHyperparams = nestHyperparams
    if parentParam is None:
      parentParam = Parameter.create(name=self.algName, type='group')
    else:
      parentParam = getParamChild(parentParam, self.algName)
    self.parentParam = parentParam
    self.addStage(self.processor)

  def addStage(self, stage: ProcessStage):
    addStageToParam(stage, self.parentParam, argNameFormat=self.argNameFormat,
                    treatAsAtomic=self.treatAsAtomic, nestHyperparams=self.nestHyperparams)

  def setStageEnabled(self, stageIdx: Sequence[str], enabled: bool):
    paramForStage: CustomMenuParameter = self.parentParam.child(*stageIdx)
    prevEnabled = paramForStage.opts['enabled']
    if prevEnabled != enabled:
      paramForStage.menuActTriggered('Toggle Enable')

  def __repr__(self) -> str:
    selfCls = type(self)
    oldName: str = super().__repr__()
    # Remove module name for brevity
    oldName = oldName.replace(f'{selfCls.__module__}.{selfCls.__name__}',
                              f'{selfCls.__name__} \'{self.algName}\'')
    return oldName

  @classmethod
  def getNestedName(cls, curProc: ProcessStage, nestedName: List[str]):
    if len(nestedName) == 0 or isinstance(curProc, AtomicProcess):
      return curProc
    # noinspection PyUnresolvedReferences
    for stage in curProc.stages:
      if stage.name == nestedName[0]:
        if len(nestedName) == 1:
          return stage
        else:
          return cls.getNestedName(stage, nestedName[1:])

def _prependFuncs():
  formatStage = ImageProcess.fromFunction(format_vertices)
  formatStage.allowDisable = False
  cropStage = ImageProcess.fromFunction(crop_to_local_area)
  return [formatStage, cropStage]

def _appendFuncs():
  applyStage = ImageProcess.fromFunction(apply_process_result)
  applyStage.allowDisable = False
  resizeStage = ImageProcess.fromFunction(return_to_full_size)
  resizeStage.allowDisable = False

  return [applyStage, basicOpsCombo(), resizeStage]

class ImgProcWrapper(GeneralProcWrapper):
  preProcStages = _prependFuncs()
  postProcStages = _appendFuncs()

  def __init__(self, processor: ImageProcess, *,
               excludedStages: List[List[str]]=None, disabledStages: List[List[str]]=None,
               **kwargs):
    # Each processor is encapsulated in processes that crop the image to the region of
    # interest specified by the user, and re-expand the area after processing
    self.output = np.zeros((0,0), bool)

    preStages = [*map(deepcopy, self.preProcStages)]
    finalStages = [*map(deepcopy, self.postProcStages)]
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
    super().__init__(processor, **kwargs)

  def run(self, **kwargs):
    newIo = self._ioDictFromRunKwargs(kwargs)

    try:
      result = self.processor.run(newIo)
    except Exception as ex:
      augmentException(ex, 'Exception during processor run:\n')
      result = ProcessIO(image=kwargs['prevCompMask'])
      warn(str(ex), S3AWarning)

    outImg = result['image'].astype(bool)
    if outImg.ndim > 2:
      outImg = np.bitwise_or.reduce(outImg, 2)
    self.output = outImg
    return outImg

  def resultAsVerts(self, localEstimate=True):
    initialList = ComplexXYVertices.fromBwMask(self.output)
    if len(initialList) == 0:
      return initialList
    if not localEstimate:
      return [ComplexXYVertices([lst]) for lst in initialList]
    # else, all vertices belong to the same component
    else:
      return [initialList]

  @staticmethod
  def _ioDictFromRunKwargs(runKwargs):
    image = runKwargs.get('image', None)
    if image is None:
      raise AlgProcessorError('Cannot run processor without an image')

    runKwargs.setdefault('firstRun', True)
    for name in 'fgVerts', 'bgVerts':
      runKwargs.setdefault(name, XYVertices())

    if runKwargs.get('prevCompMask', None) is None:
      noPrevMask = True
      runKwargs['prevCompMask'] = np.zeros(image.shape[:2], bool)
    else:
      noPrevMask = False

    return ProcessIO(**runKwargs, noPrevMask=noPrevMask)