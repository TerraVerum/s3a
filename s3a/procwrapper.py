from __future__ import annotations

from abc import ABC
from functools import wraps
from typing import Tuple, List, Sequence, Optional
from warnings import warn

import numpy as np
import cv2 as cv
from pyqtgraph.parametertree import Parameter

import s3a
from imageprocessing.processing import ImageIO, ProcessStage, AtomicProcess, Process, \
  ImageProcess, ProcessIO

from .frgraphics.graphicsutils import raiseErrorLater
from .frgraphics.parameditors import genericeditor
from .frgraphics.parameditors.pgregistered import FRCustomMenuParameter
from .generalutils import augmentException
from .processingimpls import crop_to_verts, apply_process_result, basicOpsCombo, \
  return_to_full_size, format_vertices
from .structures import FRParam, FRComplexVertices, FRAlgProcessorError, FRVertices, \
  GrayImg


def atomicRunWrapper(proc: AtomicProcess, names: List[str], params: List[Parameter]):
  oldRun = proc.run
  @wraps(oldRun)
  def newRun(io: ProcessIO = None, force=False, disable=False, verbose=False) -> ProcessIO:
    newIo = {name: param.value() for name, param in zip(names, params)}
    proc.updateParams(**newIo)
    return oldRun(io, force, disable, verbose)
  return newRun

def procRunWrapper(proc: Process, groupParam: Parameter):
  oldRun = proc.run
  @wraps(oldRun)
  def newRun(io: ProcessIO = None, force=False, disable=False, verbose=False):
    proc.disabled = not groupParam.opts['enabled']
    return oldRun(io, force=force, disable=disable, verbose=verbose)
  return newRun

class FRGeneralProcWrapper(ABC):
  def __init__(self, processor: ImageProcess, editor: s3a.frgraphics.parameditors.genericeditor.FRParamEditor):
    self.processor = processor
    self.algName = processor.name
    self.algParam = FRParam(self.algName)
    self.output = np.zeros((0,0), bool)

    self.editor = editor
    editor.registerGroup(self.algParam, nameFromParam=True, forceCreate=True)
    self.unpackStages(self.processor)

  def unpackStages(self, stage: ProcessStage, paramParent: Tuple[str,...]=()):
    if isinstance(stage, AtomicProcess):
      params: List[Parameter] = []
      for key, val in stage.input.hyperParams.items():
        curParam = FRParam(name=key, value=val)
        pgParam = self.editor.registerProp(self.algName, curParam, paramParent, asProperty=False)
        params.append(pgParam)
      stage.run = atomicRunWrapper(stage, stage.input.hyperParamKeys, params)
      return
    # else: # Process
    stage: Process
    curGroup = self.editor.params.child(self.algName, *paramParent)
    stage.run = procRunWrapper(stage, curGroup)
    # Special case of a process comprised of just one atomic function
    if len(stage.stages) == 1 and isinstance(stage.stages[0], AtomicProcess):
      self.unpackStages(stage.stages[0], paramParent=paramParent)
      return
    for childStage in stage.stages:
      valType = 'atomicgroup'
      if isinstance(childStage, Process) and childStage.allowDisable:
        valType = 'procgroup'
      curGroup = FRParam(name=childStage.name, valType=valType, value=[],)
      self.editor.registerProp(self.algName, curGroup, paramParent, asProperty=False,
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
  def getNestedName(cls, curProc: ProcessStage, nestedName: List[str]):
    for stage in curProc.stages:
      if stage.name == nestedName[0]:
        if len(nestedName) == 1:
          return stage
        else:
          return cls.getNestedName(stage, nestedName[1:])

class FRImgProcWrapper(FRGeneralProcWrapper):
  def __init__(self, processor: ImageProcess, editor: s3a.frgraphics.parameditors.genericeditor.FRParamEditor):
    # Each processor is encapsulated in processes that crop the image to the region of
    # interest specified by the user, and re-expand the area after processing
    formatStage = ImageProcess.fromFunction(format_vertices, name='Format Vertices')
    formatStage.allowDisable = False
    cropStage = ImageProcess.fromFunction(crop_to_verts, name='Crop to Local Area')

    updateStage = ImageProcess.fromFunction(apply_process_result, 'Update Cropped Area')
    updateStage.allowDisable = False
    resizeStage = ImageProcess.fromFunction(return_to_full_size, 'Return to Full Size')
    resizeStage.allowDisable = False

    finalStages = [updateStage, basicOpsCombo(), resizeStage]
    processor.stages = [formatStage, cropStage] + processor.stages + finalStages

    if hasattr(processor, 'disabledStages'):
      for namePath in processor.disabledStages:
        proc = self.getNestedName(processor, namePath)
        proc.disabled = True
    super().__init__(processor, editor)

  def run(self, **kwargs):
    newIo = self._ioDictFromRunKwargs(kwargs)

    try:
      result = self.processor.run(newIo, force=True)
    except Exception as ex:
      augmentException(ex, 'Exception during processor run:')
      result = ImageIO(image=kwargs['prevCompMask'])
      raiseErrorLater(ex)

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

    return ImageIO(**runKwargs, noPrevMask=noPrevMask)