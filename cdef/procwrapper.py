from __future__ import annotations

from abc import ABC
from functools import wraps
from typing import Tuple, List, Callable

import numpy as np
from pyqtgraph.parametertree import Parameter

from imageprocessing.processing import ImageIO, ProcessStage, AtomicFunction, Process, \
  ImageProcess, ProcessIO
from .frgraphics import parameditors
from .interfaceimpls import cropImgToROI, updateCroppedArea, basicOpsCombo
from .processingutils import getVertsFromBwComps
from .structures import FRParam, NChanImg, FRComplexVertices, FRVertices

def atomicRunWrapper(proc: AtomicFunction, names: List[str], params: List[Parameter]):
  oldRun = proc.run
  @wraps(oldRun)
  def newRun(io: ProcessIO = None, force=False, disable=False) -> ProcessIO:
    newIo = {name: param.value() for name, param in zip(names, params)}
    proc.updateParams(**newIo)
    return oldRun(io, force, disable)
  return newRun

def procRunWrapper(proc: Process, groupParam: Parameter):
  oldRun = proc.run
  @wraps(oldRun)
  def newRun(io: ProcessIO = None, force=False, disable=False):
    proc.disabled = not groupParam.opts['enabled']
    return oldRun(io, force, disable)
  return newRun

class FRGeneralProcWrapper(ABC):
  def __init__(self, processor: ImageProcess, editor: parameditors.FRParamEditor):
    self.processor = processor
    self.algName = processor.name
    self.algParam = FRParam(self.algName)
    self.output = np.zeros((0,0), bool)

    self.editor = editor
    editor.registerClass(self.algParam, overrideName=self.algName, forceCreate=True)()
    self.unpackStages(self.processor)

  def unpackStages(self, stage: ProcessStage, paramParent: Tuple[str,...]=()):
    if isinstance(stage, AtomicFunction):
      procInpt = stage.input
      params: List[Parameter] = []
      for inptKey in stage.hyperParamKeys:
        val = procInpt[inptKey]
        curParam = FRParam(name=inptKey, value=val)
        pgParam = self.editor.registerProp(self.algName, curParam, paramParent, asProperty=False)
        params.append(pgParam)
      stage.run = atomicRunWrapper(stage, stage.hyperParamKeys, params)
      return
    # else: # Process
    stage: Process
    curGroup = self.editor.params.child(self.algName, *paramParent)
    stage.run = procRunWrapper(stage, curGroup)
    # Special case of a process comprised of just one atomic function
    if len(stage.stages) == 1 and isinstance(stage.stages[0], AtomicFunction):
      self.unpackStages(stage.stages[0], paramParent=paramParent)
      return
    for childStage in stage.stages:
      valType = 'group'
      if isinstance(childStage, Process):
        valType = 'procgroup'
      curGroup = FRParam(name=childStage.name, valType=valType, value=[])
      self.editor.registerProp(self.algName, curGroup, paramParent)
      self.unpackStages(childStage, paramParent=paramParent + (childStage.name,))

  def run(self, **kwargs):
    raise NotImplementedError

class FRImgProcWrapper(FRGeneralProcWrapper):
  def __init__(self, processor: ImageProcess, editor: parameditors.FRParamEditor):
    # Each processor is encapsulated in processes that crop the image to the region of
    # interest specified by the user, and re-expand the area after processing
    processor.stages = [cropImgToROI()] + processor.stages + [updateCroppedArea(),
                                                              basicOpsCombo()]
    super().__init__(processor, editor)
    self.image: NChanImg = np.zeros((0,0), bool)

  def run(self, **kwargs):
    if kwargs.get('prevCompMask', None) is None:
      kwargs['prevCompMask'] = np.zeros(self.image.shape[:2], bool)
    newIo = ImageIO(image=self.image, **kwargs)
    result = self.processor.run(newIo, force=True)
    outImg = result['image']
    if outImg.ndim > 2:
      outImg = np.bitwise_or.reduce(outImg, 2)
    self.output = outImg
    return outImg

  def resultAsVerts(self, localEstimate=True):
    initialList = getVertsFromBwComps(self.output)

    if not localEstimate:
      return [FRComplexVertices([lst]) for lst in initialList]
    # else, all vertices belong to the same component
    else:
      return FRComplexVertices(initialList)