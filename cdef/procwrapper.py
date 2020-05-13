from __future__ import annotations

from abc import ABC
from functools import wraps
from typing import Tuple, List, Callable

import numpy as np
from pyqtgraph.parametertree import Parameter

import cdef
from imageprocessing.processing import ImageIO, ProcessStage, AtomicProcess, Process, \
  ImageProcess, ProcessIO
from .frgraphics import parameditors
from .interfaceimpls import crop_to_verts, update_area, basicOpsCombo
from .processingutils import getVertsFromBwComps
from .structures import FRParam, NChanImg, FRComplexVertices, FRVertices
from .projectvars import TEMPLATE_COMP as TC

def atomicRunWrapper(proc: AtomicProcess, names: List[str], params: List[Parameter]):
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
      if isinstance(childStage, Process):
        valType = 'procgroup'
      curGroup = FRParam(name=childStage.name, valType=valType, value=[])
      self.editor.registerProp(self.algName, curGroup, paramParent, asProperty=False)
      self.unpackStages(childStage, paramParent=paramParent + (childStage.name,))

  def run(self, **kwargs):
    raise NotImplementedError

class FRImgProcWrapper(FRGeneralProcWrapper):
  def __init__(self, processor: ImageProcess, editor: parameditors.FRParamEditor):
    # Each processor is encapsulated in processes that crop the image to the region of
    # interest specified by the user, and re-expand the area after processing
    cropStage = ImageProcess.fromFunction(crop_to_verts, name='Crop to Vertices')
    cropStage.allowDisable = False
    updateStage = ImageProcess.fromFunction(update_area, 'Update Cropped Area')
    updateStage.allowDisable = False
    processor.stages = [cropStage] + processor.stages + [updateStage, basicOpsCombo()]
    super().__init__(processor, editor)
    self.image: NChanImg = np.zeros((0,0), bool)

  def run(self, **kwargs):
    if kwargs.get('prevCompMask', None) is None:
      kwargs['prevCompMask'] = np.zeros(self.image.shape[:2], bool)
    newIo = ImageIO(image=self.image, **kwargs)
    result = self.processor.run(newIo, force=True)
    outImg = result['image'].astype(bool)
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
      return [FRComplexVertices(initialList)]

  def resultAsCompDf(self, localEstimate=True):
    compVertices = self.resultAsVerts(localEstimate=localEstimate)
    components = cdef.makeCompDf(len(compVertices))
    components[TC.VERTICES] = compVertices
    return components