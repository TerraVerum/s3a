from __future__ import annotations
from typing import Tuple, List

import numpy as np
from pyqtgraph.parametertree import Parameter

from imageprocessing.processing import ImageIO, ProcessStage, AtomicFunction, Process, \
  ImageProcess, ProcessIO
from .frgraphics import parameditors
from .processingutils import getVertsFromBwComps
from .structures import FRParam, NChanImg, FRComplexVertices, FRVertices


class FRRunWrapper:
  def __init__(self, proc: ProcessStage, names: List[str], params: List[Parameter]):
    self.oldRun = proc.run
    self.params = params
    self.names = names

  def __call__(self, io: ProcessIO = None, force=False) -> ProcessIO:
    for name, param in zip(self.names, self.params):
      if name not in io:
        io[name] = param.value()
    return self.oldRun(io, force)


class FRAlgWrapper:
  def __init__(self, processor: ImageProcess, editor: parameditors.FRParamEditor):
    self.processor = processor
    self.algName = processor.name
    self.algParam = FRParam(self.algName)
    self.image: NChanImg = np.zeros((0,0), bool)
    self.output = np.zeros((0,0), bool)

    self.editor = editor
    editor.registerClass(self.algParam, overrideName=self.algName)()
    self.unpackStages(self.processor)

  def unpackStages(self, stage: ProcessStage, paramParent: Tuple[str,...]=()):
    if isinstance(stage, AtomicFunction):
      return
    # else: # Process
    stage: Process
    procInpt = stage.input
    params: List[Parameter] = []
    for inptKey in stage.requiredInputs:
      val = procInpt[inptKey]
      curParam = FRParam(name=inptKey, value=val)
      pgParam = self.editor.registerProp(self.algName, curParam, paramParent, asProperty=False)
      params.append(pgParam)
    stage.run = FRRunWrapper(stage, stage.requiredInputs, params)
    # Special case of a process comprised of just one atomic function
    if len(stage.stages) == 1 and isinstance(stage.stages[0], AtomicFunction):
      return
    for childStage in stage.stages:
      curGroup = FRParam(name=childStage.name, valType='group', value=[])
      self.editor.registerProp(self.algName, curGroup, paramParent)
      self.unpackStages(childStage, paramParent=paramParent + (childStage.name,))

  def run(self, **kwargs):
    if kwargs.get('fgVerts') is None and kwargs.get('bgVerts') is None:
      # Assume global estimate
      shape = self.image.shape[:2][::-1]
      kwargs['fgVerts'] = FRVertices([[0,0], [0, shape[1]],
                                      [shape[0], shape[1]], [shape[0], 0]])
    newIo = ImageIO(image=self.image, **kwargs)
    result = self.processor.run(newIo, force=True)
    outImg = result['image'].data
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

