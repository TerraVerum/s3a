from __future__ import annotations

import traceback
from copy import deepcopy
from typing import List

import numpy as np
from utilitys import NestedProcWrapper
from utilitys.fns import warnLater

from s3a.generalutils import augmentException
from s3a.processing.algorithms import crop_to_local_area, apply_process_result, \
  basicOpsCombo, return_to_full_size, format_vertices
from s3a.structures import ComplexXYVertices, XYVertices
from .processing import *

__all__ = ['ImgProcWrapper', 'NestedProcWrapper']

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

class ImgProcWrapper(NestedProcWrapper):
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
      augmentException(ex, 'Exception during processor run:\n'
                       + ''.join(traceback.format_stack(limit=5)))
      result = ProcessIO(image=kwargs['prevCompMask'])
      warnLater(str(ex), UserWarning)

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
      raise RuntimeError('Cannot run processor without an image')

    runKwargs.setdefault('firstRun', True)
    for name in 'fgVerts', 'bgVerts':
      runKwargs.setdefault(name, XYVertices())

    if runKwargs.get('prevCompMask', None) is None:
      noPrevMask = True
      runKwargs['prevCompMask'] = np.zeros(image.shape[:2], bool)
    else:
      noPrevMask = False

    return ProcessIO(**runKwargs, noPrevMask=noPrevMask)