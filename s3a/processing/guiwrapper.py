from __future__ import annotations

import traceback

import numpy as np

from s3a.generalutils import augmentException
from s3a.structures import ComplexXYVertices, XYVertices
from utilitys import NestedProcWrapper
from utilitys.fns import warnLater
from .processing import *

__all__ = ['ImgProcWrapper', 'NestedProcWrapper']

class ImgProcWrapper(NestedProcWrapper):
  output: np.ndarray

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