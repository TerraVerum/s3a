from __future__ import annotations

import warnings

import numpy as np

from s3a.generalutils import augmentException
from s3a.structures import ComplexXYVertices, XYVertices
from utilitys import NestedProcWrapper
from .processing import *

__all__ = ['ImgProcWrapper', 'NestedProcWrapper']

class ImgProcWrapper(NestedProcWrapper):
  output: np.ndarray

  def run(self, **kwargs):
    newIo = self._ioDictFromRunKwargs(kwargs)
    try:
      with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        result = self.processor.run(newIo)
    except Exception as ex:
      msg = 'Exception during processor run:\n'
      # Convert stage info into a more readable format
      stages = [a.name for a in ex.args if isinstance(a, ProcessStage)]
      stagePath = ' > '.join(stages)
      # Cut these out of the current arguments
      ex.args = (f'Stage: {stagePath}\nMessage: ' + '\n'.join(ex.args[len(stages):]),)
      augmentException(ex, msg)
      warnings.warn(str(ex), UserWarning)
      return

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