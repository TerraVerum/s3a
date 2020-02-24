from abc import ABC, abstractmethod

import numpy as np
import cv2 as cv
import pandas as pd
from typing import Tuple

from Annotator.generalutils import splitListAtNans
from Annotator.params import FRParamGroup, FRVertices


class FRProcFunc(ABC):
  sharedProps: FRParamGroup

  @abstractmethod
  def __call__(self, *args, **kwargs):
    pass

class FRVertexDefinedImg:
  def __init__(self):
    self.image_np = np.zeros((1, 1), dtype='uint8')
    self._offset = FRVertices()
    self.verts = FRVertices()

  def embedMaskInImg(self, toEmbedShape: Tuple[int, int]):
    outImg = np.zeros(toEmbedShape, dtype=bool)
    selfShape = self.image_np.shape
    # Offset is x-y, shape is row-col. So, swap order of offset relative to current axis
    embedSlices = [slice(self._offset[1 - ii], selfShape[ii] + self._offset[1 - ii]) for ii in range(2)]
    outImg[embedSlices[0], embedSlices[1]] = self.image_np
    return outImg

  def updateVertices(self, newVerts: FRVertices):
    self.verts = newVerts.copy()

    if len(newVerts) == 0:
      self.image_np = np.zeros((1, 1), dtype='bool')
      return
    self._offset = newVerts.min(0)
    newVerts -= self._offset

    # cv.fillPoly requires list-of-lists format
    fillPolyArg = splitListAtNans(newVerts)
    nonNanVerts = newVerts.nonNanEntries().astype(int)
    newImgShape = nonNanVerts.max(0)[::-1] + 1
    regionData = np.zeros(newImgShape, dtype='uint8')
    cv.fillPoly(regionData, fillPolyArg, 1)
    # Make vertices full brightness
    regionData[nonNanVerts.rows, nonNanVerts.cols] = 2
    self.image_np = regionData

class FRImageProcessor(ABC):
  image: np.ndarray

  @abstractmethod
  def localCompEstimate(self, prevCompMask: np.ndarray, fgVerts: FRVertices=None, bgVerts: FRVertices=None) -> \
      np.ndarray:
    pass

  @abstractmethod
  def globalCompEstimate(self) -> np.ndarray:
    pass


