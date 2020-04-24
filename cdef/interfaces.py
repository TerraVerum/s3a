from abc import ABC, abstractmethod
from typing import List

import numpy as np

from cdef.structures.typeoverloads import NChanImg, BlackWhiteImg
from .structures.vertices import FRVertices, FRComplexVertices


class FRImageProcessor(ABC):
  image: NChanImg = np.zeros((1,1), dtype=bool)

  @abstractmethod
  def localCompEstimate(self, prevCompMask: BlackWhiteImg, fgVerts: FRVertices=None, bgVerts: FRVertices=None) -> \
      BlackWhiteImg:
    pass

  @abstractmethod
  def globalCompEstimate(self) -> List[FRComplexVertices]:
    pass

