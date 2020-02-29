from abc import ABC, abstractmethod

import numpy as np
from typing import List

from Annotator.params import FRParamGroup, FRVertices


class FRImageProcessor(ABC):
  image: np.ndarray = np.zeros((1,1), dtype=bool)

  @abstractmethod
  def localCompEstimate(self, prevCompMask: np.ndarray, fgVerts: FRVertices=None, bgVerts: FRVertices=None) -> \
      FRVertices:
    pass

  @abstractmethod
  def globalCompEstimate(self) -> List[FRVertices]:
    pass


