from __future__ import annotations

from ast import literal_eval
from typing import Union, List, Sequence
from warnings import warn

import cv2 as cv
import numpy as np

from .typeoverloads import NChanImg, BlackWhiteImg

class XYVertices(np.ndarray):
  connected = True

  def __new__(cls, inputArr: Union[list, np.ndarray, tuple]=None, connected=True, dtype=int,
              **kwargs):
    # Default to integer type if not specified, since this is how pixel coordinates will be represented anyway
    # See numpy docs on subclassing ndarray
    if inputArr is None:
      inputArr = np.zeros((0,2))
    arr = np.asarray(inputArr, dtype=dtype, **kwargs).view(cls)
    arr.connected = connected
    return arr

  def __array_finalize__(self, obj):
    shape = self.shape
    shapeLen = len(shape)
    # indicates point, so the one dimension must have only 2 elements
    if 1 < shapeLen < 2 and shape[0] != 2:
      raise ValueError(f'A one-dimensional vertex array must be shape (2,).'
                                f' Receieved array of shape {shape}')
    elif shapeLen > 2 or shapeLen > 1 and shape[1] != 2:
      raise ValueError(f'Vertex list must be Nx2. Received shape {shape}.')
    if obj is None: return
    self.connected = getattr(obj, 'connected', True)

  @property
  def empty(self):
      return len(self) == 0

  def serialize(self):
    return str(self.tolist())

  @classmethod
  def deserialize(cls, strVal: str):
    out = cls(literal_eval(strVal))
    if out.size == 0:
      # Make sure size is 0x2
      return cls()
    return out

  # def asPoint(self):
  #   if self.size == 2:
  #     return self.reshape(-1)
  #   # Reaching here means the user requested vertices as point when
  #   # more than one point is in the list
  #   raise ValueError(f'asPoint() can only be called when one vertex is in'
  #                             f' the vertex list. Currently has shape {self.shape}')

  # def asRowCol(self):
  #   return np.fliplr(self)

  @property
  def x(self):
    # Copy to array first so dimensionality checks are no longer required
    return np.array(self).reshape(-1,2)[:,[0]]
  @x.setter
  def x(self, newX): self.reshape(-1,2)[:,0] = newX

  @property
  def y(self):
    return np.array(self).reshape(-1,2)[:,[1]]
  @y.setter
  def y(self, newY): self.reshape(-1,2)[:,1] = newY

  @property
  def rows(self): return self.y
  @rows.setter
  def rows(self, newRows): self.y = newRows

  @property
  def cols(self):return self.x
  @cols.setter
  def cols(self, newCols):self.x = newCols


class ComplexXYVertices(list):
  """
  Allows holes in the component shape. Subclassing ndarray instead of list allows primitive algebraic ops on the list
  contents (e.g. subtracting/adding offset). Since normal usage doesn't typically require a mutable structure, the
  loss is minimal.
  """
  hierarchy = np.ones((0,4), dtype=int)
  """See cv.findContours for hierarchy explanation. Used in cv.RETR_CCOMP mode."""

  def __init__(self, inputArr: Union[List[XYVertices], np.ndarray]=None,
               hierarchy:np.ndarray=None,
               coerceListElements=False):

    if inputArr is None:
      inputArr = []
    numInpts = len(inputArr)
    if coerceListElements:
      inputArr = [XYVertices(el) for el in inputArr]
    if hierarchy is None:
      hierarchy = np.ones((numInpts, 4), dtype=int)*-1
    super().__init__(inputArr)
    # No hierarchy required unless list is longer than length 1
    self.hierarchy = hierarchy

  def append(self, verts:XYVertices=None) -> None:
    if verts is not None:
      super().append(verts)

  def isEmpty(self):
    return len(self.stack()) == 0

  @property
  def x(self):
    return [lst.x for lst in self]
  @x.setter
  def x(self, newX):
    for lst, newLstX in zip(self, newX):
      lst.x = newLstX

  @property
  def y(self):
    return [lst.y for lst in self]

  @y.setter
  def y(self, newY):
    for lst, newLstY in zip(self, newY):
      lst.y = newLstY

  def asPoint(self):
    if len(self) == 1 and self[0].shape[0] == 1:
      return self[0][0]
    else:
      raise ValueError(f'Can only treat ComplexXYVertices with one inner list as a point.'
                       f' Current list has {len(self)} element(s), '
                       f'where element 0 has shape {self[0].shape}.')

  def stack(self, newDtype=int) -> XYVertices:
    if len(self) == 0:
      # Check required for np vstack since it won't work with a 0-element array
      return XYVertices()
    else:
      return XYVertices(np.vstack(self), dtype=newDtype)

  @classmethod
  def stackedMax(cls, complexVertList: Sequence[ComplexXYVertices]):
    """
    Returns the max along dimension 0 for a list of complex vertices
    """
    return np.vstack([v.stack() for v in complexVertList]).max(0)

  @classmethod
  def stackedMin(cls, complexVertList: Sequence[ComplexXYVertices]):
    return np.vstack([v.stack() for v in complexVertList]).min(0)


  def filledVerts(self) -> ComplexXYVertices:
    """
    Retrieves all vertex lists corresponding to filled regions in the complex shape
    """
    idxs = np.nonzero(self.hierarchy[:,3] == -1)[0]
    return ComplexXYVertices([self[ii] for ii in idxs])

  def holeVerts(self) -> ComplexXYVertices:
    """
    Retrieves all vertex lists corresponding to holes in the complex shape
    """
    idxs = np.nonzero(self.hierarchy[:,3] != -1)[0]
    return ComplexXYVertices([self[ii] for ii in idxs])

  def toMask(self, maskShape: Union[Sequence, NChanImg]=None,
             fillColor: Union[int, float, np.ndarray]=None,
             asBool=True, checkForDisconnectedVerts=False, warnIfTooSmall=True):
    if maskShape is None:
      try:
        maskShape = tuple(self.stack().max(0)[::-1]+1)
      except ValueError:
        # Mask is zero-sized
        dtype = 'bool' if asBool else 'uint16'
        return np.zeros((0,0), dtype)
      # Guaranteed not to be too small
      warnIfTooSmall = False
    if warnIfTooSmall:
      cmpShape = maskShape if isinstance(maskShape, Sequence) else maskShape.shape[:2]
      # Wait until inside 'if' so max isn't unnecessarily calculated
      # Edge case: Empty vertices set will trigger a value warning
      if len(self) == 0:
        vertMax = 0
      else:
        vertMax = self.stack().max(0)[::-1]
      if np.any(vertMax > np.array(cmpShape[:2])):
        warn('Vertices don\'t fit in the provided mask size.\n'
             f'Vertex shape: {vertMax}, mask shape: {cmpShape}')
    if checkForDisconnectedVerts:
      fillArg = []
      for verts in self: # type: XYVertices
        if verts.connected:
          fillArg.append(verts)
        else:
          # Make sure each point is treated separately, not part of a shape
          # to fill
          fillArg.extend(verts)
    else:
      fillArg = self
    if isinstance(maskShape, NChanImg):
      out = maskShape
    else:
      out = np.zeros(maskShape, 'uint16')
    nChans = 1 if out.ndim < 3 else out.shape[2]
    if fillColor is None:
      fillColor = tuple([1 for _ in range(nChans)])
    fillColorCmp = np.array(fillColor)
    if np.any((np.iinfo(out.dtype).min > fillColorCmp)
              | (fillColorCmp > np.iinfo(out.dtype).max)):
      raise ValueError('Fill color is larger or smaller than mask range can represent')
    cv.fillPoly(out, fillArg, fillColor)
    if asBool:
      return out > 0
    else:
      return out

  @staticmethod
  def fromBwMask(bwMask: BlackWhiteImg, simplifyVerts=True, externOnly=False) -> ComplexXYVertices:
    approxMethod = cv.CHAIN_APPROX_SIMPLE
    if not simplifyVerts:
      approxMethod = cv.CHAIN_APPROX_NONE
    retrMethod = cv.RETR_CCOMP
    if externOnly:
      retrMethod = cv.RETR_EXTERNAL
    # Contours are on the inside of components, so dilate first to make sure they are on the
    # outside
    #bwmask = dilation(bwmask, np.ones((3,3), dtype=bool))
    if bwMask.dtype != np.uint8:
      bwMask = bwMask.astype('uint8')
    contours, hierarchy = cv.findContours(bwMask, retrMethod, approxMethod)
    compVertices = ComplexXYVertices()
    for contour in contours:
      compVertices.append(XYVertices(contour[:,0,:]))
    if hierarchy is None:
      hierarchy = np.ones((0,1,4), int)*-1
    else:
      hierarchy = hierarchy[0,:,:]
    compVertices.hierarchy = hierarchy
    return compVertices

  def __str__(self) -> str:
    """
    Improve the readability of vertex list in table by just displaying stats of larger arrays
    :return: Human readable string representation
    """
    concatVerts = self.stack()
    if len(concatVerts) <= 4: return str(concatVerts)
    return f'Mean:\t{np.round(concatVerts.mean(0), 1)}\n' \
           f'# Points:\t{len(concatVerts)}\n' \
           f'Min:\t{concatVerts.min(0)}\n' \
           f'Max:\t{concatVerts.max(0)}'

  def __eq__(self, other: ComplexXYVertices):
    if len(self) != len(other):
      return False
    for selfVerts, otherVerts in zip(self, other):
      if not np.array_equal(selfVerts, otherVerts):
        return False
    return True

  def __ne__(self, other):
    return not self == other

  def copy(self) -> ComplexXYVertices:
    return ComplexXYVertices([lst.copy() for lst in self], self.hierarchy)

  def serialize(self):
    return str([arr.tolist() for arr in self])

  @staticmethod
  def deserialize(strObj: str) -> ComplexXYVertices:
    # TODO: Infer appropriate hierarchy from the serialized string. It is possible by finding whether vertices are given
    #  in CW or CCW order. This doesn't affect how they are drawn, but it does effect the return values of "holeVerts()"
    #  and "filledVerts()"
    outerLst = literal_eval(strObj)
    return ComplexXYVertices([XYVertices(lst) for lst in outerLst])