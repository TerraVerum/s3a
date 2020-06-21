from __future__ import annotations

from ast import literal_eval
from typing import Union, List, Sequence
from warnings import warn

import cv2 as cv
import numpy as np

from .exceptions import FRIllFormedVerticesError
from .typeoverloads import BlackWhiteImg, NChanImg
from ..projectvars.enums import FR_ENUMS


class FRVertices(np.ndarray):
  connected = True

  def __new__(cls, inputArr: Union[list, np.ndarray, tuple]=None, connected=True, **kwargs):
    # See numpy docs on subclassing ndarray
    if inputArr is None:
      inputArr = np.zeros((0,2))
    # Default to integer type if not specified, since this is how pixel coordinates will be represented anyway
    if 'dtype' not in kwargs:
      kwargs['dtype'] = int
    arr = np.asarray(inputArr, **kwargs).view(cls)
    arr.connected = connected
    return arr

  def __array_finalize__(self, obj):
    shape = self.shape
    shapeLen = len(shape)
    # indicates point, so the one dimension must have only 2 elements
    if 1 < shapeLen < 2 and shape[0] != 2:
      raise FRIllFormedVerticesError(f'A one-dimensional vertex array must be shape (2,).'
                                f' Receieved array of shape {shape}')
    elif shapeLen > 2 or shapeLen > 1 and shape[1] != 2:
      raise FRIllFormedVerticesError(f'Vertex list must be Nx2. Received shape {shape}.')
    if obj is None: return
    self.connected = getattr(obj, 'connected', True)

  @property
  def empty(self):
      return len(self) == 0

  def asPoint(self):
    if self.size == 2:
      return self.reshape(-1)
    # Reaching here means the user requested vertices as point when
    # more than one point is in the list
    raise FRIllFormedVerticesError(f'asPoint() can only be called when one vertex is in'
                              f' the vertex list. Currently has shape {self.shape}')

  def asRowCol(self):
    return np.fliplr(self)

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


class FRComplexVertices(list):
  """
  Allows holes in the component shape. Subclassing ndarray instead of list allows primitive algebraic ops on the list
  contents (e.g. subtracting/adding offset). Since normal usage doesn't typically require a mutable structure, the
  loss is minimal.
  """
  hierarchy = np.ones((0,4), dtype=int)
  """See cv.findContours for hierarchy explanation. Used in cv.RETR_CCOMP mode."""

  def __init__(self, inputArr: Union[List[FRVertices], np.ndarray]=None,
               hierarchy: Union[np.ndarray, FR_ENUMS]=None,
               coerceListElements=False):
    if hierarchy is None:
      hierarchy = FR_ENUMS.HIER_ALL_FILLED
    if inputArr is None:
      inputArr = []
    if coerceListElements:
      inputArr = [FRVertices(el) for el in inputArr]
    super().__init__(inputArr)
    # No hierarchy required unless list is longer than length 1
    numInpts = len(inputArr)
    if numInpts > 1 and hierarchy is None:
      raise FRIllFormedVerticesError(f'Must pass a hierarchy with any complex vertices of more than one vertex list, '
                                f'received vertex list of length {numInpts}')
    elif (hierarchy is None and numInpts <= 1) \
        or hierarchy is FR_ENUMS.HIER_ALL_FILLED:
      # Default hierarchy for a one- or zero-object contour list
      hierarchy = np.ones((numInpts, 4), dtype=int)*-1
    self.hierarchy = hierarchy

  def append(self, verts:FRVertices=None) -> None:
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
    if len(self) == 1:
      return self[0].asPoint()
    else:
      raise FRIllFormedVerticesError(f'Can only treat FRComplexVertices with one inner list as a point.'
                                f' Current list has {len(self)} elements.')

  def stack(self, newDtype=int) -> FRVertices:
    if len(self) == 0:
      # Check required for np vstack since it won't work with a 0-element array
      return FRVertices()
    else:
      return FRVertices(np.vstack(self), dtype=newDtype)

  def filledVerts(self) -> FRComplexVertices:
    """
    Retrieves all vertex lists corresponding to filled regions in the complex shape
    """
    idxs = np.nonzero(self.hierarchy[:,3] == -1)[0]
    return FRComplexVertices([self[ii] for ii in idxs])

  def holeVerts(self) -> FRComplexVertices:
    """
    Retrieves all vertex lists corresponding to holes in the complex shape
    """
    idxs = np.nonzero(self.hierarchy[:,3] != -1)[0]
    return FRComplexVertices([self[ii] for ii in idxs])

  def toMask(self, maskShape: Union[Sequence, NChanImg], fillColor=None, asBool=True,
             checkForDisconnectedVerts=False, warnIfTooSmall=True):
    if warnIfTooSmall:
      cmpShape = maskShape if isinstance(maskShape, Sequence) else maskShape.shape[:2]
      # Wait until inside 'if' so max isn't unnecessarily calculated
      vertMax = self.stack().max(0)[::-1]
      if np.any(vertMax > np.array(maskShape[:2])):
        warn('Vertices don\'t fit in the provided mask size.\n'
             f'Vertex shape: {vertMax}, mask shape: {maskShape}')
    if checkForDisconnectedVerts:
      fillArg = []
      for verts in self: # type: FRVertices
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
      out = np.zeros(maskShape, 'uint8')
    nChans = 1 if out.ndim < 3 else out.shape[2]
    if fillColor is None:
      fillColor = tuple([fillColor for _ in range(nChans)])
    cv.fillPoly(out, fillArg, fillColor)
    if asBool:
      return out > 0
    else:
      return out

  @staticmethod
  def fromBwMask(bwMask: BlackWhiteImg, simplifyVerts=True, externOnly=False) -> FRComplexVertices:
    approxMethod = cv.CHAIN_APPROX_SIMPLE
    if not simplifyVerts:
      approxMethod = cv.CHAIN_APPROX_NONE
    retrMethod = cv.RETR_CCOMP
    if externOnly:
      retrMethod = cv.RETR_EXTERNAL
    # Contours are on the inside of components, so dilate first to make sure they are on the
    # outside
    #bwmask = dilation(bwmask, np.ones((3,3), dtype=bool))
    contours, hierarchy = cv.findContours(bwMask.astype('uint8'), retrMethod, approxMethod)
    compVertices = []
    for contour in contours:
      compVertices.append(FRVertices(contour[:,0,:]))
    if hierarchy is None:
      hierarchy = np.ones((0,1,4), int)*-1
    return FRComplexVertices(compVertices, hierarchy[:,0,:])

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

  def __eq__(self, other: FRComplexVertices):
    # lstLens = lambda lst: np.array([len(el) for el in lst])
    return np.array_equal(self, other)

  def __ne__(self, other):
    return not self == other

  def copy(self) -> FRComplexVertices:
    return FRComplexVertices([lst.copy() for lst in self], self.hierarchy)

  def serialize(self):
    return str([arr.tolist() for arr in self])

  @staticmethod
  def deserialize(strObj: str) -> FRComplexVertices:
    # TODO: Infer appropriate hierarchy from the serialized string. It is possible by finding whether vertices are given
    #  in CW or CCW order. This doesn't affect how they are drawn, but it does effect the return values of "holeVerts()"
    #  and "filledVerts()"
    outerLst = literal_eval(strObj)
    return FRComplexVertices([FRVertices(lst) for lst in outerLst])