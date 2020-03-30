from collections import deque
from typing import Any, Optional, List

import numpy as np

from cdef.structures.typeoverloads import TwoDArr
from .structures import FRVertices, FRParamGroup
from pandas import DataFrame as df


def nanConcatList(vertList) -> FRVertices:
  """
  Utility for concatenating all vertices within a list while adding
  NaN entries between each separate list
  """
  nanSep = np.ones((1,2), dtype=int)*np.nan
  allVerts = []
  for curVerts in vertList:
    allVerts.append(curVerts)
    if len(curVerts) == 0: continue
    # Close the current shape
    allVerts.append(curVerts[0,:])
    allVerts.append(nanSep)
  # Take away last nan if it exists
  # if len(allVerts) > 0:
  #   allVerts.pop()
  return FRVertices(np.vstack(allVerts), dtype=float)
  #return FRVertices(dtype=float)


def splitListAtNans(concatVerts:FRVertices):
  """
  Utility for taking a single list of nan-separated region vertices
  and breaking it into several regions with no nans.
  """
  allVerts = []
  nanEntries = np.nonzero(np.isnan(concatVerts[:,0]))[0]
  curIdx = 0
  for nanEntry in nanEntries:
    curVerts = concatVerts[curIdx:nanEntry,:].astype('int')
    allVerts.append(curVerts)
    curIdx = nanEntry+1
  # Account for final grouping of verts
  allVerts.append(concatVerts[curIdx:,:].astype('int'))
  return allVerts


def sliceToArray(keySlice: slice, arrToSlice: np.ndarray):
  """
  Converts array slice into concrete array values
  """
  start, stop, step = keySlice.start, keySlice.stop, keySlice.step
  if start is None:
    start = 0
  if stop is None:
    stop = len(arrToSlice)
  outArr = np.arange(start, stop, step)
  # Remove elements that don't correspond to list indices
  outArr = outArr[np.isin(outArr, arrToSlice)]
  return outArr


def getClippedBbox(arrShape: tuple, bbox: TwoDArr, margin: int):
  """
  Given a bounding box and margin, create a clipped bounding box that does not extend
  past any dimension size from arrShape

  Parameters
  ----------
  arrShape :    2-element tuple
     Refrence array dimensions

  bbox     :    2x2 array
     [minX minY; maxX maxY] bounding box coordinates

  margin   :    int
     Offset from bounding box coords. This will not fully be added to the bounding box
     if the new margin causes coordinates to fall off either end of the reference array shape.
  """
  for ii in range(2):
    bbox[0,ii] = max(0, min(bbox[0,ii]-margin, arrShape[1-ii]))
    bbox[1,ii] = min(arrShape[1-ii], max(0, bbox[1,ii]+margin+1))
  return bbox.astype(int)

def coerceDfTypes(dataframe: df, constParams: FRParamGroup):
  """
  Pandas currently has a bug where datatypes are not preserved after update operations.
  Current workaround is to coerce all types to their original values after each operation
  """
  for field in constParams:
    try:
      dataframe[field] = dataframe[field].astype(type(field.value))
    except TypeError:
      # Coercion isn't possible, nothing to do here
      pass

def largestList(verts: List[FRVertices]) -> FRVertices:
  maxLenList = []
  for vertList in verts:
    if len(vertList) > len(maxLenList): maxLenList = vertList
  # for vertList in newVerts:
  # vertList += cropOffset[0:2]
  return FRVertices(maxLenList)


class ObjUndoBuffer:
  _maxBufferLen:Optional[int] = None
  # Used to reduce the memory requirements for the undo buffer. More steps between
  # saves means fewer required buffer entries
  _maxStepsBetweenBufSave = 1
  _stepsSinceBufSave = 0
  _OLDEST_BUF_IDX = 0
  _NEWEST_BUF_IDX = -1

  # Used to keep track of where we are in the undo stack
  _oldestId = -1
  _newestId = -1

  # Main structure of the class
  _buffer: deque

  def __init__(self, maxBufferLen=None, stepsBetweenBufSave=1):
    self._buffer = deque(maxlen=maxBufferLen)
    self._maxStepsBetweenBufSave = stepsBetweenBufSave

  def undo_getObj(self) -> Any:
    # Can't undo if the current index is the oldest in the deque
    if self._oldestId != id(self._buffer[self._NEWEST_BUF_IDX]):
      self._buffer.rotate()
    return self._buffer[self._NEWEST_BUF_IDX]

  def redo_getObj(self) -> Any:
    # Can't undo if the current index is the oldest in the deque
    if self._newestId != id(self._buffer[self._NEWEST_BUF_IDX]):
      self._buffer.rotate(-1)
    return self._buffer[self._NEWEST_BUF_IDX]

  def update(self, newObj, alternateUpdateCondtn=False):
    # If the incoming vertices are part of a brand new region, throw away all history
    # Also append if no verts are already in the deque
    if not self._buffer:
      self._buffer = deque(maxlen=self._maxBufferLen)
      shouldAppendVerts = True
    else:
      # Otherwise, proceed as normal
      # Need to clean out invalid entries when an undo was performed before the current
      # operation
      while self._buffer and self._oldestId != id(self._buffer[self._OLDEST_BUF_IDX]):
        self._buffer.popleft()

      # Check if current operation should go on the deque
      self._stepsSinceBufSave += 1
      shouldAppendVerts =  self._stepsSinceBufSave > self._maxStepsBetweenBufSave \
          or alternateUpdateCondtn
    if shouldAppendVerts:
      # Time to save the action.
      # Note: we need to save the id of the first/last deque object so we know when the
      # end of the undo stack has been reached
      # This also changes when the max size has been reached, so we should reset each
      # time a new object is added
      self._buffer.append(newObj)
      self._oldestId = id(self._buffer[self._OLDEST_BUF_IDX])
      self._newestId = id(self._buffer[self._NEWEST_BUF_IDX])
      self._stepsSinceBufSave = 0