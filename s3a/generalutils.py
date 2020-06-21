from collections import deque
from pathlib import Path
from typing import Any, Optional, List, Collection

import numpy as np
from pandas import DataFrame as df

from s3a.projectvars import ANN_AUTH_DIR
from s3a.structures.typeoverloads import TwoDArr
from .structures import FRVertices, FRParam, FRComplexVertices


def nanConcatList(vertList) -> FRVertices:
  """
  Utility for concatenating all vertices within a list while adding
  NaN entries between each separate list
  """
  nanSep = np.ones((1,2), dtype=int)*np.nan
  allVerts = [np.zeros((0,2))]
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
  return FRComplexVertices(allVerts, coerceListElements=True)


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

def coerceDfTypes(dataframe: df, constParams: Collection[FRParam]=None):
  """
  Pandas currently has a bug where datatypes are not preserved after update operations.
  Current workaround is to coerce all types to their original values after each operation
  """
  if constParams is None:
    constParams = dataframe.columns
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


def resolveAuthorName(providedAuthName: Optional[str]) -> Optional[str]:
  authPath = Path(ANN_AUTH_DIR)
  authFile = authPath.joinpath('defaultAuthor.txt')
  if providedAuthName is not None:
    # New default author provided
    with open(authFile.absolute(), 'w') as ofile:
      ofile.write(providedAuthName)
      return providedAuthName
  # Fetch default author
  if not authFile.exists():
    authFile.touch()
  with open(str(authFile), 'r') as ifile:
    lines = ifile.readlines()
    if not lines:
      return None
  return lines[0]

def augmentException(ex: Exception, prependedMsg: str):
  ex.args = (prependedMsg, *ex.args)

def makeUniqueBaseClass(obj: Any):
  """
  Overwrites obj's class to a mixin base class.
  Property objects only work in Python if assigned to the *class* of an object, e.g.
  >>> class b:
  >>>   num = property(lambda self: 4)
  >>> ob = b()
  >>> ob.num # works as expected
  >>> ob.num2 = property(lambda self: 6)
  >>> ob.num2 # Property object at ... -- NOT AS EXPECTED!
  To work around this, simply use <type(ob).num2 = ...>. However, for regisetering properties,
  this means *all* objects of that type will have the properties of all other objects.
  To fix this, a mixin is added to this object and the property is added to the mixin.
  That way, the original object class is not altered, and each separate object will not
  end up sharing the same parameters.
  In summary, this feature enables the assignment
  >>> type(ob).a = property(...)
  without all other `b` objects also receiving this property.
  """
  class mixin(type(obj)): pass
  obj.__class__ = mixin
  return mixin