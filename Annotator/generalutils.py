import numpy as np

from Annotator.params import ABParamGroup
from pandas import DataFrame as df


def nanConcatList(vertList):
  """
  Utility for concatenating all vertices within a list while adding
  NaN entries between each separate list
  """
  if isinstance(vertList, np.ndarray):
    vertList = [vertList]
  nanSep = np.ones((1,2), dtype=int)*np.nan
  allVerts = []
  for curVerts in vertList:
    allVerts.append(curVerts)
    allVerts.append(nanSep)
  # Take away last nan if it exists
  if len(allVerts) > 0:
    allVerts.pop()
    return np.vstack(allVerts)
  return np.array([]).reshape(-1,2)


def splitListAtNans(concatVerts:np.ndarray):
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


def getClippedBbox(arrShape: tuple, bbox: np.ndarray, margin: int):
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
    bbox[0,ii] = np.maximum(0, bbox[0,ii]-margin)
    bbox[1,ii] = np.minimum(arrShape[1-ii], bbox[1,ii]+margin)
  return bbox.astype(int)

def coerceDfTypes(dataframe: df, constParams: ABParamGroup):
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