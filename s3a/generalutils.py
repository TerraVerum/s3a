from functools import lru_cache
from pathlib import Path
from typing import Any, Optional, Callable, Tuple, Union, Sequence, List, Collection
from contextlib import contextmanager

import cv2 as cv
import numpy as np
from pandas import DataFrame as df
from utilitys import PrjParam
from utilitys.typeoverloads import FilePath

from .structures import TwoDArr, XYVertices, ComplexXYVertices, NChanImg, BlackWhiteImg
from utilitys.fns import hierarchicalUpdate

def stackedVertsPlusConnections(vertList: ComplexXYVertices) -> (XYVertices, np.ndarray):
  """
  Utility for concatenating all vertices within a list while recording where separations
  occurred
  """
  allVerts = [np.zeros((0,2))]
  separationIdxs = []
  idxOffset = 0
  for curVerts in vertList:
    allVerts.append(curVerts)
    vertsLen = len(curVerts)
    if vertsLen == 0: continue
    # Close the current shape
    allVerts.append(curVerts[0,:])
    separationIdxs.append(idxOffset + vertsLen)
    idxOffset += vertsLen + 1
  # Take away last separator if it exists
  if len(separationIdxs) > 0:
    separationIdxs.pop()
  allVerts = np.vstack(allVerts)
  isfinite = np.ones(len(allVerts), bool)
  isfinite[separationIdxs] = False
  return XYVertices(allVerts, dtype=float), isfinite
  #return XYVertices(dtype=float)


def splitListAtNans(concatVerts:XYVertices):
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
  return ComplexXYVertices(allVerts, coerceListElements=True)

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
  bbox = bbox.astype(int)
  bbox[0] -= margin
  # Add extra 1 since python slicing stops before last index
  bbox[1] += margin+1
  arrShape = arrShape[:2]
  return np.clip(bbox, 0, arrShape[::-1])

def coerceDfTypes(dataframe: df, constParams: Collection[PrjParam]=None):
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
  return dataframe

def largestList(verts: List[XYVertices]) -> XYVertices:
  maxLenList = []
  for vertList in verts:
    if len(vertList) > len(maxLenList): maxLenList = vertList
  # for vertList in newVerts:
  # vertList += cropOffset[0:2]
  return XYVertices(maxLenList)

def augmentException(ex: Exception, prependedMsg: str):
  ex.args = (prependedMsg + str(ex),)

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


def lower_NoSpaces(name: str):
  return name.replace(' ', '').lower()

def safeCallFuncList(fnNames: Collection[str], funcLst: List[Callable],
                     fnArgs: List[Sequence]=None):
  errs = []
  rets = []
  if fnArgs is None:
    fnArgs = [()]*len(fnNames)
  for key, fn, args in zip(fnNames, funcLst, fnArgs):
    curRet, curErr = safeCallFunc(key, fn, *args)
    rets.append(curRet)
    if curErr: errs.append(curErr)
  return rets, errs

def safeCallFunc(fnName: str, func: Callable, *fnArgs):
  ret = err = None
  try:
    ret = func(*fnArgs)
  except Exception as ex:
    err = f'{fnName}: {ex}'
  return ret, err

def _maybeBgrToRgb(image: np.ndarray):
  """Treats 3/4-channel images as BGR/BGRA for opencv saving/reading"""
  if image.ndim > 2:
    # if image.shape[0] == 1:
    #   image = image[...,0]
    if image.shape[2] >= 3:
      lastAx = np.arange(image.shape[2], dtype='int')
      # Swap B & R
      lastAx[[0,2]] = [2,0]
      image = image[...,lastAx]
  return image

def cvImsave_rgb(fname: FilePath, image: np.ndarray, *args, **kwargs):
  image = _maybeBgrToRgb(image)
  cv.imwrite(str(fname), image, *args, **kwargs)

def cvImread_rgb(fname: FilePath, *args, **kwargs):
  image = cv.imread(str(fname), *args, **kwargs)
  return _maybeBgrToRgb(image)


def cornersToFullBoundary(cornerVerts: Union[XYVertices, ComplexXYVertices], sizeLimit: float=np.inf,
                          fillShape: Tuple[int]=None, stackResult=True) -> Union[XYVertices, ComplexXYVertices]:
  """
  From a list of corner vertices, returns a list with one vertex for every border pixel.
  Example:
  >>> cornerVerts = XYVertices([[0,0], [100,0], [100,100],[0,100]])
  >>> cornersToFullBoundary(cornerVerts)
  # [[0,0], [1,0], ..., [100,0], [100,1], ..., [100,100], ..., ..., [0,100]]
  :param cornerVerts: Corners of the represented polygon
  :param sizeLimit: The largest number of pixels from the enclosed area allowed before the full boundary is no
  longer returned. For instance:
    >>> cornerVerts = XYVertices([[0,0], [1000,0], [1000,1000],[0,1000]])
    >>> cornersToFullBoundary(cornerVerts, 10e5)
    will *NOT* return all boundary vertices, since the enclosed area (10e6) is larger than sizeLimit.
  :param fillShape: Size of mask to create. Useful if verts may extend beyond image dimensions
    and should be truncated. If None, no truncation will occur except for negative verts.
  :param stackResult: Whether the result should be ComplexXYVertices (if stackResult is False)
    or a stacked list of exterior verts (if stackResult is True)
  :return: List with one vertex for every border pixel, unless *sizeLimit* is violated.
  """
  if isinstance(cornerVerts, XYVertices):
    cornerVerts = ComplexXYVertices([cornerVerts])
  if fillShape is not None:
    fillShape = tuple(fillShape)
  filledMask = cornerVerts.toMask(fillShape, warnIfTooSmall=False)
  cornerVerts = ComplexXYVertices.fromBwMask(filledMask, simplifyVerts=False)
  if not stackResult:
    return cornerVerts
  cornerVerts = cornerVerts.filledVerts().stack()
  numCornerVerts = len(cornerVerts)
  if numCornerVerts > sizeLimit:
    spacingPerSamp = int(numCornerVerts/sizeLimit)
    cornerVerts = cornerVerts[::spacingPerSamp]
  return cornerVerts


def getCroppedImg(image: NChanImg, verts: np.ndarray, margin=0, *otherBboxes: np.ndarray,
                  coordsAsSlices=False, returnSlices=True) -> (np.ndarray, np.ndarray):
  verts = np.vstack(verts)
  img_np = image
  compCoords = np.vstack([verts.min(0), verts.max(0)])
  if len(otherBboxes) > 0:
    for dim in range(2):
      for ii, cmpFunc in zip(range(2), [min, max]):
        otherCmpVals = [curBbox[ii, dim] for curBbox in otherBboxes]
        compCoords[ii,dim] = cmpFunc(compCoords[ii,dim], *otherCmpVals)
  compCoords = getClippedBbox(img_np.shape, compCoords, margin)
  coordSlices = (slice(compCoords[0,1], compCoords[1,1]),
                 slice(compCoords[0,0],compCoords[1,0]))
  # Verts are x-y, index into image with row-col
  indexer = coordSlices
  if image.ndim > 2:
    indexer += (slice(None),)
  croppedImg = image[indexer]
  if not returnSlices:
    return croppedImg
  if coordsAsSlices:
    return croppedImg, coordSlices
  else:
    return croppedImg, compCoords

def imgCornerVertices(img: NChanImg=None):
  """Returns [x,y] vertices for each corner of the input image"""
  if img is None:
    return XYVertices()
  fullImShape_xy = img.shape[:2][::-1]
  return XYVertices([[0,                   0],
              [0,                   fullImShape_xy[1]-1],
              [fullImShape_xy[0]-1, fullImShape_xy[1]-1],
              [fullImShape_xy[0]-1, 0]
              ])

def resize_pad(img: NChanImg, newSize: Tuple[int, int], interp=cv.INTER_NEAREST, padVal=0):
  """
  Resizes image to the requested size using the specified interpolation method.
  :param img: Image to resize
  :param newSize: New size for image. Since aspect ratio is maintained, the portion of the
    image which couldn't be resized fully is padded with a constant value of `padVal`.
    For instance, if the original image is 5x10 and the requested new size is 10x15, then
    after resizing the image will be 7x15 to preserve aspect ratio. 2 pixels of padding
    will be added on the left and 1 pixel of padding will be added on the right so the final
    output is 10x15.
  :param padVal: Value to pad dimension that couldn't be fully resized
  :param interp: Interpolation method to use during resizing
  :param padVal: Constant value to use during padding
  :return: Resized and padded image
  """
  needsRotate = False
  # Make sure largest dimension is along rows for easier treatment and rotate back after
  newSize = np.array(newSize)
  resizeRatio = min(newSize/np.array(img.shape[:2]))
  paddedImg = cv.resize(img, (0, 0), fx = resizeRatio, fy = resizeRatio, interpolation=interp)
  padDimension = np.argmax(newSize - np.array(paddedImg.shape[:2]))
  if padDimension != 1:
    paddedImg = cv.rotate(paddedImg, cv.ROTATE_90_CLOCKWISE)
    newSize = newSize[::-1]
    needsRotate = True
  # Now pad dimension is guaranteed to be index 1
  padDimension = 1
  padding = (newSize[padDimension] - paddedImg.shape[padDimension]) // 2
  if padding > 0:
    paddedImg = cv.copyMakeBorder(paddedImg, 0, 0, padding, padding, cv.BORDER_CONSTANT, value=padVal)
  # Happens during off-by-one truncated division
  remainingPadding = newSize[padDimension] - paddedImg.shape[padDimension]
  if remainingPadding > 0:
    paddedImg = cv.copyMakeBorder(paddedImg, 0, 0, remainingPadding, 0, cv.BORDER_CONSTANT, value=padVal)
  if needsRotate:
    paddedImg = cv.rotate(paddedImg, cv.ROTATE_90_COUNTERCLOCKWISE)
  return paddedImg

@contextmanager
def monkeyPatch(obj, toChange: str, newVal):
  oldVal = getattr(obj, toChange, None)
  setattr(obj, toChange, newVal)
  yield
  if oldVal is None:
    delattr(obj, toChange)
  else:
    setattr(obj, toChange, oldVal)

def showMaskDiff(oldMask: BlackWhiteImg, newMask: BlackWhiteImg):
  infoMask = np.tile(oldMask[...,None].astype('uint8')*255, (1,1,3))
  # Was there, now it's not -- color red
  infoMask[oldMask & ~newMask,:] = [255,0,0]
  # Wasn't there, now it is -- color green
  infoMask[~oldMask & newMask,:] = [0,255,0]
  return infoMask

# Poor man's LRU dict, since I don't yet feel like including another pypi dependency
class MaxSizeDict(dict):
  def __init__(self, *args, maxsize:int=np.inf, **kwargs):
    super().__init__(*args, **kwargs)
    self.maxsize = maxsize

  def __setitem__(self, key, value):
    if len(self) == self.maxsize:
      # Evict oldest inserted entry
      self.pop(next(iter(self.keys())))
    super().__setitem__(key, value)

def orderContourPts(pts: XYVertices, ccw=True):
  """
  Only guaranteed to work for convex hulls, i.e. shapes not intersecting themselves. Orderes
  an arbitrary list of coordinates into one that works well line plotting, i.e. doesn't show
  excessive self-crossing when plotting
  """
  if len(pts) < 3:
    return pts
  midpt = np.mean(pts, 0)
  relPosPts = pts - midpt
  angles = np.arctan2(*relPosPts.T[::-1])
  ptOrder = np.argsort(angles)
  if not ccw:
    ptOrder = ptOrder[::-1]
  return pts[ptOrder]