import html
from typing import Callable, Tuple, Union, Sequence, List, Collection

import cv2 as cv
import numpy as np
import pyqtgraph as pg
from pandas import DataFrame as df
from skimage import io
from skimage import transform
from skimage.exposure import exposure

from utilitys import PrjParam, fns
# Needs to be visible outside this file
# noinspection PyUnresolvedReferences
from utilitys.fns import hierarchicalUpdate
from utilitys.typeoverloads import FilePath
from .structures import TwoDArr, XYVertices, ComplexXYVertices, NChanImg, BlackWhiteImg


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
  exMsg = str(ex)
  if fns.usingPostponedErrors:
    # Escape for qt dialog box
    prependedMsg = html.escape(prependedMsg)
    exMsg = html.escape(exMsg)
  ex.args = (prependedMsg + exMsg,)

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

def cvImsave_rgb(fname: FilePath, image: np.ndarray, *args, errOk=False, **kwargs):
  image = _maybeBgrToRgb(image)
  try:
    cv.imwrite(str(fname), image, *args, **kwargs)
  except cv.error:
    if not errOk:
      raise
    # Dtype incompatible
    io.imsave(fname, image)

def cvImread_rgb(fname: FilePath, *args, **kwargs):
  image = cv.imread(str(fname), *args, **kwargs)
  return _maybeBgrToRgb(image)

def tryCvResize(image: NChanImg, newSize: Union[tuple, float],
                asRatio=True, interp=cv.INTER_CUBIC):
  """
  Uses cv.resize where posisble, but if dtypes prevent this, it falls back to skimage.transform.rescale/resize

  :param image: Image to resize
  :param newSize: Either ratio to scale each axis or new image size (x, y -- not row, col)
  :param asRatio: Whether to interpret `newSize` as a ratio or new image dimensions
  :param interp: Interpolation to use, if cv.resize is available for the given dtype
  """
  if asRatio:
    if not isinstance(newSize, tuple):
      newSize = (newSize, newSize)
    args = dict(dsize=(0,0), fx=newSize[0], fy=newSize[1])
  else:
    args = dict(dsize=newSize)
  try:
    image = cv.resize(image, **args, interpolation=interp)
  except (TypeError, cv.error):
    oldRange = (image.min(), image.max())
    if asRatio:
      rescaled = transform.rescale(image, newSize, multichannel=image.ndim > 2)
    else:
      rescaled = transform.resize(image, newSize[::-1])
    image = exposure.rescale_intensity(rescaled, out_range=oldRange).astype(image.dtype)
  return image



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
                  coordsAsSlices=False, returnCoords=True) -> (np.ndarray, np.ndarray):
  """
  Crops an image according to the specified vertices such that the returned image does not extend
  past vertices plus margin (including other bboxes if specified). All bboxes and output coords
  are of the form [[xmin, xmax], [ymin, ymax]]. Slices are (row slices, col slices) if `coordsAsSlices`
  is specified.
  """
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
  if not returnCoords:
    return croppedImg
  if coordsAsSlices:
    return croppedImg, coordSlices
  else:
    return croppedImg, compCoords

def imgCornerVertices(img: NChanImg=None) -> XYVertices:
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

def _getPtAngles(pts):
  midpt = np.mean(pts, 0)
  relPosPts = (pts - midpt).view(np.ndarray)
  return np.arctan2(*relPosPts.T[::-1])

def orderContourPts(pts: XYVertices, ccw=True):
  """
  Only guaranteed to work for convex hulls, i.e. shapes not intersecting themselves. Orderes
  an arbitrary list of coordinates into one that works well line plotting, i.e. doesn't show
  excessive self-crossing when plotting
  """
  if len(pts) < 3:
    return pts
  angles = _getPtAngles(pts)
  ptOrder = np.argsort(angles)
  if not ccw:
    ptOrder = ptOrder[::-1]
  return pts[ptOrder]

# def movePtsTowardCenter(pts: XYVertices, dist=1):
#   if not pts.size:
#     return pts
#   angles = _getPtAngles(pts)
#   adjusts = np.column_stack([np.cos(angles), np.sin(angles)])
#   adjusts[np.abs(adjusts) < 0.01] = 0
#   # Adjust by whole steps
#   adjusts = np.sign(adjusts)*dist
#   return pts - adjusts

def symbolFromVerts(verts: ComplexXYVertices):
  concatRegion, isfinite = stackedVertsPlusConnections(verts)
  boundLoc = np.nanmin(concatRegion, 0, keepdims=True)
  useVerts = concatRegion - boundLoc + 0.5
  return pg.arrayToQPath(*useVerts.T, connect=isfinite), boundLoc

# Credit: https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
def bboxIou(boxA, boxB):
  """
  determine the (x, y)-coordinates of the intersection rectangle. Both boxes are formatted
   [[xmin, ymin], [xmax, ymax]]"""
  boxA = boxA.ravel()
  boxB = boxB.ravel()
  xA = max(boxA[0], boxB[0])
  yA = max(boxA[1], boxB[1])
  xB = min(boxA[2], boxB[2])
  yB = min(boxA[3], boxB[3])
  # compute the area of intersection rectangle
  interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
  # compute the area of both the prediction and ground-truth
  # rectangles
  boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
  boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
  # compute the intersection over union by taking the intersection
  # area and dividing it by the sum of prediction + ground-truth
  # areas - the interesection area
  iou = interArea / float(boxAArea + boxBArea - interArea)
  # return the intersection over union value
  return iou

# Credit: https://stackoverflow.com/a/13624858/9463643
class classproperty:

  def __init__(self, fget):
    self.fget = fget

  def __get__(self, owner_self, owner_cls):
    return self.fget(owner_cls)