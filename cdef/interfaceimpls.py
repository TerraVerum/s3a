from functools import partial
from typing import Tuple, Optional, Callable, List

import cv2 as cv
import numpy as np
from scipy.ndimage import binary_fill_holes
from skimage.measure import regionprops, label

from cdef.generalutils import splitListAtNans
from cdef.processingutils import growSeedpoint, cornersToFullBoundary, getCroppedImg, \
  area_coord_regionTbl
from cdef.structures import BlackWhiteImg, FRVertices
from imageprocessing import algorithms
from imageprocessing.algorithms import watershedProcess, graphCutSegmentation
from imageprocessing.common import Image, ColorSpace
from imageprocessing.processing import ImageIO, ImageProcess

def crop_to_verts(_image: Image, _fgVerts: FRVertices, _bgVerts: FRVertices,
                  _prevCompMask: BlackWhiteImg, _margin=10):
  maskLocs = np.nonzero(_prevCompMask)
  maskCoords = np.hstack([m[:,None] for m in reversed(maskLocs)])
  if maskCoords.size > 0:
    maskBbox = np.vstack([maskCoords.min(0), maskCoords.max(0)])
  else:
    maskBbox = None

  asForeground = True
  allVerts = []
  if _fgVerts.empty and _bgVerts.empty:
    # Give whole image as input
    shape = _image.shape[:2][::-1]
    _fgVerts = FRVertices([[0,0], [0, shape[1]-1],
                           [shape[0]-1, shape[1]-1], [shape[0]-1, 0]
                          ])
  if _fgVerts.empty:
    asForeground = False
    _fgVerts = _bgVerts
    _bgVerts = FRVertices()

  allVerts.append(_fgVerts)
  allVerts.append(_bgVerts)
  allVerts = np.vstack(allVerts)
  cropped, bounds = getCroppedImg(_image, allVerts, _margin, maskBbox)
  vertOffset = bounds.min(0)
  for vertList in _fgVerts, _bgVerts:
      vertList -= vertOffset
  boundSlices = slice(*bounds[:,1]), slice(*bounds[:,0])
  _prevCompMask = _prevCompMask[boundSlices]
  return ImageIO(image=cropped, fgVerts=_fgVerts, bgVerts=_bgVerts, prevCompMask=_prevCompMask,
                 boundSlices=boundSlices, origImg=_image, asForeground=asForeground,
                 allVerts=allVerts)

def update_area(_image: Image, _origImg: Image, _boundSlices: Tuple[slice], _asForeground: bool,
                _prevCompMask):
  _prevCompMask = _prevCompMask.copy()
  if _asForeground:
    bitOperation = np.bitwise_or
  else:
    # Add to background
    bitOperation = lambda curRegion, other: curRegion & (~other)

  outMask = np.zeros(_origImg.shape[:2])
  if _image.ndim > 2:
    _image = _image.asGrayScale()
  outMask[_boundSlices] = bitOperation(_prevCompMask, _image)
  return ImageIO(image=outMask)

def fill_holes(_image: Image):
  return ImageIO(image=binary_fill_holes(_image))

def openClose():
  proc = ImageIO().initProcess('Open -> Close')
  def cvt_to_uint(_image: Image):
    return ImageIO(image = _image.astype('uint8'))
  proc.addFunction(cvt_to_uint)
  proc.addProcess(algorithms.morphologyExProcess(cv.MORPH_OPEN))
  proc.addProcess(algorithms.morphologyExProcess(cv.MORPH_CLOSE))
  return proc

def keep_largest_comp(_image: Image):
  _regionPropTbl = area_coord_regionTbl(_image)
  out = np.zeros(_image.shape, bool)
  coords = _regionPropTbl.coords[_regionPropTbl.area.argmax()]
  if coords.size == 0:
    return ImageIO(image=out)
  out[coords[:,0], coords[:,1]] = True
  return ImageIO(image=out)

def rm_small_comp(_image: Image, _minSzThreshold=30):
  _regionPropTbl = area_coord_regionTbl(_image)
  validCoords = _regionPropTbl.coords[_regionPropTbl.area >= _minSzThreshold]
  out = np.zeros(_image.shape, bool)
  if len(validCoords) == 0:
    return ImageIO(image=out)
  coords = np.vstack(validCoords)
  out[coords[:,0], coords[:,1]] = True
  return ImageIO(image=out)

def get_basic_shapes(_image: Image, _prevCompMask: BlackWhiteImg,
                     _fgVerts: FRVertices):
  # Convert indices into boolean index masks
  mask = np.zeros(_image.shape[0:2], dtype='uint8')
  fillPolyArg = splitListAtNans(_fgVerts)
  mask = cv.fillPoly(mask, fillPolyArg, 1)
  # Foreground is additive, bg is subtractive. If both fg and bg are present, default to keeping old value
  return ImageIO(image=mask)

def convert_to_squares(_image: Image):
  outMask = np.zeros(_image.shape, dtype=bool)
  for region in regionprops(label(_image)):
    outMask[region.bbox[0]:region.bbox[2],region.bbox[1]:region.bbox[3]] = True
  return ImageIO(image=outMask)

def basicOpsCombo():
  proc = ImageIO().initProcess('Basic Region Operations')
  toAdd: List[ImageProcess] = []
  for func in fill_holes, keep_largest_comp, rm_small_comp:
    toAdd.append(ImageProcess.fromFunction(func, name=func.__name__.replace('_', ' ').title()))
  proc.addProcess(toAdd[0])
  proc.addProcess(openClose())
  proc.addProcess(toAdd[1])
  proc.addProcess(toAdd[2])
  return proc

def cv_grabcut(_image: Image, _fgVerts: FRVertices, _bgVerts: FRVertices,
               _prevCompMask: BlackWhiteImg, _asForeground: bool, _iters=5):
  if _image.size == 0:
    return np.zeros_like(_prevCompMask)
  img = cv.cvtColor(_image, cv.COLOR_RGB2BGR)
  # Turn foreground into x-y-width-height
  bgdModel = np.zeros((1,65),np.float64)
  fgdModel = np.zeros((1,65),np.float64)
  if not _asForeground:
    fgdClr = cv.GC_PR_BGD
    bgdClr = cv.GC_PR_FGD
  else:
    fgdClr = cv.GC_PR_FGD
    bgdClr = cv.GC_PR_BGD
  mask = np.zeros(_prevCompMask.shape, dtype='uint8')
  mask[_prevCompMask == 1] = fgdClr
  mask[_prevCompMask == 0] = bgdClr

  allverts = np.vstack([_fgVerts, _bgVerts])
  cvRect = np.array([allverts.min(0), allverts.max(0) - allverts.min(0)]).flatten()

  if np.any(_prevCompMask):
    mode = cv.GC_INIT_WITH_MASK
    for verts, fillClr in zip([_fgVerts, _bgVerts], [1,0]):
      if verts.connected and len(verts) > 0:
        cv.fillPoly(mask, [verts], fillClr)
      else:
        mask[verts.rows, verts.cols] = fillClr
  else:
    if cvRect[2] == 0 or cvRect[3] == 0:
      return ImageIO(image=np.zeros_like(_prevCompMask))
    mode = cv.GC_INIT_WITH_RECT
  cv.grabCut(img, mask, cvRect, bgdModel, fgdModel, _iters, mode=mode)
  outMask = np.where((mask==2)|(mask==0), False, True)
  return ImageIO(image=outMask)

def region_grow(_image: Image, _prevCompMask: BlackWhiteImg, _fgVerts: FRVertices,
                _asForeground: bool, _seedThresh=10):
  if _image.size == 0:
    return ImageIO(image=_prevCompMask)
  if _prevCompMask is None:
    _prevCompMask = np.zeros(_image.shape[:2], dtype=bool)
  # -----
  # DETERMINE BITWISE RELATIONSHIP B/W OLD AND NEW MASKS
  # -----
  if np.all(_fgVerts == _fgVerts[0,:]):
    # Remove unnecessary redundant seedpoints
    _fgVerts = _fgVerts[[0],:]
  if _fgVerts.shape[0] == 1:
    # Grow outward
    growFunc = growSeedpoint
    _fgVerts.connected = False
  else:
    # Grow inward
    growFunc = lambda *args: ~growSeedpoint(*args)

  tmpImgToFill = np.zeros(_image.shape[0:2], dtype='uint8')
  # For small enough shapes, get all boundary pixels instead of just shape vertices
  if _fgVerts.connected:
    _fgVerts = cornersToFullBoundary(_fgVerts, 50e3)

  newRegion = growFunc(_image, _fgVerts, _seedThresh)
  if _fgVerts.connected:
    # For connected vertices, zero out region locations outside the user defined area
    filledMask = cv.fillPoly(tmpImgToFill, [_fgVerts], 1) > 0
    newRegion[~filledMask] = False

  return ImageIO(image=newRegion)

class FRTopLevelProcessors:
  @staticmethod
  def b_regionGrowProcessor():
    return ImageProcess.fromFunction(region_grow, name='Region Growing')

  @staticmethod
  def d_graphCutProcessor():
    return graphCutSegmentation(numSegs=100)

  @staticmethod
  def a_grabCutProcessor():
    return ImageProcess.fromFunction(cv_grabcut, name='Primitive Grab Cut')

  @staticmethod
  def w_basicShapesProcessor():
    return ImageProcess.fromFunction(get_basic_shapes, name='Basic Shapes')

  @staticmethod
  def z_onlySquaresProcessor():
    return ImageProcess.fromFunction(convert_to_squares, name='Only Squares')

  @staticmethod
  def c_watershedProcessor():
    return watershedProcess()