from typing import Tuple, List

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
from imageprocessing.common import Image
from imageprocessing.processing import ImageIO, ImageProcess


def crop_to_verts(image: Image, fgVerts: FRVertices, bgVerts: FRVertices,
                  prevCompMask: BlackWhiteImg, margin=10):

  maskLocs = np.nonzero(prevCompMask)
  maskCoords = np.hstack([m[:,None] for m in reversed(maskLocs)])
  if maskCoords.size > 0:
    maskBbox = np.vstack([maskCoords.min(0), maskCoords.max(0)])
  else:
    maskBbox = None

  asForeground = True
  allVerts = []
  if fgVerts.empty and bgVerts.empty:
    # Give whole image as input
    shape = image.shape[:2][::-1]
    fgVerts = FRVertices([[0,0], [0, shape[1]-1],
                           [shape[0]-1, shape[1]-1], [shape[0]-1, 0]
                          ])
  if fgVerts.empty:
    asForeground = False
    fgVerts = bgVerts
    bgVerts = FRVertices()

  allVerts.append(fgVerts)
  allVerts.append(bgVerts)
  allVerts = np.vstack(allVerts)
  cropped, bounds = getCroppedImg(image, allVerts, margin, maskBbox)
  vertOffset = bounds.min(0)
  fgbg = []
  for vertList in fgVerts, bgVerts:
    vertList -= vertOffset
    vertList = np.clip(vertList, a_min=[0,0], a_max=bounds[1,:]-1)
    fgbg.append(vertList)
  boundSlices = slice(*bounds[:,1]), slice(*bounds[:,0])
  prevCompMask = prevCompMask[boundSlices]
  return ImageIO(image=cropped, fgVerts=fgbg[0], bgVerts=fgbg[1], prevCompMask=prevCompMask,
                 boundSlices=boundSlices, origImg=image, asForeground=asForeground,
                 allVerts=allVerts)

def update_area(image: Image, asForeground: bool,
                prevCompMask):
  prevCompMask = prevCompMask.copy()
  if asForeground:
    bitOperation = np.bitwise_or
  else:
    # Add to background
    bitOperation = lambda curRegion, other: curRegion & (~other)
  return ImageIO(image=bitOperation(prevCompMask, image))

def return_to_full_size(image: Image, origImg: Image, boundSlices: Tuple[slice]):
  outMask = np.zeros(origImg.shape[:2])
  if image.ndim > 2:
    image = image.asGrayScale()
  outMask[boundSlices] = image
  return ImageIO(image=outMask)

def fill_holes(image: Image):
  return ImageIO(image=binary_fill_holes(image))

def openClose():
  proc = ImageIO().initProcess('Open -> Close')
  def cvt_to_uint(image: Image):
    return ImageIO(image = image.astype('uint8'))
  proc.addFunction(cvt_to_uint)
  proc.addProcess(algorithms.morphologyExProcess(cv.MORPH_OPEN))
  proc.addProcess(algorithms.morphologyExProcess(cv.MORPH_CLOSE))
  return proc

def keep_largest_comp(image: Image):
  regionPropTbl = area_coord_regionTbl(image)
  out = np.zeros(image.shape, bool)
  coords = regionPropTbl.coords[regionPropTbl.area.argmax()]
  if coords.size == 0:
    return ImageIO(image=out)
  out[coords[:,0], coords[:,1]] = True
  return ImageIO(image=out)

def rm_small_comp(image: Image, minSzThreshold=30):
  regionPropTbl = area_coord_regionTbl(image)
  validCoords = regionPropTbl.coords[regionPropTbl.area >= minSzThreshold]
  out = np.zeros(image.shape, bool)
  if len(validCoords) == 0:
    return ImageIO(image=out)
  coords = np.vstack(validCoords)
  out[coords[:,0], coords[:,1]] = True
  return ImageIO(image=out)

def get_basic_shapes(image: Image, fgVerts: FRVertices):
  # Convert indices into boolean index masks
  mask = np.zeros(image.shape[0:2], dtype='uint8')
  fillPolyArg = splitListAtNans(fgVerts)
  mask = cv.fillPoly(mask, fillPolyArg, 1)
  # Foreground is additive, bg is subtractive. If both fg and bg are present, default to keeping old value
  return ImageIO(image=mask)

def convert_to_squares(image: Image):
  outMask = np.zeros(image.shape, dtype=bool)
  for region in regionprops(label(image)):
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

def cv_grabcut(image: Image, fgVerts: FRVertices, bgVerts: FRVertices,
               prevCompMask: BlackWhiteImg, asForeground: bool, noPrevMask: bool,
               iters=5):
  if image.size == 0:
    return ImageIO(image=np.zeros_like(prevCompMask))
  img = cv.cvtColor(image, cv.COLOR_RGB2BGR)
  # Turn foreground into x-y-width-height
  bgdModel = np.zeros((1,65),np.float64)
  fgdModel = np.zeros((1,65),np.float64)
  if not asForeground:
    fgdClr = cv.GC_PR_BGD
    bgdClr = cv.GC_PR_FGD
  else:
    fgdClr = cv.GC_PR_FGD
    bgdClr = cv.GC_PR_BGD
  mask = np.zeros(prevCompMask.shape, dtype='uint8')
  mask[prevCompMask == 1] = fgdClr
  mask[prevCompMask == 0] = bgdClr

  allverts = np.vstack([fgVerts, bgVerts])
  cvRect = np.array([allverts.min(0), allverts.max(0) - allverts.min(0)]).flatten()

  if noPrevMask:
    if cvRect[2] == 0 or cvRect[3] == 0:
      return ImageIO(image=np.zeros_like(prevCompMask))
    mode = cv.GC_INIT_WITH_RECT
  else:
    mode = cv.GC_INIT_WITH_MASK
    for verts, fillClr in zip([fgVerts, bgVerts], [1, 0]):
      # Grabcut throws errors when the mask is totally full or empty. To prevent this,
      # clip vertices to allow at least a 1-pixel boundary on all image sides
      verts = np.clip(verts, a_min=1, a_max=np.array(mask.shape[::-1])-2).view(FRVertices)
      if verts.connected and len(verts) > 0:
        cv.fillPoly(mask, [verts], fillClr)
      else:
        mask[verts.rows, verts.cols] = fillClr
  cv.grabCut(img, mask, cvRect, bgdModel, fgdModel, iters, mode=mode)
  outMask = np.where((mask==2)|(mask==0), False, True)
  return ImageIO(image=outMask)

def region_grow(image: Image, prevCompMask: BlackWhiteImg, fgVerts: FRVertices,
                seedThresh=10):
  if image.size == 0:
    return ImageIO(image=prevCompMask)
  if prevCompMask is None:
    prevCompMask = np.zeros(image.shape[:2], dtype=bool)
  # -----
  # DETERMINE BITWISE RELATIONSHIP B/W OLD AND NEW MASKS
  # -----
  if np.all(fgVerts == fgVerts[0, :]):
    # Remove unnecessary redundant seedpoints
    fgVerts = fgVerts[[0], :]
  if fgVerts.shape[0] == 1:
    # Grow outward
    growFunc = growSeedpoint
    fgVerts.connected = False
  else:
    # Grow inward
    growFunc = lambda *args: ~growSeedpoint(*args)

  tmpImgToFill = np.zeros(image.shape[0:2], dtype='uint8')
  # For small enough shapes, get all boundary pixels instead of just shape vertices
  if fgVerts.connected:
    fgVerts = cornersToFullBoundary(fgVerts, 50e3)

  newRegion = growFunc(image, fgVerts, seedThresh)
  if fgVerts.connected:
    # For connected vertices, zero out region locations outside the user defined area
    filledMask = cv.fillPoly(tmpImgToFill, [fgVerts], 1) > 0
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
    proc = FRTopLevelProcessors.w_basicShapesProcessor()
    proc.name = 'Only Squares'
    proc.addProcess(ImageProcess.fromFunction(convert_to_squares, name=convert_to_squares.__name__))
    return proc

  @staticmethod
  def c_watershedProcessor():
    return watershedProcess()
