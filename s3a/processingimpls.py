from typing import Tuple, List

import cv2 as cv
import numpy as np
import pandas as pd
from imageprocessing import algorithms
from imageprocessing.common import Image
from imageprocessing.processing import ImageIO, ImageProcess
from scipy.ndimage import binary_fill_holes
from skimage.measure import regionprops, label, regionprops_table
from skimage.morphology import flood
from skimage.segmentation import quickshift
import skimage.segmentation as seg

from s3a.generalutils import splitListAtNans, getClippedBbox
from s3a.structures import BlackWhiteImg, FRVertices, NChanImg, FRComplexVertices, \
  GrayImg, RgbImg, FRAlgProcessorError


def getCroppedImg(image: NChanImg, verts: np.ndarray, margin: int,
                  *otherBboxes: np.ndarray,
                  coordsAsSlices=False) -> (np.ndarray, np.ndarray):
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
  croppedImg = image[coordSlices + (slice(None),)]
  if coordsAsSlices:
    return croppedImg, coordSlices
  else:
    return croppedImg, compCoords

def growSeedpoint(img: NChanImg, seeds: FRVertices, thresh: float) -> BlackWhiteImg:
  shape = np.array(img.shape[0:2])
  bwOut = np.zeros(shape, dtype=bool)
  # Turn x-y vertices into row-col seeds
  seeds = seeds[:, ::-1]
  # Remove seeds that don't fit in the image
  seeds = seeds[np.all(seeds >= 0, 1)]
  seeds = seeds[np.all(seeds < shape, 1)]

  for seed in seeds:
    for chan in range(img.shape[2]):
      curBwMask = flood(img[...,chan], tuple(seed), tolerance=thresh)
      bwOut |= curBwMask
  return bwOut

def cornersToFullBoundary(cornerVerts: FRVertices, sizeLimit: float=np.inf) -> FRVertices:
  """
  From a list of corner vertices, returns a list with one vertex for every border pixel.
  Example:
  >>> cornerVerts = FRVertices([[0,0], [100,0], [100,100],[0,100]])
  >>> cornersToFullBoundary(cornerVerts)
  # [[0,0], [1,0], ..., [100,0], [100,1], ..., [100,100], ..., ..., [0,100]]
  :param cornerVerts: Corners of the represented polygon
  :param sizeLimit: The largest number of pixels from the enclosed area allowed before the full boundary is no
  longer returned. For instance:
    >>> cornerVerts = FRVertices([[0,0], [1000,0], [1000,1000],[0,1000]])
    >>> cornersToFullBoundary(cornerVerts, 10e5)
    will *NOT* return all boundary vertices, since the enclosed area (10e6) is larger than sizeLimit.
  :return: List with one vertex for every border pixel, unless *sizeLimit* is violated.
  """
  fillShape = cornerVerts.asRowCol().max(0)+1
  filledMask = FRComplexVertices([cornerVerts]).toMask(tuple(fillShape))
  cornerVerts = FRComplexVertices.fromBwMask(filledMask, simplifyVerts=False).filledVerts().stack()
  numCornerVerts = len(cornerVerts)
  if numCornerVerts > sizeLimit:
    spacingPerSamp = int(numCornerVerts/sizeLimit)
    cornerVerts = cornerVerts[::spacingPerSamp]
  return cornerVerts

def colorLabelsWithMean(labelImg: GrayImg, refImg: NChanImg) -> RgbImg:
  outImg = np.empty(refImg.shape)
  labels = np.unique(labelImg)
  for curLabel in labels:
    curmask = labelImg == curLabel
    outImg[curmask,:] = refImg[curmask,:].reshape(-1,3).mean(0)
  return outImg

def growSeedpoint_cv_fastButErratic(img: NChanImg, seeds: FRVertices, thresh: float):
  if len(seeds) == 0:
    return np.zeros(img.shape[:2], bool)
  if img.ndim > 2:
    nChans = img.shape[2]
  else:
    nChans = 1
  thresh = int(np.clip(thresh, 0, 255))
  imRCShape = np.array(img.shape[:2])
  bwOut = np.zeros(imRCShape+2, 'uint8')
  # Throw away seeds outside image boundaries
  seeds = seeds[np.all(seeds < imRCShape, axis=1)]
  seeds = np.fliplr(seeds)
  mask = np.zeros(imRCShape+2, 'uint8')
  for seed in seeds:
    mask.fill(0)
    flooded = img.copy()
    seed = tuple(seed.flatten())
    _, _, curOut, _ = cv.floodFill(flooded, mask, seed, 255, (thresh,)*nChans, (thresh,)*nChans,8)
    bwOut |= curOut
  bwOut = bwOut[1:-1,1:-1]
  return bwOut.astype(bool)

def area_coord_regionTbl(_image: Image):
  if not np.any(_image.data):
    return pd.DataFrame({'coords': [np.array([[]])], 'area': [0]})
  regionDict = regionprops_table(label(_image.data), properties=('coords', 'area'))
  _regionPropTbl = pd.DataFrame(regionDict)
  return _regionPropTbl


# 0 = unspecified, 1 = background, 2 = foreground
_historyMask = np.array([[]], 'uint8')
def format_vertices(image: Image, fgVerts: FRVertices, bgVerts: FRVertices,
                    prevCompMask: BlackWhiteImg, firstRun: bool,
                    keepVertHistory=True):
  global _historyMask

  if firstRun or not keepVertHistory:
    _historyMask = np.zeros(image.shape[:2], 'uint8')

  asForeground = True
  # 0 = unspecified, 1 = background, 2 = foreground
  for fillClr, verts in enumerate([bgVerts, fgVerts], 1):
    if not verts.empty:
      cv.fillPoly(_historyMask, [verts], fillClr)

  fullImShape_xy = image.shape[:2][::-1]
  if fgVerts.empty and bgVerts.empty:
    # Give whole image as input
    fgVerts = FRVertices([[0,                   0],
                          [0,                   fullImShape_xy[1]-1],
                          [fullImShape_xy[0]-1, fullImShape_xy[1]-1],
                          [fullImShape_xy[0]-1, 0]
                          ])
    fgVerts = cornersToFullBoundary(fgVerts)
    _historyMask[fgVerts.rows, fgVerts.cols] = True
  curHistory = _historyMask.copy()
  if fgVerts.empty:
    # Invert the mask and paint foreground pixels
    asForeground = False
    # Invert the history mask too
    curHistory[_historyMask == 2] = 1
    curHistory[_historyMask == 1] = 2
    fgVerts = bgVerts
    bgVerts = FRVertices()

  if asForeground:
    foregroundAdjustedCompMask = prevCompMask.copy()
  else:
    foregroundAdjustedCompMask = ~prevCompMask

  # Default to bound slices that encompass the whole image
  bounds = np.array([[0, 0], image.shape[:2][::-1]])
  boundSlices = slice(*bounds[:,1]), slice(*bounds[:,0])
  return ImageIO(image=image, fgVerts=fgVerts, bgVerts=bgVerts, asForeground=asForeground,
                 historyMask=curHistory, prevCompMask=foregroundAdjustedCompMask,
                 origCompMask=prevCompMask, boundSlices=boundSlices)

def crop_to_local_area(image: Image, fgVerts: FRVertices, bgVerts: FRVertices,
                       prevCompMask: BlackWhiteImg, margin_pctRoiSize=10):
  allVerts = np.vstack([fgVerts, bgVerts])
  if len(allVerts) == 1:
    # Single point, use image size as reference shape
    vertArea_rowCol = image.shape[:2]
  else:
    # Lots of points, use their bounded area
    vertArea_rowCol = (allVerts.max(0)-allVerts.min(0))[::-1]
  margin = round(max(vertArea_rowCol) * (margin_pctRoiSize / 100))
  cropped, bounds = getCroppedImg(image, allVerts, margin)
  vertOffset = bounds.min(0)
  for vertList in fgVerts, bgVerts:
    vertList -= vertOffset
    np.clip(vertList, a_min=[0,0], a_max=bounds[1,:]-1, out=vertList)
  boundSlices = slice(*bounds[:,1]), slice(*bounds[:,0])
  croppedCompMask = prevCompMask[boundSlices]
  curHistory = _historyMask[boundSlices]

  rectThickness = int(max(1, *image.shape)*0.005)
  toPlot = cv.rectangle(image.copy(), tuple(bounds[0,:]), tuple(bounds[1,:]),
                        (255,0,0), rectThickness)
  toPlot = Image(toPlot, name='Cropped Region')

  return ImageIO(image=cropped, fgVerts=fgVerts, bgVerts=bgVerts, prevCompMask=croppedCompMask,
                 boundSlices=boundSlices, historyMask=curHistory, toPlot=toPlot, display='toPlot')

def apply_process_result(image: Image, asForeground: bool,
                         prevCompMask: BlackWhiteImg, origCompMask: BlackWhiteImg,
                         boundSlices: Tuple[slice,slice]):
  if asForeground:
    bitOperation = np.bitwise_or
  else:
    # Add to background
    bitOperation = lambda curRegion, other: ~(curRegion | other)
  # The other basic operations need the rest of the component mask to work properly,
  # so expand the current area of interest only as much as needed. Returning to full size
  # now would incur unnecessary addtional processing times for the full-sized image
  outMask = origCompMask.copy()
  outMask[boundSlices] = bitOperation(prevCompMask, image)
  foregroundPixs = np.c_[np.nonzero(outMask)]
  # Keep algorithm from failing when no foreground pixels exist
  if len(foregroundPixs) == 0:
    mins = [0,0]
    maxs = [1,1]
  else:
    mins = foregroundPixs.min(0)
    maxs = foregroundPixs.max(0)
  # Add 1 to max slice so stopping value is last foreground pixel
  newSlices = (slice(mins[0], maxs[0]+1), slice(mins[1], maxs[1]+1))
  return ImageIO(image=outMask[newSlices], boundSlices=newSlices)

def return_to_full_size(image: Image, origCompMask: BlackWhiteImg,
                        boundSlices: Tuple[slice]):
  out = np.zeros_like(origCompMask)
  if image.ndim > 2:
    image = image.asGrayScale()
  out[boundSlices] = image
  return ImageIO(image=out)

def fill_holes(image: Image):
  return ImageIO(image=binary_fill_holes(image))

def cvt_to_uint(image: Image):
  if image.ndim > 2:
    image = image.asGrayScale()
  return ImageIO(image = image.astype('uint8'), display=None)

def disallow_paint_tool(_image: Image, fgVerts: FRVertices, bgVerts: FRVertices):
  if len(np.vstack([fgVerts, bgVerts])) < 2:
    raise FRAlgProcessorError('This algorithm requires an enclosed area to work.'
                              ' Only one vertex was given as an input.')
  return ImageIO(image=_image)

def openClose():
  proc = ImageIO().initProcess('Open -> Close')
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

def rm_small_comps(image: Image, minSzThreshold=30):
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
  verts = splitListAtNans(fgVerts)
  mask = verts.toMask(image.shape[0:2])
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
  for func in fill_holes, keep_largest_comp, rm_small_comps:
    toAdd.append(ImageProcess.fromFunction(func, name=func.__name__.replace('_', ' ').title()))
  proc.addProcess(toAdd[0])
  proc.addProcess(openClose())
  proc.addProcess(toAdd[1])
  proc.addProcess(toAdd[2])
  return proc


def _grabcutResultToMask(gcResult):
  return np.where((gcResult==2)|(gcResult==0), False, True)

def cv_grabcut(image: Image, fgVerts: FRVertices, bgVerts: FRVertices,
               prevCompMask: BlackWhiteImg, noPrevMask: bool,
               historyMask: GrayImg, iters=5):
  if image.size == 0:
    return ImageIO(image=np.zeros_like(prevCompMask))
  img = cv.cvtColor(image, cv.COLOR_RGB2BGR)
  # Turn foreground into x-y-width-height
  bgdModel = np.zeros((1,65),np.float64)
  fgdModel = np.zeros((1,65),np.float64)
  mask = np.zeros(prevCompMask.shape, dtype='uint8')
  mask[prevCompMask == 1] = cv.GC_PR_FGD
  mask[prevCompMask == 0] = cv.GC_PR_BGD
  mask[historyMask == 2] = cv.GC_FGD
  mask[historyMask == 1] = cv.GC_BGD

  allverts = np.vstack([fgVerts, bgVerts])
  cvRect = np.array([allverts.min(0), allverts.max(0) - allverts.min(0)]).flatten()

  if noPrevMask:
    if cvRect[2] == 0 or cvRect[3] == 0:
      return ImageIO(image=np.zeros_like(prevCompMask))
    mode = cv.GC_INIT_WITH_RECT
  else:
    mode = cv.GC_INIT_WITH_MASK
  cv.grabCut(img, mask, cvRect, bgdModel, fgdModel, iters, mode=mode)
  outMask = np.where((mask==2)|(mask==0), False, True)
  return ImageIO(image=outMask, grabcutDisplay=mask, display='grabcutDisplay')

def quickshift_seg(image: Image, fgVerts: FRVertices, maxDist=10., kernelSize=5,
               sigma=0.0):
  # For maxDist of 0, the input isn't changed and it takes a long time
  if maxDist == 0:
    return image
  segImg = quickshift(image, kernel_size=kernelSize, max_dist=maxDist,
                      sigma=sigma)

def k_means(image: Image, kVal=5, attempts=10):
  # Logic taken from https://docs.opencv.org/master/d1/d5c/tutorial_py_kmeans_opencv.html
  clrs = image.reshape(-1, image.depth)
  clrs = clrs.astype('float32')
  criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
  ret, lbls, imgMeans = cv.kmeans(clrs, kVal, None, criteria, attempts,
                             cv.KMEANS_RANDOM_CENTERS)
  # Now convert back into uint8, and make original image
  imgMeans = imgMeans.astype('uint8')
  lbls = lbls.reshape(image.shape[:2])
  return ImageIO(image=lbls, imgMeans=imgMeans,
                 disp=imgMeans[lbls], display='disp')

def binarize_kmeans(image: Image, fgVerts: FRVertices, imgMeans: np.ndarray,
                    removeBoundaryLbls=True,
                    discardLargesetLbl=False):
  if removeBoundaryLbls == discardLargesetLbl:
    raise FRAlgProcessorError('Exactly one of *removeBoundaryLbls* or'
                              ' *discardLargesetLbl* must be *True*.')
  # Binarize by turning all boundary labels into background and keeping forground
  out = np.zeros(image.shape, bool)
  numLbls = imgMeans.shape[0]
  if removeBoundaryLbls:
    discardLbls = np.unique(image[fgVerts.rows, fgVerts.cols])\
    # For a single point vertex, invert this rule
    if fgVerts.shape[0] == 1:
      discardLbls = np.setdiff1d(np.arange(numLbls), discardLbls)
  else:
    discardLbls = np.argsort(np.histogram(image, numLbls)[0])[[-1]]
  # if not asForeground:
  #   discardLbls = np.setdiff1d(np.arange(numLbls), discardLbls)
  keepMembership = ~np.isin(image, discardLbls)
  out[keepMembership] = True
  return ImageIO(image=out)


def region_grow(image: Image, fgVerts: FRVertices, seedThresh=10):
  if image.size == 0:
    return ImageIO(image=np.zeros(image.shape[:2], bool))
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

  # outMask = np.zeros(image.shape[0:2], bool)
  # For small enough shapes, get all boundary pixels instead of just shape vertices
  if fgVerts.connected:
    fgVerts = cornersToFullBoundary(fgVerts, 50e3)

  # Don't let region grow outside area of effect
  # img_aoe, coords = getCroppedImg(image, fgVerts, areaOfEffect, coordsAsSlices=True)
  # Offset vertices before filling
  # seeds = fgVerts - [coords[1].start, coords[0].start]
  outMask = growFunc(image, fgVerts, seedThresh)
  # outMask[coords] = newRegion
  if fgVerts.connected:
    # For connected vertices, zero out region locations outside the user defined area
    filledMask = cv.fillPoly(np.zeros_like(outMask, dtype='uint8'), [fgVerts], 1) > 0
    outMask[~filledMask] = False

  return ImageIO(image=outMask)

class FRTopLevelProcessors:
  @staticmethod
  def b_regionGrowProcessor():
    return ImageProcess.fromFunction(region_grow, name='Region Growing')

  @staticmethod
  def c_kMeansProcessor():
    proc = ImageProcess.fromFunction(k_means, name='K Means')
    proc.addFunction(binarize_kmeans)
    return proc

  @staticmethod
  def a_grabCutProcessor():
    proc = ImageProcess.fromFunction(cv_grabcut, name='Primitive Grab Cut')
    return proc

  @staticmethod
  def w_basicShapesProcessor():
    proc = ImageProcess.fromFunction(get_basic_shapes, name='Basic Shapes')
    proc.disabledStages = [['Basic Region Operations', 'Open -> Close']]
    return proc