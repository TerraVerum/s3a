from typing import Tuple, List

import cv2 as cv
import numpy as np
import pandas as pd
from scipy.ndimage import binary_fill_holes
from skimage.measure import regionprops, label, regionprops_table
from skimage.morphology import flood
from skimage.segmentation import quickshift

from s3a.generalutils import splitListAtNans, cornersToFullBoundary, \
  getCroppedImg, imgCornerVertices
from s3a.processing.processing import FRProcessIO, FRImageProcess
from s3a.structures import BlackWhiteImg, FRVertices, NChanImg, GrayImg, RgbImg, FRAlgProcessorError


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

def area_coord_regionTbl(_image: NChanImg):
  if not np.any(_image.data):
    return pd.DataFrame({'coords': [np.array([[]])], 'area': [0]})
  regionDict = regionprops_table(label(_image.data), properties=('coords', 'area'))
  _regionPropTbl = pd.DataFrame(regionDict)
  return _regionPropTbl


_historyMaskHolder = [np.array([[]], 'uint8')]
"""
0 = unspecified, 1 = background, 2 = foreground. Place inside list so reassignment
doesn't destroy object reference
"""
def format_vertices(image: NChanImg, fgVerts: FRVertices, bgVerts: FRVertices,
                    prevCompMask: BlackWhiteImg, firstRun: bool,
                    keepVertHistory=True):
  global _historyMaskHolder

  if firstRun or not keepVertHistory:
    _historyMask = np.zeros(image.shape[:2], 'uint8')
  else:
    _historyMask = _historyMaskHolder[0]

  asForeground = True
  # 0 = unspecified, 1 = background, 2 = foreground
  for fillClr, verts in enumerate([bgVerts, fgVerts], 1):
    if not verts.empty:
      cv.fillPoly(_historyMask, [verts], fillClr)

  if fgVerts.empty and bgVerts.empty:
    # Give whole image as input
    fgVerts = imgCornerVertices(image)
    fgVerts = cornersToFullBoundary(fgVerts)
    _historyMask[fgVerts.rows, fgVerts.cols] = True
  _historyMaskHolder[0] = _historyMask
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
  return FRProcessIO(image=image, fgVerts=fgVerts, bgVerts=bgVerts, asForeground=asForeground,
                 historyMask=curHistory, prevCompMask=foregroundAdjustedCompMask,
                 origCompMask=prevCompMask, boundSlices=boundSlices)

def crop_to_local_area(image: NChanImg, fgVerts: FRVertices, bgVerts: FRVertices,
                       prevCompMask: BlackWhiteImg, historyMask: GrayImg, margin_pctRoiSize=10):
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
  curHistory = historyMask[boundSlices]

  rectThickness = int(max(1, *image.shape)*0.005)
  toPlot = cv.rectangle(image.copy(), tuple(bounds[0,:]), tuple(bounds[1,:]),
                        (255,0,0), rectThickness)
  info = {'name': 'Selected Area', 'image': toPlot}
  return FRProcessIO(image=cropped, fgVerts=fgVerts, bgVerts=bgVerts, prevCompMask=croppedCompMask,
                 boundSlices=boundSlices, historyMask=curHistory, summaryInfo=info)

def apply_process_result(image: NChanImg, asForeground: bool,
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
  return FRProcessIO(image=outMask[newSlices], boundSlices=newSlices)

def return_to_full_size(image: NChanImg, origCompMask: BlackWhiteImg,
                        boundSlices: Tuple[slice]):
  out = np.zeros_like(origCompMask)
  if image.ndim > 2:
    image = image.asGrayScale()
  out[boundSlices] = image
  return FRProcessIO(image=out)

def fill_holes(image: NChanImg):
  return FRProcessIO(image=binary_fill_holes(image))

def cvt_to_uint(image: NChanImg):
  if image.ndim > 2:
    image = image.asGrayScale()
  return FRProcessIO(image = image.astype('uint8'), summaryInfo=None)

def disallow_paint_tool(_image: NChanImg, fgVerts: FRVertices, bgVerts: FRVertices):
  if len(np.vstack([fgVerts, bgVerts])) < 2:
    raise FRAlgProcessorError('This algorithm requires an enclosed area to work.'
                              ' Only one vertex was given as an input.')
  return FRProcessIO(image=_image)

def openClose():
  proc = FRImageProcess('Open -> Close')
  proc.addFunction(cvt_to_uint)
  def morphFactory(op):
    def morph(image: NChanImg, ksize=5):
      outImg = cv.morphologyEx(image, op, (ksize,))
      return FRProcessIO(image=outImg)
    return morph
  inner = FRImageProcess.fromFunction(morphFactory(cv.MORPH_OPEN), 'Opening')
  proc.addProcess(inner)
  inner = FRImageProcess.fromFunction(morphFactory(cv.MORPH_CLOSE), 'Closing')
  proc.addProcess(inner)
  return proc

def keep_largest_comp(image: NChanImg):
  regionPropTbl = area_coord_regionTbl(image)
  out = np.zeros(image.shape, bool)
  coords = regionPropTbl.coords[regionPropTbl.area.argmax()]
  if coords.size == 0:
    return FRProcessIO(image=out)
  out[coords[:,0], coords[:,1]] = True
  return FRProcessIO(image=out)

def rm_small_comps(image: NChanImg, minSzThreshold=30):
  regionPropTbl = area_coord_regionTbl(image)
  validCoords = regionPropTbl.coords[regionPropTbl.area >= minSzThreshold]
  out = np.zeros(image.shape, bool)
  if len(validCoords) == 0:
    return FRProcessIO(image=out)
  coords = np.vstack(validCoords)
  out[coords[:,0], coords[:,1]] = True
  return FRProcessIO(image=out)

def get_basic_shapes(image: NChanImg, fgVerts: FRVertices):
  # Convert indices into boolean index masks
  verts = splitListAtNans(fgVerts)
  mask = verts.toMask(image.shape[0:2])
  # Foreground is additive, bg is subtractive. If both fg and bg are present, default to keeping old value
  return FRProcessIO(image=mask)

def convert_to_squares(image: NChanImg):
  outMask = np.zeros(image.shape, dtype=bool)
  for region in regionprops(label(image)):
    outMask[region.bbox[0]:region.bbox[2],region.bbox[1]:region.bbox[3]] = True
  return FRProcessIO(image=outMask)

def basicOpsCombo():
  proc = FRImageProcess('Basic Region Operations')
  toAdd: List[FRImageProcess] = []
  for func in fill_holes, keep_largest_comp, rm_small_comps:
    toAdd.append(FRImageProcess.fromFunction(func))
  proc.addProcess(toAdd[0])
  proc.addProcess(openClose())
  proc.addProcess(toAdd[1])
  proc.addProcess(toAdd[2])
  return proc


def _grabcutResultToMask(gcResult):
  return np.where((gcResult==2)|(gcResult==0), False, True)

def cv_grabcut(image: NChanImg, fgVerts: FRVertices, bgVerts: FRVertices,
               prevCompMask: BlackWhiteImg, noPrevMask: bool,
               historyMask: GrayImg, iters=5):
  if image.size == 0:
    return FRProcessIO(image=np.zeros_like(prevCompMask))
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
      return FRProcessIO(image=np.zeros_like(prevCompMask))
    mode = cv.GC_INIT_WITH_RECT
  else:
    mode = cv.GC_INIT_WITH_MASK
  cv.grabCut(img, mask, cvRect, bgdModel, fgdModel, iters, mode=mode)
  outMask = np.where((mask==2)|(mask==0), False, True)
  return FRProcessIO(image=outMask, grabcutDisplay=mask, display='grabcutDisplay')

def quickshift_seg(image: NChanImg, fgVerts: FRVertices, maxDist=10., kernelSize=5,
               sigma=0.0):
  # For maxDist of 0, the input isn't changed and it takes a long time
  if maxDist == 0:
    return image
  segImg = quickshift(image, kernel_size=kernelSize, max_dist=maxDist,
                      sigma=sigma)

def k_means(image: NChanImg, kVal=5, attempts=10):
  # Logic taken from https://docs.opencv.org/master/d1/d5c/tutorial_py_kmeans_opencv.html
  numChannels = 1 if image.ndim < 3 else image.shape[2]
  clrs = image.reshape(-1, numChannels)
  clrs = clrs.astype('float32')
  criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
  ret, lbls, imgMeans = cv.kmeans(clrs, kVal, None, criteria, attempts,
                             cv.KMEANS_RANDOM_CENTERS)
  # Now convert back into uint8, and make original image
  imgMeans = imgMeans.astype('uint8')
  lbls = lbls.reshape(image.shape[:2])

  return FRProcessIO(image=lbls, imgMeans=imgMeans, summaryInfo={'image': imgMeans[lbls]})

def binarize_kmeans(image: NChanImg, fgVerts: FRVertices, imgMeans: np.ndarray,
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
  return FRProcessIO(image=out)


def region_grow(image: NChanImg, fgVerts: FRVertices, seedThresh=10):
  if image.size == 0:
    return FRProcessIO(image=np.zeros(image.shape[:2], bool))
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

  return FRProcessIO(image=outMask)

class FRTopLevelProcessors:
  @staticmethod
  def b_regionGrowProcessor():
    return FRImageProcess.fromFunction(region_grow, name='Region Growing')

  @staticmethod
  def c_kMeansProcessor():
    proc = FRImageProcess.fromFunction(k_means, name='K Means')
    proc.addFunction(binarize_kmeans)
    return proc

  @staticmethod
  def a_grabCutProcessor():
    proc = FRImageProcess.fromFunction(cv_grabcut, name='Primitive Grab Cut')
    return proc

  @staticmethod
  def w_basicShapesProcessor():
    proc = FRImageProcess.fromFunction(get_basic_shapes, name='Basic Shapes')
    proc.disabledStages = [['Basic Region Operations', 'Open -> Close']]
    return proc

# class FRVertsPredictor(FRParamEditorPlugin):
#   name = 'Vertices Predictor'
#
#   @classmethod
#   def __initEditorParams__(cls):
#     super().__initEditorParams__()