from functools import lru_cache
from typing import Tuple, List

import cv2 as cv
import numpy as np
import pandas as pd
from scipy.ndimage import binary_fill_holes, maximum_filter
from skimage import morphology as morph
from skimage.measure import regionprops, label, regionprops_table
from skimage.morphology import flood
from skimage.segmentation import quickshift
from utilitys.processing import *
from utilitys import fns

from s3a.constants import REQD_TBL_FIELDS as RTF, PRJ_ENUMS
from s3a.generalutils import cornersToFullBoundary, getCroppedImg, imgCornerVertices
from s3a.processing.processing import ImageProcess, GlobalPredictionProcess
from s3a.structures import BlackWhiteImg, XYVertices, ComplexXYVertices, NChanImg
from s3a.structures import GrayImg, RgbImg


def growSeedpoint(img: NChanImg, seeds: XYVertices, thresh: float) -> BlackWhiteImg:
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

def growSeedpoint_cv_fastButErratic(img: NChanImg, seeds: XYVertices, thresh: float):
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
  if not np.any(_image):
    return pd.DataFrame({'coords': [np.array([[]])], 'area': [0]})
  regionDict = regionprops_table(label(_image), properties=('coords', 'area'))
  _regionPropTbl = pd.DataFrame(regionDict)
  return _regionPropTbl


_historyMaskHolder = [np.array([[]], 'uint8')]

"""
0 = unspecified, 1 = background, 2 = foreground. Place inside list so reassignment
doesn't destroy object reference
"""
def format_vertices(image: NChanImg, fgVerts: XYVertices, bgVerts: XYVertices,
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
    bgVerts = XYVertices()

  if asForeground:
    foregroundAdjustedCompMask = prevCompMask.copy()
  else:
    foregroundAdjustedCompMask = ~prevCompMask

  # Default to bound slices that encompass the whole image
  bounds = np.array([[0, 0], image.shape[:2][::-1]])
  boundSlices = slice(*bounds[:,1]), slice(*bounds[:,0])
  return ProcessIO(image=image, summaryInfo=None, fgVerts=fgVerts, bgVerts=bgVerts, asForeground=asForeground,
                   historyMask=curHistory, prevCompMask=foregroundAdjustedCompMask,
                   origCompMask=prevCompMask, boundSlices=boundSlices)

def crop_to_local_area(image: NChanImg,
                       fgVerts: XYVertices,
                       bgVerts: XYVertices,
                       prevCompMask: BlackWhiteImg,
                       prevCompVerts: ComplexXYVertices,
                       viewbox: XYVertices,
                       historyMask: GrayImg,
                       reference='viewbox',
                       margin_pct=10
                       ):
  """
  :param reference:
    pType: list
    limits:
      - image
      - component
      - viewbox
      - roi
  """
  roiVerts = np.vstack([fgVerts, bgVerts])
  compVerts = np.vstack([prevCompVerts.stack(), roiVerts])
  if reference == 'image':
    allVerts = np.array([[0, 0], image.shape[:2]])
  elif reference == 'roi' and len(roiVerts) > 1:
    allVerts = roiVerts
  elif reference == 'component' and len(compVerts) > 1:
    allVerts = compVerts
  else:
    # viewbox or badly sized previous region/roi
    allVerts = np.vstack([viewbox, roiVerts])
  # Lots of points, use their bounded area
  try:
    vertArea_rowCol = (allVerts.max(0)-allVerts.min(0))[::-1]
  except ValueError:
    # 0-sized
    vertArea_rowCol = 0
  margin = int(round(max(vertArea_rowCol) * (margin_pct / 100)))
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
  return ProcessIO(image=cropped, fgVerts=fgVerts, bgVerts=bgVerts, prevCompMask=croppedCompMask,
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
  return ProcessIO(image=outMask[newSlices], boundSlices=newSlices)

def return_to_full_size(image: NChanImg, origCompMask: BlackWhiteImg,
                        boundSlices: Tuple[slice]):
  out = np.zeros_like(origCompMask)
  if image.ndim > 2:
    image = image.mean(2).astype(int)
  out[boundSlices] = image
  return ProcessIO(image=out)

def fill_holes(image: NChanImg):
  return ProcessIO(image=binary_fill_holes(image))

def disallow_paint_tool(_image: NChanImg, fgVerts: XYVertices, bgVerts: XYVertices):
  if len(np.vstack([fgVerts, bgVerts])) < 2:
    raise ValueError('This algorithm requires an enclosed area to work.'
                              ' Only one vertex was given as an input.')
  return ProcessIO(image=_image)

def openClose():
  proc = ImageProcess('Open -> Close')
  def perform_op(image: NChanImg, radius=1, shape='rectangle'):
    """
    :param radius: Radius of the structuring element. Note that the total side length
      of the structuring element will be (2*radius)+1.
    :param shape:
      helpText: Structuring element shape
      pType: list
      limits:
        - rectangle
        - disk
        - diamond
    """
    if image.ndim > 2:
      image = image.mean(2)
    image = image.astype('uint8')
    ksize = [radius]
    if shape == 'rectangle':
      ksize = [ksize[0]*2+1]*2
    strel = getattr(morph, shape)(*ksize)
    outImg = cv.morphologyEx(image.copy(), cv.MORPH_OPEN, strel)
    outImg = cv.morphologyEx(outImg, cv.MORPH_CLOSE, strel)
    return outImg
  proc.addFunction(perform_op, needsWrap=True)
  return proc

def keep_largest_comp(image: NChanImg):
  regionPropTbl = area_coord_regionTbl(image)
  out = np.zeros(image.shape, bool)
  coords = regionPropTbl.coords[regionPropTbl.area.argmax()]
  if coords.size == 0:
    return ProcessIO(image=out)
  out[coords[:,0], coords[:,1]] = True
  return ProcessIO(image=out)

def rm_small_comps(image: NChanImg, minSzThreshold=30):
  regionPropTbl = area_coord_regionTbl(image)
  validCoords = regionPropTbl.coords[regionPropTbl.area >= minSzThreshold]
  out = np.zeros(image.shape, bool)
  if len(validCoords) == 0:
    return ProcessIO(image=out)
  coords = np.vstack(validCoords)
  out[coords[:,0], coords[:,1]] = True
  return ProcessIO(image=out)

def basic_shapes(image: NChanImg, fgVerts: XYVertices, penSize=1, penShape='circle'):
  """
  Draws basic shapes with minimal pre- or post-processing.

  :param penSize: Size of the drawing pen
  :param penShape:
    helpText: Shape of the drawing pen
    pType: list
    limits:
      - circle
      - rectangle
  """
  out = np.zeros(image.shape[:2], dtype='uint8')
  drawFns = {
    'circle': lambda pt: cv.circle(out, tuple(pt), penSize//2, 1, -1),
    'rectangle': lambda pt: cv.rectangle(out, tuple(pt-penSize//2), tuple(pt+penSize//2), 1, -1)
  }
  try:
    drawFn = drawFns[penShape]
  except KeyError:
    raise ValueError(f"Can't understand shape {penShape}. Must be one of:\n"
                              f"{','.join(drawFns)}")
  if len(fgVerts) > 1 and penSize > 1:
    ComplexXYVertices([fgVerts]).toMask(out, 1, False, warnIfTooSmall=False)
  else:
    for vert in fgVerts:
      drawFn(vert)
  return out > 0

def convert_to_squares(image: NChanImg):
  outMask = np.zeros(image.shape, dtype=bool)
  for region in regionprops(label(image)):
    outMask[region.bbox[0]:region.bbox[2],region.bbox[1]:region.bbox[3]] = True
  return ProcessIO(image=outMask)

def basicOpsCombo():
  proc = ImageProcess('Basic Region Operations')
  toAdd: List[ImageProcess] = []
  for func in fill_holes, keep_largest_comp, rm_small_comps:
    toAdd.append(ImageProcess.fromFunction(func))
  proc.addProcess(toAdd[0])
  proc.addProcess(openClose())
  proc.addProcess(toAdd[1])
  proc.addProcess(toAdd[2])
  return proc


def _grabcutResultToMask(gcResult):
  return np.where((gcResult==2)|(gcResult==0), False, True)

def cv_grabcut(image: NChanImg, prevCompMask: BlackWhiteImg, fgVerts: XYVertices,
               noPrevMask: bool, historyMask: GrayImg, iters=5):
  if image.size == 0:
    return ProcessIO(image=np.zeros_like(prevCompMask))
  img = cv.cvtColor(image, cv.COLOR_RGB2BGR)
  # Turn foreground into x-y-width-height
  bgdModel = np.zeros((1,65),np.float64)
  fgdModel = np.zeros((1,65),np.float64)
  historyMask = historyMask.copy()
  historyMask[fgVerts.rows, fgVerts.cols] = 2

  mask = np.zeros(prevCompMask.shape, dtype='uint8')
  mask[prevCompMask == 1] = cv.GC_PR_FGD
  mask[prevCompMask == 0] = cv.GC_PR_BGD
  mask[historyMask == 2] = cv.GC_FGD
  mask[historyMask == 1] = cv.GC_BGD

  cvRect = np.array([fgVerts.min(0), fgVerts.max(0) - fgVerts.min(0)]).flatten()

  if noPrevMask:
    if cvRect[2] == 0 or cvRect[3] == 0:
      return ProcessIO(image=np.zeros_like(prevCompMask))
    mode = cv.GC_INIT_WITH_RECT
  else:
    mode = cv.GC_INIT_WITH_MASK
  cv.grabCut(img, mask, cvRect, bgdModel, fgdModel, iters, mode=mode)
  outMask = np.where((mask==2)|(mask==0), False, True)
  return ProcessIO(image=outMask, summaryInfo={'image': mask})

def quickshift_seg(image: NChanImg, fgVerts: XYVertices, maxDist=10., kernelSize=5,
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

  return ProcessIO(image=lbls, imgMeans=imgMeans, summaryInfo={'image': imgMeans[lbls]})

def keep_regions_touching_roi(image: BlackWhiteImg, fgVerts: XYVertices):
  """
  For a given binary image input, only keeps connected components that are directly in
  contact with at least one of the specified vertices. In essence, this function can make
  a wide variety of operations behave similarly to region growing.
  """
  if image.ndim > 2:
    raise ValueError('Cannot handle multichannel images.\n'
                              f'(image.shape={image.shape})')
  out = np.zeros_like(image)
  seeds = fgVerts[:,::-1]
  seeds = np.clip(seeds, 0, np.array(image.shape)-1)
  for seed in seeds:
    out |= flood(image, tuple(seed))
  return out

def binarize_kmeans(image: NChanImg, fgVerts: XYVertices, imgMeans: np.ndarray,
                    decisionMetric='Remove Boundary Labels'):
  """

  :param image:
  :param fgVerts:
  :param imgMeans:
  :param decisionMetric:
    helpText: "How to binarize the result of a k-means process. If `Remove Boundary Labels`,
      the binary foreground is whatever *didn't* intersect the ROI vertices for a polygon
      and whatever *did* intersect for a point. If `Discard Largest Label`, the largest
      label by area is removed."
    pType: list
    limits:
      - Discard Largest Label
      - Remove Boundary Labels
  """
  # Binarize by turning all boundary labels into background and keeping forground
  out = np.zeros(image.shape, bool)
  numLbls = imgMeans.shape[0]
  if decisionMetric == 'Remove Boundary Labels':
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
  return ProcessIO(image=out)

def region_growing(image: NChanImg, fgVerts: XYVertices, seedThresh=10):
  if image.size == 0:
    return ProcessIO(image=np.zeros(image.shape[:2], bool))
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

  return ProcessIO(image=outMask)

class TopLevelImageProcessors:
  @staticmethod
  def b_regionGrowProcessor():
    return ImageProcess.fromFunction(region_growing)

  @staticmethod
  def c_kMeansProcessor():
    proc = ImageProcess.fromFunction(k_means)
    proc.addFunction(binarize_kmeans)
    # Add as process so it can be disabled
    proc.addProcess(ImageProcess.fromFunction(keep_regions_touching_roi, needsWrap=True))
    return proc

  @staticmethod
  def a_grabCutProcessor():
    proc = ImageProcess.fromFunction(cv_grabcut, name='Primitive Grab Cut')
    proc.addProcess(ImageProcess.fromFunction(keep_regions_touching_roi, needsWrap=True))
    return proc

  @staticmethod
  def w_basicShapesProcessor():
    def basic_shapes(image: np.ndarray, fgVerts: XYVertices):
      return ProcessIO(image=ComplexXYVertices([fgVerts]).toMask(image.shape[:2], asBool=True))
    proc = ImageProcess.fromFunction(basic_shapes, name='Basic Shapes')
    proc.disabledStages = [['Basic Region Operations', 'Open -> Close']]
    return proc


# -----
# GLOBAL PROCESSING
# -----
def get_component_images(image: np.ndarray, components: pd.DataFrame):
  """
  From a main image and dataframe of components, adds an 'img' column to `components` which holds the
  subregion within the image each component occupies.
  """
  imgs = [getCroppedImg(image, verts.stack(), 0) for verts in components[RTF.VERTICES]]
  return ProcessIO(subimages=imgs)

def dispatchedTemplateMatcher(func):
  inputSpec = ProcessIO.fromFunction(func, ignoreKeys=['template'])
  inputSpec['components'] = inputSpec.FROM_PREV_IO
  def dispatcher(image: np.ndarray, components: pd.DataFrame, **kwargs):
    out = ProcessIO()
    allComps = []
    for ii, comp in components.iterrows():
      verts = comp[RTF.VERTICES].stack()
      template = getCroppedImg(image, verts, 0, returnSlices=False)
      result = func(image=image, template=template, **kwargs)
      if isinstance(result, ProcessIO):
        pts = result['matchPts']
        out.update(**result)
        out.pop('components', None)
      else:
        pts = result
      allComps.append(pts_to_components(pts, comp))
    outComps = pd.concat(allComps, ignore_index=True)
    out['components'] = outComps
    out['deleteOrig'] = True
    return out
  dispatcher.__doc__ = func.__doc__
  proc = GlobalPredictionProcess(fns.nameFormatter(func.__name__))
  proc.addFunction(dispatcher)
  proc.stages[0].input = inputSpec
  return proc

def dispatchedFocusedProcessor(func):
  inputSpec = ProcessIO.fromFunction(func)
  inputSpec['components'] = inputSpec.FROM_PREV_IO
  def dispatcher(image: np.ndarray, components: pd.DataFrame, **kwargs):
    out = ProcessIO()
    allComps = []
    for ii, comp in components.iterrows():
      verts = comp[RTF.VERTICES].stack()
      focusedImage = getCroppedImg(image, verts, 0, returnSlices=False)
      result = func(image=focusedImage, **kwargs)
      if isinstance(result, ProcessIO):
        mask = result['image']
        out.update(**result)
        out.pop('components', None)
      else:
        mask = result
      newComp = comp.copy()
      if mask is not None:
        newComp[RTF.VERTICES] = ComplexXYVertices.fromBwMask(mask)
      allComps.append(fns.serAsFrame(newComp))
    outComps = pd.concat(allComps)
    out['components'] = outComps
    out['addType'] = PRJ_ENUMS.COMP_ADD_AS_MERGE
    return out
  dispatcher.__doc__ = func.__doc__
  proc = GlobalPredictionProcess(fns.nameFormatter(func.__name__))
  proc.addFunction(dispatcher)
  proc.stages[0].input = inputSpec
  return proc


@fns.dynamicDocstring(metricTypes=[d for d in dir(cv) if d.startswith('TM')])
def cv_template_match(template: np.ndarray, image: np.ndarray, viewbox: np.ndarray,
                      threshold=0.8, metric='TM_CCOEFF_NORMED', area='viewbox'):
  """
  Performs template matching using default opencv functions
  :param template: Template image
  :param image: Main image
  :param threshold:
    helpText: Cutoff point to consider a matched template
    limits: [0, 1]
    step: 0.1
  :param metric:
    helpText: Template maching metric
    pType: list
    limits: {metricTypes}
  :param area:
    helpText: Where to apply the new components
    pType: list
    limits:
      - image
      - viewbox
  """
  if area == 'viewbox':
    image, coords = getCroppedImg(image, viewbox, 0)
  else:
    coords = np.array([[0,0]])
  if image.ndim < 3:
    grayImg = image
  else:
    grayImg = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
  if template.ndim > 2:
    template = cv.cvtColor(template, cv.COLOR_RGB2GRAY)

  metric = getattr(cv, metric)
  res = cv.matchTemplate(grayImg, template, metric)
  maxFilter = maximum_filter(res, template.shape[:2])
  # Non-max suppression to remove close-together peaks
  res[maxFilter > res] = 0
  loc = np.nonzero(res >= threshold)
  scores = res[loc]
  matchPts = np.c_[loc[::-1]] + coords[[0]]
  return ProcessIO(matchPts=matchPts, scores=scores, matchImg=maxFilter)


def pts_to_components(matchPts: np.ndarray, component: pd.Series):
  numOutComps = len(matchPts)
  # Explicit copy otherwise all rows point to the same component
  outComps = pd.concat([fns.serAsFrame(component)]*numOutComps, ignore_index=True).copy()
  origOffset = component[RTF.VERTICES].stack().min(0)
  allNewverts = []
  for ii, pt in zip(outComps.index, matchPts):
    newVerts = []
    for verts in outComps.at[ii, RTF.VERTICES]:
      newVerts.append(verts-origOffset+pt)
    allNewverts.append(ComplexXYVertices(newVerts))
  outComps[RTF.VERTICES] = allNewverts
  return outComps


@lru_cache()
def _modelFromFile(model: str):
  fname = MODEL_DIR/model
  return tf.load(fname)

def nn_model_prediction(image: NChanImg, model=''):
  """

  :param image:
  :param fgVerts:
  :param model:
    pType: list
    limits:
      - ''
      - Tool 1
      - Tool 2
      - Tool 3
  """
  if not model:
    return None
  nnModel = _modelFromFile(model)

  return np.random.random((image.shape[:2])) > 0

TOP_GLOBAL_PROCESSOR_FUNCS = [
  lambda: dispatchedTemplateMatcher(cv_template_match),
  lambda: dispatchedFocusedProcessor(nn_model_prediction)
]