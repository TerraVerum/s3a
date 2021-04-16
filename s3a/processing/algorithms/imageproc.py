import inspect
from functools import lru_cache
from typing import Tuple, List

import numpy as np
import pandas as pd
from scipy.ndimage import binary_fill_holes, maximum_filter
from skimage import morphology as morph, img_as_float
from skimage.measure import regionprops, label, regionprops_table
from skimage.morphology import flood
from skimage import segmentation as seg
from utilitys.processing import *
from utilitys import fns

from s3a.generalutils import cornersToFullBoundary, getCroppedImg, imgCornerVertices, \
  showMaskDiff, MaxSizeDict
from s3a.processing.processing import ImageProcess
from s3a.structures import BlackWhiteImg, XYVertices, ComplexXYVertices, NChanImg
from s3a.structures import GrayImg, RgbImg

import cv2 as cv


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

def _growSeedpoint_cv_fastButErratic(img: NChanImg, seeds: XYVertices, thresh: float):
  if len(seeds) == 0:
    return np.zeros(img.shape[:2], bool)
  nChans = img.shape[2] if img.ndim > 2 else 1
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

def _area_coord_regionTbl(_image: NChanImg):
  if not np.any(_image):
    return pd.DataFrame({'coords': [np.array([[]])], 'area': [0]})
  regionDict = regionprops_table(label(_image), properties=('coords', 'area'))
  return pd.DataFrame(regionDict)


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
    _historyMask[fgVerts.rows, fgVerts.cols] = 1
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
  change = bitOperation(prevCompMask, image)
  outMask[boundSlices] = change
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

  infoMask = showMaskDiff(origCompMask[boundSlices], image)

  return ProcessIO(image=out, summaryInfo={'image': infoMask, 'name': 'Finalize Region'})

def fill_holes(image: NChanImg):
  return ProcessIO(image=binary_fill_holes(image))

def disallow_paint_tool(_image: NChanImg, fgVerts: XYVertices, bgVerts: XYVertices):
  if len(np.vstack([fgVerts, bgVerts])) < 2:
    raise ValueError('This algorithm requires an enclosed area to work.'
                              ' Only one vertex was given as an input.')
  return ProcessIO(image=_image)


@fns.dynamicDocstring(morphOps=[d for d in dir(cv) if d.startswith('MORPH_')])
def morph_op(image: NChanImg, radius=1, op: str='', shape='rectangle'):
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
  :param op:
    pType: list
    limits: {morphOps}
  """
  opType = getattr(cv, op)
  if image.ndim > 2:
    image = image.mean(2)
  image = image.astype('uint8')
  ksize = [radius]
  if shape == 'rectangle':
    ksize = [ksize[0]*2+1]*2
  strel = getattr(morph, shape)(*ksize)
  outImg = cv.morphologyEx(image.copy(), opType, strel)
  return ProcessIO(image=outImg)

def opening_factory():
  return AtomicProcess(morph_op, 'Opening', op='MORPH_OPEN')

def closing_factory():
  return AtomicProcess(morph_op, 'Closing', op='MORPH_CLOSE')

def _openClose():
  proc = ImageProcess('Open -> Close')
  proc.addFunction(morph_op, name='Open', needsWrap=True, op='MORPH_OPEN')
  proc.addFunction(morph_op, name='Close', needsWrap=True, op='MORPH_CLOSE')
  return proc

def keep_largest_comp(image: NChanImg):
  regionPropTbl = _area_coord_regionTbl(image)
  out = np.zeros(image.shape, bool)
  coords = regionPropTbl.coords[regionPropTbl.area.argmax()]
  if coords.size == 0:
    return ProcessIO(image=out)
  out[coords[:,0], coords[:,1]] = True
  return ProcessIO(image=out)

def rm_small_comps(image: NChanImg, minSzThreshold=30):
  regionPropTbl = _area_coord_regionTbl(image)
  validCoords = regionPropTbl.coords[regionPropTbl.area >= minSzThreshold]
  out = np.zeros(image.shape, bool)
  if len(validCoords) == 0:
    return ProcessIO(image=out)
  coords = np.vstack(validCoords)
  out[coords[:,0], coords[:,1]] = True
  return ProcessIO(image=out)

def _draw_vertices_old(image: NChanImg, fgVerts: XYVertices, penSize=1, penShape='circle'):
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
  return ProcessIO(image=out > 0)

def draw_vertices(image: NChanImg, fgVerts: XYVertices):
  return ProcessIO(image=ComplexXYVertices([fgVerts]).toMask(image.shape[:2], asBool=True))

def convert_to_squares(image: NChanImg):
  outMask = np.zeros(image.shape, dtype=bool)
  for region in regionprops(label(image)):
    outMask[region.bbox[0]:region.bbox[2],region.bbox[1]:region.bbox[3]] = True
  return ProcessIO(image=outMask)

def _basicOpsCombo():
  proc = ImageProcess('Basic Region Operations')
  toAdd: List[ProcessStage] = []
  for func in fill_holes, keep_largest_comp, rm_small_comps:
    nextProc = AtomicProcess(func)
    nextProc.allowDisable = False
    toAdd.append(nextProc)
  proc.addProcess(toAdd[0])
  proc.addProcess(_openClose())
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
  return ProcessIO(labels=outMask)

_qsInCache = MaxSizeDict(maxsize=5)
_qsOutCache = MaxSizeDict(maxsize=5)

def quickshift_seg(image: NChanImg, max_dist=10., kernel_size=5,
                   sigma=0.0):
  global _qsInCache, _qsOutCache
  # For max_dist of 0, the input isn't changed and it takes a long time
  key = (max_dist, kernel_size, sigma)
  if max_dist == 0:
    # Make sure output is still 1-channel
    segImg = image.mean(2).astype(int) if image.ndim > 2 else image
  else:
    if image.ndim < 3:
      image = np.tile(image[:,:,None], (1,1,3))
    # First check if this image was the same as last time
    if np.array_equal(_qsInCache.get(key, None), image) and key in _qsOutCache:
      segImg =  _qsOutCache[key]
    else:
      segImg = seg.quickshift(image, kernel_size=kernel_size, max_dist=max_dist, sigma=sigma)
  _qsInCache[key] = image
  _qsOutCache[key] = segImg.copy()
  return ProcessIO(labels=segImg)
quickshift_seg.__doc__ = seg.quickshift.__doc__

# Taken from example page: https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_morphsnakes.html
def morph_acwe(image: NChanImg, initialCheckerSize=6, iters=35, smoothing=3):
  image = img_as_float(image)
  if image.ndim > 2:
    image = image.mean(2)
  initLs = seg.checkerboard_level_set(image.shape, initialCheckerSize)
  outLevels = seg.morphological_chan_vese(image, iters, init_level_set=initLs, smoothing=smoothing)
  return ProcessIO(labels=outLevels)

def k_means_segmentation(image: NChanImg, kVal=5, attempts=10):
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

  return ProcessIO(labels=lbls, means=imgMeans)

def _labelBoundaries_cv(labels: np.ndarray, thickness: int):
  """Code stolen and reinterpreted for cv from skimage.segmentation.boundaries"""
  if thickness % 2 == 0:
    thickness += 1
  thickness = max(thickness, 3)
  if labels.dtype not in [np.uint8, np.uint16, np.int16, np.float16, np.float32]:
    labels = labels.astype(np.uint16)
  strel = cv.getStructuringElement(cv.MORPH_RECT, (thickness, thickness))
  return cv.morphologyEx(labels, cv.MORPH_DILATE, strel) \
               != cv.morphologyEx(labels, cv.MORPH_ERODE, strel)

def binarize_labels(image: NChanImg, labels: BlackWhiteImg, fgVerts: XYVertices,
                    historyMask: BlackWhiteImg, touchingRoiOnly=True, useMeanColor=True,
                    lineThickness=2):
  """
  For a given binary image input, only keeps connected components that are directly in
  contact with at least one of the specified vertices. In essence, this function can make
  a wide variety of operations behave similarly to region growing.

  :param touchingRoiOnly: Whether to only keep labeled regions that are in contact
    with the current ROI
  :param useMeanColor: Whether to color the summary info image with mean values or
    (if *False*) just draw the boundaries around each label.
  :param lineThickness:
    helpText: How thick to draw label boundary and ROI vertices lines
  """
  if labels.ndim > 2:
    raise ValueError('Cannot handle multichannel labels.\n'
                     f'(labelss.shape={labels.shape})')
  seeds = cornersToFullBoundary(fgVerts, 50e3)[:, ::-1]
  seeds = np.clip(seeds, 0, np.array(labels.shape)-1)
  if image.ndim < 3:
    image = image[...,None]
  if touchingRoiOnly:
    out = np.zeros_like(labels, dtype=bool)
    for seed in seeds:
      out |= flood(labels, tuple(seed),)
  else:
    keepColors = labels[seeds[:,0], seeds[:,1]]
    out = np.isin(labels, keepColors)
  # Zero out negative regions from previous runs
  if historyMask.shape == out.shape:
    out[historyMask == 1] = False
  nChans = image.shape[2]
  if useMeanColor:
    summaryImg = np.zeros_like(image)
    # Offset by 1 to avoid missing 0-labels
    for lbl in regionprops(labels+1):
      coords = lbl.coords
      intensity = image[coords[:,0], coords[:,1],...].mean(0)
      summaryImg[coords[:,0], coords[:,1], :] = intensity
  else:
    if np.issubdtype(labels.dtype, np.bool_):
      labels = labels.astype('uint8')
    boundaries = _labelBoundaries_cv(labels, lineThickness)
    summaryImg = image.copy()
    summaryImg[boundaries,...] = [255 for _ in range(nChans)]
  color = (255,) + tuple(0 for _ in range(1, nChans))
  cv.drawContours(summaryImg, [fgVerts], -1, color, lineThickness)
  return ProcessIO(image=out, summaryInfo={'image': summaryImg})

def region_grow_segmentation(image: NChanImg, fgVerts: XYVertices, seedThresh=10):
  if image.size == 0:
    return ProcessIO(image=np.zeros(image.shape[:2], bool))
  if np.all(fgVerts == fgVerts[0, :]):
    # Remove unnecessary redundant seedpoints
    fgVerts = fgVerts[[0], :]
  # outMask = np.zeros(image.shape[0:2], bool)
  # For small enough shapes, get all boundary pixels instead of just shape vertices
  if fgVerts.connected:
    fgVerts = cornersToFullBoundary(fgVerts, 50e3)

  # Don't let region grow outside area of effect
  # img_aoe, coords = getCroppedImg(image, fgVerts, areaOfEffect, coordsAsSlices=True)
  # Offset vertices before filling
  # seeds = fgVerts - [coords[1].start, coords[0].start]
  outMask = growSeedpoint(image, fgVerts, seedThresh)

  return ProcessIO(image=outMask)

def slic_segmentation(image, n_segments=100, compactness=10.0, sigma=0, min_size_factor=0.5, max_size_factor=3):
  return ProcessIO(labels=seg.slic(**locals(), start_label=1))
slic_segmentation.__doc__ = seg.slic.__doc__