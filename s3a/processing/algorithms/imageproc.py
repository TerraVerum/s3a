from typing import Tuple, Union

import cv2 as cv
import numpy as np
from scipy.ndimage import binary_fill_holes
from skimage import morphology as morph, img_as_float
from skimage import segmentation as seg
from skimage.measure import regionprops, label
from skimage.morphology import flood

from s3a.generalutils import cornersToFullBoundary, getCroppedImg, imgCornerVertices, \
  showMaskDiff, MaxSizeDict, tryCvResize
from s3a.structures import BlackWhiteImg, XYVertices, ComplexXYVertices, NChanImg
from s3a.structures import GrayImg
from utilitys import fns
from utilitys.processing import *


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

# def _growSeedpoint_cv_fastButErratic(img: NChanImg, seeds: XYVertices, thresh: float):
#   if len(seeds) == 0:
#     return np.zeros(img.shape[:2], bool)
#   nChans = img.shape[2] if img.ndim > 2 else 1
#   thresh = int(np.clip(thresh, 0, 255))
#   imRCShape = np.array(img.shape[:2])
#   bwOut = np.zeros(imRCShape+2, 'uint8')
#   # Throw away seeds outside image boundaries
#   seeds = seeds[np.all(seeds < imRCShape, axis=1)]
#   seeds = np.fliplr(seeds)
#   mask = np.zeros(imRCShape+2, 'uint8')
#   for seed in seeds:
#     mask.fill(0)
#     flooded = img.copy()
#     seed = tuple(seed.flatten())
#     _, _, curOut, _ = cv.floodFill(flooded, mask, seed, 255, (thresh,)*nChans, (thresh,)*nChans,8)
#     bwOut |= curOut
#   bwOut = bwOut[1:-1,1:-1]
#   return bwOut.astype(bool)

def _cvConnComps(image: np.ndarray, returnLabels=True, areaOnly=True, removeBg=True):
  if image.dtype != 'uint8':
    image = image.astype('uint8')
  _, labels, conncomps, _ = cv.connectedComponentsWithStats(image)
  startIdx = 1 if removeBg else 0
  if areaOnly:
    conncomps = conncomps[:, cv.CC_STAT_AREA]
  if returnLabels:
    return conncomps[startIdx:], labels
  return conncomps[startIdx:]

_historyMaskHolder = [np.array([[]], 'uint8')]

"""
0 = unspecified, 1 = background, 2 = foreground. Place inside list so reassignment
doesn't destroy object reference
"""
def format_vertices(image: NChanImg, fgVerts: XYVertices, bgVerts: XYVertices,
                    prevCompMask: BlackWhiteImg, firstRun: bool,
                    useFullBoundary=True,
                    keepVertHistory=True):
  global _historyMaskHolder

  if firstRun or not keepVertHistory:
    _historyMask = np.zeros(image.shape[:2], 'uint8')
  else:
    _historyMask = _historyMaskHolder[0]

  asForeground = True
  # 0 = unspecified, 1 = background, 2 = foreground
  for fillClr, verts in enumerate([bgVerts, fgVerts], 1):
    if not verts.empty and verts.connected:
      cv.fillPoly(_historyMask, [verts], fillClr)

  if useFullBoundary:
    if not fgVerts.empty:
      fgVerts = cornersToFullBoundary(fgVerts)
    if not bgVerts.empty:
      bgVerts = cornersToFullBoundary(bgVerts)

  if fgVerts.empty and bgVerts.empty:
    # Give whole image as input, trim from edges
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
                       margin_pct=10,
                       maxSize=0
                       ):
  """
  :param reference:
    pType: list
    limits:
      - image
      - component
      - viewbox
      - roi
  :param maxSize: Maximum side length for a local portion of the image. If the local area exceeds this, it will be
    rescaled to match this size. It can be beneficial for algorithms that take a long time to run, and quality of
    segmentation can be retained. Set to <= 0 to have no maximum size
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
  ratio = 1
  curMaxDim = np.max(cropped.shape[:2])
  if 0 < maxSize < curMaxDim:
    ratio = maxSize / curMaxDim

  vertOffset = bounds.min(0)
  useVerts = [fgVerts, bgVerts]
  for ii in range(2):
    # Add additional offset
    tmp = (((useVerts[ii] - vertOffset)*ratio)).astype(int)
    useVerts[ii] = np.clip(tmp, a_min=[0,0], a_max=(bounds[1,:]-1)*ratio, dtype=int, casting='unsafe')
  fgVerts, bgVerts = useVerts

  boundSlices = slice(*bounds[:,1]), slice(*bounds[:,0])
  croppedCompMask = prevCompMask[boundSlices]
  curHistory = historyMask[boundSlices]

  rectThickness = int(max(1, *image.shape)*0.005)
  toPlot = image.copy()
  borderMask = cv.rectangle(np.zeros(image.shape[:2], dtype='uint8'), tuple(bounds[0,:]), tuple(bounds[1,:]),
                        255, rectThickness) > 0
  if image.ndim < 3:
    borderClr = 255
    borderSlice = borderMask,
  else:
    borderClr = [255] + [0 for _ in range(image.shape[2]-1)]
    borderSlice = borderMask, slice(None, None)
  borderClr = np.clip(np.array(borderClr, dtype=image.dtype), image.min(), image.max(), dtype=image.dtype)
  toPlot[borderSlice] = borderClr
  info = {'name': 'Selected Area', 'image': toPlot}
  out = ProcessIO(image=cropped, fgVerts=fgVerts, bgVerts=bgVerts, prevCompMask=croppedCompMask,
                   boundSlices=boundSlices, historyMask=curHistory, resizeRatio=ratio, summaryInfo=info)
  if ratio < 1:
    for kk in 'image', 'prevCompMask', 'historyMask':
      out[kk] = cv_resize(out[kk], ratio, interpolation='INTER_NEAREST')
  return out

def apply_process_result(image: NChanImg, asForeground: bool,
                         prevCompMask: BlackWhiteImg, origCompMask: BlackWhiteImg,
                         boundSlices: Tuple[slice,slice],
                         resizeRatio: float):
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
  if resizeRatio < 1:
    origSize = (boundSlices[0].stop - boundSlices[0].start,
                boundSlices[1].stop - boundSlices[1].start)
    # Without first converting to float, interpolation will be cliped to True/False. This causes
    # 'jagged' edges in the output
    change = cv_resize(change.astype(float), origSize[::-1], asRatio=False, interpolation='INTER_LINEAR')
    change = change > change[change > 0].mean()
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

def keep_largest_comp(image: NChanImg):
  if not np.any(image):
    return ProcessIO(image=image)
  areas, labels = _cvConnComps(image)
  # 0 is background, so skip it
  out = np.zeros_like(image, shape=image.shape[:2])
  # Offset by 1 since 0 was removed earlier
  maxAreaIdx = np.argmax(areas) + 1
  out[labels == maxAreaIdx] = True
  return ProcessIO(image=out)

def remove_small_comps(image: NChanImg, minSzThreshold=30):
  areas, labels = _cvConnComps(image, areaOnly=True)
  validLabels = np.flatnonzero(areas >= minSzThreshold) + 1
  out = np.isin(labels, validLabels)
  return ProcessIO(image=out)

def draw_vertices(image: NChanImg, fgVerts: XYVertices):
  return ProcessIO(image=ComplexXYVertices([fgVerts]).toMask(image.shape[:2], asBool=True))

def convert_to_squares(image: NChanImg):
  outMask = np.zeros(image.shape, dtype=bool)
  for region in regionprops(label(image)):
    outMask[region.bbox[0]:region.bbox[2],region.bbox[1]:region.bbox[3]] = True
  return ProcessIO(image=outMask)

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
  if historyMask.size:
    historyMask[fgVerts.rows, fgVerts.cols] = 2

  mask = np.zeros(prevCompMask.shape, dtype='uint8')
  if historyMask.shape == mask.shape:
    mask[prevCompMask == 1] = cv.GC_PR_FGD
    mask[prevCompMask == 0] = cv.GC_PR_BGD
    mask[historyMask == 2] = cv.GC_FGD
    mask[historyMask == 1] = cv.GC_BGD

  cvRect = np.array([fgVerts.min(0), fgVerts.max(0) - fgVerts.min(0)]).flatten()

  if noPrevMask or not np.any(mask):
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

def quickshift_seg(image: NChanImg, ratio=1.0, max_dist=10.0, kernel_size=5,
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
      segImg = seg.quickshift(image, ratio=ratio, kernel_size=kernel_size, max_dist=max_dist, sigma=sigma)
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
  # seeds = cornersToFullBoundary(fgVerts, 50e3)[:, ::-1]
  seeds = np.clip(fgVerts[:, ::-1], 0, np.array(labels.shape)-1)
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
  if np.issubdtype(labels.dtype, np.bool_):
    # Done for stage summary only
    labels = label(labels)
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

@fns.dynamicDocstring(inters=[attr for attr in vars(cv) if attr.startswith('INTER_')])
def cv_resize(image: np.ndarray, newSize: Union[float, tuple]=0.5, asRatio=True, interpolation='INTER_CUBIC'):
  """
  :param image: Image to resize
  :param interpolation:
    pType: list
    limits: {inters}
  :param newSize:
    pType: float
    step: 0.1
  :param asRatio:
    readonly: True
  """
  if isinstance(interpolation, str):
    interpolation = getattr(cv, interpolation)
  return tryCvResize(image, newSize, asRatio, interpolation)