import cv2 as cv
import numpy as np
import pandas as pd
from skimage.measure import regionprops, label, regionprops_table
from skimage.morphology import closing, dilation, opening
from skimage.morphology import disk
from skimage.segmentation import quickshift, flood

from cdef.generalutils import getClippedBbox
from cdef.structures import NChanImg, FRVertices

from cdef.structures.typeoverloads import RgbImg, GrayImg, NChanImg, BlackWhiteImg, \
  TwoDArr
from cdef.structures.vertices import FRVertices, FRComplexVertices
from imageprocessing.common import Image


def getBwComps(img: RgbImg, minSz=30) -> BlackWhiteImg:
  bwOut = bwBgMask(img)
  bwOut = opening(bwOut, np.ones((10,10), dtype=bool))
  bwOut = closing(bwOut, np.ones((7,7), dtype=bool))
  return rmSmallComps(bwOut, minSz)


def colorLabelsWithMean(labelImg: GrayImg, refImg: NChanImg) -> RgbImg:
  outImg = np.empty(refImg.shape)
  labels = np.unique(labelImg)
  for curLabel in labels:
    curmask = labelImg == curLabel
    outImg[curmask,:] = refImg[curmask,:].reshape(-1,3).mean(0)
  return outImg

def bwBgMask(img: RgbImg) -> BlackWhiteImg:
  if img.dtype != 'uint8':
    img = img.astype('uint8')

  if img.dtype != 'uint8':
    img = img.astype('uint8')
  chans = cv.split(img)
  mask = np.bitwise_and(1.25*chans[0] < chans[1],
                        1.25*chans[2] < chans[1])
  mask = np.invert(closing(mask, disk(5)))
  return mask


def getVertsFromBwComps(bwmask: BlackWhiteImg, simplifyVerts=True, externOnly=False) -> FRComplexVertices:
  approxMethod = cv.CHAIN_APPROX_SIMPLE
  if not simplifyVerts:
    approxMethod = cv.CHAIN_APPROX_NONE
  retrMethod = cv.RETR_CCOMP
  if externOnly:
    retrMethod = cv.RETR_EXTERNAL
  # Contours are on the inside of components, so dilate first to make sure they are on the
  # outside
  #bwmask = dilation(bwmask, np.ones((3,3), dtype=bool))
  contours, hierarchy = cv.findContours(bwmask.astype('uint8'), retrMethod, approxMethod)
  compVertices = []
  for contour in contours:
    compVertices.append(FRVertices(contour[:,0,:]))
  if hierarchy is None:
    hierarchy = np.ones((0,1,4), int)*-1
  return FRComplexVertices(compVertices, hierarchy[:,0,:])

def segmentComp(compImg: RgbImg, maxDist: np.float, kernSz=10) -> RgbImg:
  # For maxDist of 0, the input isn't changed and it takes a long time
  if maxDist < 1:
    return compImg
  segImg = quickshift(compImg, kernel_size=kernSz, max_dist=maxDist)
  # Color segmented image with mean values
  return colorLabelsWithMean(segImg, compImg)

def rmSmallComps(bwMask: BlackWhiteImg, minSz: int=0) -> BlackWhiteImg:
  """
  Removes components smaller than :param:`minSz` from the input mask.

  :param bwMask: Input mask
  :param minSz: Minimum individual component size allowed. That is, regions smaller
        than :param:`minSz` connected pixels will be removed from the output.
  :return: output mask without small components
  """
  # make sure we don't modify the input data
  bwMask = bwMask.copy()
  if minSz < 1:
    return bwMask
  regions = regionprops(label(bwMask))
  for region in regions:
    if region.area < minSz:
      coords = region.coords
      bwMask[coords[:, 0], coords[:, 1]] = False
  return bwMask

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

def growSeedpoint_cv_fastButErratic(img: NChanImg, seeds: TwoDArr, thresh: float, minSz: int=0):
  if thresh > 255:
    thresh = 255
  if thresh < 0:
    thresh = 0
  imRCShape = img.shape[:2]
  bwOut = np.zeros((imRCShape[0]+2, imRCShape[1]+2), np.uint8)
  flooded = img.copy()
  # Throw away seeds outside image boundaries
  seeds = seeds[np.all(seeds < imRCShape, axis=1)]
  seeds = np.fliplr(seeds)
  for seed in seeds:
    mask = np.zeros((imRCShape[0]+2, imRCShape[1]+2), np.uint8)
    valAtSeed = img[seed[1],seed[0],:]
    seed = tuple(seed.flatten())
    cv.floodFill(flooded, mask, seed, 1, (thresh,)*3, (thresh,)*3,
                 8 | cv.FLOODFILL_MASK_ONLY)
    bwOut |= mask
  bwOut = bwOut[1:-1,1:-1]
  bwOut = closing(bwOut, np.ones((3,3), dtype=bool))
  # Remove components smaller than minSz
  return rmSmallComps(bwOut, minSz)

def growSeedpoint_custom_slowButWorks(img: np.array, seeds: np.array, thresh: float, minSz: int=0) -> \
    np.array:
  """
  Starting from *seed*, fills each connected pixel if the difference between
  neighboring pixels and the current component is less than *thresh*.
  Places one 'on' pixel in each seed location, rerunning the algorithm for each
  new seed. **Note**: only one label is  generated for each call to this
  function. I.e. if multiple seeds are specified, they are assumed to belong
  to the same label.

  :param img:    MxNxChan
    Input image

  :param seeds:  Mx2 np array
    Contains locations in output mask that are 'on'
    at the start of the algorithm. Pixels connected to these are
    iteratively added to the components if their intensities are
    close enough to each seed component.

  :param thresh: float
    Threshold between component and neighbors. If neighbor pixels
    are below this value, they are added to the seed component.

  :param minSz: Minimum individual component size allowed. That is, regions smaller
    than :param:`minSz` connected pixels will be removed from the output.
  """
  nChans = img.shape[2] if img.ndim > 2 else 1
  if nChans < 1:
    img = img[:,:,None]
  imRCShape = np.array(img.shape[0:2])[None,:]
  finalBwOut = np.zeros(img.shape[0:2], dtype=bool)
  nChans = img.shape[2] if len(img.shape) > 2 else 1
  # Computationally cheaper to compare square of thresh instead of using
  # euclidean distance
  thresh = thresh**2
  # Throw away seeds outside image boundaries
  seeds = seeds[np.all(seeds < imRCShape, axis=1)]
  for seed in seeds:
    curBwOut = np.zeros(img.shape[0:2], dtype=bool)
    curBwOut[seed[0], seed[1]] = True
    changed = True
    while changed:
      neighbors = dilation(curBwOut, np.ones((3,3)))
      neighbors[curBwOut] = False
      compMean = img[curBwOut,:].reshape(-1,nChans).mean(0)
      # Add neighbor pixels close to this mean value
      valsAtNeighbors = img[neighbors,:].reshape(-1,nChans)
      diffFromMean = ((valsAtNeighbors-compMean)**2).sum(1)
      # Invalidate idxs not passing the threshold
      nbrIdxs = np.nonzero(neighbors)
      invalidIdxs = []
      for idxList in nbrIdxs:
        idxList = idxList[diffFromMean >= thresh]
        invalidIdxs.append(idxList)
      neighbors[invalidIdxs[0], invalidIdxs[1]] = False
      newBwOut = curBwOut | neighbors
      changed = np.any(newBwOut != curBwOut)
      curBwOut = newBwOut
    finalBwOut |= curBwOut



  finalBwOut = closing(finalBwOut, np.ones((3,3), dtype=bool))
  # Remove components smaller than minSz
  return rmSmallComps(finalBwOut, minSz)

def growBoundarySeeds(img: NChanImg, seedThresh: float, minSz: int,
                      segThresh: float=0., useAllBounds=False) -> BlackWhiteImg:
  """
  Treats all border pixels of :param:`img` as seedpoints for growing. Once these are
  grown, all regions are united, and the inverse area is returned. This has the effect
  of thinning the boundary around a component to only return the component itself.

  :param img: See :func:`growSeedpoint` *img* param.

  :param seedThresh: See :func:`growSeedpoint` *thresh* param.

  :param minSz: See :func:`growSeedpoint` *minSz* param.

  :param useAllBounds: Whether the function should consider every single boundary pixel
    as a seed. This can lead to very poor performance for large components.

  :return: Mask without any regoions formed from border pixels.
  """
  img = segmentComp(img, segThresh)
  nrows, ncols, *_ = img.shape
  maxRow, maxCol = nrows-1, ncols-1
  if useAllBounds:
    seedRows = np.concatenate([np.repeat(0, maxCol+1), np.repeat(maxRow, maxCol+1),
                np.arange(nrows, dtype=int), np.arange(nrows, dtype=int)])
    seedCols = np.concatenate([np.arange(ncols, dtype=int), np.arange(ncols, dtype=int),
                np.repeat(0, maxRow+1), np.repeat(maxCol, maxRow+1)])
    seeds = np.hstack((seedRows[:,None], seedCols[:,None]))
  else:
    # Just use image corners
    seeds = np.array([[0,0], [0, maxCol], [maxRow, 0], [maxRow, maxCol]])
  seeds = FRVertices(seeds)
  # Since these are background components, we don't want to remove small components until
  # after inverting the mask
  bwBgSeedGrow = growSeedpoint(img, seeds, seedThresh)
  bwOut = ~bwBgSeedGrow

  # Remove sparsely connected regions
  bwOut = opening(bwOut)

  # For now, just keep the largest component
  regions = regionprops(label(bwOut))
  if len(regions) > 0:
    biggestRegion = regions[0]
    for region in regions[1:]:
      if region.area > biggestRegion.area:
        biggestRegion = region
    bwOut[:,:] = False
    bwOut[biggestRegion.coords[:,0], biggestRegion.coords[:,1]] = True

  return rmSmallComps(bwOut, minSz)

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
  if np.prod(fillShape) > sizeLimit:
    return cornerVerts

  tmpImgToFill = np.zeros(fillShape, dtype='uint8')

  filledMask = cv.fillPoly(tmpImgToFill, [cornerVerts], 1) > 0
  return getVertsFromBwComps(filledMask, simplifyVerts=False).filledVerts().stack()


def getCroppedImg(image: NChanImg, verts: np.ndarray, margin: int, otherBbox: np.ndarray=None) -> (np.ndarray, np.ndarray):
  verts = np.vstack(verts)
  img_np = image
  compCoords = np.vstack([verts.min(0), verts.max(0)])
  if otherBbox is not None:
    for dim in range(2):
      for ii, cmpFunc in zip(range(2), [min, max]):
        compCoords[ii,dim] = cmpFunc(compCoords[ii,dim], otherBbox[ii,dim])
  compCoords = getClippedBbox(img_np.shape, compCoords, margin)
  # Verts are x-y, index into image with row-col
  croppedImg = image[compCoords[0,1]:compCoords[1,1], compCoords[0,0]:compCoords[1,0], :]
  return croppedImg, compCoords


def area_coord_regionTbl(_image: Image):
  if not np.any(_image.data):
    return pd.DataFrame({'coords': [np.array([[]])], 'area': [0]})
  regionDict = regionprops_table(label(_image.data), properties=('coords', 'area'))
  _regionPropTbl = pd.DataFrame(regionDict)
  return _regionPropTbl