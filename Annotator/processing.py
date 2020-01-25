import cv2 as cv
import numpy as np
from skimage.filters import gaussian
from skimage.measure import regionprops, label
from skimage.morphology import closing, dilation
from skimage.morphology import disk
from skimage.segmentation import quickshift

from .graphicseval import overlayImgs


def getBwComps(img: np.ndarray) -> np.ndarray:
  return bwBgMask(img)

def getBwComps_segmentation(img: np.ndarray) -> np.ndarray:
  img = (gaussian(img, 1)*255).astype('uint8')
  margin = 5
  bwHullImg = regionConvHulls(img)
  bwHullImg = dilation(bwHullImg, np.ones((margin,margin)))
  segImg = quickshift(img, ratio=0.9, max_dist=25)
  colorBroadcast = np.zeros((1,1,3),dtype='uint8')
  colorBroadcast[0,0,2] = 255

  win = overlayImgs(colorLabelsWithMean(segImg, img), img, bwHullImg.astype('uint8')[:,:,None]*colorBroadcast)
  win.show()
  seedpoints = getVertsFromBwComps(bwHullImg, simplifyVerts=False)
  for compVerts in seedpoints:
    # Vertices are x-y, convert to row-col
    edgeLabels = np.unique(segImg[compVerts[:,1], compVerts[:,0]])
    comparisonMat = segImg[:,:,None] == edgeLabels[None,None,:]
    isEdgePix = np.any(comparisonMat, axis=2)
    bwHullImg[isEdgePix] = False
  return bwHullImg

def regionConvHulls(img: np.ndarray):
  compMask = bwBgMask(img)
  outImg = np.zeros(img.shape[0:2], dtype=bool)
  labeledImg = label(compMask)
  regions = regionprops(labeledImg)
  for region in regions:
    bbox = region.bbox
    convHull = region.convex_image
    outImg[bbox[0]:bbox[2],bbox[1]:bbox[3]] = convHull
  return outImg

def colorLabelsWithMean(labelImg, refImg) -> np.ndarray:
  outImg = np.empty(refImg.shape)
  labels = np.unique(labelImg)
  for curLabel in labels:
    curmask = labelImg == curLabel
    outImg[curmask,:] = refImg[curmask,:].reshape(-1,3).mean(0)
  return outImg


def bwBgMask(img: np.array) -> np.array:
  if img.dtype != 'uint8':
    img = img.astype('uint8')
  chans = cv.split(img)
  mask = np.bitwise_and(1.25*chans[0] < chans[1],
                        1.25*chans[2] < chans[1])
  mask = np.invert(closing(mask, disk(5)))
  return mask

def getVertsFromBwComps(bwmask: np.array, simplifyVerts=True) -> np.array:
  approxMethod = cv.CHAIN_APPROX_SIMPLE
  if not simplifyVerts:
    approxMethod = cv.CHAIN_APPROX_NONE
  # Contours are on the inside of components, so dilate first to make sure they are on the
  # outside
  #bwmask = dilation(bwmask, np.ones((3,3), dtype=bool))
  contours, _ = cv.findContours(bwmask.astype('uint8'), cv.RETR_EXTERNAL, approxMethod)
  compVertices = []
  for contour in contours:
    compVertices.append(contour[:,0,:])
  return compVertices

def segmentComp(compImg: np.array, maxDist: np.float, kernSz=10) -> np.array:
  # For maxDist of 0, the input isn't changed and it takes a long time
  if maxDist < 1:
    return compImg
  segImg = quickshift(compImg, kernel_size=kernSz, max_dist=maxDist)
  # Color segmented image with mean values
  return colorLabelsWithMean(segImg, compImg)

def rmSmallComps(bwMask: np.ndarray, minSz: int=0) -> np.ndarray:
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

def growSeedpoint(img: np.array, seeds: np.array, thresh: float, minSz: int=0) -> \
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
  bwOut = np.zeros(img.shape[0:2], dtype=bool)
  nChans = img.shape[2] if len(img.shape) > 2 else 1
  # Computationally cheaper to compare square of thresh instead of using
  # euclidean distance
  thresh = thresh**2
  for seed in seeds:
    bwOut[seed[0], seed[1]] = True
    changed = True
    while changed:
      neighbors = dilation(bwOut, np.ones((3,3)))
      neighbors[bwOut] = False
      compMean = img[bwOut,:].reshape(-1,nChans).mean(0)
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
      newBwOut = bwOut | neighbors
      changed = np.any(newBwOut != bwOut)
      bwOut = newBwOut
  # Remove components smaller than minSz
  return rmSmallComps(bwOut, minSz)

def growBoundarySeeds(img: np.ndarray, seedThresh: float, minSz: int,
                      segThresh: float=0, useAllBounds=False) -> np.ndarray:
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
  img = segmentComp(img, seedThresh)
  nrows, ncols, *_ = img.shape
  maxRow, maxCol = nrows-1, ncols-1
  if useAllBounds:
    seedRows = np.concatenate([np.repeat(0, maxCol), np.repeat(maxRow, maxCol),
                np.arange(nrows, dtype=int), np.arange(nrows, dtype=int)])
    seedCols = np.concatenate([np.arange(ncols, dtype=int), np.arange(ncols, dtype=int),
                np.repeat(0, maxRow), np.repeat(maxCol, maxRow)])
    seeds = np.hstack((seedRows[:,None], seedCols[:,None]))
  else:
    # Just use image corners
    seeds = np.array([[0,0], [0, maxCol], [maxRow, 0], [maxRow, maxCol]])
  # Since these are background components, we don't want to remove small components until
  # after inverting the mask
  bwBgSeedGrow = growSeedpoint(img, seeds, seedThresh, 0)
  bwOut = ~bwBgSeedGrow

  # Merge sparsely connected regions
  bwOut = closing(bwOut, np.ones((3,3), dtype=bool))

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


def nanConcatList(vertList):
  """
  Utility for concatenating all vertices within a list while adding
  NaN entries between each separate list
  """
  if isinstance(vertList, np.ndarray):
    vertList = [vertList]
  nanSep = np.ones((1,2), dtype=int)*np.nan
  allVerts = []
  for curVerts in vertList:
    allVerts.append(curVerts)
    allVerts.append(nanSep)
  # Take away last nan if it exists
  if len(allVerts) > 0:
    allVerts.pop()
    return np.vstack(allVerts)
  return np.array([]).reshape(-1,2)

def splitListAtNans(concatVerts:np.ndarray):
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
  return allVerts

def sliceToArray(keySlice: slice, arrToSlice: np.ndarray):
  """
  Converts array slice into concrete array values
  """
  start, stop, step = keySlice.start, keySlice.stop, keySlice.step
  if start is None:
    start = 0
  if stop is None:
    stop = len(arrToSlice)
  outArr = np.arange(start, stop, step)
  # Remove elements that don't correspond to list indices
  outArr = outArr[np.isin(outArr, arrToSlice)]
  return outArr

def getClippedBbox(arrShape: tuple, bbox: np.ndarray, margin: int):
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
  for ii in range(2):
    bbox[0,ii] = np.maximum(0, bbox[0,ii]-margin)
    bbox[1,ii] = np.minimum(arrShape[1-ii], bbox[1,ii]+margin)
  return bbox.astype(int)


if __name__ == '__main__':
  from PIL import Image
  import pyqtgraph as pg
  im = np.array(Image.open('../med.tif'))
  im = getBwComps(im)
  pg.image(im)
  pg.show()