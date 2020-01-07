import numpy as np
import cv2 as cv

from skimage.morphology import closing, dilation
from skimage.morphology import disk
from skimage.segmentation import quickshift

def getBwComps(img: np.array) -> np.array:
  if img.dtype != 'uint8':
    img = img.astype('uint8')
  chans = cv.split(img)
  mask = np.bitwise_and(1.25*chans[0] < chans[1],
                        1.25*chans[2] < chans[1])
  mask = np.invert(closing(mask, disk(5)))
  return mask

def getVertsFromBwComps(bwmask: np.array) -> np.array:
  contours, _ = cv.findContours(bwmask.astype('uint8'), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
  compVertices = []
  for contour in contours:
    compVertices.append(contour[:,0,:])
  return compVertices

def segmentComp(compImg: np.array, maxDist: np.float) -> np.array:
  segImg = quickshift(compImg, kernel_size=10, max_dist=maxDist)
  # Color segmented image with mean values
  labels = np.unique(segImg)
  outImg = np.empty(compImg.shape)
  for label in labels:
    curmask = segImg == label
    outImg[curmask,:] = compImg[curmask,:].reshape(-1,3).mean(0)
  return outImg

def growSeedpoint(img: np.array, seeds: np.array, thresh: float) -> np.array:
  '''
  Starting from *seed*, fills each connected pixel if the difference between
  neighboring pixels and the current component is less than *thresh*.
  Places one 'on' pixel in each seed location, rerunning the algorithm for each
  new seed. **Note**: only one label is  generated for each call to this
  function. I.e. if multiple seeds are specified, they are assumed to belong
  to the same label.

  Parameters
  ----------
  img :    MxNxChan
    Input image

  seeds :  Mx2 np array
    Contains locations in output mask that are 'on'
    at the start of the algorithm. Pixels connected to these are
    iteratively added to the components if their intensities are
    close enough to each seed component.

  thresh : float
    Threshold between component and neighbors. If neighbor pixels
    are below this value, they are added to the seed component.
  '''
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
  return bwOut

def nanConcatList(vertList):
  '''
  Utility for concatenating all vertices within a list while adding
  NaN entries between each separate list
  '''
  if isinstance(vertList, np.ndarray):
    vertList = [vertList]
  nanSep = np.ones((1,2), dtype=int)*np.nan
  allVerts = []
  for curVerts in vertList:
    allVerts.append(curVerts)
    allVerts.append(nanSep)
  return np.vstack(allVerts)

def splitListAtNans(concatVerts:np.ndarray):
  '''
  Utility for taking a single list of nan-separated region vertices
  and breaking it into several regions with no nans.
  '''
  # concatVerts must end with nan if it came from nanConcatList
  if not np.isnan(concatVerts[-1,0]):
    concatVerts = nanConcatList(concatVerts)
  allVerts = []
  nanEntries = np.nonzero(np.isnan(concatVerts[:,0]))[0]
  curIdx = 0
  for nanEntry in nanEntries:
    curVerts = concatVerts[curIdx:nanEntry,:].astype('int')
    allVerts.append(curVerts)
  return allVerts

def sliceToArray(keySlice: slice, arrToSlice: np.ndarray):
  """
  Converts array slice into concrete array values
  """
  start, stop, step = keySlice.start, keySlice.stop, keySlice.step
  if start == None:
    start = 0
  if stop == None:
    stop = len(arrToSlice)
  outArr = np.arange(start, stop, step)
  # Remove elements that don't correspond to list indices
  outArr = outArr[np.isin(outArr, arrToSlice)]
  return outArr


