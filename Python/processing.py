import numpy as np
import cv2 as cv

from skimage.morphology import closing
from skimage.morphology import disk
from skimage.segmentation import quickshift

def getCompBounds(img: np.array) -> np.array:
  chans = cv.split(img)
  mask = np.bitwise_and(1.25*chans[0] < chans[1],
                        1.25*chans[2] < chans[1])
  mask = np.invert(closing(mask, disk(5)))
  contours, _ = cv.findContours(mask.astype('uint8'), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
  compVertices = []
  for contour in contours:
    compVertices.append(contour[:,0,:])
  return compVertices

def segmentComp(comp: np.array, maxDist: np.float) -> np.array:
  segImg = quickshift(comp, kernel_size=10, max_dist=maxDist)
  # Color segmented image with mean values
  labels = np.unique(segImg)
  outImg = np.empty(comp.shape)
  for label in labels:
    curmask = segImg == label
    outImg[curmask,:] = comp[curmask,:].reshape(-1,3).mean(0)
  return outImg
