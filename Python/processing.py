import numpy as np
import cv2 as cv

from skimage.morphology import closing
from skimage.morphology import disk
from skimage.segmentation import quickshift

def getComps(img: np.array) -> np.array:
  chans = cv.split(img)
  mask = np.bitwise_and(1.25*chans[0] < chans[1],
                        1.25*chans[2] < chans[1])
  mask = closing(mask, disk(5))
  return np.invert(mask)

def segmentComp(comp: np.array, maxDist: np.float) -> np.array:
  segImg = quickshift(comp, kernel_size=10, max_dist=maxDist)
  # Color segmented image with mean values
  labels = np.unique(segImg)
  outImg = np.empty(comp.shape)
  for label in labels:
    curmask = segImg == label
    outImg[curmask,:] = comp[curmask,:].reshape(-1,3).mean(0)
  return outImg