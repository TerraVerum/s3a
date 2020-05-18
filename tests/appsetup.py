from skimage import io

from cdef.structures import FRComplexVertices
import random
from random import randint

import cv2 as cv
import numpy as np
from pandas import DataFrame as df

from cdef import FR_SINGLETON
from cdef.projectvars import REQD_TBL_FIELDS, BASE_DIR
from cdef.structures import FRComplexVertices

makeCompDf = FR_SINGLETON.tableData.makeCompDf

TESTS_DIR = BASE_DIR.parent/'tests'
IMG_DIR = BASE_DIR.parent/'images'
NUM_COMPS = 15
SAMPLE_IMG_DIR = IMG_DIR/'circuitBoard.png'
SAMPLE_IMG = io.imread(SAMPLE_IMG_DIR)

random.seed(42)
np.random.seed(42)

class CompDfTester:
  def __init__(self, numComps):
    self.compDf = makeCompDf(numComps)
    self.compDf.set_index(np.arange(numComps, dtype=int), inplace=True)
    self.numComps = numComps

  def fillRandomVerts(self, imShape=(2000, 2000)):
    mask = np.zeros(imShape[:2], 'uint8')

    for ii in range(self.numComps):
      radius = randint(0, max(imShape)//5)
      o_x = randint(0, imShape[1])
      o_y = randint(0, imShape[0])
      verts = FRComplexVertices.fromBwMask(cv.circle(mask, (o_x, o_y), radius, 1))
      self.compDf.at[ii, REQD_TBL_FIELDS.VERTICES] = verts
      mask.fill(0)

  def fillRandomClasses(self):
    # Encapsulate in np array for random indexing
    npClasses = np.array(FR_SINGLETON.tableData.compClasses)
    randomIdxs = np.random.randint(0, len(npClasses), size=self.numComps)

    newClasses = npClasses[randomIdxs]
    self.compDf.loc[:, REQD_TBL_FIELDS.COMP_CLASS] = newClasses