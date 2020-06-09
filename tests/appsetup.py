import random
import stat
from typing import List

import cv2 as cv
import numpy as np
from skimage import io
from pandas import DataFrame as df

from cdef import FR_SINGLETON, FRCdefApp
from cdef.projectvars import REQD_TBL_FIELDS, BASE_DIR
from cdef.structures import FRComplexVertices

makeCompDf = FR_SINGLETON.tableData.makeCompDf

TESTS_DIR = BASE_DIR.parent/'tests'
IMG_DIR = BASE_DIR.parent/'images'
EXPORT_DIR = TESTS_DIR/'files'

NUM_COMPS = 15
SAMPLE_IMG_DIR = IMG_DIR/'circuitBoard.png'
SAMPLE_IMG = io.imread(SAMPLE_IMG_DIR)

RND = np.random.default_rng(seed=42)

class CompDfTester:
  def __init__(self, numComps):
    self.compDf = makeCompDf(numComps)
    self.compDf.set_index(np.arange(numComps, dtype=int), inplace=True)
    self.numComps = numComps

  def fillRandomVerts(self, imShape=(2000, 2000), compDf: df=None):
    if compDf is None:
      compDf = self.compDf
    mask = np.zeros(imShape[:2], 'uint8')

    retVal = []
    for ii in range(self.numComps):
      radius = RND.integers(5, max(imShape)//5)
      o_x = RND.integers(0, imShape[1])
      o_y = RND.integers(0, imShape[0])
      verts = FRComplexVertices.fromBwMask(cv.circle(mask, (o_x, o_y), radius, 1))
      compDf.at[ii, REQD_TBL_FIELDS.VERTICES] = verts
      retVal.append(verts)
      mask.fill(0)
    return retVal

  def fillRandomClasses(self, compDf: df=None):
    if compDf is None:
      compDf = self.compDf
    # Encapsulate in np array for random indexing
    npClasses = np.array(FR_SINGLETON.tableData.compClasses)
    randomIdxs = RND.integers(0, len(npClasses), size=self.numComps)

    newClasses = npClasses[randomIdxs]
    compDf.loc[:, REQD_TBL_FIELDS.COMP_CLASS] = newClasses
    return newClasses

def clearTmpFiles(exceptFiles: List[str] =None):
  if exceptFiles is None:
    exceptFiles: List[str] = []
  for fileExt in 'csv', 'png':
    for file in EXPORT_DIR.glob(f'*.{fileExt}'):
      if str(file) not in exceptFiles:
        file.chmod(stat.S_IWRITE)
        file.unlink()

def defaultApp_tester():
  app = FRCdefApp(Image=SAMPLE_IMG_DIR)
  dfTester = CompDfTester(NUM_COMPS)
  dfTester.fillRandomVerts(imShape=SAMPLE_IMG.shape)
  dfTester.fillRandomClasses()
  return app, dfTester
