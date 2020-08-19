import stat
from pathlib import Path
from typing import List

import cv2 as cv
import numpy as np
from pandas import DataFrame as df

from s3a import FR_SINGLETON
from s3a.constants import REQD_TBL_FIELDS
from s3a.structures import FRComplexVertices
from testingconsts import RND, IMG_DIR

class CompDfTester:
  def __init__(self, numComps):
    self.compDf = FR_SINGLETON.tableData.makeCompDf(numComps)
    self.compDf.set_index(np.arange(numComps, dtype=int), inplace=True)
    self.numComps = numComps
    self.fillRandomClasses()
    self.fillRandomVerts()


  def fillRandomVerts(self, imShape=(2000, 2000), compDf: df=None):
    if compDf is None:
      compDf = self.compDf
    mask = np.zeros(imShape[:2], 'uint8')

    retVal = []
    for ii in range(len(compDf)):
      radius = RND.integers(5, max(imShape) // 5)
      o_x = RND.integers(0, imShape[1])
      o_y = RND.integers(0, imShape[0])
      verts = FRComplexVertices.fromBwMask(cv.circle(mask, (o_x, o_y), radius, 1, -1))
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


def clearTmpFiles(exceptFiles: List[Path] =None):
  if exceptFiles is None:
    exceptFiles: List[Path] = []
  for file in IMG_DIR.glob('*'):
    if file not in exceptFiles:
      file.chmod(stat.S_IWRITE)
      file.unlink()