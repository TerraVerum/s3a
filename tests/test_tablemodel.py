from cdef import appInst, FRCdefApp, makeCompDf
from cdef.structures import FRComplexVertices, FRVertices
from cdef.projectvars import TEMPLATE_COMP as TC, TEMPLATE_COMP_CLASSES as COMP_CLASSES
from cdef.projectvars import FR_ENUMS

import numpy as np
import pandas as pd
import cv2 as cv
from pandas import DataFrame as df
from PIL import Image
import random
from random import randint


from skimage import io
from skimage import morphology as morph
import skimage

NUM_COMPS = 5

from unittest import TestCase
class ModelTester(TestCase):
  def setUp(self):
    random.seed(42)
    np.random.seed(42)
    self.app = FRCdefApp()
    self.mgr = self.app.compMgr

    self.img: np.ndarray = io.imread('../images/circuitBoard.png')
    mask = np.zeros(self.img.shape[:2], 'uint8')

    sampleComps = makeCompDf(NUM_COMPS)
    sampleComps.set_index(np.arange(NUM_COMPS, dtype=int), inplace=True)

    for ii in range(NUM_COMPS):
      radius = randint(0, 100)
      o_x = randint(0, self.img.shape[1])
      o_y = randint(0, self.img.shape[0])
      sampleComps.loc[ii, TC.VERTICES] = [FRComplexVertices.fromBwMask(
        cv.circle(mask, (o_x, o_y), radius, 1)
      )]
      mask.fill(0)
    self.sampleComps = sampleComps

    self.emptyArr = np.array([], int)


  def test_add_comps(self):
    # Standard add
    oldIds = np.arange(NUM_COMPS, dtype=int)
    comps = self.sampleComps.copy(deep=True)
    changeList = self.mgr.addComps(comps)
    self.cmpChangeList(changeList, oldIds)

    # Empty add
    changeList = self.mgr.addComps(makeCompDf(0))
    self.cmpChangeList(changeList)

    # Add existing should change during merge
    changeList = self.mgr.addComps(comps, FR_ENUMS.COMP_ADD_AS_MERGE)
    self.cmpChangeList(changeList, changed=oldIds)

    # Should be new IDs during 'add as new'
    changeList = self.mgr.addComps(comps)
    self.cmpChangeList(changeList, added=oldIds + NUM_COMPS)


  def test_rm_comps(self):
    comps = self.sampleComps.copy(deep=True)
    ids = np.arange(NUM_COMPS, dtype=int)

    # Standard remove
    self.mgr.addComps(comps)
    changeList = self.mgr.rmComps(ids)
    self.cmpChangeList(changeList, deleted=ids)

    # Remove when ids don't exist
    changeList = self.mgr.rmComps(ids)
    self.cmpChangeList(changeList)

    # Remove all
    for _ in range(10):
      self.mgr.addComps(comps)
    oldIds = self.mgr.compDf[TC.INST_ID].values
    changeList = self.mgr.rmComps('all')
    self.cmpChangeList(changeList,deleted=oldIds)


  @staticmethod
  def cmpChangeList(changeList: dict, added: np.ndarray=None, deleted: np.ndarray=None,
                    changed: np.ndarray=None):
    emptyArr = np.array([], int)
    arrs = locals()
    for name in 'added', 'deleted', 'changed':
      if arrs[name] is None:
        arrCmp = emptyArr
      else:
        arrCmp = arrs[name]
      np.testing.assert_equal(changeList[name], arrCmp)
