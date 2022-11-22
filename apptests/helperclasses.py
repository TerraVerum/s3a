import stat
from typing import Union

import cv2 as cv
import numpy as np
import pandas as pd
from qtextras import OptionsDict

from apptests.testingconsts import RND, TEST_FILE_DIR
from s3a.constants import REQD_TBL_FIELDS
from s3a.structures import ComplexXYVertices
from s3a.tabledata import TableData


class CompDfTester:
    def __init__(self, numComps, fillInfo=True, tableData=None):
        if tableData is None:
            tableData = TableData()
        self.tableData = tableData
        self.compDf = tableData.makeComponentDf(numComps)
        self.compDf.set_index(np.arange(numComps, dtype=int), inplace=True)
        self.numComps = numComps
        if fillInfo:
            self.fillRandomVerts()
            for field in tableData.allFields:
                # Silently fails for non-limit based fields
                self.fillRandomCategorical(field)

    def fillRandomCategorical(
        self, field: Union[str, OptionsDict], compDf: pd.DataFrame = None
    ):
        if compDf is None:
            compDf = self.compDf
        field = self.tableData.fieldFromName(field)
        lims = field.opts.get("limits", None)
        if lims is None:
            # Nothing to do
            return compDf[field]
        lims = np.array(lims)
        newData = lims[RND.integers(0, len(lims), len(compDf))]
        compDf[field] = newData

    def fillRandomVerts(
        self, imageShape=(2000, 2000), compDf: pd.DataFrame = None, vertType="circle"
    ):
        if compDf is None:
            compDf = self.compDf
        mask = np.zeros(imageShape[:2], "uint8")

        retVal = []
        for ii in range(len(compDf)):
            radius = RND.integers(5, max(imageShape) // 5)
            o_x = RND.integers(0, imageShape[1])
            o_y = RND.integers(0, imageShape[0])
            if vertType == "circle":
                cv.circle(mask, (o_x, o_y), radius, 1, -1)
            elif vertType == "rectangle":
                cv.rectangle(mask, (o_x, o_y), (o_x + radius, o_y + radius), 1, -1)
            verts = ComplexXYVertices.fromBinaryMask(mask)
            compDf.at[ii, REQD_TBL_FIELDS.VERTICES] = verts
            retVal.append(verts)
            mask.fill(0)
        return retVal


def clearTmpFiles(globexpr="*"):
    for curPath in TEST_FILE_DIR.glob(globexpr):
        if curPath.is_file():
            curPath.chmod(stat.S_IWRITE)
            curPath.unlink()
