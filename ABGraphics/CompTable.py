import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets

import numpy as np

from constants import ComponentTableFields as CTF
import component
from processing import sliceToArray

from typing import Union

class CompTable(pg.TableWidget):
  colTitles = [field.value for field in CTF]

  def __init__(self, *args, **kwargs):
    super().__init__(editable=True)
    self.setColumnCount(len(self.colTitles))
    self.setHorizontalHeaderLabels(self.colTitles)

    # Sort by vertices will defer to index sort, since value sort crashes the
    # application
    self.setSortMode(self.colTitles.index('Vertices'), 'index')

    # Create list of component fields that correspond to table columns
    # These are camel-cased
    xpondingCompFields = []
    compFields = list(component.Component().__dict__.keys())
    lowercaseCompFields = [field.lower() for field in compFields]
    compareColNames = [name.lower().replace(' ', '') for name in self.colTitles]
    for name in compareColNames:
      try:
        compFieldIdx = lowercaseCompFields.index(name)
        xpondingCompFields.append(compFields[compFieldIdx])
      except ValueError:
        pass
    self._xpondingCompFields = xpondingCompFields

  def addComps(self, compDf):
    compArr = np.array(compDf.loc[:,self._xpondingCompFields])
    self.appendData(compArr)

  def updateComps(self, compDf):
    compArr = np.array(compDf.loc[:,self._xpondingCompFields])
    idsToUpdate = compArr[:,self._xpondingCompFields.index('instanceId')]
    rowsToUpdate = self.idsToRowIdxs(idsToUpdate)
    for ii, rowIdx in enumerate(rowsToUpdate):
      self.setRow(rowIdx, compArr[ii,:])

  def resetComps(self, compDf):
    self.setRowCount(0)
    self.addComps(compDf)

  def idsToRowIdxs(self, idList: Union[np.ndarray, int]):
    if not hasattr(idList, '__iter__'):
      idList = np.array([idList])
    idColIdx = self._xpondingCompFields.index('instanceId')
    existingIds = np.array([self.item(ii, idColIdx).value
                            for ii in range(self.rowCount())])
    xrefMat = idList[None,:] == existingIds[:,None]
    # Xrefmat has len(existingIds) x len(idList) entries
    # argsort by columns gives existingIds that match each
    # initial entry in idList
    xpondingIdxs = np.nonzero(xrefMat)
    listOrder = np.argsort(xpondingIdxs[1])
    return xpondingIdxs[0][listOrder]

if __name__ == '__main__':
  from sys import path
  app = pg.mkQApp()
  t = CompTable()
  t.show()
  app.exec()