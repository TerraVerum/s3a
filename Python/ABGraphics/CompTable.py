import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets

import numpy as np

from constants import ComponentTableFields as CTF
import component

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

  def addComps(self, compList):
    compArr = np.array(compList.loc[:,self._xpondingCompFields])
    self.appendData(compArr)

  def resetComps(self, compList):
    self.setRowCount(0)
    self.addComps(compList)

  def _comp2TableRow(self, comp):
    return [getattr(comp, field) for field in self._xpondingCompFields]

if __name__ == '__main__':
  from sys import path
  app = pg.mkQApp()
  t = CompTable()
  t.show()
  app.exec()