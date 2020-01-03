import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets

from constants import ComponentTableFields as CTF
import component

class CompTable(pg.TableWidget):
  colTitles = [field.value for field in CTF]

  def __init__(self, *args, **kwargs):
    super().__init__()
    self.setColumnCount(len(self.colTitles))
    for ii, curTitle in enumerate(self.colTitles):
      curCol = QtWidgets.QTableWidgetItem(curTitle)
      self.setHorizontalHeaderItem(ii, curCol)

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
    for comp in compList:
      curRow = [getattr(comp, field) for field in self._xpondingCompFields]
      self.addRow(curRow)

if __name__ == '__main__':
  from sys import path
  app = pg.mkQApp()
  path.append('..')
  t = CompTable()
  t.show()
  app.exec()