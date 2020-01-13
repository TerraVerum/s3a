from __future__ import annotations
from pyqtgraph.Qt import QtWidgets, QtCore
Slot = QtCore.pyqtSlot


from constants import ComponentTableFields as CTF, ComponentTypes
import component

from typing import Union

class CompTableView(QtWidgets.QTableView):
  def __init__(self, parent):
    super().__init__(parent)

    self.setItemDelegate(TextDelegate(self))

    colTitles = [field.value for field in CTF]
    validColIdx = colTitles.index('Validated')
    validOpts = ['True', 'False']
    self.setItemDelegateForColumn(validColIdx, ComboBoxDelegate(self, comboValues=validOpts))

    devTypeColIdx = colTitles.index('Device Type')
    devTypeOpts = [devType for devType in ComponentTypes]
    self.setItemDelegateForColumn(devTypeColIdx, ComboBoxDelegate(self, comboValues=devTypeOpts))

class CompTableModel(QtCore.QAbstractTableModel):
  colTitles = [field.value for field in CTF]

  def __init__(self, compMgr: component.ComponentMgr):
    super().__init__()
    self.compMgr = compMgr

    compMgr.sigCompsAboutToChange.connect(self.layoutAboutToBeChanged.emit)
    compMgr.sigCompsChanged.connect(self.layoutChanged.emit)

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

  # Helper for delegates
  def indexToRowCol(self, index: QtCore.QModelIndex):
    row = index.row()
    col = self._xpondingCompFields[index.column()]
    return row, col

  # ------
  # Functions required to implement table model
  # ------
  def columnCount(self, *args, **kwargs):
    return len(self.colTitles)

  def rowCount(self, *args, **kwargs):
    return len(self.compMgr._compList)

  def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
    if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
      return self.colTitles[section]

  def data(self, index, role=QtCore.Qt.DisplayRole):
    dataIdx = self.indexToRowCol(index)
    outData = self.compMgr._compList.loc[dataIdx]
    if role == QtCore.Qt.DisplayRole:
      return str(outData)
    elif role == QtCore.Qt.EditRole:
      return outData
    else:
      return None

  def setData(self, index, value, role=QtCore.Qt.EditRole):
    dataIdx = self.indexToRowCol(index)
    self.compMgr._compList.loc[dataIdx] = value
    return True

  def flags(self, index):
    noEditColIdxs = [self.colTitles.index(col) for col in ['Instance ID', 'Vertices']]
    if index.column() not in noEditColIdxs:
      return QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable
    else:
      return QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled

class TextDelegate(QtWidgets.QItemDelegate):
  def createEditor(self, parent, option, index):
    editor = QtWidgets.QPlainTextEdit(parent)
    editor.setTabChangesFocus(True)
    return editor

  def setEditorData(self, editor: QtWidgets.QPlainTextEdit, index):
    text = index.data(QtCore.Qt.DisplayRole)
    editor.setPlainText(text)

  def setModelData(self, editor: QtWidgets.QTextEdit,
                           model: CompTableModel,
                           index: QtCore.QModelIndex):
    model.setData(index, editor.toPlainText())

  def updateEditorGeometry(self, editor: QtWidgets.QPlainTextEdit,
                           option: QtWidgets.QStyleOptionViewItem,
                           index: QtCore.QModelIndex):
    editor.setGeometry(option.rect)


class ComboBoxDelegate(QtWidgets.QStyledItemDelegate):
  def __init__(self, parent=None, comboValues=[], comboNames=None):
    super().__init__(parent)
    self.comboValues: list = comboValues
    if comboNames is None:
      self.comboNames = [str(val) for val in comboValues]

  def createEditor(self, parent, option, index):
    editor = QtWidgets.QComboBox(parent)
    editor.addItems(self.comboNames)
    return editor

  def setEditorData(self, editor: QtWidgets.QComboBox, index):
    curVal = index.data(QtCore.Qt.DisplayRole)
    editor.setCurrentIndex(self.comboNames.index(curVal))

  def setModelData(self, editor: QtWidgets.QComboBox,
                   model: CompTableModel,
                           index: QtCore.QModelIndex):
    model.setData(index, self.comboValues[editor.currentIndex()])

  def updateEditorGeometry(self, editor: QtWidgets.QPlainTextEdit,
                           option: QtWidgets.QStyleOptionViewItem,
                           index: QtCore.QModelIndex):
    editor.setGeometry(option.rect)