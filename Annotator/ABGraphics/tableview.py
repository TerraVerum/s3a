from __future__ import annotations

from enum import Enum

from pyqtgraph.Qt import QtWidgets, QtCore

from ..constants import TEMPLATE_COMP
from ..tablemodel import CompTableModel

Slot = QtCore.pyqtSlot

class CompTableView(QtWidgets.QTableView):
  def __init__(self, *args):
    super().__init__(*args)
    self.setSortingEnabled(True)

    # Default to text box delegate
    self.setItemDelegate(TextDelegate(self))

    validOpts = [True, False]
    boolDelegate = ComboBoxDelegate(self, comboValues=validOpts)

    colTitles = []
    for ii, field in enumerate(TEMPLATE_COMP):
      colTitles.append(field.name)
      curval = field.value
      if isinstance(curval, bool):
        self.setItemDelegateForColumn(ii, boolDelegate)
      elif isinstance(curval, Enum):
        self.setItemDelegateForColumn(ii, ComboBoxDelegate(self, comboValues=list(type(curval))))
      else:
        # Default to text box
        pass

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
  def __init__(self, parent=None, comboValues=None, comboNames=None):
    super().__init__(parent)
    if comboValues is None:
      comboValues = []
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