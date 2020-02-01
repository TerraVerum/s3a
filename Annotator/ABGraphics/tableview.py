from __future__ import annotations

from enum import Enum

from pyqtgraph.Qt import QtWidgets, QtCore, QtGui

import numpy as np
import pandas as pd
from pandas import DataFrame as df

from typing import Sequence

from ..constants import TEMPLATE_COMP
from ..tablemodel import CompTableModel, ComponentMgr, ModelOpts

Slot = QtCore.pyqtSlot
Signal = QtCore.pyqtSignal

class PopupTableDialog(QtWidgets.QDialog):
  def __init__(self, *args):
    super().__init__(*args)
    self.setModal(True)
    # -----------
    # Table View
    # -----------
    self.tbl = CompTableView(minimal=True)

    # -----------
    # Warning Message
    # -----------
    self.titles = np.array(self.tbl.mgr.colTitles)
    self.warnLbl = QtWidgets.QLabel(self)
    self.warnLbl.setStyleSheet("font-weight: bold; color:red; font-size:14")
    self.updateWarnMsg(self.titles)

    # -----------
    # Widget buttons
    # -----------
    self.applyBtn = QtWidgets.QPushButton('Apply')
    self.closeBtn = QtWidgets.QPushButton('Close')

    # -----------
    # Widget layout
    # -----------
    btnLayout = QtWidgets.QHBoxLayout()
    btnLayout.addWidget(self.applyBtn)
    btnLayout.addWidget(self.closeBtn)

    centralLayout = QtWidgets.QVBoxLayout()
    centralLayout.addWidget(self.warnLbl)
    centralLayout.addWidget(self.tbl)
    centralLayout.addLayout(btnLayout)
    self.setLayout(centralLayout)
    self.setMinimumWidth(self.tbl.width())

    # -----------
    # UI Element Signals
    # -----------
    self.closeBtn.clicked.connect(self.close)
    self.applyBtn.clicked.connect(self.accept)

  def updateWarnMsg(self, updatableCols: List[str]):
    warnMsg = f'Note! Only {", ".join(updatableCols)} will be updated from this view.'
    self.warnLbl.setText(warnMsg)


  @property
  def data(self):
    return self.tbl.mgr.compDf.iloc[[0],:]

  def setData(self, compDf: df, colIdxs: Sequence):
    self.tbl.mgr.rmComps()
    self.tbl.mgr.addComps(compDf, addtype=ModelOpts.ADD_AS_MERGE)
    self.updateWarnMsg(self.titles[colIdxs])

class CompTableView(QtWidgets.QTableView):
  """
  Table for displaying :class:`ComponentMgr` data.
  """
  sigSelectionChanged = Signal(object)

  def __init__(self, *args, minimal=False):
    """
    Creates the table.

    :param minimal: Whether to make a table with minimal features.
       If false, creates extensible table with context menu options.
       Otherwise, only contains minimal features.
    """
    super().__init__(*args)
    self.setSortingEnabled(True)

    self.mgr = ComponentMgr()
    self.setModel(self.mgr)

    if not minimal:
      self.popup = PopupTableDialog()
      # Create context menu for changing table rows
      self.menu = self.createContextMenu()
      self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
      cursor = QtGui.QCursor()
      self.customContextMenuRequested.connect(lambda: self.menu.exec_(cursor.pos()))
    else:
      # Don't forget to disable custom delete option
      self.keyPressEvent = super().keyPressEvent

    # Default to text box delegate
    self.setItemDelegate(TextDelegate(self))

    validOpts = [True, False]
    boolDelegate = ComboBoxDelegate(self, comboValues=validOpts)

    self.instIdColIdx = TEMPLATE_COMP.paramNames().index(TEMPLATE_COMP.INST_ID.name)

    for ii, field in enumerate(TEMPLATE_COMP):
      curval = field.value
      if isinstance(curval, bool):
        self.setItemDelegateForColumn(ii, boolDelegate)
      elif isinstance(curval, Enum):
        self.setItemDelegateForColumn(ii, ComboBoxDelegate(self, comboValues=list(type(curval))))
      else:
        # Default to text box
        pass

  # When the model is changed, get a reference to the ComponentMgr
  def setModel(self, modelOrProxy: QtCore.QAbstractTableModel):
    super().setModel(modelOrProxy)
    try:
      # If successful we were given a proxy model
      self.mgr = modelOrProxy.sourceModel()
    except AttributeError:
      self.mgr = modelOrProxy

  def selectionChanged(self, curSel: QtCore.QItemSelection, prevSel: QtCore.QItemSelection):
    """
    When the selected rows in the table change, this retrieves the corresponding previously
    and newly selected IDs. They are then emitted in sigSelectionChanged.
    """
    super().selectionChanged(curSel, prevSel)
    selectedIds = []
    selection = self.selectionModel().selectedIndexes()
    for item in selection:
      selectedIds.append(item.sibling(item.row(),self.instIdColIdx).data(QtCore.Qt.EditRole))
    self.sigSelectionChanged.emit(pd.unique(selectedIds))

  def keyPressEvent(self, ev: QtGui.QKeyEvent) -> None:
    # Only delete rows if at least on cell is currently selected
    pressedKey = ev.key()
    modifiers = ev.modifiers()
    isItemSelected = len(self.selectionModel().selectedIndexes()) > 0
    if isItemSelected:
      if pressedKey == QtCore.Qt.Key_Delete:
        self.removeTriggered()
      elif pressedKey == QtCore.Qt.Key_D and modifiers == QtCore.Qt.ControlModifier:
        self.overwriteTriggered()
    super().keyPressEvent(ev)

  def createContextMenu(self):
    menu = QtWidgets.QMenu(self)
    menu.setTitle('Table Actions')

    remAct = QtGui.QAction("Remove", menu)
    remAct.triggered.connect(self.removeTriggered)
    menu.addAction(remAct)

    overwriteAct = QtGui.QAction("Set Same As First", menu)
    overwriteAct.triggered.connect(self.overwriteTriggered)
    menu.addAction(overwriteAct)

    setAsAct = QtGui.QAction("Set As...", menu)
    menu.addAction(setAsAct)
    setAsAct.triggered.connect(self.setAsTriggered)

    return menu

  def removeTriggered(self):
    # Make sure the user actually wants this
    dlg = QtWidgets.QMessageBox()
    confirm  = dlg.question(self, 'Remove Rows', 'Are you sure you want to remove these rows?',
                 dlg.Yes | dlg.Cancel)
    if confirm == dlg.Yes:
      # Proceed with operation
      idList = [idx.sibling(idx.row(), self.instIdColIdx).data(QtCore.Qt.EditRole) for idx in self.selectedIndexes()]
      # Since each selection represents a row, remove duplicate row indices
      idList = pd.unique(idList)
      self.mgr.rmComps(idList)
      self.clearSelection()

  def overwriteTriggered(self):
    # Make sure the user actually wants this
    dlg = QtWidgets.QMessageBox()
    warnMsg = f'This operation will overwrite *ALL* selected columns with the corresponding column values from'\
              f' the FIRST row in your selection. PLEASE USE CAUTION. Do you wish to proceed?'
    confirm  = dlg.question(self, 'Overwrite Rows', warnMsg, dlg.Yes | dlg.Cancel)
    if confirm != dlg.Yes:
      return
    idList, colIdxs = self.getIds_colsFromSelection()
    if len(idList) <= 1:
      return
    toOverwrite = self.mgr.compDf.loc[idList].copy()
    # Some bug is preventing the single assignment value from broadcasting
    setVals = [toOverwrite.iloc[0,colIdxs] for _ in range(len(idList)-1)]
    toOverwrite.iloc[1:, colIdxs] = setVals
    self.mgr.addComps(toOverwrite, addtype=ModelOpts.ADD_AS_MERGE)
    self.clearSelection()

  def getIds_colsFromSelection(self):
    selectedIdxs = self.selectedIndexes()
    idList = []
    colIdxs = []
    for idx in selectedIdxs:
      idAtIdx = idx.sibling(idx.row(), 0).data(QtCore.Qt.EditRole)
      idList.append(idAtIdx)
      colIdxs.append(idx.column())
    idList = pd.unique(idList)
    colIdxs = pd.unique(colIdxs)
    return idList, colIdxs


  def setAsTriggered(self):
    idList, colIdxs = self.getIds_colsFromSelection()
    dataToSet = self.mgr.compDf.iloc[[idList[0]],:].copy()
    self.popup.setData(dataToSet, colIdxs)
    wasAccepted = self.popup.exec()
    if wasAccepted:
      toOverwrite = self.mgr.compDf.loc[idList].copy()
      # Some bug is preventing the single assignment value from broadcasting
      overwriteData = self.popup.data
      setVals = [overwriteData.iloc[0,colIdxs] for _ in range(len(idList))]
      toOverwrite.iloc[:, colIdxs] = setVals
      self.mgr.addComps(toOverwrite, addtype=ModelOpts.ADD_AS_MERGE)

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