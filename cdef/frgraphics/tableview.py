from functools import partial
from typing import Sequence

import numpy as np
import pandas as pd
import pyqtgraph as pg
from pandas import DataFrame as df
from pyqtgraph.Qt import QtWidgets, QtCore, QtGui

from cdef import FR_SINGLETON
from cdef.projectvars import FR_CONSTS, FR_ENUMS, REQD_TBL_FIELDS
from cdef.tablemodel import FRCompTableModel, FRComponentMgr

Slot = QtCore.pyqtSlot
Signal = QtCore.pyqtSignal

TBL_FIELDS = FR_SINGLETON.tableData.allFields

class FRPopupTableDialog(QtWidgets.QDialog):
  def __init__(self, *args):
    super().__init__(*args)
    self.setModal(True)
    # -----------
    # Table View
    # -----------
    self.tbl = FRCompTableView(minimal=True)
    # Keeps track of which columns were edited by the user
    self.dirtyColIdxs = []

    # -----------
    # Warning Message
    # -----------
    self.titles = np.array(self.tbl.mgr.colTitles)
    self.warnLbl = QtWidgets.QLabel(self)
    self.warnLbl.setStyleSheet("font-weight: bold; color:red; font-size:14")
    self.updateWarnMsg([])

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
    # TODO: Find if there's a better way to see if changes happen in a table
    for colIdx in range(len(self.titles)):
      deleg = self.tbl.itemDelegateForColumn(colIdx)
      deleg.commitData.connect(partial(self.reflectDataChanged, colIdx))

  def updateWarnMsg(self, updatableCols: Sequence[str]):
    warnMsg = 'Note! '
    if len(updatableCols) == 0:
      tblInfo = 'No information'
    else:
      tblInfo = "Only " + ", ".join(updatableCols)
    warnMsg += f'{tblInfo} will be updated from this view.'
    self.warnLbl.setText(warnMsg)

  @property
  def data(self):
    return self.tbl.mgr.compDf.iloc[[0],:]

  def setData(self, compDf: df, colIdxs: Sequence):
    # New selection, so reset dirty columns
    self.dirtyColIdxs = []
    # Hide columns that weren't selected by the user since these changes
    # Won't propagate
    for ii in range(len(self.titles)):
      if ii not in colIdxs:
        self.tbl.hideColumn(ii)
    self.tbl.mgr.rmComps()
    self.tbl.mgr.addComps(compDf, addtype=FR_ENUMS.COMP_ADD_AS_MERGE)
    self.updateWarnMsg([])

  def reflectDataChanged(self, editedColIdx: int):
    """Responds to user editing the value in a cell by setting a dirty bit for that column"""
    if editedColIdx not in self.dirtyColIdxs:
      self.dirtyColIdxs.append(editedColIdx)
      self.dirtyColIdxs.sort()
      self.updateWarnMsg(self.titles[self.dirtyColIdxs])

  def reject(self):
    # On dialog close be sure to unhide all columns / reset dirty cols
    self.dirtyColIdxs = []
    for ii in range(len(self.titles)):
      self.tbl.showColumn(ii)
    super().reject()

@FR_SINGLETON.registerGroup(FR_CONSTS.CLS_COMP_TBL)
class FRCompTableView(QtWidgets.QTableView):
  """
  Table for displaying :class:`FRComponentMgr` data.
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

    self.mgr = FRComponentMgr()
    self.setModel(self.mgr)

    self.minimal = minimal
    if not minimal:
      self.popup = FRPopupTableDialog(*args)
      # Create context menu for changing table rows
      self.menu = self.createContextMenu()
      self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
      cursor = QtGui.QCursor()
      self.customContextMenuRequested.connect(lambda: self.menu.exec_(cursor.pos()))


    # Default to text box delegate
    # self.setItemDelegate(FRTextDelegate(self))

    validOpts = [True, False]
    boolDelegate = FRComboBoxDelegate(self, comboValues=validOpts)

    self.instIdColIdx = TBL_FIELDS.index(REQD_TBL_FIELDS.INST_ID)

    for ii, field in enumerate(TBL_FIELDS):
      curType = field.valType
      curval = field.value
      if curType == 'bool':
        self.setItemDelegateForColumn(ii, boolDelegate)
      elif curType == 'Enum':
        self.setItemDelegateForColumn(ii, FRComboBoxDelegate(self, comboValues=list(type(curval))))
      elif curType == 'FRParam':
        self.setItemDelegateForColumn(ii, FRComboBoxDelegate(self, comboValues=list(curval.group)))
      elif curType in ['int', 'float']:
        self.setItemDelegateForColumn(ii, FRSpinBoxDelegate(self, spinType=curType))
      else:
        self.setItemDelegateForColumn(ii, FRTextDelegate(self))

    self.horizontalHeader().setSectionsMovable(True)

  # When the model is changed, get a reference to the FRComponentMgr
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

  def createContextMenu(self):
    menu = QtWidgets.QMenu(self)
    menu.setTitle('Table Actions')

    remAct = QtWidgets.QAction("Remove", menu)
    remAct.triggered.connect(self.removeTriggered)
    menu.addAction(remAct)

    overwriteAct = QtWidgets.QAction("Set Same As First", menu)
    overwriteAct.triggered.connect(self.overwriteTriggered)
    menu.addAction(overwriteAct)

    setAsAct = QtWidgets.QAction("Set As...", menu)
    menu.addAction(setAsAct)
    setAsAct.triggered.connect(self.setAsTriggered)

    return menu

  @FR_SINGLETON.shortcuts.registerMethod(FR_CONSTS.SHC_TBL_DEL_ROWS)
  def removeTriggered(self):
    if self.minimal: return

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


  @FR_SINGLETON.shortcuts.registerMethod(FR_CONSTS.SHC_TBL_SET_SAME_AS_FIRST)
  def overwriteTriggered(self):
    if self.minimal: return

    # Make sure the user actually wants this
    dlg = QtWidgets.QMessageBox()
    warnMsg = f'This operation will overwrite *ALL* selected columns with the corresponding column values from'\
              f' the FIRST row in your selection. PLEASE USE CAUTION. Do you wish to proceed?'
    confirm  = dlg.question(self, 'Overwrite Rows', warnMsg, dlg.Yes | dlg.Cancel)
    if confirm != dlg.Yes:
      return
    idList, colIdxs = self.getIds_colsFromSelection()
    if idList is None:
      return
    toOverwrite = self.mgr.compDf.loc[idList].copy()
    # Some bug is preventing the single assignment value from broadcasting
    setVals = [toOverwrite.iloc[0,colIdxs] for _ in range(len(idList)-1)]
    toOverwrite.iloc[1:, colIdxs] = setVals
    self.mgr.addComps(toOverwrite, addtype=FR_ENUMS.COMP_ADD_AS_MERGE)
    self.clearSelection()

  def getIds_colsFromSelection(self):
    selectedIdxs = self.selectedIndexes()
    idList = []
    colIdxs = []
    for idx in selectedIdxs:
      # 0th row contains instance ID
      # TODO: If the user is allowed to reorder columns this needs to be revisited
      idAtIdx = idx.sibling(idx.row(), 0).data(QtCore.Qt.EditRole)
      idList.append(idAtIdx)
      colIdxs.append(idx.column())
    idList = pd.unique(idList)
    colIdxs = pd.unique(colIdxs)
    colIdxs = np.setdiff1d(colIdxs, self.model().sourceModel().noEditColIdxs)
    if len(idList) == 0 or len(colIdxs) == 0:
      QtWidgets.QMessageBox.information(self, 'No Editable Data',
                                        'No editable columns selected. Nothing to do.',
                                        QtWidgets.QMessageBox.Ok)
      return None, None
    return idList, colIdxs

  @FR_SINGLETON.shortcuts.registerMethod(FR_CONSTS.SHC_TBL_SET_AS)
  def setAsTriggered(self):
    if self.minimal: return

    idList, colIdxs = self.getIds_colsFromSelection()
    if idList is None:
      return

    dataToSet = self.mgr.compDf.loc[[idList[0]],:].copy()
    with FR_SINGLETON.undoStack.ignoreActions():
      self.popup.setData(dataToSet, colIdxs)
      wasAccepted = self.popup.exec()
    if wasAccepted and len(self.popup.dirtyColIdxs) > 0:
      toOverwrite = self.mgr.compDf.loc[idList].copy()
      # Only overwrite dirty columns from popup

      colIdxs = np.intersect1d(colIdxs, self.popup.dirtyColIdxs)
      # Some bug is preventing the single assignment value from broadcasting
      overwriteData = self.popup.data
      setVals = [overwriteData.iloc[0,colIdxs] for _ in range(len(idList))]
      toOverwrite.iloc[:, colIdxs] = setVals
      self.mgr.addComps(toOverwrite, addtype=FR_ENUMS.COMP_ADD_AS_MERGE)

class FRTextDelegate(QtWidgets.QItemDelegate):
  def createEditor(self, parent, option, index):
    editor = QtWidgets.QPlainTextEdit(parent)
    editor.setTabChangesFocus(True)
    return editor

  def setEditorData(self, editor: QtWidgets.QPlainTextEdit, index):
    text = index.data(QtCore.Qt.DisplayRole)
    editor.setPlainText(text)

  def setModelData(self, editor: QtWidgets.QTextEdit,
                   model: FRCompTableModel,
                   index: QtCore.QModelIndex):
    model.setData(index, editor.toPlainText())

  def updateEditorGeometry(self, editor: QtWidgets.QPlainTextEdit,
                           option: QtWidgets.QStyleOptionViewItem,
                           index: QtCore.QModelIndex):
    editor.setGeometry(option.rect)


class FRComboBoxDelegate(QtWidgets.QStyledItemDelegate):
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

  def setEditorData(self, editor: QtWidgets.QComboBox, index: QtCore.QModelIndex):
    curVal = index.data(QtCore.Qt.DisplayRole)
    editor.setCurrentIndex(self.comboNames.index(curVal))

  def setModelData(self, editor: QtWidgets.QComboBox,
                   model: FRCompTableModel,
                   index: QtCore.QModelIndex):
    model.setData(index, self.comboValues[editor.currentIndex()])

  def updateEditorGeometry(self, editor: QtWidgets.QPlainTextEdit,
                           option: QtWidgets.QStyleOptionViewItem,
                           index: QtCore.QModelIndex):
    editor.setGeometry(option.rect)

class FRSpinBoxDelegate(QtWidgets.QStyledItemDelegate):
  def __init__(self, parent=None, spinType = 'int'):
    super().__init__(parent)
    self.defs = {
      'value': 0, 'min': None, 'max': None,
      'step': 1.0,
      'siPrefix': False, 'suffix': '', 'decimals': 3,
    }
    if spinType == 'int':
      self.defs['minStep'] = 1.0
    else:
      self.defs['step'] = 0.1
    self.defs['dec'] = spinType != 'int'
    self.defs['int'] = spinType == 'int'


  def createEditor(self, parent, option, index: QtCore.QModelIndex):
    editor = pg.SpinBox(parent, **self.defs)
    editor.setMaximumHeight(editor.maximumWidth())
    return editor

  def setEditorData(self, editor: pg.SpinBox, index: QtCore.QModelIndex):
    curVal = index.data(QtCore.Qt.DisplayRole)
    editor.setValue(curVal)

  def setModelData(self, editor: pg.SpinBox,
                   model: FRCompTableModel,
                   index: QtCore.QModelIndex):
    model.setData(index, editor.value())

  def updateEditorGeometry(self, editor: QtWidgets.QPlainTextEdit,
                           option: QtWidgets.QStyleOptionViewItem,
                           index: QtCore.QModelIndex):
    editor.setGeometry(option.rect)