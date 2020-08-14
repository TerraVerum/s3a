from functools import partial
from typing import Sequence
from warnings import warn

import numpy as np
import pandas as pd
from pandas import DataFrame as df
from pyqtgraph.Qt import QtWidgets, QtCore, QtGui

from s3a import FR_SINGLETON
from s3a.models.tablemodel import FRComponentMgr
from s3a.projectvars import FR_CONSTS, FR_ENUMS, REQD_TBL_FIELDS
from s3a.structures import FRS3AException, FRS3AWarning, OneDArr, TwoDArr

__all__ = ['FRCompTableView']

from .parameditors import pgregistered, FRParamEditor, FRParamEditorDockGrouping
from .parameditors.table import genParamList
from ..graphicsutils import contextMenuFromEditorActions

Signal = QtCore.Signal

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
      else:
        self.tbl.showColumn(ii)
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

  @classmethod
  def __initEditorParams__(cls):
    cls.showOnCreate = FR_SINGLETON.generalProps.registerProp(cls, FR_CONSTS.PROP_SHOW_TBL_ON_COMP_CREATE)
    cls.toolsEditor = FRParamEditor.buildClsToolsEditor(cls, name='Component Table Tools')
    (cls.setCellsAsAct, cls.setSameAsFirstAct, cls.removeRowsAct) = cls.toolsEditor.registerProps(
      cls,[FR_CONSTS.TOOL_TBL_SET_AS, FR_CONSTS.TOOL_TBL_SET_SAME_AS_FIRST,
           FR_CONSTS.TOOL_TBL_DEL_ROWS], asProperty=False, ownerObj=cls)
    nameFilters = []
    for field in FR_SINGLETON.tableData.allFields:
      show = not field.opts.get('colHidden', False)
      nameFilters.append(dict(name=field.name, type='bool', value=show))
    FR_CONSTS.PROP_COLS_TO_SHOW.value = nameFilters
    cls.colsVisibleProps = FR_SINGLETON.generalProps.registerProp(
      cls, FR_CONSTS.PROP_COLS_TO_SHOW, asProperty=False)

    dockGroup = FRParamEditorDockGrouping([cls.toolsEditor, FR_SINGLETON.filter], 'Component Table')
    FR_SINGLETON.addDocks(dockGroup)

  def __init__(self, *args, minimal=False):
    """
    Creates the table.

    :param minimal: Whether to make a table with minimal features.
       If false, creates extensible table with context menu options.
       Otherwise, only contains minimal features.
    """
    super().__init__(*args)

    self._prevSelRows = np.array([])
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
      self.setCellsAsAct.sigActivated.connect(lambda: self.setSelectedCellsAs_gui())
      self.setSameAsFirstAct.sigActivated.connect(lambda: self.setSelectedCellsAsFirst())
      self.removeRowsAct.sigActivated.connect(lambda: self.removeSelectedRows_gui())
      for ii, child in enumerate(self.colsVisibleProps):
        child.sigValueChanged.connect(lambda param, value, idx=ii: self.setColumnHidden(idx, not value))
        # Trigger initial hide/show
        child.sigValueChanged.emit(child, child.value())


    self.instIdColIdx = TBL_FIELDS.index(REQD_TBL_FIELDS.INST_ID)

    for ii, field in enumerate(TBL_FIELDS):
      curType = field.pType
      curval = field.value
      paramDict = dict(type=curType, default=curval, **field.opts)
      if curType == 'Enum':
        paramDict['type'] = 'list'
        paramDict.update(values=list(type(curval)))
      elif curType == 'FRParam':
        paramDict['type'] = 'list'
        paramDict.update(values=list(curval.group))
      elif curType == 'bool':
        # TODO: Get checkbox to stay in table after editing for a smoother appearance.
        #   For now, the easiest solution is to just use dropdown
        paramDict['type'] = 'list'
        paramDict.update(values={'True': True, 'False': False})
      try:
        self.setItemDelegateForColumn(ii, pgregistered.FRPgParamDelegate(paramDict, self))
      except FRS3AException:
        # Parameter doesn't have a registered pyqtgraph editor, so default to
        # generic text editor
        paramDict['type'] = 'text'
        paramDict['default'] = str(curval)
        self.setItemDelegateForColumn(ii, pgregistered.FRPgParamDelegate(paramDict, self))

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
    newRows = pd.unique(selectedIds)
    if np.array_equal(newRows, self._prevSelRows):
      return
    self._prevSelRows = newRows
    self.sigSelectionChanged.emit(pd.unique(selectedIds))

  def createContextMenu(self):
    menu = contextMenuFromEditorActions(self.toolsEditor, 'Table Tools', self)
    return menu

  def removeSelectedRows_gui(self):
    if self.minimal: return

    idList = [idx.siblingAtColumn(self.instIdColIdx).data(QtCore.Qt.EditRole)
              for idx in self.selectedIndexes()]
    if len(idList) == 0:
      return
    # Make sure the user actually wants this
    dlg = QtWidgets.QMessageBox()
    confirm  = dlg.question(self, 'Remove Rows', 'Are you sure you want to remove these rows?',
                 dlg.Yes | dlg.Cancel)
    if confirm == dlg.Yes:
      # Proceed with operation
      # Since each selection represents a row, remove duplicate row indices
      idList = pd.unique(idList)
      self.mgr.rmComps(idList)
      self.clearSelection()

  def ids_rows_colsFromSelection(self, excludeNoEditCols=True, warnNoneSelection=True):
    """Returns Nx3 np array of (ids, rows, cols) from current table selection"""
    selectedIdxs = self.selectedIndexes()
    retLists = [] # (Ids, rows, cols)
    for idx in selectedIdxs:
      row = idx.row()
      # 0th row contains instance ID
      # TODO: If the user is allowed to reorder columns this needs to be revisited
      idAtIdx = idx.siblingAtColumn(self.instIdColIdx).data(QtCore.Qt.EditRole)
      retLists.append([idAtIdx, row, idx.column()])
    retLists = np.array(retLists)
    if excludeNoEditCols and len(retLists) > 0:
      # Set diff will eliminate any repeats, so use a slower op that at least preserves
      # duplicates
      retLists = retLists[~np.isin(retLists[:,2], self.mgr.noEditColIdxs)]
    if warnNoneSelection and len(retLists) == 0:
      warn('No editable columns selected.', FRS3AWarning)
    return retLists

  def setSelectedCellsAsFirst(self):
    selection = self.ids_rows_colsFromSelection()
    overwriteData = self.mgr.compDf.loc[selection[0,0]]
    self.setSelectedCellsAs(selection, overwriteData)

  def setSelectedCellsAs_gui(self, selectionIdxs: TwoDArr=None):
    if selectionIdxs is None:
      selectionIdxs = self.ids_rows_colsFromSelection()
    overwriteData = self.mgr.compDf.loc[[selectionIdxs[0,0]],:].copy()
    with FR_SINGLETON.actionStack.ignoreActions():
      self.popup.setData(overwriteData, pd.unique(selectionIdxs[:,2]))
      wasAccepted = self.popup.exec_()
    if not wasAccepted or len(self.popup.dirtyColIdxs) == 0:
      return

    selectionIdxs = selectionIdxs[np.isin(selectionIdxs[:,2], self.popup.dirtyColIdxs)]
    self.setSelectedCellsAs(selectionIdxs, self.popup.data)

  def setSelectedCellsAs(self, selectionIdxs: TwoDArr, overwriteData: df):
    """
    Overwrites the data from rows and cols with the information in *overwriteData*.
    Each (id, row, col) index is treated as a single index
    :param selectionIdxs: Selection idxs to overwrite. If *None*, defaults to
      current selection.
    :param overwriteData: What to fill in the overwrite locations. If *None*, a popup table
      is displayed and its data is used.
    """
    if self.minimal: return

    if len(selectionIdxs) == 0:
      return
    overwriteData = overwriteData.squeeze()
    uniqueIds = pd.unique(selectionIdxs[:,0])
    newDataDf = self.mgr.compDf.loc[uniqueIds].copy()
    # New data ilocs will no longer match, fix this using loc + indexed columns
    colsForLoc = self.mgr.compDf.columns[selectionIdxs[:,2]]
    for idxTriplet, colForLoc in zip(selectionIdxs, colsForLoc):
      newDataDf.at[idxTriplet[0], colForLoc] = overwriteData.iat[idxTriplet[2]]
    self.mgr.addComps(newDataDf, addtype=FR_ENUMS.COMP_ADD_AS_MERGE)