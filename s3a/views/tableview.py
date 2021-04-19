from functools import partial
from typing import Sequence
from warnings import warn

import numpy as np
import pandas as pd
from pandas import DataFrame as df
from pyqtgraph.Qt import QtWidgets, QtCore, QtGui

from s3a.constants import PRJ_CONSTS, REQD_TBL_FIELDS, PRJ_ENUMS
from s3a.models.tablemodel import ComponentMgr
from s3a.parameditors.singleton import PRJ_SINGLETON
from s3a.structures import TwoDArr

__all__ = ['CompTableView']

from utilitys import ParamEditor, EditorPropsMixin, RunOpts
from utilitys.params.pgregistered import PgParamDelegate

Signal = QtCore.Signal

TBL_FIELDS = PRJ_SINGLETON.tableData.allFields

class PopupTableDialog(QtWidgets.QDialog):
  def __init__(self, *args):
    super().__init__(*args)
    self.setModal(True)
    # -----------
    # Table View
    # -----------
    self.tbl = CompTableView(minimal=True)
    # Keeps track of which columns were edited by the user
    self.dirtyColIdxs = []

    # -----------
    # Warning Message
    # -----------
    self.reflectDelegateChange()
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

  def updateWarnMsg(self, updatableCols: Sequence[str]):
    warnMsg = 'Note! '
    if len(updatableCols) == 0:
      tblInfo = 'No information'
    else:
      tblInfo = "Only " + ", ".join(updatableCols)
    warnMsg += f'{tblInfo} will be updated from this view.'
    self.warnLbl.setText(warnMsg)

  def reflectDelegateChange(self):
    # TODO: Find if there's a better way to see if changes happen in a table
    self.titles = np.array(list([f.name for f in TBL_FIELDS]))
    self.tbl.setColDelegates()
    for colIdx in range(len(self.titles)):
      deleg = self.tbl.itemDelegateForColumn(colIdx)
      deleg.commitData.connect(partial(self.reflectDataChanged, colIdx))


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
    self.tbl.mgr.addComps(compDf, addtype=PRJ_ENUMS.COMP_ADD_AS_MERGE)
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

class CompTableView(EditorPropsMixin, QtWidgets.QTableView):
  __groupingName__ = 'Component Table'
  """
  Table for displaying :class:`ComponentMgr` data.
  """
  sigSelectionChanged = Signal(object)

  @classmethod
  def __initEditorParams__(cls):
    cls.showOnCreate = PRJ_SINGLETON.generalProps.registerProp(PRJ_CONSTS.PROP_SHOW_TBL_ON_COMP_CREATE)
    cls.toolsEditor = ParamEditor.buildClsToolsEditor(cls, name='Component Table Tools')

  def __init__(self, *args, minimal=False):
    """
    Creates the table.

    :param minimal: Whether to make a table with minimal features.
       If false, creates extensible table with context menu options.
       Otherwise, only contains minimal features.
    """
    super().__init__(*args)

    self.setStyleSheet("QTableView { selection-color: white; selection-background-color: #0078d7; }")
    self._prevSelRows = np.array([])
    self.setSortingEnabled(True)

    self.mgr = ComponentMgr()
    self.minimal = minimal
    self.setModel(self.mgr)
    self.setColDelegates()

    with PRJ_SINGLETON.generalProps.setBaseRegisterPath(self.__groupingName__):
      proc, params = PRJ_SINGLETON.generalProps.registerFunc(
        self.setVisibleColumns, runOpts=RunOpts.ON_CHANGED, nest=False,
        returnParam=True, visibleColumns=[])
    def onChange(*_args):
      params.child('visibleColumns').setLimits([f.name for f in TBL_FIELDS])
    onChange()
    PRJ_SINGLETON.tableData.sigCfgUpdated.connect(onChange)
    proc.run(**params)

    if not minimal:
      self.popup = PopupTableDialog(*args)
      # Create context menu for changing table rows
      self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
      cursor = QtGui.QCursor()
      self.customContextMenuRequested.connect(lambda: self.menu.exec_(cursor.pos()))

    self.instIdColIdx = TBL_FIELDS.index(REQD_TBL_FIELDS.INST_ID)

  def setVisibleColumns(self, visibleColumns: Sequence[str]):
    """
    Determines which columns to show. All unspecified columns will be hidden.
    :param visibleColumns:
      pType: checklist
    """
    for ii, col in enumerate(self.mgr.colTitles):
      self.setColumnHidden(ii, col not in visibleColumns)

  def setColDelegates(self):
    for ii, field in enumerate(TBL_FIELDS):
      curType = field.pType.lower()
      curval = field.value
      paramDict = dict(type=curType, default=curval, **field.opts)
      if curType == 'enum':
        paramDict['type'] = 'list'
        paramDict.update(values=list(type(curval)))
      elif curType == 'prjparam':
        paramDict['type'] = 'list'
        paramDict.update(values=list(curval.group))
      elif curType == 'bool':
        # TODO: Get checkbox to stay in table after editing for a smoother appearance.
        #   For now, the easiest solution is to just use dropdown
        paramDict['type'] = 'list'
        paramDict.update(values={'True': True, 'False': False})
      try:
        self.setItemDelegateForColumn(ii, PgParamDelegate(paramDict, self))
      except TypeError:
        # Parameter doesn't have a registered pyqtgraph editor, so default to
        # generic text editor
        paramDict['type'] = 'text'
        paramDict['default'] = str(curval)
        self.setItemDelegateForColumn(ii, PgParamDelegate(paramDict, self))

    self.horizontalHeader().setSectionsMovable(True)

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
    newRows = pd.unique(selectedIds)
    if np.array_equal(newRows, self._prevSelRows):
      return
    self._prevSelRows = newRows
    self.sigSelectionChanged.emit(pd.unique(selectedIds))

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
      warn('No editable columns selected.', UserWarning)
    return retLists

  def setSelectedCellsAsFirst(self):
    """
    Sets all cells in the selection to be the same as the first row in the selection.
    See the project wiki for a detailed description
    """
    selection = self.ids_rows_colsFromSelection()
    overwriteData = self.mgr.compDf.loc[selection[0,0]]
    self.setSelectedCellsAs(selection, overwriteData)

  def setSelectedCellsAs_gui(self, selectionIdxs: TwoDArr=None):
    """
    Sets all cells in the selection to the values specified in the popup table. See
    the project wiki for a detailed description
    """
    if selectionIdxs is None:
      selectionIdxs = self.ids_rows_colsFromSelection()
    overwriteData = self.mgr.compDf.loc[[selectionIdxs[0,0]]].copy()
    with PRJ_SINGLETON.actionStack.ignoreActions():
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
    self.mgr.addComps(newDataDf, addtype=PRJ_ENUMS.COMP_ADD_AS_MERGE)