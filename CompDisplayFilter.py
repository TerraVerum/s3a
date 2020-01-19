# Required to avoid cyclic dependency from CompTable annotation
from __future__ import annotations

import pyqtgraph as pg

from ABGraphics import tableview
from ABGraphics.clickables import ClickableTextItem
from ABGraphics.parameditors import SchemeEditor, TableFilterEditor
from ABGraphics.regions import MultiRegionPlot
from constants import (ComponentTypes)
from tablemodel import *

Signal = QtCore.pyqtSignal
Slot = QtCore.pyqtSlot
# Ensure an application instance is running
app = pg.mkQApp()

class CompDisplayFilter(QtCore.QObject):
  sigCompClicked = Signal(object)

  def __init__(self, compMgr: ComponentMgr, mainImg: pg.PlotWindow,
               compTbl: tableview.CompTableView, filterEditor: TableFilterEditor):
    super().__init__()
    self._mainImgArea = mainImg
    self._filter = filterEditor.params.getValues()
    self._compTbl = compTbl
    self._compMgr = compMgr

    # Attach to manager signals
    self._compMgr.sigCompsChanged.connect(self.redrawComps)

    # Retrieve filter changes
    filterEditor.sigParamStateUpdated.connect(self._updateFilter)

    self._compBounds = MultiRegionPlot()
    self._displayedIds = np.array([], dtype=int)
    # Keep copy of old id plots to delete when they are removed from compMgr
    # No need to keep vertices, since deleted vertices are handled by the
    # MultiRegionPlot and table rows will still have the associated ID
    self._oldPlotsDf = df(columns=['idPlot'])
    self._oldPlotsDf.index.set_names([TC.INST_ID.name], inplace=True)

    mainImg.addItem(self._compBounds)

  def redrawComps(self, idLists):
    # Following mix of cases are possible:
    # Components: DELETED, UNCHANGED, CHANGED, NEW
    # New is different from changed since id plot already exists (unhide vs. create)
    # Plots: DRAWN, UNDRAWN
    # Note that hiding the ID is chosen instead of deleting, since that is a costly graphics
    # operation
    id_indexDf = self._compMgr.compDf

    # For new components: Add hidden id plot. This will be shown later if filter allows
    addedIds = idLists['added']
    verts = id_indexDf.loc[addedIds, TC.VERTICES.name].values
    valids = id_indexDf.loc[addedIds, TC.VALIDATED.name].values
    newIdPlots = [None]*len(addedIds)
    for pltIdx, (curId, curVerts, curValid) in enumerate(zip(addedIds, verts, valids)):
      newPlt = self._createIdPlot(curId, curVerts, curValid)
      # noinspection PyTypeChecker
      newIdPlots[pltIdx] = newPlt
    newIdPlots_Df = df(newIdPlots, index=addedIds, columns=self._oldPlotsDf.columns)
    self._oldPlotsDf = pd.concat((self._oldPlotsDf, newIdPlots_Df), sort=False)

    # Hide all other ids and table rows, since they will be reshown as needed after display filtering
    for rowIdx, plt in enumerate(self._oldPlotsDf['idPlot']):
      plt.hide()
      self._compTbl.hideRow(rowIdx)

    # Component deleted: Delete hidden id plot
    idsToRm = idLists['deleted']
    pltsToRm = self._oldPlotsDf['idPlot'].loc[idsToRm].dropna()
    for plt in pltsToRm.values:
      self._mainImgArea.removeItem(plt)

    # Component changed: update text plot
    # No need to update regions, since the whole list is reset at the end of
    # this function
    idsToChange = idLists['changed']
    changedVerts = id_indexDf.loc[idsToChange, TC.VERTICES.name]
    changedValid = id_indexDf.loc[idsToChange, TC.VALIDATED.name]
    plotsToChange = self._oldPlotsDf['idPlot'].loc[idsToChange]
    for curId, curVerts, curValid, idPlot in \
        zip(idsToChange, changedVerts, changedValid, plotsToChange):
      idPlot.update(str(curId), curVerts, curValid)

    # Update filter list: hide/unhide ids and verts as needed. This should occur in other
    # functions that hook into signals sent from filter widget
    # TODO: Implement this logic. Will be placed in self._displayedIds
    # Only plot shown vertices
    self._populateDisplayedIds()

    pltsToShow = self._oldPlotsDf.loc[self._displayedIds, 'idPlot']
    tblIdxsToShow = np.nonzero(np.in1d(id_indexDf.index, self._displayedIds))[0]
    for plt in pltsToShow:
      plt.show()
    for rowIdx in tblIdxsToShow:
      self._compTbl.showRow(rowIdx)

    displayVerts = id_indexDf.loc[self._displayedIds, TC.VERTICES.name]
    self._compBounds.resetRegionList(self._displayedIds, displayVerts)

  def _updateFilter(self, newFilterDict):
    self._filter = newFilterDict
    self.redrawComps(self._compMgr.defaultEmitDict)

  def _populateDisplayedIds(self):
    curComps = self._compMgr.compDf

    # idx 0 = value, 1 = children
    # ------
    # ID FILTERING
    # ------
    curParam = self._filter[TC.INST_ID.name][1]
    curmin, curmax = [curParam[name][0] for name in ['min', 'max']]

    idList = curComps.index
    curComps = curComps.loc[(idList >= curmin) & (idList <= curmax),:]

    # ------
    # VALIDATED FILTERING
    # ------
    curParam = self._filter[TC.VALIDATED.name][1]
    allowValid, allowInvalid = [curParam[name][0] for name in ['Validated', 'Not Validated']]

    validList = np.array(curComps.loc[:, TC.VALIDATED.name], dtype=bool)
    if not allowValid:
      curComps = curComps.loc[~validList, :]
    if not allowInvalid:
      curComps = curComps.loc[validList, :]

    # ------
    # DEVICE TYPE FILTERING
    # ------
    compTypes = np.array(curComps.loc[:, TC.DEV_TYPE.name])
    curParam = self._filter[TC.DEV_TYPE.name][1]
    allowedTypes = []
    for curType in ComponentTypes:
      isAllowed = curParam[curType.value][0]
      if isAllowed:
        allowedTypes.append(curType)
    curComps = curComps.loc[np.isin(compTypes, allowedTypes),:]

    # ------
    # LOGO, NOTES, BOARD, DEVICE TEXT FILTERING
    # ------
    nextParamNames = [param.name for param in [TC.LOGO, TC.NOTES, TC.BOARD_TEXT, TC.DEV_TEXT]]
    for param in nextParamNames:
      compParamVals = curComps.loc[:, param]
      allowedRegex = self._filter[param][0]
      isCompAllowed = compParamVals.str.contains(allowedRegex, regex=True, case=False)
      curComps = curComps.loc[isCompAllowed,:]

    # ------
    # VERTEX FILTERING
    # ------
    compVerts = curComps.loc[:, TC.VERTICES.name]
    vertsAllowed = np.ones(len(compVerts), dtype=bool)

    vertParam = self._filter[TC.VERTICES.name][1]
    xParam = vertParam['X Bounds'][1]
    yParam = vertParam['Y Bounds'][1]
    xmin, xmax, ymin, ymax = [param[val][0] for param in (xParam, yParam) for val in ['min', 'max']]

    for vertIdx, verts in enumerate(compVerts):
      xVerts = verts[:,0]
      yVerts = verts[:,1]
      isAllowed = np.all((xVerts >= xmin) & (xVerts <= xmax)) & \
                  np.all((yVerts >= ymin) & (yVerts <= ymax))
      vertsAllowed[vertIdx] = isAllowed
    curComps = curComps.loc[vertsAllowed,:]

    # Give self the id list of surviving comps
    self._displayedIds = curComps.index

  @Slot()
  def resetCompBounds(self):
    self._compBounds.resetRegionList()

  def _createIdPlot(self, instId, verts, validated):
    idPlot = ClickableTextItem()
    idPlot.sigClicked.connect(self._rethrowCompClick)
    idPlot.update(str(instId), verts, validated)
    self._mainImgArea.addItem(idPlot)
    return idPlot

  @Slot()
  def _rethrowCompClick(self):
    # noinspection PyTypeChecker
    idPlot: ClickableTextItem = self.sender()
    clickedId = int(idPlot.textItem.toPlainText())
    self.sigCompClicked.emit(self._compMgr.compDf.loc[clickedId,:])

  @staticmethod
  def setScheme(scheme: SchemeEditor):
    # Pass this scheme to drawing elements
    MultiRegionPlot.setScheme(scheme)
    ClickableTextItem.setScheme(scheme)

if __name__ == '__main__':
  from PIL import Image
  from ABGraphics.tableview import CompTableView
  mw = CompTableView()
  item = pg.ImageItem(np.array(Image.open('./Images/fast.tif')))
  #mw.addItem(item)
  #mw.setAspectLocked(True)
  mgr = ComponentMgr()
  mw.setModel(mgr)
  mw.show()

  c = makeCompDf(5)
  for ii in range(len(c)):
    c.loc[ii,TC.VERTICES.name] = [np.random.randint(0,100,size=(10,2))]
    c.loc[ii,TC.INST_ID.name] = 5
    c.loc[ii,TC.BOARD_TEXT.name] = 'test'
  mgr.addComps(c)

  newComp = makeCompDf(1)
  newComp[TC.INST_ID.name] = 100
  mgr.addComps(newComp)

  app.exec()