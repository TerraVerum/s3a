# -*- coding: utf-8 -*-

import sys
from functools import partial
from os.path import join
from pathlib import Path
from typing import Callable

import pyqtgraph as pg
from pandas import DataFrame as df
from pyqtgraph.Qt import QtCore, QtWidgets, QtGui

from .frgraphics.annotator_ui import FRAnnotatorUI
from .frgraphics.graphicsutils import applyWaitCursor, dialogSaveToFile, addDirItemsToMenu, \
  attemptLoadSettings, popupFilePicker, disableAppDuringFunc, dialogGetAuthorName
from .frgraphics.parameditors import FRParamEditor, FR_SINGLETON
from .projectvars.constants import FR_CONSTS, ANN_AUTH_DIR
from .projectvars.constants import LAYOUTS_DIR, TEMPLATE_COMP as TC
from .projectvars.enums import FR_ENUMS
from .tablemodel import ComponentMgr as ComponentMgr, makeCompDf
from .tableviewproxy import CompDisplayFilter, CompSortFilter

Slot = QtCore.pyqtSlot
Signal = QtCore.pyqtSignal

# Configure pg to correctly read image dimensions
pg.setConfigOptions(imageAxisOrder='row-major')

@FR_SINGLETON.registerClass(FR_CONSTS.CLS_ANNOTATOR)
class MainWindow(FRAnnotatorUI):
  """
  Top-level widget for producing component bounding boxes from an input image.
  """
  # Alerts GUI that a layout (either new or overwriting old) was saved
  sigLayoutSaved = Signal()

  def __init__(self, startImgFpath=None, authorName:str=None):
    super().__init__()
    
    # ---------------
    # DATA ATTRIBUTES
    # ---------------
    self.mainImgFpath = None
    if startImgFpath is not None:
      # Make sure to simplify the incoming path
      self.mainImgFpath = str(Path(startImgFpath).resolve())
    self.hasUnsavedChanges = False


    self.statBar = QtWidgets.QStatusBar(self)
    self.setStatusBar(self.statBar)
    #self.setWindowIcon(QtGui.QIcon(BASE_DIR + './ficsLogo.png'))
    # Flesh out pg components
    # ---------------
    # MAIN IMAGE
    # ---------------
    self.mainImg.sigComponentCreated.connect(self._add_focusComp)
    self.mainImg.setImage(startImgFpath)
    # ---------------
    # FOCUSED IMAGE
    # ---------------

    # ---------------
    # COMPONENT MANAGER
    # ---------------
    self.compMgr = ComponentMgr()
    self.compMgr.sigCompsChanged.connect(self._recordCompChange)

    # Allow filtering/sorting
    self.sortFilterProxy = CompSortFilter(self.compMgr, self)

    self.compTbl.setModel(self.sortFilterProxy)

    # ---------------
    # COMPONENT DISPLAY FILTER
    # ---------------
    self.compDisplay = CompDisplayFilter(self.compMgr, self.mainImg, self.compTbl,
                                         self)

    self.mainImg.imgItem.sigImageChanged.connect(self.clearBoundaries)
    self.compDisplay.sigCompClicked.connect(self.updateCurComp)

    # ---------------
    # UI ELEMENT SIGNALS
    # ---------------
    # Buttons
    self.openImgAct.triggered.connect(self.openImgActionTriggered)
    self.clearRegionBtn.clicked.connect(self.clearRegionBtnClicked)
    self.resetRegionBtn.clicked.connect(self.resetRegionBtnClicked)
    self.acceptRegionBtn.clicked.connect(self.acceptRegionBtnClicked)


    # Dropdowns
    # self.addRmCombo.currentIndexChanged.connect(self.addRmComboChanged)

    # Checkboxes
    # self.allowEditsChk.stateChanged.connect(self.allowEditsChkChanged)


    # Same with estimating boundaries
    if startImgFpath is not None \
        and self.estBoundsOnStart:
      self.estimateBoundaries()

    # Menu options
    # FILE
    self.saveLayout.triggered.connect(self.saveLayoutActionTriggered)
    self.sigLayoutSaved.connect(self.populateLoadLayoutOptions)

    self.saveComps.triggered.connect(self.saveCompsActionTriggered)
    self.loadComps_merge.triggered.connect(lambda: self.loadCompsActionTriggered(FR_ENUMS.COMP_ADD_AS_MERGE))
    self.loadComps_new.triggered.connect(lambda: self.loadCompsActionTriggered(FR_ENUMS.COMP_ADD_AS_NEW))

    # SETTINGS
    self.createSettingsMenus()

    # ---------------
    # LOAD LAYOUT OPTIONS
    # ---------------
    self.populateLoadLayoutOptions()
    # Start with docks in default position, hide error if default file doesn't exist
    self.loadLayoutActionTriggered('Default', showError=False)

    # Placing in a single shot timer ensures the app has enough time to load and assess screen
    # dimensions before resizing. Otherwise, the maximize doesn't work properly
    QtCore.QTimer.singleShot(0, self.showMaximized)
    # self.showMaximized()
    if authorName is None:
      authorName = self.getAuthorName()
    FR_SINGLETON.annotationAuthor = authorName
    self.statBar.showMessage(FR_SINGLETON.annotationAuthor)

  # -----------------------------
  # MainWindow CLASS FUNCTIONS
  # -----------------------------
  def getAuthorName(self):
    annPath = Path(ANN_AUTH_DIR)
    annFile = annPath.joinpath('defaultAuthor.txt')
    quitApp, name = dialogGetAuthorName(self, annFile)

    if quitApp:
      sys.exit(0)

    with open(annFile, 'w') as ofile:
      ofile.write(name)
    return name

  @FR_SINGLETON.generalProps.registerProp(FR_CONSTS.PROP_EST_BOUNDS_ON_START)
  def estBoundsOnStart(self): pass

  def closeEvent(self, ev: QtGui.QCloseEvent):
    # Confirm all components have been saved
    shouldExit = False
    if self.hasUnsavedChanges:
      ev.ignore()
      if (QtWidgets.QMessageBox.question(self, 'Confirm Exit',
          'Component table has unsaved changes.\nAre you sure you want to exit?',
          QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Cancel)
        == QtWidgets.QMessageBox.Ok):
        shouldExit = True
    else:
      shouldExit = True
    if shouldExit:
      # Clean up all editor windows, which could potentially be left open
      ev.accept()
      FR_SINGLETON.close()


  def createSettingsMenus(self):
    for editor, name in zip(FR_SINGLETON.editors, FR_SINGLETON.editorNames): \
        #type: FRParamEditor, str
      menu = QtWidgets.QMenu(name, self)
      editAct = QtWidgets.QAction('Edit ' + name, self)
      menu.addAction(editAct)
      menu.addSeparator()
      editAct.triggered.connect(editor.show)
      loadFunc = partial(self.genericLoadActionTriggered, editor)
      populateFunc = partial(self.genericPopulateMenuOptions, editor, menu, loadFunc)
      editor.sigParamStateCreated.connect(populateFunc)
      # Initialize default menus
      populateFunc()
      self.menuSettings.addMenu(menu)

  @Slot(object)
  def _recordCompChange(self):
    self.hasUnsavedChanges = True

  @Slot(object)
  def _add_focusComp(self, newComp):
    self.compMgr.addComps(newComp)
    # Make sure index matches ID before updating current component
    newComp = newComp.set_index(TC.INST_ID, drop=False)
    # Set this component as active in the focused view
    self.updateCurComp(newComp.squeeze())

  # ---------------
  # MENU CALLBACKS
  # ---------------

  @Slot()
  @applyWaitCursor
  def openImgActionTriggered(self):
    fileFilter = "Image Files (*.png; *.tif; *.jpg; *.jpeg; *.bmp; *.jfif);; All files(*.*)"
    fname = popupFilePicker(self, 'Select Main Image', fileFilter)

    if fname is not None:
      self.compMgr.rmComps()
      self.mainImg.setImage(fname)
      self.mainImgFpath = fname
      self.compImg.resetImage()
      if self.estBoundsOnStart:
        self.estimateBoundaries()

  def populateLoadLayoutOptions(self):
    layoutGlob = join(LAYOUTS_DIR, '*.dockstate')
    addDirItemsToMenu(self.menuLayout, layoutGlob, self.loadLayoutActionTriggered)

  @Slot(str)
  def loadLayoutActionTriggered(self, layoutName, showError=True):
    layoutFilename = join(LAYOUTS_DIR, f'{layoutName}.dockstate')
    dockStates = attemptLoadSettings(layoutFilename, showErrorOnFail=showError)
    if dockStates is not None:
      self.restoreState(dockStates)

  @Slot()
  def saveLayoutActionTriggered(self):
    dockStates = self.saveState()
    dialogSaveToFile(self, dockStates, 'Layout Name', LAYOUTS_DIR, 'dockstate')
    self.sigLayoutSaved.emit()

  @Slot()
  def saveCompsActionTriggered(self):
    onlyExportFiltered = self.compMgr.exportOnlyVis
    if onlyExportFiltered:
      exportIds = self.compDisplay.displayedIds
    else:
      exportIds = FR_ENUMS.COMP_EXPORT_ALL
    fileDlg = QtWidgets.QFileDialog()
    fileFilter = "CSV Files (*.csv)"
    fname, _ = fileDlg.getSaveFileName(self, 'Select Save File', '', fileFilter)
    if len(fname) > 0:
      self.compMgr.csvExport(fname, self.mainImgFpath, exportIds)
      self.hasUnsavedChanges = False

  def loadCompsActionTriggered(self, loadType=FR_ENUMS.COMP_ADD_AS_NEW):
    fileFilter = "CSV Files (*.csv)"
    fname = popupFilePicker(self, 'Select Load File', fileFilter)
    if fname is not None:
      # Operation may take a long time, but we don't want to start the wait cursor until
      # after dialog selection
      err = applyWaitCursor(self.compMgr.csvImport)(fname, loadType,
                                                    self.mainImg.image.shape)
      if err is not None:
        # Something went wrong. Inform the user.
        errMsg = f'Failed to import components. {type(err)}:\n{err}'
        QtWidgets.QMessageBox().information(self, 'Error During Import', errMsg)

  @staticmethod
  def genericPopulateMenuOptions(objForMenu: FRParamEditor, winMenu: QtWidgets.QMenu, triggerFn: Callable):
    addDirItemsToMenu(winMenu,
                      join(objForMenu.saveDir, f'*.{objForMenu.fileType}'),
                      triggerFn)

  @staticmethod
  def genericLoadActionTriggered(objForMenu: FRParamEditor, nameToLoad: str):
    dictFilename = join(objForMenu.saveDir, f'{nameToLoad}.{objForMenu.fileType}')
    loadDict = attemptLoadSettings(dictFilename)
    if loadDict is None:
      return
    objForMenu.loadState(loadDict)
    objForMenu.applyBtnClicked()

  # ---------------
  # BUTTON CALLBACKS
  # ---------------
  # Push buttons
  @Slot()
  def clearRegionBtnClicked(self):
    # Reset drawn comp vertices to nothing
    # Only perform action if image currently exists
    if self.compImg.imgItem.image is None:
      return
    self.compImg.updateRegionFromVerts(None)

  @Slot()
  def resetRegionBtnClicked(self):
    # Reset drawn comp vertices to nothing
    # Only perform action if image currently exists
    if self.compImg.imgItem.image is None:
      return
    self.compImg.updateRegionFromVerts(self.compImg.compSer[TC.VERTICES])

  @FR_SINGLETON.shortcuts.registerMethod(FR_CONSTS.SHC_ACCEPT_REGION)
  @Slot()
  def acceptRegionBtnClicked(self):
    self.compImg.saveNewVerts()
    modifiedComp = self.compImg.compSer
    self.compMgr.addComps(modifiedComp.to_frame().T, addtype=FR_ENUMS.COMP_ADD_AS_MERGE)

  @disableAppDuringFunc
  @FR_SINGLETON.shortcuts.registerMethod(FR_CONSTS.SHC_ESTIMATE_BOUNDARIES)
  def estimateBoundaries(self):
    compVertices = self.mainImg.procCollection.curProcessor.globalCompEstimate()
    components = makeCompDf(len(compVertices))
    components[TC.VERTICES] = compVertices
    self.compMgr.addComps(components)

  @Slot()
  @applyWaitCursor
  @FR_SINGLETON.shortcuts.registerMethod(FR_CONSTS.SHC_CLEAR_BOUNDARIES)
  def clearBoundaries(self):
    self.compMgr.rmComps()

  # ---------------
  # CUSTOM UI ELEMENT CALLBACKS
  # ---------------
  @Slot(object)
  @applyWaitCursor
  def updateCurComp(self, newComp: df):
    mainImg = self.mainImg.image
    self.compImg.updateAll(mainImg, newComp)
    self.curCompIdLbl.setText(f'Component ID: {newComp[TC.INST_ID]}')

## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
  app = pg.mkQApp()
  win = MainWindow()
  app.exec()