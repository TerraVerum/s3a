import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Union

import pandas as pd
import pyqtgraph as pg
import qdarkstyle
from pyqtgraph.parametertree.parameterTypes.file import popupFilePicker
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from qtextras import OptionsDict, ParameterEditor, fns, widgets

from ..constants import ICON_DIR, LAYOUTS_DIR, PRJ_CONSTS, PRJ_ENUMS, REQD_TBL_FIELDS
from ..generalutils import hierarchicalUpdate
from ..logger import getAppLogger
from ..models.s3abase import S3ABase
from ..plugins.mainimage import MainImagePlugin
from ..plugins.table import ComponentTablePlugin
from ..structures import FilePath, NChanImg

__all__ = ["S3A"]


class S3A(S3ABase):
    sigLayoutSaved = QtCore.Signal()

    def __init__(
        self,
        log: Union[str, Sequence[str]] = PRJ_ENUMS.LOG_TERM,
        loadLastState=True,
        **startupSettings,
    ):
        # Wait to import quick loader profiles until after self initialization so
        # customized loading functions also get called
        super().__init__(**startupSettings)

        self.setWindowIcon(QtGui.QIcon(str(ICON_DIR / "s3alogo.svg")))
        logger = getAppLogger()
        if PRJ_ENUMS.LOG_GUI in log:
            logger.registerExceptions()
            logger.registerWarnings()
            logger.addHandler(
                widgets.FadeNotifyHandler(
                    PRJ_ENUMS.LOG_LEVEL_ATTENTION,
                    self,
                    maxLevel=PRJ_ENUMS.LOG_LEVEL_ATTENTION,
                )
            )
            logger.addHandler(
                widgets.StatusBarHandler(logging.INFO, self, maxLevel=logging.INFO)
            )
            # This logger isn't supposed to propagate, since everything is handled in
            # the terminal on accepted events unless 'terminal' is also specified
            if PRJ_ENUMS.LOG_TERM not in log:
                logger.propagate = False
        self.APP_TITLE = "Semi-Supervised Semantic Annotator"
        self.CUR_COMP_LBL = "Current Component ID:"
        self.setWindowTitle(self.APP_TITLE)
        self.setWindowIconText(self.APP_TITLE)

        self.currentComponentLabel = QtWidgets.QLabel(self.CUR_COMP_LBL)

        # -----
        # LAOYUT MANAGER
        # -----
        # Dummy editor for layout options since it doesn't really have editable settings
        # Maybe later this can be elevated to have more options
        self.layoutEditor = ParameterEditor(
            name="Layout", directory=LAYOUTS_DIR, suffix=".dockstate"
        )

        def loadLayout(layoutName: Union[str, Path]):
            layoutName = Path(layoutName)
            if not layoutName.is_absolute():
                layoutName = LAYOUTS_DIR / f"{layoutName}.dockstate"
            state = fns.attemptFileLoad(layoutName)
            self.restoreState(state["docks"])

        def saveRecentLayout(_folderName: Path):
            outFile = _folderName / "layout.dockstate"
            self.saveLayout(outFile)
            return str(outFile)

        self.layoutEditor.loadParameterValues = loadLayout
        self.layoutEditor.saveParameterValues = saveRecentLayout
        self.appStateEditor.addImportExportOptions(
            "layout", loadLayout, saveRecentLayout
        )

        self._buildGui()
        self._buildMenu()

        # Load in startup settings
        stateDict = None if loadLastState else {}
        hierarchicalUpdate(self.appStateEditor.startupSettings, startupSettings)
        self.appStateEditor.loadParameterValues(stateDict=stateDict)

    def _buildMenu(self):
        # Nothing to do for now
        pass

    def _buildGui(self):
        self.setDockOptions(QtWidgets.QMainWindow.DockOption.ForceTabbedDocks)
        self.setTabPosition(
            QtCore.Qt.AllDockWidgetAreas, QtWidgets.QTabWidget.TabPosition.North
        )
        centralWidget = QtWidgets.QWidget()
        self.setCentralWidget(centralWidget)
        layout = QtWidgets.QVBoxLayout(centralWidget)

        self.toolbarWidgets: Dict[OptionsDict, List[QtGui.QAction]] = defaultdict(list)
        layout.addWidget(self.mainImage)

        self.tableFieldToolbar.setObjectName("Table Field Plugins")
        # self.addToolBar(self.tableFieldToolbar)
        self.tableFieldToolbar.hide()
        self.generalToolbar.setObjectName("General")
        self.addToolBar(self.generalToolbar)

        _plugins = [
            self.classPluginMap[c] for c in [MainImagePlugin, ComponentTablePlugin]
        ]
        parents = [self.mainImage, self.tableView]
        for plugin, parent in zip(_plugins, reversed(parents)):
            newMenu = plugin.createActionsFromFunctions(stealShortcuts=False)
            parent.menu.addMenu(newMenu)

        tableDock = QtWidgets.QDockWidget("Component Table Window", self)
        feat = QtWidgets.QDockWidget.DockWidgetFeature
        tableDock.setFeatures(feat.DockWidgetMovable | feat.DockWidgetFloatable)

        tableDock.setObjectName("Component Table Dock")
        tableContents = QtWidgets.QWidget(tableDock)
        tableLayout = QtWidgets.QVBoxLayout(tableContents)
        tableLayout.addWidget(self.tableView)
        tableDock.setWidget(tableContents)

        self.addDockWidget(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, tableDock)

        # STATUS BAR
        statusBar = self.statusBar()

        self.mousePosLabel = QtWidgets.QLabel()
        self.pixelColorLabel = QtWidgets.QLabel()

        self.imageLabel = QtWidgets.QLabel(f"Image: None")

        statusBar.show()
        statusBar.addPermanentWidget(self.imageLabel)

        statusBar.addPermanentWidget(self.mousePosLabel)
        self.mainImage.mouseCoordsLbl = self.mousePosLabel

        statusBar.addPermanentWidget(self.pixelColorLabel)
        self.mainImage.pxColorLbl = self.pixelColorLabel

    def saveLayout(self, layoutName: Union[str, Path] = None):
        dockStates = self.saveState().data()
        if Path(layoutName).is_absolute():
            savePathPlusStem = layoutName
        else:
            savePathPlusStem = LAYOUTS_DIR / layoutName
        saveFile = savePathPlusStem.with_suffix(f".dockstate")
        fns.saveToFile({"docks": dockStates}, saveFile)
        self.sigLayoutSaved.emit()

    def changeFocusedComponent(self, ids: Union[int, Sequence[int]] = None):
        ret = super().changeFocusedComponent(ids)
        self.currentComponentLabel.setText(
            f"Component ID: {self.componentManager.focusedComponent[REQD_TBL_FIELDS.ID]}"
        )
        return ret

    def resetTableFieldsGui(self):
        outFname = popupFilePicker(
            None, "Select Table Config File", "All Files (*.*);; Config Files (*.yml)"
        )
        if outFname is not None:
            self.tableData.loadConfig(outFname)

    def setMainImage(
        self,
        file: FilePath = None,
        imageData: NChanImg = None,
        clearExistingComponents=True,
    ):
        gen = super().setMainImage(file, imageData, clearExistingComponents)
        ret = fns.gracefulNext(gen)
        img = self.sourceImagePath
        if img is not None:
            img = img.name
        self.imageLabel.setText(f"Image: {img}")

        yield ret
        yield fns.gracefulNext(gen)

    def setMainImageGui(self):
        fileFilter = (
            "Image Files (*.png *.tif *.jpg *.jpeg *.bmp *.jfif);;All files(*.*)"
        )
        fname = popupFilePicker(None, "Select Main Image", fileFilter)
        if fname is not None:
            with pg.BusyCursor():
                self.setMainImage(fname)

    def exportAnnotationsGui(self):
        """Saves the component table to a file"""
        fileFilters = self.componentIo.ioFileFilter(**{"*": "All Files"})
        outFname = popupFilePicker(
            None, "Select Save File", fileFilters, existing=False
        )
        if outFname is not None:
            super().exportCurrentAnnotation(outFname)

    def openAnnotationGui(self):
        # TODO: See note about exporting components. Delegate the filepicker activity to
        #  importer
        fileFilter = self.componentIo.ioFileFilter(which=PRJ_ENUMS.IO_IMPORT)
        fname = popupFilePicker(None, "Select Load File", fileFilter)
        if fname is None:
            return
        self.openAnnotations(fname)

    def saveLayoutGui(self):
        outName = fns.dialogGetSaveFileName(self, "Layout Name")
        if outName is None or outName == "":
            return
        self.saveLayout(outName)

    # ---------------
    # BUTTON CALLBACKS
    # ---------------
    def closeEvent(self, ev: QtGui.QCloseEvent):
        # Confirm all components have been saved
        shouldExit = True
        forceClose = False
        if self.hasUnsavedChanges:
            ev.ignore()
            forceClose = False
            msg = QtWidgets.QMessageBox()
            btnTypes = QtWidgets.QMessageBox.StandardButton
            msg.setWindowTitle("Confirm Exit")
            msg.setText(
                "Component table has unsaved changes.\nYou can choose to save and exit "
                "or discard changes "
            )
            msg.setDefaultButton(btnTypes.Save)
            msg.setStandardButtons(btnTypes.Discard | btnTypes.Cancel | btnTypes.Save)
            code = msg.exec_()
            if code == btnTypes.Discard:
                forceClose = True
            elif code == btnTypes.Cancel:
                shouldExit = False
        if shouldExit:
            # Clean up all editor windows, which could potentially be left open
            ev.accept()
            if not forceClose:
                self.appStateEditor.saveParameterValues()

    def forceClose(self):
        """
        Allows the app to close even if it has unsaved changes. Useful for closing
        within a script
        """
        self.hasUnsavedChanges = False
        self.close()

    def _populateLoadLayoutOptions(self):
        self.menuLayout = self.menuBar().addMenu("Layout")
        self.layoutEditor.stateManager.addSavedStatesToMenu(self.menuLayout)

    def updateTheme(self, useDarkTheme=False):
        style = ""
        if useDarkTheme:
            style = qdarkstyle.load_stylesheet()
        self.setStyleSheet(style)

    def addAndFocusComponents(
        self, components: pd.DataFrame, addType=PRJ_ENUMS.COMPONENT_ADD_AS_NEW
    ):
        gen = super().addAndFocusComponents(components, addType=addType)
        changeDict = fns.gracefulNext(gen)
        keepIds = changeDict["ids"]
        keepIds = keepIds[keepIds >= 0]
        selection = self.componentController.selectRowsById(keepIds)
        if (
            self.isVisible()
            and self.tableView.props[PRJ_CONSTS.PROP_SHOW_TBL_ON_COMP_CREATE]
        ):
            # For some reason sometimes the actual table selection doesn't propagate in
            # time, so directly forward the selection here
            self.tableView.setSelectedCellsAsGui(selection)
        yield changeDict
        yield fns.gracefulNext(gen)


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication([])
    win = S3A()
    win.showMaximized()
    sys.exit(app.exec_())
