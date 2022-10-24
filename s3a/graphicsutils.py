from __future__ import annotations

from functools import wraps
from typing import List, Union

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from utilitys import RunOpts
from utilitys.widgets import EasyWidget, ImageViewer

Signal = QtCore.Signal
QCursor = QtGui.QCursor


def disableAppDuringFunc(func):
    @wraps(func)
    def disableApp(*args, **kwargs):
        # Captures 'self' instance
        mainWin = args[0]
        try:
            mainWin.setEnabled(False)
            return func(*args, **kwargs)
        finally:
            mainWin.setEnabled(True)

    return disableApp


def createAndAddMenuAction(
    mainWindow: QtWidgets.QWidget, parentMenu: QtWidgets.QMenu, title: str, asMenu=False
) -> Union[QtWidgets.QMenu, QtGui.QAction]:
    menu = None
    if asMenu:
        menu = QtWidgets.QMenu(title, mainWindow)
        act = menu.menuAction()
    else:
        act = QtGui.QAction(title)
    parentMenu.addAction(act)
    if asMenu:
        return menu
    else:
        return act


def autosaveOptionsDialog(parent):
    dlg = QtWidgets.QDialog(parent)
    layout = QtWidgets.QGridLayout(dlg)
    dlg.setLayout(layout)

    fileBtn = QtWidgets.QPushButton("Select Output Folder")
    folderNameDisplay = QtWidgets.QLabel("", dlg)

    def retrieveFolderName():
        folderDlg = QtWidgets.QFileDialog(dlg)
        folderDlg.setModal(True)
        folderName = folderDlg.getExistingDirectory(dlg, "Select Output Folder")

        dlg.folderName = folderName
        folderNameDisplay.setText(folderName)

    fileBtn.clicked.connect(retrieveFolderName)

    layout.addWidget(folderNameDisplay, 0, 0)
    layout.addWidget(fileBtn, 0, 1)

    saveLbl = QtWidgets.QLabel("Save Name:", dlg)
    baseFileNameEdit = QtWidgets.QLineEdit(dlg)
    dlg.baseFileNameEdit = baseFileNameEdit

    layout.addWidget(saveLbl, 1, 0)
    layout.addWidget(baseFileNameEdit, 1, 1)

    intervalLbl = QtWidgets.QLabel("Save Interval (mins):", dlg)
    intervalEdit = QtWidgets.QSpinBox(dlg)
    intervalEdit.setMinimum(1)
    dlg.intervalEdit = intervalEdit

    layout.addWidget(intervalLbl, 2, 0)
    layout.addWidget(intervalEdit, 2, 1)

    saveDescr = QtWidgets.QLabel(
        "Every <em>interval</em> minutes, a new autosave is"
        " created from the component list. Its name is"
        " [Parent Folder]/[base name]_[counter].csv, where"
        " counter is the current save file number.",
        dlg,
    )
    saveDescr.setWordWrap(True)
    layout.addWidget(saveDescr, 3, 0, 1, 2)

    okBtn = QtWidgets.QPushButton("Ok")
    cancelBtn = QtWidgets.QPushButton("Cancel")
    cancelBtn.clicked.connect(dlg.reject)
    okBtn.clicked.connect(dlg.accept)
    dlg.okBtn = okBtn

    layout.addWidget(okBtn, 4, 0)
    layout.addWidget(cancelBtn, 4, 1)
    return dlg


# Taken directly from
# https://stackoverflow.com/questions/60663793/drop-one-or-more-files-into-listwidget-or-lineedit  # noqa
class DropList(QtWidgets.QListWidget):
    def __init__(self, parent=None):
        super(DropList, self).__init__(parent)
        self.setAcceptDrops(True)
        self.setSelectionMode(self.ExtendedSelection)
        seq = QtGui.QKeySequence(QtCore.Qt.Key.Key_Delete)
        self.delShc = QtGui.QShortcut(seq, self, self.deleteSelected)

    def deleteSelected(self):
        selectedIdxs = self.selectionModel().selectedIndexes()
        selectedRows = np.sort([i.row() for i in selectedIdxs])[::-1]
        for row in selectedRows:
            self.takeItem(row)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        return self.dragEnterEvent(event)

    def dropEvent(self, event):
        md = event.mimeData()
        if md.hasUrls():
            for url in md.urls():
                self.addItem(url.toLocalFile())
            event.acceptProposedAction()

    @property
    def files(self):
        model = self.model()
        return [model.index(ii, 0).data() for ii in range(model.rowCount())]


class RegionHistoryViewer(QtWidgets.QMainWindow):
    sigImageBufferChanged = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.differenceImages: List[np.ndarray] = []
        self.histTimer = QtCore.QTimer(self)

        self.differenceImage = pg.ImageItem()
        dp = self.displayPlot = ImageViewer()
        self.displayPlot.addItem(self.differenceImage)
        self.differenceImage.setOpacity(0.5)

        _, param = dp.toolsEditor.registerFunc(
            self.updateImage, runOpts=RunOpts.ON_CHANGED, returnParam=True
        )
        self.slider = param.child("curSlice")
        self.sigImageBufferChanged.connect(
            lambda: self.slider.setLimits([0, len(self.differenceImages) - 1])
        )
        self.histTimer.timeout.connect(self.incrSlicer)

        dp.toolsEditor.registerFunc(self.autoPlay)
        dp.toolsEditor.registerFunc(lambda: self.histTimer.stop(), name="Stop Autoplay")
        dp.toolsEditor.registerFunc(
            self.discardLeftEntries, name="Discard Entries Left of Slider"
        )

        EasyWidget.buildMainWin([dp], layout="H", win=self)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, dp.toolsEditor)

    def setDifferenceImages(self, differenceImages: List[np.ndarray]):
        self.differenceImages = differenceImages
        self.sigImageBufferChanged.emit()

    def autoPlay(self, timestep=500):
        """
        Parameters
        ----------
        timestep
            title: Timestep (ms)
        """
        self.histTimer.start(timestep)

    def incrSlicer(self):
        if self.slider.value() < len(self.differenceImages) - 1:
            self.slider.setValue(self.slider.value() + 1)
        else:
            self.histTimer.stop()

    def show(self):
        super().show()
        self.displayPlot.toolsEditor.show()

    def updateImage(self, curSlice=0):
        """
        Parameters
        ----------
        curSlice
            pType: slider
            limits: [0, 10]
        """
        self.differenceImage.setImage(self.differenceImages[curSlice])

    def discardLeftEntries(self):
        self.setDifferenceImages(self.differenceImages[self.slider.value() :])
