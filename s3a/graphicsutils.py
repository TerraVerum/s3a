from __future__ import annotations

from functools import wraps
from pathlib import Path
from typing import Dict, Union, List

import numpy as np
from pyqtgraph.Qt import QtWidgets, QtCore, QtGui
import pyqtgraph as pg

from utilitys import RunOpts
from utilitys.widgets import ImageViewer, EasyWidget

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

def create_addMenuAct(mainWin: QtWidgets.QWidget, parentMenu: QtWidgets.QMenu, title: str, asMenu=False) \
    -> Union[QtWidgets.QMenu, QtWidgets.QAction]:
  menu = None
  if asMenu:
    menu = QtWidgets.QMenu(title, mainWin)
    act = menu.menuAction()
  else:
    act = QtWidgets.QAction(title)
  parentMenu.addAction(act)
  if asMenu:
    return menu
  else:
    return act


def autosaveOptsDialog(parent):
  dlg = QtWidgets.QDialog(parent)
  layout = QtWidgets.QGridLayout(dlg)
  dlg.setLayout(layout)

  fileBtn = QtWidgets.QPushButton('Select Output Folder')
  folderNameDisplay = QtWidgets.QLabel('', dlg)
  def retrieveFolderName():
    folderDlg = QtWidgets.QFileDialog(dlg)
    folderDlg.setModal(True)
    folderName = folderDlg.getExistingDirectory(dlg, 'Select Output Folder')

    dlg.folderName = folderName
    folderNameDisplay.setText(folderName)
  fileBtn.clicked.connect(retrieveFolderName)

  layout.addWidget(folderNameDisplay, 0, 0)
  layout.addWidget(fileBtn, 0, 1)

  saveLbl = QtWidgets.QLabel('Save Name:', dlg)
  baseFileNameEdit = QtWidgets.QLineEdit(dlg)
  dlg.baseFileNameEdit = baseFileNameEdit

  layout.addWidget(saveLbl, 1, 0)
  layout.addWidget(baseFileNameEdit, 1, 1)

  intervalLbl = QtWidgets.QLabel('Save Interval (mins):', dlg)
  intervalEdit = QtWidgets.QSpinBox(dlg)
  intervalEdit.setMinimum(1)
  dlg.intervalEdit = intervalEdit

  layout.addWidget(intervalLbl, 2, 0)
  layout.addWidget(intervalEdit, 2, 1)

  saveDescr = QtWidgets.QLabel('Every <em>interval</em> minutes, a new autosave is'
                               ' created from the component list. Its name is'
                               ' [Parent Folder]/[base name]_[counter].csv, where'
                               ' counter is the current save file number.', dlg)
  saveDescr.setWordWrap(True)
  layout.addWidget(saveDescr, 3, 0, 1, 2)


  okBtn = QtWidgets.QPushButton('Ok')
  cancelBtn = QtWidgets.QPushButton('Cancel')
  cancelBtn.clicked.connect(dlg.reject)
  okBtn.clicked.connect(dlg.accept)
  dlg.okBtn = okBtn

  layout.addWidget(okBtn, 4, 0)
  layout.addWidget(cancelBtn, 4, 1)
  return dlg

class ThumbnailViewer(QtWidgets.QListWidget):
  sigDeleteRequested = QtCore.Signal(object)
  """List[Selected image paths]"""
  sigImageSelected = QtCore.Signal(object)
  """Full path of selected image"""

  def __init__(self, parent=None):
    super().__init__(parent)
    self.nameToFullPathMapping: Dict[str, Path] = {}
    self.setViewMode(self.IconMode)
    self.setIconSize(QtCore.QSize(200,200))
    self.setResizeMode(self.Adjust)
    self.itemActivated.connect(lambda item: self.sigImageSelected.emit(self.nameToFullPathMapping[item.text()]))

    # def findDelImgs():
    #   selection = self.selectedImages
    #   self.sigDeleteRequested.emit(selection)
    # TODO: Incorporate deletion. Just using "delete" button isn't great since
    #   when shortcuts conflict it is ambiguous as to whether an image will
    #   actually be deleted
    # self.delShc = QtWidgets.QShortcut(QtCore.Qt.Key_Delete, self, findDelImgs)

  def addThumbnail(self, fullName: Path, force=False):
    icon = QtGui.QIcon(str(fullName))
    if fullName.name in self.nameToFullPathMapping:
      if force:
        self.removeThumbnail(fullName.name)
      else:
        raise IOError('Name already exists in image list')
    newItem = QtWidgets.QListWidgetItem(fullName.name)
    newItem.setIcon(icon)
    self.addItem(newItem)
    self.nameToFullPathMapping[fullName.name] = fullName

  @property
  def selectedImages(self):
    return [self.nameToFullPathMapping[idx.data()] for idx in self.selectedIndexes()]

  def removeThumbnail(self, name: str):
    items = self.findItems(name, QtCore.Qt.MatchExactly)
    if len(items) == 0:
      return
    item = items[0]
    self.takeItem(self.row(item))
    del self.nameToFullPathMapping[name]

  def clear(self):
    super().clear()
    self.nameToFullPathMapping.clear()

# Taken directly from https://stackoverflow.com/questions/60663793/drop-one-or-more-files-into-listwidget-or-lineedit
class DropList(QtWidgets.QListWidget):
  def __init__(self, parent=None):
    super(DropList, self).__init__(parent)
    self.setAcceptDrops(True)
    self.setSelectionMode(self.ExtendedSelection)
    self.delShc = QtWidgets.QShortcut(QtCore.Qt.Key_Delete, self, self.deleteSelected)

  def deleteSelected(self):
    selectedIdxs = self.selectionModel().selectedIndexes()
    selectedRows = reversed(sorted([i.row() for i in selectedIdxs]))
    for row in selectedRows:
      self.takeItem(row)

  def dragEnterEvent(self, event):
    if event.mimeData().hasUrls():
      event.acceptProposedAction()
    else:
      event.ignore()

  def dragMoveEvent(self, event):
    if event.mimeData().hasUrls():
      event.acceptProposedAction()
    else:
      event.ignore()

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
  sigDiffsChanged = QtCore.Signal()

  def __init__(self, parent=None):
    super().__init__(parent)
    self.diffs: List[np.ndarray] = []
    self.histTimer = QtCore.QTimer(self)

    self.diffImg = pg.ImageItem()
    dp = self.displayPlt = ImageViewer()
    self.displayPlt.addItem(self.diffImg)
    self.diffImg.setOpacity(0.5)

    _, param = dp.toolsEditor.registerFunc(self.updateImg, runOpts=RunOpts.ON_CHANGED, returnParam=True)
    self.slider = param.child('curSlice')
    self.sigDiffsChanged.connect(lambda: self.slider.setLimits([0, len(self.diffs)-1]))
    self.histTimer.timeout.connect(self.incrSlicer)

    dp.toolsEditor.registerFunc(self.autoPlay)
    dp.toolsEditor.registerFunc(lambda: self.histTimer.stop(), name='Stop Autoplay')
    dp.toolsEditor.registerFunc(self.discardLeftEntries, name='Discard Entries Left of Slider')

    EasyWidget.buildMainWin([dp], layout='H', win=self)
    self.addDockWidget(QtCore.Qt.RightDockWidgetArea, dp.toolsEditor)


  def setDiffs(self, diffs: List[np.ndarray]):
    self.diffs = diffs
    self.sigDiffsChanged.emit()

  def autoPlay(self, timestep=500):
    """
    :param timestep:
      title: Timestep (ms)
    """
    self.histTimer.start(timestep)

  def incrSlicer(self):
    if self.slider.value() < len(self.diffs)-1:
      self.slider.setValue(self.slider.value()+1)
    else:
      self.histTimer.stop()

  def show(self):
    super().show()
    self.displayPlt.toolsEditor.show()

  def updateImg(self, curSlice=0):
    """
    :param curSlice:
      pType: slider
      limits: [0, 10]
    :return:
    """
    self.diffImg.setImage(self.diffs[curSlice])

  def discardLeftEntries(self):
    self.setDiffs(self.diffs[self.slider.value():])