# -*- coding: utf-8 -*-
from abc import abstractmethod

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from pyqtgraph.parametertree import Parameter, ParameterTree

from constants import (
  DEFAULT_SCHEME_DICT, SCHEMES_DIR, SchemeValues as SV,
  TABLE_EDIT_DICT, ComponentTableFields as CTF)
from ABGraphics.utils import dialogSaveToFile

import pickle as pkl

from typing import List, Union
from os.path import join

Signal = QtCore.pyqtSignal
# Ensure app instance is running
app = pg.mkQApp()


class ConstParamWidget(QtWidgets.QDialog):
  def __init__(self, parent=None):
    # Place in list so an empty value gets unpacked into super constructor
    parentArg = []
    if parent is not None:
      parentArg.append(parent)
    super().__init__(*parentArg)
    self.resize(500, 400)
    self.setModal(True)
    # -----------
    # Construct parameter tree
    # -----------
    self.params = Parameter(name='Parameters', type='group')
    self.tree = ParameterTree()
    self.tree.setParameters(self.params, showTop=False)

    # Allow the user to change column widths
    for colIdx in range(2):
      self.tree.header().setSectionResizeMode(colIdx, QtWidgets.QHeaderView.Interactive)

    # -----------
    # Additional widget buttons
    # -----------
    self.acceptBtn = QtWidgets.QPushButton('Accept')
    self.cancelBtn = QtWidgets.QPushButton('Cancel')

    # -----------
    # Widget layout
    # -----------
    btnLayout = QtWidgets.QHBoxLayout()
    btnLayout.addWidget(self.acceptBtn)
    btnLayout.addWidget(self.cancelBtn)

    centralLayout = QtWidgets.QVBoxLayout()
    centralLayout.addWidget(self.tree)
    centralLayout.addLayout(btnLayout)
    self.setLayout(centralLayout)
    # -----------
    # UI Element Signals
    # -----------
    self.acceptBtn.clicked.connect(self.acceptBtnClicked)
    self.cancelBtn.clicked.connect(self.close)

  @abstractmethod
  def acceptBtnClicked(self):
    return


class TableRowEditor(ConstParamWidget):
  sigEditFinished = Signal(list)
  def __init__(self):
    super().__init__()
    self.params.addChildren(TABLE_EDIT_DICT)

  def acceptBtnClicked(self) -> List:
    """
    :return: List where each index corresponds to the tree's parameter.
    This is suitable for extending with an ID and vertex list, after which
    it can be placed into the component table.
    """
    outList = []
    for param in self.params.children():
      outList.append(param.value())
    self.sigEditFinished.emit(outList)
    return outList


class SchemeEditor(ConstParamWidget):
  sigSchemeSaved = Signal(str)

  def __init__(self, parent=None):
    super().__init__()
    self.params.addChildren(DEFAULT_SCHEME_DICT)

    # Init default layout as export of DEFAULT_SCHEME_DICT
    with open(join(SCHEMES_DIR, 'Default.scheme'), 'wb') as ofile:
      pkl.dump(self.params.saveState(), ofile)

  def acceptBtnClicked(self, saveName=False):
    newScheme = self.params.saveState()
    if saveName is False or saveName is None:
      saveName = dialogSaveToFile(self, newScheme, 'Scheme Name', SCHEMES_DIR, 'scheme', allowOverwriteDefault=False)
    else:
      with open(saveName, 'wb') as saveFile:
        pkl.dump(newScheme, saveFile)
    if saveName is not None:
      self.sigSchemeSaved.emit(saveName)
      self.close()

  def loadScheme(self, schemeDict):
    self.params.restoreState(schemeDict, addChildren=False)

  def _getProps(self, compOrFocIm: SV, whichProps: Union[SV, List[SV]]):
    returnList = True
    if isinstance(whichProps, SV):
      whichProps = [whichProps]
      returnList = False
    outProps = []
    for prop in whichProps:
      outProps.append(self.params.child(compOrFocIm.value, prop.value).value())
    if not returnList:
      outProps = outProps[0]
    return outProps

  def getCompProps(self, whichProps: List[SV]):
    return self._getProps(SV.COMP_PARAMS, whichProps)

  def getFocImgProps(self, whichProps: List[SV]):
    return self._getProps(SV.FOC_IMG_PARAMS, whichProps)


if __name__ == '__main__':
  win = QtWidgets.QWidget()
  layout = QtWidgets.QGridLayout()
  win.setLayout(layout)
  # layout.addWidget(SchemeEditor())
  te = TableRowEditor()
  te.sigEditFinished.connect(lambda lst: print(lst))
  layout.addWidget(te)
  win.show()
  app.exec()
