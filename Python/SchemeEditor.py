# -*- coding: utf-8 -*-

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from pyqtgraph.parametertree import Parameter, ParameterTree
Signal = QtCore.pyqtSignal

from constants import DEFAULT_SCHEME_DICT, SchemeValues as SV
from ABGraphics.utils import dialogSaveToFile

import pickle as pkl

from typing import List, Union

# Ensure app instance is running
app = pg.mkQApp()

class SchemeEditor(QtWidgets.QWidget):
  '''
  Only one instance of this class is intended for use for a given application.
  So, don't import and instantiate this class. Rather, import the SCHEME variable
  defined below this class definition.
  '''
  sigSchemeSaved = Signal(str)

  def __init__(self):
    super().__init__()
    self.resize(500, 300)
    # -----------
    # Construct parameter tree
    # -----------
    self.params = Parameter(name='Parameters', type='group')
    self.params.addChildren(DEFAULT_SCHEME_DICT)
    self.tree = ParameterTree()
    self.tree.setParameters(self.params, showTop=False)

    # Init default layout as export of DEFAULT_SCHEME_DICT
    with open('./Schemes/Default.scheme', 'wb') as ofile:
      pkl.dump(self.params.saveState(), ofile)

    # Allow the user to change column widths
    for colIdx in range(2):
      self.tree.header().setSectionResizeMode(colIdx, QtWidgets.QHeaderView.Interactive)

    # -----------
    # Additional widget buttons
    # -----------
    self.saveBtn = QtWidgets.QPushButton('Save and Switch')
    self.cancelBtn = QtWidgets.QPushButton('Cancel')

    # -----------
    # Widget layout
    # -----------
    btnLayout = QtWidgets.QHBoxLayout()
    btnLayout.addWidget(self.saveBtn)
    btnLayout.addWidget(self.cancelBtn)

    centralLayout = QtWidgets.QVBoxLayout()
    centralLayout.addWidget(self.tree)
    centralLayout.addLayout(btnLayout)
    self.setLayout(centralLayout)
    # -----------
    # UI Element Signals
    # -----------
    self.saveBtn.clicked.connect(self.saveBtnClicked)
    self.cancelBtn.clicked.connect(self.close)


  def saveBtnClicked(self, saveName=False):
    newScheme = self.params.saveState()
    if saveName is False or saveName is None:
      saveName = dialogSaveToFile(self, newScheme, 'Scheme Name', './Schemes/', 'scheme', allowOverwriteDefault=False)
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

if __name__== '__main__':
  win = QtWidgets.QWidget()
  layout = QtWidgets.QGridLayout()
  win.setLayout(layout)
  layout.addWidget(SchemeEditor())
  win.show()
  app.exec()
