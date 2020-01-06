# -*- coding: utf-8 -*-
from abc import abstractmethod

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from pyqtgraph.parametertree import (Parameter, ParameterTree, parameterTypes as pTypes,
                                     registerParameterType)

from constants import (
  SCHEMES_DIR, SchemeValues as SV,
  ComponentTableFields as CTF, ComponentTypes)
from ABGraphics.utils import dialogSaveToFile

import pickle as pkl
import sys

from typing import List, Union
from os.path import join

Signal = QtCore.pyqtSignal
# Ensure app instance is running
app = pg.mkQApp()

def _genList(nameIter, paramType, defaultVal, defaultParam='value'):
  """Helper for generating children elements"""
  return [{'name': name, 'type': paramType, defaultParam: defaultVal} for name in nameIter]


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

  def _paramsToList(self) -> list:
    """
    :return: List where each index corresponds to the tree's parameter.
    This is suitable for extending with an ID and vertex list, after which
    it can be placed into the component table.
    """
    outList = []
    for param in self.params.children():
      outList.append(param.value())
    return outList

  @abstractmethod
  def acceptBtnClicked(self):
    return



class TableRowEditor(ConstParamWidget):
  # Emits list where each element corresponds to an edit field
  sigEditFinished = Signal(list)
  def __init__(self):
    super().__init__()
    _TABLE_EDIT_DICT = [
      {'name': CTF.VALIDATED.value, 'type': 'bool', 'value': False},
      {'name': CTF.DEVICE_TYPE.value, 'type': 'list', 'value': ComponentTypes.N_A.value,
       'limits': [compType for compType in ComponentTypes]
       },
      {'name': CTF.LOGO.value, 'type': 'text', 'value': ''},
      {'name': CTF.NOTES.value, 'type': 'text', 'value': ''},
      {'name': CTF.BOARD_TEXT.value, 'type': 'text', 'value': ''},
      {'name': CTF.DEVICE_TEXT.value, 'type': 'text', 'value': ''}
    ]
    self.params.addChildren(_TABLE_EDIT_DICT)

  def acceptBtnClicked(self) -> List:
    """
    :return: List where each index corresponds to the tree's parameter.
    This is suitable for extending with an ID and vertex list, after which
    it can be placed into the component table.
    """
    outList = self[:]
    self.sigEditFinished.emit(outList)
    return outList


class TableFilterEditor(ConstParamWidget):
  # Emits key-value pair of input filter options
  sigFilterChanged = Signal(dict)

  def __init__(self):
    super().__init__()
    minMaxParam = _genList(['min', 'max'], 'int', 0)
    # Make max 'infinity'
    minMaxParam[1]['value'] = sys.maxsize
    validatedParms = _genList(['Validated', 'Not Validated'], 'bool', True)
    devTypeParam = _genList((name.value for name in ComponentTypes), 'bool', True)
    xyVerts = _genList(['X Bounds', 'Y Bounds'], 'group', minMaxParam, 'children')

    _FILTER_DICT = [
        {'name': CTF.INST_ID.value, 'type': 'group', 'children': minMaxParam},
        {'name': CTF.VALIDATED.value, 'type': 'group', 'children': validatedParms},
        {'name': CTF.DEVICE_TYPE.value, 'type': 'group', 'children': devTypeParam},
        {'name': CTF.LOGO.value, 'type': 'str', 'value': '.*'},
        {'name': CTF.NOTES.value, 'type': 'str', 'value': '.*'},
        {'name': CTF.BOARD_TEXT.value, 'type': 'str', 'value': '.*'},
        {'name': CTF.DEVICE_TEXT.value, 'type': 'str', 'value': '.*'},
        {'name': CTF.VERTICES.value, 'type': 'group', 'children': xyVerts}
      ]
    self.params.addChildren(_FILTER_DICT)

  def acceptBtnClicked(self) -> List:
    """
    :return: List where each index corresponds to the tree's parameter.
    This is suitable for extending with an ID and vertex list, after which
    it can be placed into the component table.
    """
    # TODO: Add method to save this filter to a menu like schemes and layouts
    outDict = self.params.getValues()
    self.sigFilterChanged.emit(outDict)
    return outDict

class SchemeEditor(ConstParamWidget):
  sigSchemeSaved = Signal(str)

  def __init__(self, parent=None):
    super().__init__()
    _DEFAULT_SCHEME_DICT = [
      {'name': SV.COMP_PARAMS.value, 'type': 'group', 'children': [
        {'name': SV.VALID_ID_COLOR.value, 'type': 'color', 'value': '0f0'},
        {'name': SV.NONVALID_ID_COLOR.value, 'type': 'color', 'value': 'f00'},
        {'name': SV.BOUNDARY_COLOR.value, 'type': 'color', 'value': 'ff0'},
        {'name': SV.BOUNDARY_WIDTH.value, 'type': 'int', 'value': 2},
        {'name': SV.ID_FONT_SIZE.value, 'type': 'int', 'value': 10}
      ]},
      {'name': SV.FOC_IMG_PARAMS.value, 'type': 'group', 'children': [
        {'name': SV.REG_VERT_COLOR.value, 'type': 'color', 'value': '0f0'},
        {'name': SV.REG_FILL_COLOR.value, 'type': 'color', 'value': '00ff0046'}
      ]},
    ]
    self.params.addChildren(_DEFAULT_SCHEME_DICT)

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