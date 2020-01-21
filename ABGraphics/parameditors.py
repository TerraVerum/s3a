# -*- coding: utf-8 -*-
import pickle as pkl
import sys
from enum import Enum
from os.path import join

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from pyqtgraph.parametertree import (Parameter, ParameterTree)

from ABGraphics.utils import dialogSaveToFile
from constants import (
  SCHEMES_DIR, SchemeValues as SV,
  ComponentTableFields as CTF, ComponentTypes,
  REGION_CTRL_DIR, RegionControlsEditorValues as RCEV,
  FILTERS_DIR)

Signal = QtCore.pyqtSignal
# Ensure app instance is running
app = pg.mkQApp()

def _genList(nameIter, paramType, defaultVal, defaultParam='value'):
  """Helper for generating children elements"""
  return [{'name': name, 'type': paramType, defaultParam: defaultVal} for name in nameIter]


class ConstParamWidget(QtWidgets.QDialog):
  sigParamStateCreated = Signal(str)
  sigParamStateUpdated = Signal(dict)

  def __init__(self, parent=None, paramDict=None, saveDir='.',
               saveExt='param', saveDlgName='Save As', createDefaultOutput=True):
    # Place in list so an empty value gets unpacked into super constructor
    if paramDict is None:
      paramDict = []
    parentArg = []
    if parent is not None:
      parentArg.append(parent)
    super().__init__(*parentArg)
    self.resize(500, 400)
    #self.setModal(True)

    # -----------
    # Construct parameter tree
    # -----------
    self.params = Parameter(name='Parameters', type='group', children=paramDict)
    self.tree = ParameterTree()
    self.tree.setParameters(self.params, showTop=False)

    # Allow the user to change column widths
    for colIdx in range(2):
      self.tree.header().setSectionResizeMode(colIdx, QtWidgets.QHeaderView.Interactive)

    # -----------
    # Internal parameters for saving settings
    # -----------
    self.SAVE_DIR = saveDir
    self.FILE_TYPE = saveExt
    self._saveDlgName = saveDlgName
    self._stateBeforeEdit = self.params.saveState()

    # -----------
    # Save default state if desired
    # -----------
    if createDefaultOutput:
      with open(join(saveDir, f'Default.{saveExt}'), 'wb') as ofile:
        pkl.dump(self.params.saveState(), ofile)

    # -----------
    # Additional widget buttons
    # -----------
    self.saveAsBtn = QtWidgets.QPushButton('Save As...')
    self.applyBtn = QtWidgets.QPushButton('Apply')
    self.closeBtn = QtWidgets.QPushButton('Close')

    # -----------
    # Widget layout
    # -----------
    btnLayout = QtWidgets.QHBoxLayout()
    btnLayout.addWidget(self.saveAsBtn)
    btnLayout.addWidget(self.applyBtn)
    btnLayout.addWidget(self.closeBtn)

    centralLayout = QtWidgets.QVBoxLayout()
    centralLayout.addWidget(self.tree)
    centralLayout.addLayout(btnLayout)
    self.setLayout(centralLayout)
    # -----------
    # UI Element Signals
    # -----------
    self.saveAsBtn.clicked.connect(self.saveAsBtnClicked)
    self.closeBtn.clicked.connect(self.close)
    self.applyBtn.clicked.connect(self.applyBtnClicked)

  # Helper method for accessing simple parameter values
  def __getitem__(self, key: Enum):
    return self.params.child(key.value)

  def close(self):
    """
    If window is closed apart from pressing 'accept', restore pre-edit state
    """
    self.params.restoreState(self._stateBeforeEdit)
    super().close()

  def applyBtnClicked(self):
    self._stateBeforeEdit = self.params.saveState()
    outDict = self.params.getValues()
    self.sigParamStateUpdated.emit(outDict)
    return outDict

  def saveAsBtnClicked(self, saveName=None):
    """
    :return: List where each index corresponds to the tree's parameter.
    This is suitable for extending with an ID and vertex list, after which
    it can be placed into the component table.
    """
    paramState = self.params.saveState()
    if saveName is False or saveName is None:
      saveName = dialogSaveToFile(self, paramState, self._saveDlgName,
                                  self.SAVE_DIR, self.FILE_TYPE, allowOverwriteDefault=False)
    else:
      with open(saveName, 'wb') as saveFile:
        pkl.dump(paramState, saveFile)
    if saveName is not None:
      # Accept new param state after saving
      self.applyBtnClicked()
      outDict = self.params.getValues()
      self.sigParamStateCreated.emit(saveName)
      return outDict
    # If no name specified
    return None

  def loadState(self, newStateDict):
    self.params.restoreState(newStateDict, addChildren=False)


class RegionControlsEditor(ConstParamWidget):
  def __init__(self, parent=None):
    _CONTROLS_DICT = [
        {'name': RCEV.MARGIN.value, 'type': 'int', 'value': 5},
        {'name': RCEV.SEG_THRESH.value, 'type': 'float', 'value': 6.},
        {'name': RCEV.SEED_THRESH.value, 'type': 'float', 'value': 15.},
        {'name': RCEV.NEW_COMP_SZ.value, 'type': 'int', 'value': 10},
      ]
    super().__init__(parent, paramDict=_CONTROLS_DICT, saveDir=REGION_CTRL_DIR, saveExt='regctrl')

class TableFilterEditor(ConstParamWidget):
  def __init__(self, parent=None):
    super().__init__(parent)
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
    super().__init__(paramDict=_FILTER_DICT, saveDir=FILTERS_DIR, saveExt='filter')

class SchemeEditor(ConstParamWidget):
  def __init__(self, parent=None):
    super().__init__(parent)
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
    super().__init__(paramDict=_DEFAULT_SCHEME_DICT, saveDir=SCHEMES_DIR, saveExt='scheme')

  def _getProps(self, compOrFocIm: SV, whichProps):
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

  def getCompProps(self, whichProps):
    return self._getProps(SV.COMP_PARAMS, whichProps)

  def getFocImgProps(self, whichProps):
    return self._getProps(SV.FOC_IMG_PARAMS, whichProps)