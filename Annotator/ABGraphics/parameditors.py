# -*- coding: utf-8 -*-
import pickle as pkl
import sys
from os.path import join

from pyqtgraph.Qt import QtCore, QtWidgets
from pyqtgraph.parametertree import (Parameter, ParameterTree)

from .utils import dialogSaveToFile
from ..constants import (
  SCHEMES_DIR, REGION_CTRL_DIR, FILTERS_DIR,
  TEMPLATE_SCHEME_VALUES as SV,
  TEMPLATE_COMP as TC, ComponentTypes,
  TEMPLATE_REG_CTRLS as REG_CTRLS)
from Annotator.params import ABParam

Signal = QtCore.pyqtSignal

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
    super().__init__(parent)
    self.resize(500, 400)

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
  def __getitem__(self, key: ABParam):
    return self.params.child(key.name)

  def show(self):
    super().show()
    self.setWindowState(QtCore.Qt.WindowActive)

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
        {'name': REG_CTRLS.MARGIN.name, 'type': 'int', 'value': 5},
        {'name': REG_CTRLS.SEG_THRESH.name, 'type': 'float', 'value': 6.},
        {'name': REG_CTRLS.SEED_THRESH.name, 'type': 'float', 'value': 15.},
        {'name': REG_CTRLS.NEW_COMP_SZ.name, 'type': 'int', 'value': 10},
        {'name': REG_CTRLS.EST_BOUNDS_ON_START.name, 'type': 'bool', 'value': True}
      ]
    super().__init__(parent, paramDict=_CONTROLS_DICT, saveDir=REGION_CTRL_DIR, saveExt='regctrl')

class TableFilterEditor(ConstParamWidget):
  def __init__(self, parent=None):
    minMaxParam = _genList(['min', 'max'], 'int', 0)
    # Make max 'infinity'
    minMaxParam[1]['value'] = sys.maxsize
    validatedParms = _genList(['Validated', 'Not Validated'], 'bool', True)
    devTypeParam = _genList((name.value for name in ComponentTypes), 'bool', True)
    xyVerts = _genList(['X Bounds', 'Y Bounds'], 'group', minMaxParam, 'children')
    _FILTER_DICT = [
        {'name': TC.INST_ID.name, 'type': 'group', 'children': minMaxParam},
        {'name': TC.VALIDATED.name, 'type': 'group', 'children': validatedParms},
        {'name': TC.DEV_TYPE.name, 'type': 'group', 'children': devTypeParam},
        {'name': TC.LOGO.name, 'type': 'str', 'value': '.*'},
        {'name': TC.NOTES.name, 'type': 'str', 'value': '.*'},
        {'name': TC.BOARD_TEXT.name, 'type': 'str', 'value': '.*'},
        {'name': TC.DEV_TEXT.name, 'type': 'str', 'value': '.*'},
        {'name': TC.VERTICES.name, 'type': 'group', 'children': xyVerts}
      ]
    super().__init__(parent, paramDict=_FILTER_DICT, saveDir=FILTERS_DIR, saveExt='filter')

class SchemeEditor(ConstParamWidget):
  def __init__(self, parent=None):
    _DEFAULT_SCHEME_DICT = [
      {'name': SV.COMP_PARAMS.name, 'type': 'group', 'children': [
        {'name': SV.VALID_ID_COLOR.name, 'type': 'color', 'value': '0f0'},
        {'name': SV.NONVALID_ID_COLOR.name, 'type': 'color', 'value': 'f00'},
        {'name': SV.BOUNDARY_COLOR.name, 'type': 'color', 'value': 'ff0'},
        {'name': SV.BOUNDARY_WIDTH.name, 'type': 'int', 'value': 2},
        {'name': SV.ID_FONT_SIZE.name, 'type': 'int', 'value': 10}
      ]},
      {'name': SV.FOC_IMG_PARAMS.name, 'type': 'group', 'children': [
        {'name': SV.REG_VERT_COLOR.name, 'type': 'color', 'value': '0f0'},
        {'name': SV.REG_FILL_COLOR.name, 'type': 'color', 'value': '00ff0046'}
      ]},
    ]
    super().__init__(parent, paramDict=_DEFAULT_SCHEME_DICT, saveDir=SCHEMES_DIR,
                     saveExt='scheme')

  def _getProps(self, compOrFocIm: SV, whichProps):
    returnList = True
    if isinstance(whichProps, ABParam):
      whichProps = [whichProps]
      returnList = False
    outProps = []
    for prop in whichProps:
      outProps.append(self.params.child(compOrFocIm.name, prop.name).value())
    if not returnList:
      outProps = outProps[0]
    return outProps

  def getCompProps(self, whichProps):
    return self._getProps(SV.COMP_PARAMS, whichProps)

  def getFocImgProps(self, whichProps):
    return self._getProps(SV.FOC_IMG_PARAMS, whichProps)

class _SchemeSingleton:
  _scheme = SchemeEditor()

  # Using properties intead of raw member ensures the same scheme is used across all class instances
  @property
  def scheme(self):
    return _SchemeSingleton._scheme

  @scheme.setter
  def scheme(self, newScheme: SchemeEditor):
    _SchemeSingleton._scheme = newScheme

# Encapsulate scheme within class so that changes to the scheme propagate to all GUI elements
SCHEME_HOLDER = _SchemeSingleton()