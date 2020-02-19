from typing import Tuple, Dict

from PyQt5 import QtWidgets, QtGui, QtCore

from Annotator.params import FRParamGroup
from ..constants import FR_CONSTS
from ..params import FRParam

class FRDrawOpts(QtWidgets.QWidget):
  def __init__(self, parent=None, shapes: Tuple[FRParamGroup, ...]=None, actions: Tuple[FRParamGroup, ...]=None):
    super().__init__(parent)
    # Create 2 layout versions so on resize the group boxes can 'wrap'
    self.topLayout = QtWidgets.QHBoxLayout()
    self.setLayout(self.topLayout)

    # SHAPES
    shapeUiGroup, self.shapeBtnGroup, self.shapeBtns = self._create_addBtnToGroup(shapes)
    self.topLayout.addWidget(shapeUiGroup)
    # ACTIONS
    actionUiGroup, self.actionBtnGroup, self.actionBtns = self._create_addBtnToGroup(actions)
    self.topLayout.addWidget(actionUiGroup)
    self.topLayout.setDirection(self.topLayout.LeftToRight)
    self.horizWidth = self.layout().minimumSize().width()

  def resizeEvent(self, ev: QtGui.QResizeEvent) -> None:
    if ev.size().width() < self.horizWidth + 10:
      self.topLayout.setDirection(self.topLayout.TopToBottom)
    else:
      self.topLayout.setDirection(self.topLayout.LeftToRight)
    super().resizeEvent(ev)

    # RESIZE BEHAVIOR
  def _create_addBtnToGroup(self, whichBtns: FR_CONSTS) -> Tuple:
    uiGroup = QtWidgets.QGroupBox(self)
    uiLayout = QtWidgets.QHBoxLayout(uiGroup)
    btnGroup = QtWidgets.QButtonGroup(uiGroup)
    btnDict: Dict[FRParam, QtWidgets.QPushButton] = {}
    for btnParam in whichBtns: # type: FRParam
      # TODO: Use picture instead of name
      newBtn = QtWidgets.QPushButton(btnParam.name, uiGroup)
      newBtn.setCheckable(True)
      btnGroup.addButton(newBtn)
      uiLayout.addWidget(newBtn)
      btnDict[btnParam] = newBtn
    return uiGroup, btnGroup, btnDict

def _create_addActionsToToolbar(toolbar: QtWidgets.QToolBar, whichBtns: FR_CONSTS) -> QtWidgets.QActionGroup:
  uiGroup = QtWidgets.QGroupBox('Test', toolbar)
  actionGroup = QtWidgets.QActionGroup(uiGroup)
  actionGroup.setExclusive(True)
  for btnParam in whichBtns: # type: FRParam
    # TODO: Use picture instead of name
    newAction = actionGroup.addAction(btnParam.name)
    newAction.setCheckable(True)
  toolbar.addWidget(uiGroup)
  return actionGroup