from typing import Tuple

from bidict import bidict
from pyqtgraph.Qt import QtWidgets, QtGui

from s3a.projectvars import FR_CONSTS
from s3a.structures import FRParam, FRParamGroup


class FRDrawOpts(QtWidgets.QWidget):
  def __init__(self, parent=None, shapes: Tuple[FRParamGroup, ...]=None, actions: Tuple[FRParamGroup, ...]=None):
    """
    Creates a draw options widget hosting both shape and action selection buttons.
    :param parent: UI widget whose destruction will also destroy these widgets
    :param shapes: Shape options that will appear in the widget
    :param actions: Action optiosn that will appear in the widget
    """
    super().__init__(parent)
    # Create 2 layout versions so on resize the group boxes can 'wrap'
    self.topLayout = QtWidgets.QHBoxLayout()
    self.setLayout(self.topLayout)

    # SHAPES
    shapeUiGroup, self.shapeBtnGroup, self.shapeBtnParamMap = self._create_addBtnToGroup(shapes, "Shapes")
    self.topLayout.addWidget(shapeUiGroup)
    # ACTIONS
    actionUiGroup, self.actionBtnGroup, self.actionBtnParamMap = self._create_addBtnToGroup(actions, "Actions")
    self.topLayout.addWidget(actionUiGroup)
    self.topLayout.setDirection(self.topLayout.LeftToRight)
    self.horizWidth = self.layout().minimumSize().width()

  def resizeEvent(self, ev: QtGui.QResizeEvent) -> None:
    if self.width() < self.horizWidth + 30:
      self.topLayout.setDirection(self.topLayout.TopToBottom)
    else:
      self.topLayout.setDirection(self.topLayout.LeftToRight)
    super().resizeEvent(ev)

    # RESIZE BEHAVIOR
  def _create_addBtnToGroup(self, whichBtns: FR_CONSTS, groupTitle: str) \
      -> Tuple[QtWidgets.QGroupBox, QtWidgets.QButtonGroup, bidict]:
    uiGroup = QtWidgets.QGroupBox(self)
    uiGroup.setTitle(groupTitle)
    uiLayout = QtWidgets.QHBoxLayout(uiGroup)
    btnGroup = QtWidgets.QButtonGroup(uiGroup)
    # btnDict: Dict[QtWidgets.QPushButton, FRParam] = {}
    btnDict = bidict()
    for btnParam in whichBtns: # type: FRParam
      # TODO: Use picture instead of name
      if btnParam.value is not None:
        newBtn = QtWidgets.QPushButton(QtGui.QIcon(btnParam.value), '', uiGroup)
        newBtn.setToolTip(btnParam.name)
      else:
        newBtn = QtWidgets.QPushButton(btnParam.name, uiGroup)
      newBtn.setCheckable(True)
      btnGroup.addButton(newBtn)
      uiLayout.addWidget(newBtn)
      btnDict[newBtn] = btnParam
    return uiGroup, btnGroup, btnDict
    
  def selectOpt(self, shapeOrAction: FRParam):
    """
    Programmatically selects a shape or action from the existing button group.
    Whether a shape or action is passed in is inferred from which button group
    :param:`shapeOrAction` belongs to

    :param shapeOrAction: The button to select
    :return: None
    """
    # TODO: This should probably be more robust
    if shapeOrAction in self.shapeBtnParamMap.inverse:
      self.shapeBtnParamMap.inverse[shapeOrAction].setChecked(True)
    elif shapeOrAction in self.actionBtnParamMap.inverse:
      self.actionBtnParamMap.inverse[shapeOrAction].setChecked(True)
  

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