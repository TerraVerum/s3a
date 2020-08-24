from typing import Union

from pyqtgraph.Qt import QtCore, QtWidgets

from s3a import FR_SINGLETON, FRParamEditor
from s3a.generalutils import frPascalCaseToTitle
from s3a.parameditors import FRParamEditorDockGrouping
from s3a.parameditors.genericeditor import FRTableFieldAssistant
from s3a.views.imageareas import FRFocusedImage


class FRTableVertsPlugin(FRTableFieldAssistant):
  name = 'Vertices'
  focusedImg: FRFocusedImage = None
  curCompIdLbl: QtWidgets.QLabel = None

  @classmethod
  def __initEditorParams__(cls):
    FRFocusedImage.__initEditorParams__()
    cls.toolsEditor = FRFocusedImage.toolsEditor
    cls.toolsEditor.name = 'Tools'
    FRFocusedImage.procCollection.name = 'Processor'
    dockGroup = FRParamEditorDockGrouping([cls.toolsEditor, FRFocusedImage.procCollection],
                                          frPascalCaseToTitle(cls.name))
    cls.docks = dockGroup

  def makeWidget(self):
    if self.widget is not None:
      return self.widget
    self.focusedImg = FRFocusedImage()
    self.curCompIdLbl = QtWidgets.QLabel('Current Component ID:')

    focusedImgContents = QtWidgets.QWidget()
    focusedLayout = QtWidgets.QVBoxLayout(focusedImgContents)
    focusedLayout.addWidget(self.focusedImg.drawOptsWidget)
    focusedLayout.addWidget(self.focusedImg.toolsGrp)
    focusedLayout.addWidget(self.curCompIdLbl, 0, QtCore.Qt.AlignHCenter)
    focusedLayout.addWidget(self.focusedImg)

    self.focusedImg.showGuiBtns.sigValueChanged.connect(
      lambda _p, val: self.focusedImg.toolsGrp.setVisible(val))

    self.compSer = self.focusedImg.compSer

    self.widget = focusedImgContents
    return focusedImgContents