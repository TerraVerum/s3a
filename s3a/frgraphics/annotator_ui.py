from pyqtgraph.Qt import  QtCore, QtWidgets

Slot = QtCore.Slot

from .graphicsutils import create_addMenuAct, FRPopupLineEditor
from .imageareas import FRMainImage, FRFocusedImage
from .tableview import FRCompTableView


class FRAnnotatorUI(QtWidgets.QMainWindow):
  def __init__(self):
    super().__init__()
    self.APP_TITLE = 'FICS Semi-Supervised Semantic Annotator'
    self.CUR_COMP_LBL = 'Current Component ID:'

    self.setWindowTitle(self.APP_TITLE)

    self.setDockNestingEnabled(True)
    self.setTabPosition(QtCore.Qt.AllDockWidgetAreas, QtWidgets.QTabWidget.North)


    # -----
    # MAIN IMAGE AREA
    # -----
    # Bookkeeping widgets
    centralwidget = QtWidgets.QWidget(self)
    layout = QtWidgets.QVBoxLayout(centralwidget)

    # Important widgets
    self.mainImg = FRMainImage(centralwidget)
    # Hookup
    self.setCentralWidget(centralwidget)
    layout.addWidget(self.mainImg.drawOptsWidget)
    layout.addWidget(self.mainImg)

    # -----
    # FOCUSED IMAGE
    # -----
    # Bookkeeping widgets
    focusedImgDock = QtWidgets.QDockWidget('Focused Image', self)
    focusedImgContents = QtWidgets.QWidget(self)
    focusedLayout = QtWidgets.QVBoxLayout(focusedImgContents)
    focusedImgDock.setWidget(focusedImgContents)
    focusedImgDock.setObjectName('Focused Image Dock')
    regionBtnLayout = QtWidgets.QHBoxLayout()

    # Important widgets
    self.focusedImg = FRFocusedImage(focusedImgContents)
    self.curCompIdLbl = QtWidgets.QLabel(self.CUR_COMP_LBL)
    self.clearRegionBtn = QtWidgets.QPushButton('Clear', focusedImgContents)
    self.resetRegionBtn = QtWidgets.QPushButton('Reset', focusedImgContents)
    self.acceptRegionBtn = QtWidgets.QPushButton('Accept', focusedImgContents)
    self.acceptRegionBtn.setStyleSheet("background-color:lightgreen")

    # Hookup
    regionBtnLayout.addWidget(self.clearRegionBtn)
    regionBtnLayout.addWidget(self.resetRegionBtn)
    regionBtnLayout.addWidget(self.acceptRegionBtn)

    focusedLayout.addWidget(self.focusedImg.drawOptsWidget)
    focusedLayout.addWidget(self.curCompIdLbl, 0, QtCore.Qt.AlignHCenter)
    focusedLayout.addWidget(self.focusedImg)
    focusedLayout.addLayout(regionBtnLayout)

    self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, focusedImgDock)

    # -----
    # COMPONENT TABLE
    # -----
    # Bookkeeping widgets
    tableDock = QtWidgets.QDockWidget('Component Table', self)
    tableDock.setObjectName('Component Table Dock')
    tableContents = QtWidgets.QWidget(tableDock)
    tableLayout = QtWidgets.QVBoxLayout(tableContents)

    # Important widgets
    self.compTbl = FRCompTableView(tableDock)
    self.compTbl.setSortingEnabled(True)
    self.compTbl.setAlternatingRowColors(True)

    # UI creation
    tableLayout.addWidget(self.compTbl)
    tableDock.setWidget(tableContents)

    self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, tableDock)

    # -----
    # MENU BAR
    # -----
    # Top Level
    self.menubar = QtWidgets.QMenuBar(self)
    self.menuFile = QtWidgets.QMenu('&File', self.menubar)
    self.menuEdit = QtWidgets.QMenu('&Edit', self.menubar)
    self.menuAnalytics = QtWidgets.QMenu('&Analytics', self.menubar)

    toolbar = self.addToolBar('Parameter Editors')
    toolbar.setObjectName('Parameter Edtor Toolbar')
    self.paramTools = QtWidgets.QMenuBar()
    toolbar.addWidget(self.paramTools)

    self.menubar.addMenu(self.menuFile)
    self.menubar.addMenu(self.menuEdit)
    self.menubar.addMenu(self.menuAnalytics)

    # File / Image
    self.openImgAct = create_addMenuAct(self, self.menuFile, '&Open Image')

    # File / layout
    self.menuLayout = create_addMenuAct(self, self.menuFile, '&Layout', True)
    self.saveLayoutAct = create_addMenuAct(self, self.menuLayout, 'Save Layout')
    self.menuLayout.addSeparator()

    # File / components
    self.menuExport = create_addMenuAct(self, self.menuFile, '&Export...', True)
    self.exportCompListAct = create_addMenuAct(self, self.menuExport, '&Component List')
    self.exportLabelImgAct = create_addMenuAct(self, self.menuExport, '&Labeled Image')

    self.menuLoad_Components = create_addMenuAct(self, self.menuFile, '&Import', True)
    self.loadCompsAct_merge = create_addMenuAct(self, self.menuLoad_Components, 'Update as &Merge')
    self.loadCompsAct_new = create_addMenuAct(self, self.menuLoad_Components, 'Append as &New')

    # File / autosave
    self.menuAutosave = create_addMenuAct(self, self.menuFile, '&Autosave...', True)
    self.startAutosaveAct = create_addMenuAct(self, self.menuAutosave, 'Star&t Autosave')
    self.stopAutosaveAct = create_addMenuAct(self, self.menuAutosave, 'Sto&p Autosave')


    # Edit
    self.undoAct = create_addMenuAct(self, self.menuEdit, '&Undo')
    self.undoAct.setShortcut('Ctrl+Z')
    self.redoAct = create_addMenuAct(self, self.menuEdit, '&Redo')
    self.redoAct.setShortcut('Ctrl+Y')

    # Analytics
    self.newCompAnalyticsAct = create_addMenuAct(self, self.menuAnalytics, 'Newest Added Component')
    self.modCompAnalyticsAct = create_addMenuAct(self, self.menuAnalytics, 'Modified Component')


    self.setMenuBar(self.menubar)




if __name__ == '__main__':
  import sys
  app = QtWidgets.QApplication([])
  win = FRAnnotatorUI()
  win.showMaximized()
  sys.exit(app.exec_())