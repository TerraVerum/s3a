from pyqtgraph.Qt import  QtCore, QtWidgets

from Annotator.FRGraphics.drawopts import FRDrawOpts
from Annotator.constants import FR_CONSTS

Slot = QtCore.pyqtSlot

from .graphicsutils import create_addMenuAct
from .imageareas import MainImageArea, FocusedComp
from .tableview import CompTableView


class FRAnnotatorUI(QtWidgets.QMainWindow):
  def __init__(self):
    super().__init__()
    self.APP_TITLE = 'FICS Automated Component Detection and Evaluation Tool'
    self.CUR_COMP_LBL = 'Current Component ID:'

    self.setWindowTitle(self.APP_TITLE)

    self.setDockNestingEnabled(True)

    ALL_DRAW_SHAPES = (FR_CONSTS.DRAW_SHAPE_PAINT, FR_CONSTS.DRAW_SHAPE_FREE, FR_CONSTS.DRAW_SHAPE_RECT,
                       FR_CONSTS.DRAW_SHAPE_POLY, FR_CONSTS.DRAW_SHAPE_FG_BG)

    # -----
    # MAIN IMAGE AREA
    # -----
    # Bookkeeping widgets
    centralwidget = QtWidgets.QWidget(self)
    layout = QtWidgets.QVBoxLayout(centralwidget)

    # Important widgets
    self.mainImg = MainImageArea(centralwidget)
    self.mainDrawOpts = FRDrawOpts(centralwidget, ALL_DRAW_SHAPES,
                                   (FR_CONSTS.DRAW_ACT_ADD, FR_CONSTS.DRAW_ACT_SELECT, FR_CONSTS.DRAW_ACT_PAN))
    # TODO: Implement these options
    self.mainDrawOpts.shapeBtns[FR_CONSTS.DRAW_SHAPE_FREE].setEnabled(False)
    self.mainDrawOpts.shapeBtns[FR_CONSTS.DRAW_SHAPE_POLY].setEnabled(False)
    self.mainDrawOpts.shapeBtns[FR_CONSTS.DRAW_SHAPE_FG_BG].setEnabled(False)
    # Hookup
    self.setCentralWidget(centralwidget)
    layout.addWidget(self.mainDrawOpts)
    layout.addWidget(self.mainImg)

    # -----
    # FOCUSED IMAGE
    # -----
    # Bookkeeping widgets
    focusedImgDock = QtWidgets.QDockWidget('Focused Component', self)
    focusedImgContents = QtWidgets.QWidget(self)
    focusedLayout = QtWidgets.QVBoxLayout(focusedImgContents)
    focusedImgDock.setWidget(focusedImgContents)
    regionBtnLayout = QtWidgets.QHBoxLayout()

    # Important widgets
    self.focusedDrawOpts = FRDrawOpts(centralwidget, ALL_DRAW_SHAPES,
                                      (FR_CONSTS.DRAW_ACT_ADD, FR_CONSTS.DRAW_ACT_REM, FR_CONSTS.DRAW_ACT_PAN))
    # TODO: Implement these options
    for shape in ALL_DRAW_SHAPES:
      self.focusedDrawOpts.shapeBtns[shape].setEnabled(False)
    self.focusedDrawOpts.shapeBtns[FR_CONSTS.DRAW_SHAPE_PAINT].setEnabled(True)
    self.compImg = FocusedComp(focusedImgContents)
    self.curCompIdLbl = QtWidgets.QLabel(self.CUR_COMP_LBL)
    self.clearRegionBtn = QtWidgets.QPushButton('Clear', focusedImgContents)
    self.resetRegionBtn = QtWidgets.QPushButton('Reset', focusedImgContents)
    self.acceptRegionBtn = QtWidgets.QPushButton('Accept', focusedImgContents)
    self.acceptRegionBtn.setStyleSheet("background-color:lightgreen")

    # Hookup
    regionBtnLayout.addWidget(self.clearRegionBtn)
    regionBtnLayout.addWidget(self.resetRegionBtn)
    regionBtnLayout.addWidget(self.acceptRegionBtn)

    focusedLayout.addWidget(self.focusedDrawOpts)
    focusedLayout.addWidget(self.curCompIdLbl, 0, QtCore.Qt.AlignHCenter)
    focusedLayout.addWidget(self.compImg)
    focusedLayout.addLayout(regionBtnLayout)

    self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, focusedImgDock)

    # -----
    # COMPONENT TABLE
    # -----
    # Bookkeeping widgets
    tableDock = QtWidgets.QDockWidget('Component Table', self)
    tableContents = QtWidgets.QWidget(tableDock)
    tableLayout = QtWidgets.QVBoxLayout(tableContents)

    # Important widgets
    self.compTbl = CompTableView(tableDock)
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
    self.menuSettings = QtWidgets.QMenu('&Settings', self.menubar)
    self.menubar.addMenu(self.menuFile)
    self.menubar.addMenu(self.menuSettings)

    # File / Image
    self.openImgAct = create_addMenuAct(self.menuFile, '&Open Image')

    # File / layout
    self.menuLayout = create_addMenuAct(self.menuFile, 'Layout', True)
    self.saveLayout = create_addMenuAct(self.menuLayout, 'Save Layout')
    self.menuLayout.addSeparator()

    # File / components
    self.saveComps = create_addMenuAct(self.menuFile, 'Save Components')

    self.menuLoad_Components = create_addMenuAct(self.menuFile, 'Load Components', True)
    self.loadComps_merge = create_addMenuAct(self.menuLoad_Components, 'Update as Merge')
    self.loadComps_new = create_addMenuAct(self.menuLoad_Components, 'Append as New')

    self.setMenuBar(self.menubar)



if __name__ == '__main__':
  import sys
  app = QtWidgets.QApplication([])
  win = FRAnnotatorUI()
  win.showMaximized()
  sys.exit(app.exec_())