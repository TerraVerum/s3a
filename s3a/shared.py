from pyqtgraph.Qt import QtCore

class SharedAppSettings(QtCore.QObject):
  def __init__(self, parent=None):
    from s3a.constants import IMG_PROC_DIR, MULT_PRED_DIR, CFG_DIR
    from s3a.parameditors.algcollection import AlgCollection
    from s3a.parameditors.table import TableData
    from s3a.plugins.misc import SettingsPlugin, ShortcutsPlugin
    from s3a.processing import ImgProcWrapper, ImageProcess
    from utilitys import ActionStack

    super().__init__(parent)
    self.actionStack = ActionStack()

    self.tableData = TableData()
    self.filter = self.tableData.filter

    self.imgProcClctn = AlgCollection(ImgProcWrapper, ImageProcess, saveDir=IMG_PROC_DIR,
                                      template=CFG_DIR/'imageproc.yml')
    self.multiPredClctn = AlgCollection(saveDir=MULT_PRED_DIR, template=CFG_DIR/'multipred.yml')

    self.settingsPlg = SettingsPlugin()
    self.colorScheme = self.settingsPlg.colorScheme
    self.generalProps = self.settingsPlg.generalProps

    self.shortcutsPlg = ShortcutsPlugin()
    self.shortcuts = self.shortcutsPlg.shortcuts
    self.quickLoader = self.shortcutsPlg.quickLoader
