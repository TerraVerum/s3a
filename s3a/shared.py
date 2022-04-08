from utilitys import ActionStack
from .constants import IMG_PROC_DIR, MULT_PRED_DIR, CFG_DIR
from .parameditors.algcollection import AlgCollection
from .parameditors.table import TableData
from .plugins.settings import SettingsPlugin
from .plugins.shortucts import ShortcutsPlugin
from .processing import ImgProcWrapper, ImageProcess


class SharedAppSettings:
    def __init__(self):
        self.actionStack = ActionStack()

        self.tableData = TableData(makeFilter=True)
        self.filter = self.tableData.filter

        self.imgProcClctn = AlgCollection(
            ImgProcWrapper,
            ImageProcess,
            saveDir=IMG_PROC_DIR,
            template=CFG_DIR / "imageproc.yml",
        )
        self.multiPredClctn = AlgCollection(
            saveDir=MULT_PRED_DIR, template=CFG_DIR / "multipred.yml"
        )

        self.settingsPlg = SettingsPlugin()
        self.colorScheme = self.settingsPlg.colorScheme
        self.generalProps = self.settingsPlg.generalProps

        self.shortcutsPlg = ShortcutsPlugin()
        self.shortcuts = self.shortcutsPlg.shortcuts
        self.quickLoader = self.shortcutsPlg.quickLoader
