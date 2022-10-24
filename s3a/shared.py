from utilitys import ActionStack

from .constants import CONFIG_DIR, IMAGE_PROCESSORS_DIR, MULTI_PREDICTIONS_DIR
from .parameditors.algcollection import AlgorithmCollection
from .parameditors.table import TableData
from .plugins.settings import SettingsPlugin
from .plugins.shortucts import ShortcutsPlugin
from .processing import ImageProcess, ImgProcWrapper


class SharedAppSettings:
    def __init__(self):
        self.actionStack = ActionStack()

        self.tableData = TableData(makeFilter=True)
        self.filter = self.tableData.filter

        self.imageProcessCollection = AlgorithmCollection(
            ImgProcWrapper,
            ImageProcess,
            saveDir=IMAGE_PROCESSORS_DIR,
            template=CONFIG_DIR / "imageproc.yml",
        )
        self.multiPredictionCollection = AlgorithmCollection(
            saveDir=MULTI_PREDICTIONS_DIR, template=CONFIG_DIR / "multipred.yml"
        )

        self.settingsPlugin = SettingsPlugin()
        self.colorScheme = self.settingsPlugin.colorScheme
        self.generalProperties = self.settingsPlugin.generalProps

        self.shortcutsPlugin = ShortcutsPlugin()
        self.shortcuts = self.shortcutsPlugin.shortcuts
        self.quickLoader = self.shortcutsPlugin.quickLoader
