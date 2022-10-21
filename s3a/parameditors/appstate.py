import warnings
from pathlib import Path
from typing import List, Dict, Union, Callable, Any

import pandas as pd
from pyqtgraph.parametertree import Parameter
from utilitys import ParamEditor, EditorPropsMixin
from utilitys.fns import serAsFrame, attemptFileLoad

from ..constants import APP_STATE_DIR
from ..generalutils import safeCallFunctionList, hierarchicalUpdate, safeCallFunction
from ..logger import getAppLogger
from ..shared import SharedAppSettings
from ..structures import FilePath


class AppStateEditor(EditorPropsMixin, ParamEditor):
    def __initEditorParams__(self, shared: SharedAppSettings):
        self.quickLoader = shared.quickLoader

    def __init__(
        self,
        parent=None,
        paramList: List[Dict] = None,
        saveDir: FilePath = APP_STATE_DIR,
        fileType="parameter",
        name=None,
        topTreeChild: Parameter = None,
    ):
        # TODO: Add params to choose which features are saved, etc.
        super().__init__(parent, paramList, saveDir, fileType, name, topTreeChild)
        self.stateFuncsDf = pd.DataFrame(
            columns=["importFunction", "exportFunction", "required"]
        )
        self.loading = False

        self.startupSettings = {}

    def saveParamValues(self, saveName: str = None, paramState: dict = None, **kwargs):
        if saveName is None:
            saveName = self.RECENT_STATE_FNAME
        if paramState is None:
            # TODO: May be good in the future to be able to choose which should be saved
            legitKeys = self.stateFuncsDf.index
            exportFuncs = self.stateFuncsDf.exportFunction
            saveOnExitDir = self.saveDir / "saved_on_exit"
            saveOnExitDir.mkdir(exist_ok=True)
            rets, errs = safeCallFunctionList(
                legitKeys, exportFuncs, [[saveOnExitDir]] * len(legitKeys)
            )
            updateDict = {k: ret for k, ret in zip(legitKeys, rets) if ret is not None}
            paramState = dict(**updateDict)
            for editor in self.quickLoader.listModel.uniqueEditors:
                if editor.stateName == "Default":
                    curSaveName = str(saveOnExitDir / editor.name)
                else:
                    curSaveName = editor.stateName
                formattedName = editor.name.replace(" ", "").lower()
                editor.saveParamValues(curSaveName)
                paramState.update({formattedName: curSaveName})
        else:
            errs = []

        ret = super().saveParamValues(saveName, paramState, **kwargs)
        self.raiseErrorMessageIfNeeded(errs)
        return ret

    def loadParamValues(
        self,
        stateName: Union[str, Path] = None,
        stateDict: dict = None,
        **kwargs,
    ):
        self.loading = True
        # Copy old settings to put them back after loading
        oldStartup = self.startupSettings.copy()
        try:  # try block to ensure loading is false after
            if stateName is None:
                stateName = self.RECENT_STATE_FNAME
            stateName = self.formatFileName(stateName)
            if not stateName.exists() and stateDict is None:
                stateDict = {}
            if isinstance(stateDict, str):
                stateDict = {"quickloader": stateDict}
            stateDict = self._parseStateDictIncludeRequired(stateName, stateDict)
            paramDict = stateDict.pop("Parameters", {}) or {}

            # It's possible for some functions (e.g. project load) to add or remove
            # startup args, so chack for this
            hierarchicalUpdate(self.startupSettings, kwargs)

            def nextKey():
                hierarchicalUpdate(stateDict, self.startupSettings)
                self.startupSettings.clear()
                legitKeys = self.stateFuncsDf.index.intersection(stateDict)
                if legitKeys.size > 0:
                    return legitKeys[0]

            key = nextKey()
            rets, errs = [], {}
            while key:
                importFunc = self.stateFuncsDf.loc[key, "importFunction"]
                arg = stateDict.pop(key, None)
                curRet, curErr = safeCallFunction(key, importFunc, arg)
                rets.append(curRet)
                if curErr:
                    errs[key] = curErr
                key = nextKey()
            if errs:
                warnings.warn(
                    "The following settings could not be loaded (shown as [setting]: "
                    "[exception])\n" + "\n\n".join(errs.values()),
                    UserWarning,
                )
            if stateDict:
                self.quickLoader.buildFromStartupParameters(stateDict)
            ret = super().loadParamValues(stateName, paramDict)
        finally:
            self.loading = False
            hierarchicalUpdate(self.startupSettings, oldStartup)
        return ret

    def _parseStateDictIncludeRequired(
        self,
        stateName: Union[str, Path],
        stateDict: dict = None,
    ):
        if self.RECENT_STATE_FNAME.exists():
            defaults = attemptFileLoad(self.RECENT_STATE_FNAME)
        else:
            defaults = {}
        try:
            out = self._parseStateDict(stateName, stateDict)
        except FileNotFoundError:
            out = {}
        for k in self.stateFuncsDf.index[self.stateFuncsDf["required"]]:
            out.setdefault(k, defaults.get(k))
        return out

    @staticmethod
    def raiseErrorMessageIfNeeded(errorMessages: List[str]):
        if len(errorMessages) > 0:
            err = IOError(
                "Errors were encountered for the following parameters"
                " (shown as [parameter]: [exception])\n" + "\n\n".join(errorMessages)
            )
            getAppLogger(__name__).critical(err)

    def addImportExportOptions(
        self,
        optionName: str,
        importFunction: Callable[[str], Any],
        exportFunction: Callable[[Path], str],
        index: int = None,
        required=False,
    ):
        """
        Main interface to the app state editor. By providing import and export functions,
        various aspects of the program state can be loaded and saved on demand.

        Parameters
        ----------
        optionName
            What should this save option be called? E.g. when providing a load and save
            for annotation data, this is 'annotations'.
        importFunction
            Function called when importing saved data. Takes in a full file path
        exportFunction
            Function to save the app data. Input is a full folder path. Expects the
            output to be a full file path of the saved file. This file is then passed
            to 'importFunction' on loading a parameter state. If *None* is returned, the value
            is not stored.
        index
            Where to place this function. In most cases, this won't matter. However,
            some imports must be performed first / last otherwise app behavior may be
            undefined. In these cases, passing a value for index ensures correct
            placement of the import/export pair. By default, the function is added to
            the end of the import/export list.
        required
            If *True*, this parameter is required every time parameter values are loaded.
            In the case it is missing from a load, the parameter editor first attempts to
            fetch this option from the most recent saved state.
        """
        newRow = pd.Series(
            [importFunction, exportFunction, required],
            name=optionName,
            index=self.stateFuncsDf.columns,
        )
        if index is not None:
            # First, shift old entries
            df = self.stateFuncsDf
            self.stateFuncsDf = pd.concat(
                [df.iloc[:index], serAsFrame(newRow), df.iloc[index:]]
            )
        else:
            self.stateFuncsDf: pd.DataFrame
            self.stateFuncsDf.loc[optionName] = newRow

    @property
    def RECENT_STATE_FNAME(self):
        return self.saveDir / f"recent.{self.fileType}"
