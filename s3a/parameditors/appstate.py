from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, List, Union

import pandas as pd
from qtextras import ParameterEditor, attemptFileLoad, seriesAsFrame

from ..constants import APP_STATE_DIR
from ..generalutils import (
    hierarchicalUpdate,
    safeCallFunction,
    safeCallFunctionList,
    concatAllowEmpty,
)
from ..logger import getAppLogger
from ..structures import FilePath

if TYPE_CHECKING:
    from .quickloader import QuickLoaderEditor


class AppStateEditor(ParameterEditor):
    def __init__(
        self,
        name=None,
        quickLoader: QuickLoaderEditor = None,
        directory: FilePath = APP_STATE_DIR,
        suffix=".appstate",
    ):
        # TODO: Add params to choose which features are saved, etc.
        super().__init__(name=name, directory=directory, suffix=suffix)
        self.quickLoader = quickLoader
        self.stateFunctionsDf = pd.DataFrame(
            columns=["importFunction", "exportFunction", "required"]
        )
        self.loading = False

        self.startupSettings = {}

    def saveParameterValues(
        self, saveName: str = None, stateDict: dict = None, **kwargs
    ):
        if saveName is None:
            saveName = self.RECENT_STATE_FILENAME
        if stateDict is None:
            # TODO: May be good in the future to be able to choose which should be saved
            legitKeys = self.stateFunctionsDf.index
            exportFuncs = self.stateFunctionsDf.exportFunction
            saveOnExitDir = self.directory / "saved_on_exit"
            saveOnExitDir.mkdir(exist_ok=True)
            rets, errs = safeCallFunctionList(
                legitKeys, exportFuncs, [[saveOnExitDir]] * len(legitKeys)
            )
            updateDict = {k: ret for k, ret in zip(legitKeys, rets) if ret is not None}
            stateDict = dict(**updateDict)
            for editor in self.quickLoader.listModel.uniqueEditors:
                if editor.stateName == editor.getDefaultStateName().stem:
                    curSaveName = str(saveOnExitDir / editor.name)
                else:
                    curSaveName = editor.stateName
                formattedName = editor.name.replace(" ", "").lower()
                editor.saveParameterValues(curSaveName)
                stateDict.update({formattedName: curSaveName})
        else:
            errs = []

        ret = super().saveParameterValues(saveName, stateDict, **kwargs)
        self.raiseErrorMessageIfNeeded(errs)
        return ret

    def loadParameterValues(
        self,
        stateName: Union[str, Path] = None,
        stateDict: dict = None,
        **kwargs,
    ):
        self.loading = True
        # Copy old settings to put them back after loading
        oldStartup = self.startupSettings.copy()
        if stateName is None:
            stateName = self.RECENT_STATE_FILENAME
        stateName = self.stateManager.formatFileName(stateName)
        if isinstance(stateDict, str):
            stateDict = {"quickloader": stateDict}

        try:  # try block to ensure `loading` is false after
            stateDict = self._parseStateDictIncludeRequired(stateName, stateDict)
            initialStateDict = stateDict.copy()

            # It's possible for some functions (e.g. project load) to add or remove
            # startup args, so chack for this
            hierarchicalUpdate(self.startupSettings, kwargs)

            def nextKey():
                hierarchicalUpdate(stateDict, self.startupSettings)
                self.startupSettings.clear()
                legitKeys = self.stateFunctionsDf.index.intersection(stateDict)
                if legitKeys.size > 0:
                    return legitKeys[0]

            key = nextKey()
            rets, errs = [], {}
            while key:
                importFunc = self.stateFunctionsDf.loc[key, "importFunction"]
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
                    stacklevel=3,
                )
            if stateDict:
                self.quickLoader.buildFromStartupParameters(stateDict)
            ret = super().loadParameterValues(stateName, initialStateDict)
        finally:
            self.loading = False
            hierarchicalUpdate(self.startupSettings, oldStartup)
        return ret

    def _parseStateDictIncludeRequired(
        self,
        stateName: Union[str, Path],
        stateDict: dict = None,
    ):
        defaults = attemptFileLoad(self.RECENT_STATE_FILENAME, missingOk=True) or {}
        if stateDict is None:
            stateDict = attemptFileLoad(stateName, missingOk=True) or {}
        for k in self.stateFunctionsDf.index[self.stateFunctionsDf["required"]]:
            stateDict.setdefault(k, defaults.get(k))
        return stateDict

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
            index=self.stateFunctionsDf.columns,
            dtype=object,
        )
        if index is not None:
            # First, shift old entries
            df = self.stateFunctionsDf
            self.stateFunctionsDf = concatAllowEmpty(
                [df.iloc[:index], seriesAsFrame(newRow), df.iloc[index:]]
            )
        else:
            self.stateFunctionsDf = concatAllowEmpty(
                [self.stateFunctionsDf, seriesAsFrame(newRow)]
            )

    @property
    def RECENT_STATE_FILENAME(self):
        return self.directory / f"recent{self.suffix}"
