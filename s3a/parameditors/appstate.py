import typing as t
from pathlib import Path
from typing import List, Dict, Union, Callable, Any

import numpy as np
import pandas as pd
from pyqtgraph.parametertree import Parameter
from utilitys import ParamEditor, fns

from s3a import PRJ_SINGLETON
from s3a.constants import APP_STATE_DIR
from s3a.generalutils import safeCallFuncList
from s3a.structures import FilePath
from utilitys.fns import serAsFrame


class AppStateEditor(ParamEditor):

  def __init__(self, parent=None, paramList: List[Dict] = None,
               saveDir: FilePath = APP_STATE_DIR, fileType='param', name=None,
               topTreeChild: Parameter = None):
    # TODO: Add params to choose which features are saved, etc.
    super().__init__(parent, paramList, saveDir, fileType, name, topTreeChild)
    self._stateFuncsDf = pd.DataFrame(columns=['importFuncs', 'exportFuncs', 'required'])

  def saveParamValues(self, saveName: str=None, paramState: dict=None, **kwargs):
    if saveName is None:
      saveName = self.RECENT_STATE_FNAME
    if paramState is None:
      # TODO: May be good in the future to be able to choose which should be saved
      legitKeys = self._stateFuncsDf.index
      exportFuncs = self._stateFuncsDf.exportFuncs
      saveOnExitDir = self.saveDir/'saved_on_exit'
      saveOnExitDir.mkdir(exist_ok=True)
      rets, errs = safeCallFuncList(legitKeys, exportFuncs, [[saveOnExitDir]] * len(legitKeys))
      updateDict = {k: ret for k, ret in zip(legitKeys, rets) if ret is not None}
      paramState = dict(Parameters=paramState, **updateDict)
      for editor in PRJ_SINGLETON.quickLoader.listModel.uniqueEditors:
        editor.applyChanges()
        if editor.lastAppliedName == 'Default':
          continue
        curSaveName = str(saveOnExitDir/editor.name)
        formattedName = editor.name.replace(' ', '').lower()
        editor.saveParamValues(curSaveName, blockWrite=False)
        paramState.update({formattedName: curSaveName})
    else:
      errs = []

    ret = super().saveParamValues(saveName, paramState, **kwargs)
    self.raiseErrMsgIfNeeded(errs)
    return ret

  def loadParamValues(self, stateName: Union[str, Path]=None, stateDict: dict = None,
                     overrideDict: dict=None, **kwargs):
    if stateName is None:
      stateName = self.RECENT_STATE_FNAME
    if not stateName.exists() and stateDict is None:
      stateDict = {}
    if isinstance(stateDict, str):
      stateDict = {'quickloader': stateDict}
    stateDict = self._parseStateDict_includeRequired(stateName, stateDict)
    if overrideDict is not None:
      stateDict.update(overrideDict)
    paramDict = stateDict.pop('Parameters', {}) or {}
    stateDictKeys = list(stateDict.keys())
    legitKeys = self._stateFuncsDf.index.intersection(stateDictKeys)
    importFuncs = self._stateFuncsDf.loc[legitKeys, 'importFuncs']
    args = []
    for k in legitKeys:
      args.append((stateDict.pop(k, None),))
    _, errs = safeCallFuncList(legitKeys, importFuncs, args)
    if len(np.setdiff1d(stateDict.keys(), legitKeys)) > 0:
      PRJ_SINGLETON.quickLoader.buildFromStartupParams(stateDict)
    ret = super().loadParamValues(stateName, paramDict, **kwargs)
    return ret

  def _parseStateDict_includeRequired(self, stateName: t.Union[str, Path], stateDict: dict = None):
    try:
      out = self._parseStateDict(stateName, stateDict)
    except FileNotFoundError:
      out = {}
    for k in self._stateFuncsDf.index[self._stateFuncsDf['required']]:
      out.setdefault(k, None)
    return out

  @staticmethod
  def raiseErrMsgIfNeeded(errMsgs: List[str]):
    if len(errMsgs) > 0:
      err = IOError('Errors were encountered for the following parameters'
                         ' (shown as <parameter>: <exception>)\n'
                       + "\n\n".join(errMsgs))
      fns.raiseErrorLater(err)


  def addImportExportOpts(self, optName: str, importFunc: Callable[[str], Any],
                          exportFunc: Callable[[Path], str], index:int=None,
                          required=False):
    """
    Main interface to the app state editor. By providing import and export functions,
    various aspects of the program state can be loaded and saved on demand.

    :param optName: What should this save option be called? E.g. when providing a
      load and save for annotation data, this is 'annotations'.
    :param importFunc: Function called when importing saved data. Takes in a
      full file path
    :param exportFunc: Function to save the app data. Input is a full folder path. Expects
      the output to be a full file path of the saved file. This file is then passed to
      'importFunc' on loading a param state
    :param index: Where to place this function. In most cases, this won't matter. However, some imports must be
      performed first / last otherwise app behavior may be undefined. In these cases, passing a value for index ensures
      correct placement of the import/export pair. By default, the function is added to the end of the import/export list.
    :param required: If *True*, this parameter is required every time param values are loaded.
      In the case it is missing from a load, the param editor first attempts to fetch this option
      from the most recent saved state.
    """
    newRow = pd.Series([importFunc, exportFunc, required], name=optName,
                       index=self._stateFuncsDf.columns)
    if index is not None:
      # First, shift old entries
      df = self._stateFuncsDf
      self._stateFuncsDf = pd.concat([df.iloc[:index], serAsFrame(newRow), df.iloc[index:]])
    else:
      self._stateFuncsDf: pd.DataFrame = self._stateFuncsDf.append(newRow)

  @property
  def RECENT_STATE_FNAME(self):
      return self.saveDir/f'recent.{self.fileType}'
