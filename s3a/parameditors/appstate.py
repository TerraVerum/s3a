from pathlib import Path
from typing import List, Dict, Type, Union, Callable, Any

import pandas as pd
import numpy as np
from pyqtgraph.parametertree import Parameter

from s3a.generalutils import safeCallFuncList
from s3a.graphicsutils import raiseErrorLater
from s3a.constants import APP_STATE_DIR
from s3a.structures import FilePath, PrjParam, S3AIOError
from s3a.parameditors import ParamEditor
from s3a import FR_SINGLETON


class AppStateEditor(ParamEditor):

  def __init__(self, parent=None, paramList: List[Dict] = None,
               saveDir: FilePath = APP_STATE_DIR, fileType='param', name=None,
               topTreeChild: Parameter = None):
    # TODO: Add params to choose which features are saved, etc.
    super().__init__(parent, paramList, saveDir, fileType, name, topTreeChild)
    self._stateFuncsDf = pd.DataFrame(columns=['importFuncs', 'exportFuncs'])

  def saveParamState(self, saveName: str=None, paramState: dict=None, **kwargs):
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
      for editor in FR_SINGLETON.quickLoader.listModel.uniqueEditors:
        editor.applyChanges()
        if editor.lastAppliedName == 'Default':
          continue
        curSaveName = str(saveOnExitDir/editor.name)
        formattedName = editor.name.replace(' ', '').lower()
        editor.saveParamState(curSaveName, blockWrite=False)
        paramState.update({formattedName: curSaveName})
    else:
      errs = []

    ret = super().saveParamState(saveName, paramState, **kwargs)
    self.raiseErrMsgIfNeeded(errs)
    return ret

  def loadParamState(self, stateName: Union[str, Path]=None, stateDict: dict = None,
                     overrideDict: dict=None, **kwargs):
    if stateName is None:
      stateName = self.RECENT_STATE_FNAME
    if not stateName.exists() and stateDict is None:
      stateDict = {}
    if isinstance(stateDict, str):
      stateDict = {'quickloader': stateDict}
    stateDict = self._parseStateDict(stateName, stateDict)
    if overrideDict is not None:
      stateDict.update(overrideDict)
    paramDict = stateDict.pop('Parameters', {})
    stateDictKeys = list(stateDict.keys())
    legitKeys = self._stateFuncsDf.index.intersection(stateDictKeys)
    importFuncs = self._stateFuncsDf.loc[legitKeys, 'importFuncs']
    args = []
    for k in legitKeys:
      args.append((stateDict.pop(k, None),))
    _, errs = safeCallFuncList(legitKeys, importFuncs, args)
    if len(np.setdiff1d(stateDict.keys(), legitKeys)) > 0:
      FR_SINGLETON.quickLoader.buildFromStartupParams(stateDict)
    ret = super().loadParamState(stateName, paramDict, **kwargs)
    return ret

  @staticmethod
  def raiseErrMsgIfNeeded(errMsgs: List[str]):
    if len(errMsgs) > 0:
      err = S3AIOError('Errors were encountered for the following parameters'
                         ' (shown as <parameter>: <exception>)\n'
                       + "\n\n".join(errMsgs))
      raiseErrorLater(err)


  def addImportExportOpts(self, optName: str, importFunc: Callable[[str], Any],
                          exportFunc: Callable[[Path], str], index:int=None):
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
    """
    newRow = pd.Series([importFunc, exportFunc], name=optName,
                       index=self._stateFuncsDf.columns)
    if index is not None:
      # First, shift old entries
      df = self._stateFuncsDf
      self._stateFuncsDf = pd.concat([df.iloc[:index], newRow.to_frame().T, df.iloc[index:]])
    else:
      self._stateFuncsDf: pd.DataFrame = self._stateFuncsDf.append(newRow)

  @property
  def RECENT_STATE_FNAME(self):
      return self.saveDir/f'recent.{self.fileType}'
