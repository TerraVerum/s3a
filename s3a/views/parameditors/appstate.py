from pathlib import Path
from typing import List, Dict, Type, Union, Callable, Any, Tuple

import pandas as pd
import numpy as np
from pyqtgraph.parametertree import Parameter

from s3a.generalutils import _safeCallFuncList
from s3a.graphicsutils import raiseErrorLater
from s3a.projectvars import APP_STATE_DIR
from s3a.structures import FilePath, FRParam, FRAppIOError
from s3a.views.parameditors import FRParamEditor
from s3a import FR_SINGLETON


class FRAppStateEditor(FRParamEditor):

  def __init__(self, parent=None, paramList: List[Dict] = None,
               saveDir: FilePath = APP_STATE_DIR, fileType='param', name=None,
               topTreeChild: Parameter = None, registerCls: Type = None,
               registerParam: FRParam = None):
    # TODO: Add params to choose which features are saved, etc.
    super().__init__(parent, paramList, saveDir, fileType, name, topTreeChild,
                     registerCls, registerParam)
    self._stateFuncsDf = pd.DataFrame(columns=['importFuncs', 'exportFuncs'])

  def saveParamState(self, saveName: str=None, paramState: dict=None,
                     allowOverwriteDefault=False, blockWrite=False):
    if saveName is None:
      saveName = self.RECENT_STATE_FNAME
    if paramState is None:
      # TODO: May be good in the future to be able to choose which should be saved
      legitKeys = self._stateFuncsDf.index
      exportFuncs = self._stateFuncsDf.exportFuncs
      rets, errs = _safeCallFuncList(legitKeys, exportFuncs)
      updateDict = {k: ret for k, ret in zip(legitKeys, rets) if ret is not None}
      paramState = dict(Parameters=paramState, **updateDict)
      for editor in FR_SINGLETON.quickLoader.listModel.uniqueEditors:
        editor.applyChanges()
        curSaveName = str(self.saveDir/editor.name)
        formattedName = editor.name.replace(' ', '').lower()
        editor.saveParamState(curSaveName, blockWrite=False)
        paramState.update({formattedName: curSaveName})
    else:
      errs = []

    ret = super().saveParamState(saveName, paramState, allowOverwriteDefault, blockWrite)
    self.raiseErrMsgIfNeeded(errs)
    return ret

  def loadParamState(self, stateName: Union[str, Path]=None, stateDict: dict = None,
                     addChildren=False, removeChildren=False, applyChanges=True,
                     overrideDict: dict=None):
    if stateName is None:
      stateName = self.RECENT_STATE_FNAME
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
      args.append((stateDict.pop(k),))
    _, errs = _safeCallFuncList(legitKeys, importFuncs, args)
    if len(np.setdiff1d(stateDictKeys, legitKeys)) > 0:
      FR_SINGLETON.quickLoader.buildFromUserProfile(stateDict)
    ret = super().loadParamState(stateName, paramDict, addChildren, removeChildren,
                                 applyChanges)
    return ret

  @staticmethod
  def raiseErrMsgIfNeeded(errMsgs: List[str]):
    if len(errMsgs) > 0:
      err = FRAppIOError('Errors were encountered for the following parameters'
                         ' (shown as <parameter>: <exception>)\n'
                         + "\n\n".join(errMsgs))
      raiseErrorLater(err)


  def addImportExportOpts(self, optName: str, importFunc: Callable[[str], Any],
                          exportFunc: Callable[[], str]):
    newRow = pd.Series([importFunc, exportFunc], name=optName,
                       index=self._stateFuncsDf.columns)
    self._stateFuncsDf: pd.DataFrame = self._stateFuncsDf.append(newRow)

  @property
  def RECENT_STATE_FNAME(self):
      return self.saveDir/f'recent.{self.fileType}'
