from __future__ import annotations
import typing as t
import inspect
from abc import ABC, abstractmethod
from warnings import warn
import copy
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
import numpy as np

from s3a.generalutils import frPascalCaseToTitle
from s3a.structures import FRS3AWarning, FRAlgProcessorError

__all__ = ['FRProcessIO', 'FRProcessStage', 'FRGeneralProcess', 'FRImageProcess',
           'FRAtomicProcess']

class FRProcessIO(dict):
  """
  The object through which the processor pipeline communicates data. Inputs to one process
  become updated by the results from that process, and this updated ProcessIO is used
  as the input to the *next* process.
  """

  class FROM_PREV_IO:
    """
    Helper class to indicate whether a key in this IO is supposed to come from
    a previous process stage. Typical usage:
    ```if not hyperparam: self[k] = self.FROM_PREV_IO```

    Typically, process objects will have two IO dictionaries: One that hold the input spec
    (which makes use of `FROM_PREV_IO`) and one that holds the runtime process values. The
    latter IO will not make use of `FROM_PREV_IO`.
    """

  def __init__(self, hyperParamKeys: t.Sequence[str]=None, **kwargs) -> None:
    """
    :param hyperParamKeys: Hyperparameters for this process that aren't expected to come
      from the previous stage in the process. Forwarded keys (that were passed from a
      previous stage) are inferred by everything that is not a hyperparameter key.
    :param kwargs: see *dict* init
    """
    if hyperParamKeys is None:
      hyperParamKeys = []
    self.hyperParamKeys = hyperParamKeys
    warnKeys = []
    for k in self.hyperParamKeys:
      if k not in kwargs:
        warnKeys.append(k)
        kwargs[k] = None
    if len(warnKeys) > 0:
      warn(f'Hyperparameter keys were specified, but did not exist in provided'
           f' inputs:\n{warnKeys}\n'
           f'Defaulting to `None` for those keys.', FRS3AWarning)
    super().__init__(**kwargs)
    self.keysFromPrevIO = set(self.keys()) - set(self.hyperParamKeys)

  @classmethod
  def fromFunction(cls, func: t.Callable, **overriddenDefaults):
    """
    In the ProcessIO scheme, default arguments in a function signature constitute algorithm
    hyperparameters, while required arguments must be provided each time the function is
    run. If `**overriddenDefaults` is given, this will override any default arguments from
    `func`'s signature.
    :param func: Function whose input signature should be parsed
    :param overriddenDefaults: Keys here that match default argument names in `func` will
      override those defaults
    """
    outDict = {}
    hyperParamKeys = []
    spec = inspect.signature(func).parameters
    for k, v in spec.items():
      formattedV = v.default if k not in overriddenDefaults else overriddenDefaults[k]
      if formattedV is v.empty:
        formattedV = cls.FROM_PREV_IO
        # Not a hyperparameter
      else:
        hyperParamKeys.append(k)
      outDict[k] = formattedV
    return cls(hyperParamKeys, **outDict)


class FRProcessStage(ABC):
  name: str
  input: FRProcessIO = None
  allowDisable = False
  disabled = False
  result: FRProcessIO = None
  mainResultKeys: t.List[str] = None


  def __repr__(self) -> str:
    selfCls = type(self)
    oldName: str = super().__repr__()
    # Remove module name for brevity
    oldName = oldName.replace(f'{selfCls.__module__}.{selfCls.__name__}',
                              f'{selfCls.__name__} \'{self.name}\'')
    return oldName

  def __str__(self) -> str:
    return repr(self)

  def updateInput(self, prevIo: FRProcessIO):
    """
    Helper function to update current inputs from previous ones while ignoring leading
    underscores.
    """
    selfFmtToUnfmt = {k.lstrip('_'): k for k in self.input.keysFromPrevIO}
    prevIoKeyToFmt = {k.lstrip('_'): k for k in prevIo}
    missingKeys = []
    for fmtK, trueK in selfFmtToUnfmt.items():
      if fmtK in prevIoKeyToFmt:
        self.input[trueK] = prevIo[prevIoKeyToFmt[fmtK]]
      else:
        missingKeys.append(fmtK)
    if len(missingKeys) > 0:
      raise FRAlgProcessorError(f'Missing Following keys from {self}: {missingKeys}')

  def run(self, *args, **kwargs):
    raise NotImplementedError

  @property
  @abstractmethod
  def stages_flattened(self):
    raise NotImplementedError

class FRAtomicProcess(FRProcessStage):
  """
  Often, process functions return a single argument (e.g. string of text,
  processed image, etc.). In these cases, it is beneficial to know what name should
  be assigned to that result.
  """

  def __init__(self, func: t.Callable, name:str=None, **overriddenDefaults):
    """
    :param func: Function to wrap
    :param name: Name of this process. If `None`, defaults to the function name with
      camel case or underscores converted to title case.
    :param overriddenDefaults: Passed directly to ProcessIO when creating this function's
      input specifications.
    """
    if name is None:
      name = frPascalCaseToTitle(func.__name__)
    self.name = name
    self.func = func
    self.input = FRProcessIO.fromFunction(func, **overriddenDefaults)
    self.result: t.Optional[FRProcessIO] = None

  @property
  def keysFromPrevIO(self):
    return self.input.keysFromPrevIO

  def run(self, prevIO: FRProcessIO, disable=False):
    self.updateInput(prevIO)
    if not disable:
      self.result = self.func(**self.input)
    else:
      self.result = self.input
    return self.result

  @property
  def stages_flattened(self):
    return [self]

class FRGeneralProcess(FRProcessStage):

  def __init__(self, name: str=None):
    self.stages: t.List[FRProcessStage] = []
    self.name = name
    self.allowDisable = True

  def addFunction(self, func: t.Callable, name: str=None, **overriddenDefaults):
    """See function signature for AtomicProcess for input explanation"""
    atomic = FRAtomicProcess(func, name, **overriddenDefaults)
    atomic.mainResultKeys = self.mainResultKeys
    numSameNames = 0
    for stage in self.stages:
      if atomic.name == stage.name.split('#')[0]:
        numSameNames += 1
    if numSameNames > 0:
      atomic.name = f'{atomic.name}#{numSameNames+1}'
    if self.name is None:
      self.name = atomic.name
    self.stages.append(atomic)
    return atomic

  @classmethod
  def fromFunction(cls, func: t.Callable, name: str=None, **overriddenDefaults):
    out = cls(name)
    out.addFunction(func, name, **overriddenDefaults)
    return out

  def addProcess(self, process: FRGeneralProcess):
    if self.name is None:
      self.name = process.name
    self.stages.append(process)
    return process


  def run(self, io: FRProcessIO = None, disable=False):
    if io is None:
      _activeIO = FRProcessIO()
    else:
      _activeIO = copy.copy(io)

    for i, stage in enumerate(self.stages):
      newIO = stage.run(_activeIO, disable=self.disabled or disable)
      _activeIO.update(newIO)

    return self.result

  @property
  def result(self):
    return self.stages[-1].result

  @property
  def input(self):
    return self.stages[0].input

  @property
  def stages_flattened(self):
    outStages: t.List[FRProcessStage] = []
    for stage in self.stages:
      outStages.extend(stage.stages_flattened)
    return outStages

  def _stageSummaryWidget(self):
    raise NotImplementedError

  def _nonDisabledStages_flattened(self):
    out = []
    for stage in self.stages:
      if isinstance(stage, FRAtomicProcess):
        out.append(stage)
      elif not stage.disabled:
        stage: FRGeneralProcess
        out.extend(stage._nonDisabledStages_flattened())
    return out

  def stageSummary_gui(self):
    if self.result is None:
      raise FRAlgProcessorError('Analytics can only be shown after the algorithmwas run.')
    outGrid = self._stageSummaryWidget()
    outGrid.showMaximized()
    def fixedShow():
      for item in outGrid.ci.items:
        item.getViewBox().autoRange()
    QtCore.QTimer.singleShot(0, fixedShow)

  def getStageInfos(self, ignoreDuplicates=True):
    allInfos: t.List[t.Union[t.List, t.Dict[str, t.Any]]] = []
    for stage in self._nonDisabledStages_flattened():
      if stage.disabled:
        continue
      res = stage.result
      if 'summaryInfo' not in res:
        defaultSummaryInfo = {k: res[k] for k in self.mainResultKeys}
        defaultSummaryInfo.update(name=stage.name)
        res['summaryInfo'] = defaultSummaryInfo
      if res['summaryInfo'] is None:
        continue
      infos = stage.result['summaryInfo']
      if not isinstance(infos, t.Sequence):
        infos = [infos]
      for info in infos:
        stageNameCount = 0
        if info.get('name', None) is None:
          newName = stage.name
          if stageNameCount > 0:
            newName = f'{newName}#{stageNameCount}'
          info['name'] = newName
        stageNameCount += 1
      allInfos.extend(infos)
    return allInfos

class FRImageProcess(FRGeneralProcess):
  mainResultKeys = ['image']

  def _stageSummaryWidget(self):
    infoToDisplay = self.getStageInfos()

    numStages = len(infoToDisplay)
    nrows = np.sqrt(numStages).astype(int)
    ncols = np.ceil(numStages/nrows)
    outGrid = pg.GraphicsLayoutWidget()
    sizeToAxMapping: t.Dict[tuple, pg.PlotItem] = {}
    for ii, info in enumerate(infoToDisplay):
      pltItem: pg.PlotItem = outGrid.addPlot(title=info.get('name', None))
      pltItem.getViewBox().invertY(True)
      npImg = info['image']
      sameSizePlt = sizeToAxMapping.get(npImg.shape[:2], None)
      if sameSizePlt is not None:
        pltItem.setXLink(sameSizePlt)
        pltItem.setYLink(sameSizePlt)
      sizeToAxMapping[npImg.shape[:2]] = pltItem
      imgItem = pg.ImageItem(npImg)
      pltItem.addItem(imgItem)

      if ii % ncols == ncols-1:
        outGrid.nextRow()
    # See https://github.com/pyqtgraph/pyqtgraph/issues/1348. strange zooming occurs
    # if aspect is locked on all figures
    for ax in sizeToAxMapping.values():
      ax.getViewBox().setAspectLocked(True)
    oldClose = outGrid.closeEvent
    def newClose(ev):
      del _winRefs[outGrid]
      oldClose(ev)
    # Windows that go out of scope get garbage collected. Prevent that here
    _winRefs[outGrid] = outGrid
    outGrid.closeEvent = newClose

    return outGrid

  def getStageInfos(self, ignoreDuplicates=True):
    infos = super().getStageInfos(ignoreDuplicates)
    keepInfos = [{'name': 'Initial Image', 'image': self.input['image']}]
    if ignoreDuplicates:
      for info in infos:
        if not np.array_equal(info['image'], keepInfos[-1]['image']):
          keepInfos.append(info)
    return keepInfos

_winRefs = {}
