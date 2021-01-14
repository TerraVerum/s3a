from __future__ import annotations

import copy
import inspect
import typing as t
from abc import ABC, abstractmethod
from functools import wraps
from warnings import warn

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from s3a.generalutils import pascalCaseToTitle, coerceAnnotation
from s3a.structures import S3AWarning, AlgProcessorError

__all__ = ['ProcessIO', 'ProcessStage', 'GeneralProcess', 'ImageProcess',
           'AtomicProcess']

_infoType = t.List[t.Union[t.List, t.Dict[str, t.Any]]]
StrList = t.List[str]
StrCol = t.Collection[str]
class _DUPLICATE_INFO: pass

class ProcessIO(dict):
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
           f'Defaulting to `None` for those keys.', S3AWarning)
    super().__init__(**kwargs)
    self.keysFromPrevIO = set(self.keys()) - set(self.hyperParamKeys)

  @classmethod
  def fromFunction(cls, func: t.Callable, ignoreKeys: StrCol=None, forceKeys: StrCol=None,
                   **overriddenDefaults):
    """
    In the ProcessIO scheme, default arguments in a function signature constitute algorithm
    hyperparameters, while required arguments must be provided each time the function is
    run. If `**overriddenDefaults` is given, this will override any default arguments from
    `func`'s signature.
    :param func: Function whose input signature should be parsed
    :param ignoreKeys: Keys to disregard entirely from the incoming function. This is useful for cases like adding
      a function at the class instead of instance level and `self` shouldn't be regarded by the parser.
    :param forceKeys: Keys without defaults but that should count as hyperparameters.
      It is strongly advised that `overriddenDefaults` should be provided for these values,
      or the value should be specified in the parameter's docstring
    :param overriddenDefaults: Keys here that match default argument names in `func` will
      override those defaults
    """
    if ignoreKeys is None:
      ignoreKeys = []
    if forceKeys is None:
      forceKeys = []
    outDict = {}
    hyperParamKeys = []
    spec = inspect.signature(func).parameters
    for k, v in spec.items():
      if k in ignoreKeys: continue
      formattedV = overriddenDefaults.get(k, v.default)
      if formattedV is v.empty and k in forceKeys:
        # Replace with *None* as described in func doc
        formattedV = coerceAnnotation(v.annotation)
      if formattedV is v.empty:
        formattedV = cls.FROM_PREV_IO
        # Not a hyperparameter
      else:
        hyperParamKeys.append(k)
      outDict[k] = formattedV
    # Make sure to avoid 'init got multiple values' error
    initSig = inspect.signature(cls.__init__).parameters
    keys = list(outDict)
    for key in keys:
      if key in initSig:
        newName = '_' + key
        outDict[newName] = outDict[key]
        del outDict[key]
    return cls(hyperParamKeys, **outDict)

class ProcessStage(ABC):
  name: str
  input: ProcessIO = None
  allowDisable = False
  disabled = False
  result: ProcessIO = None
  mainResultKeys: StrList = None
  mainInputKeys: StrList = None


  def __repr__(self) -> str:
    selfCls = type(self)
    oldName: str = super().__repr__()
    # Remove module name for brevity
    oldName = oldName.replace(f'{selfCls.__module__}.{selfCls.__name__}',
                              f'{selfCls.__name__} \'{self.name}\'')
    return oldName

  def __str__(self) -> str:
    return repr(self)

  def updateInput(self, prevIo: ProcessIO, **kwargs):
    """
    Helper function to update current inputs from previous ones while ignoring leading
    underscores.
    """
    selfFmtToUnfmt = {k.lstrip('_'): k for k in self.input}
    requiredKeyFmt = {k: v for k, v in selfFmtToUnfmt.items() if v in self.input.keysFromPrevIO}
    prevIoKeyToFmt = {k.lstrip('_'): k for k in {**prevIo, **kwargs}}
    missingKeys = []
    for fmtK, trueK in selfFmtToUnfmt.items():
      if fmtK in prevIoKeyToFmt:
        self.input[trueK] = prevIo[prevIoKeyToFmt[fmtK]]
      elif fmtK in requiredKeyFmt:
        missingKeys.append(fmtK)
    if len(missingKeys) > 0:
      raise AlgProcessorError(f'Missing Following keys from {self}: {missingKeys}')

  def run(self, io: ProcessIO=None, disable=False, **runKwargs):
    raise NotImplementedError

  def __call__(self, **kwargs):
    return self.run(ProcessIO(**kwargs))

  @property
  @abstractmethod
  def stages_flattened(self):
    raise NotImplementedError

class AtomicProcess(ProcessStage):
  """
  Often, process functions return a single argument (e.g. string of text,
  processed image, etc.). In these cases, it is beneficial to know what name should
  be assigned to that result.
  """

  def __init__(self, func: t.Callable, name:str=None, *, needsWrap=False,
               mainResultKeys: StrList=None, mainInputKeys: StrList=None,
               **procIoKwargs):
    """
    :param func: Function to wrap
    :param name: Name of this process. If `None`, defaults to the function name with
      camel case or underscores converted to title case.
    :param needsWrap: For functions not defined by the user, it is often inconvenient if they have
    to be redefined just to return a FRProcessIO object. If `func` does not return a `FRProcessIO`
    object, `needsWrap` can be set to `True`. In this case, `func` is assumed to
    returns either one result or a list of results. It is converted into a function
    returning a FRProcessIO object instead. Each `mainResultKey` is assigned to each output
    of the function in order. If only one main result key exists, then the output of the
    function is assumed to be that key. I.e. in the case where `len(cls.mainResultKeys) == 1`,
    the output is expected to be the direct result, not a sequence of results per key.
    :param mainResultKeys: Set by parent process as needed
    :param mainInputKeys: Set by parent process as needed
    :param procIoKwargs: Passed to ProcessIO.fromFunction
    """
    if name is None:
      name = pascalCaseToTitle(func.__name__)
    if mainResultKeys is not None:
      self.mainResultKeys = mainResultKeys
    if mainInputKeys is not None:
      self.mainInputKeys = mainInputKeys

    self.name = name
    self.input = ProcessIO.fromFunction(func, **procIoKwargs)
    self.result: t.Optional[ProcessIO] = None

    if mainInputKeys is not None:
      keys = set(self.input.keys())
      missingKeys = set(mainInputKeys) - keys
      if missingKeys:
        raise AlgProcessorError(f'{name} input signature is missing the following required input keys:\n'
                                f'{missingKeys}')

    if needsWrap:
      func = self._wrappedFunc(func, self.mainResultKeys)
    self.func = func

  @classmethod
  def _wrappedFunc(cls, func, mainResultKeys: StrList=None, mainInputKeys: StrList=None):
    """
    Wraps a function returining either a result or list of results, instead making the
    return value an `FRProcessIO` object where each `cls.mainResultkey` corresponds
    to a returned value
    """
    if mainResultKeys is None:
      mainResultKeys = cls.mainResultKeys
    if len(mainResultKeys) == 1:
      @wraps(func)
      def newFunc(*args, **kwargs):
        return ProcessIO(**{mainResultKeys[0]: func(*args, **kwargs)})
    else:
      @wraps(func)
      def newFunc(*args, **kwargs):
        return ProcessIO(**{k: val for k, val in zip(mainResultKeys, func(*args, **kwargs))})
    return newFunc

  @property
  def keysFromPrevIO(self):
    return self.input.keysFromPrevIO

  def run(self, prevIO: ProcessIO=None, disable=False, **runKwargs):
    if prevIO is not None:
      self.updateInput(prevIO, **runKwargs)
    if not disable:
      self.result = self.func(**self.input)
    else:
      self.result = self.input
    return self.result

  @property
  def stages_flattened(self):
    return [self]

class GeneralProcess(ProcessStage):

  def __init__(self, name: str=None, mainInputKeys: StrList=None, mainResultKeys: StrList=None):
    self.stages: t.List[ProcessStage] = []
    self.name = name
    self.allowDisable = True
    if mainInputKeys is not None:
      self.mainInputKeys = mainInputKeys
    if mainResultKeys is not None:
      self.mainResultKeys = mainResultKeys

  def addFunction(self, func: t.Callable, keySpec: t.Union[t.Type[GeneralProcess], GeneralProcess]=None, **kwargs):
    """
    Wraps the provided function in an AtomicProcess and adds it to the current process.
    :param func: Forwarded to AtomicProcess
    :param kwargs: Forwarded to AtomicProcess
    :param keySpec: This argument should have 'mainInputKeys' and 'mainResultKeys' that are used
      when adding a function to this process. This can be beneficial when an Atomic Process
      is added with different keys than the current process type
    """
    if keySpec is None:
      keySpec = self
    atomic = AtomicProcess(func, mainResultKeys=keySpec.mainResultKeys, mainInputKeys=keySpec.mainInputKeys, **kwargs)
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
  def fromFunction(cls, func: t.Callable, **kwargs):
    name = kwargs.get('name', None)
    out = cls(name)
    out.addFunction(func, **kwargs)
    return out

  def addProcess(self, process: GeneralProcess):
    if self.name is None:
      self.name = process.name
    self.stages.append(process)
    return process

  def run(self, io: ProcessIO = None, disable=False, **runKwargs):
    if io is None:
      _activeIO = ProcessIO()
    else:
      _activeIO = copy.copy(io)
    _activeIO.update(runKwargs)

    for i, stage in enumerate(self.stages):
      newIO = stage.run(_activeIO, disable=self.disabled or disable)
      if isinstance(newIO, ProcessIO):
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
    outStages: t.List[ProcessStage] = []
    for stage in self.stages:
      outStages.extend(stage.stages_flattened)
    return outStages

  def _stageSummaryWidget(self):
    raise NotImplementedError

  def _nonDisabledStages_flattened(self):
    out = []
    for stage in self.stages:
      if isinstance(stage, AtomicProcess):
        out.append(stage)
      elif not stage.disabled:
        stage: GeneralProcess
        out.extend(stage._nonDisabledStages_flattened())
    return out

  def stageSummary_gui(self):
    if self.result is None:
      raise AlgProcessorError('Analytics can only be shown after the algorithm was run.')
    outGrid = self._stageSummaryWidget()
    outGrid.showMaximized()
    def fixedShow():
      for item in outGrid.ci.items:
        item.getViewBox().autoRange()
    QtCore.QTimer.singleShot(0, fixedShow)

  def getStageInfos(self, ignoreDuplicates=True):
    allInfos: _infoType = []
    lastInfos = []
    for stage in self._nonDisabledStages_flattened():
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
      if not ignoreDuplicates:
        validInfos = infos
      else:
        validInfos = self._cmpPrevCurInfos(lastInfos, infos)
      lastInfos = infos
      for info in validInfos:
        stageNameCount = 0
        if info.get('name', None) is None:
          newName = stage.name
          if stageNameCount > 0:
            newName = f'{newName}#{stageNameCount}'
          info['name'] = newName
        stageNameCount += 1
      allInfos.extend(validInfos)
    return allInfos

  @classmethod
  def _cmpPrevCurInfos(cls, prevInfos: t.List[dict], infos: t.List[dict]):
    """
    This comparison allows keys from the last result which exactly match keyts from the
    current result to be discarded for brevity.
    """
    validInfos = []
    for info in infos:
      validInfo = copy.copy(info)
      for lastInfo in prevInfos:
        for key in set(info.keys()).intersection(lastInfo.keys()) - {'name'}:
          if np.array_equal(info[key], lastInfo[key]):
            validInfo[key] = _DUPLICATE_INFO
      validInfos.append(validInfo)
    return validInfos

class ImageProcess(GeneralProcess):
  mainInputKeys = ['image']
  mainResultKeys = ['image']

  @classmethod
  def _cmpPrevCurInfos(cls, prevInfos: t.List[dict], infos: t.List[dict]):
    validInfos = super()._cmpPrevCurInfos(prevInfos, infos)
    # Iterate backwards to facilitate entry deletion
    for ii in range(len(validInfos)-1,-1,-1):
      info = validInfos[ii]
      duplicateKeys = {'name'}
      for k, v in info.items():
        if v is _DUPLICATE_INFO:
          duplicateKeys.add(k)
      if len(info.keys() - duplicateKeys) == 0:
        del validInfos[ii]
    return validInfos


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
    # Add entry for initial image since it will be missed when just searching for
    # stage outputs
    infos.insert(0, {'name': 'Initial Image', 'image': self.input['image']})
    return infos

_winRefs = {}

class GlobalPredictionProcess(ImageProcess):
  def _stageSummaryWidget(self):
    return QtWidgets.QWidget()
  mainResultKeys = ['components']

class CategoricalProcess(GeneralProcess):
  def _stageSummaryWidget(self):
    pass

  mainResultKeys = ['categories', 'confidences']