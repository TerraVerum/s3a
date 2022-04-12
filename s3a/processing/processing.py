from __future__ import annotations

import typing as t

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
from utilitys.processing import *

__all__ = [
    "ProcessIO",
    "ProcessStage",
    "NestedProcess",
    "ImageProcess",
    "AtomicProcess",
    "ThreadedFuncWrapper",
    "AbortableThreadContainer",
    "RunnableFuncWrapper",
    "RunnableThreadContainer",
]

_infoType = t.List[t.Union[t.List, t.Dict[str, t.Any]]]
StrList = t.List[str]
StrCol = t.Collection[str]


class ImageProcess(NestedProcess):
    inMap = ["image"]
    outMap = ["image"]

    @classmethod
    def _cmpPrevCurInfos(cls, prevInfos: t.List[dict], infos: t.List[dict]):
        validInfos = super()._cmpPrevCurInfos(prevInfos, infos)
        # Iterate backwards to facilitate entry deletion
        for ii in range(len(validInfos) - 1, -1, -1):
            info = validInfos[ii]
            duplicateKeys = {"name"}
            for k, v in info.items():
                if v is cls._DUPLICATE_INFO:
                    duplicateKeys.add(k)
            if len(info.keys() - duplicateKeys) == 0:
                del validInfos[ii]
        return validInfos

    def _stageSummaryWidget(self, displayInfo=None):
        if displayInfo is None:
            displayInfo = self.getAllStageInfos()

        numStages = len(displayInfo)
        nrows = np.sqrt(numStages).astype(int)
        ncols = np.ceil(numStages / nrows)
        outGrid = pg.GraphicsLayoutWidget()
        sizeToAxMapping: t.Dict[tuple, pg.PlotItem] = {}
        for ii, info in enumerate(displayInfo):
            pltItem: pg.PlotItem = outGrid.addPlot(title=info.get("name", None))
            vb = pltItem.getViewBox()
            vb.invertY(True)
            vb.setAspectLocked(True)
            npImg = info["image"]
            imShp = npImg.shape[:2]
            margin = np.array(imShp) * 2
            lim = max(margin + imShp)
            vb.setLimits(maxXRange=lim, maxYRange=lim)
            sameSizePlt = sizeToAxMapping.get(imShp, None)
            if sameSizePlt is not None:
                pltItem.setXLink(sameSizePlt)
                pltItem.setYLink(sameSizePlt)
            sizeToAxMapping[imShp] = pltItem
            imgItem = pg.ImageItem(npImg)
            pltItem.addItem(imgItem)

            if ii % ncols == ncols - 1:
                outGrid.nextRow()
        # See https://github.com/pyqtgraph/pyqtgraph/issues/1348. strange zooming occurs
        # if aspect is locked on all figures
        # for ax in sizeToAxMapping.values():
        #   ax.getViewBox().setAspectLocked(True)
        oldClose = outGrid.closeEvent

        def newClose(ev):
            del _winRefs[outGrid]
            oldClose(ev)

        oldResize = outGrid.resizeEvent

        def newResize(ev):
            oldResize(ev)
            for ax in sizeToAxMapping.values():
                ax.getViewBox().autoRange()

        # Windows that go out of scope get garbage collected. Prevent that here
        _winRefs[outGrid] = outGrid
        outGrid.closeEvent = newClose
        outGrid.resizeEvent = newResize

        return outGrid

    def getAllStageInfos(self, ignoreDuplicates=True):
        infos = super().getAllStageInfos(ignoreDuplicates)
        # Add entry for initial image since it will be missed when just searching for
        # stage outputs
        infos.insert(0, {"name": "Initial Image", "image": self.input["image"]})
        return infos


_winRefs = {}


class RunnableFuncWrapperSignals(QtCore.QObject):
    resultReady = QtCore.Signal(object)
    """ThreadedFuncWrapper instance, emmitted on successful function run"""
    failed = QtCore.Signal(object, object)
    """ThreadedFuncWrapper instance, exception instance, emmitted on any exception during function run"""


class RunnableFuncWrapper(QtCore.QRunnable):
    signals = RunnableFuncWrapperSignals()

    def __init__(self, func, **kwargs):
        super().__init__()
        if not isinstance(func, AtomicProcess):
            kwargs.update(interactive=False)
            func = AtomicProcess(func, **kwargs)
        self.proc = func
        self.inProgress = False

    def run(self):
        self.inProgress = True
        try:
            self.proc.run()
            # This could go in a "finally" block to avoid duplication below, but now "inProgress" will be false before
            # the signal is emitted
            self.inProgress = False
            self.signals.resultReady.emit(self)
        except Exception as ex:
            self.inProgress = False
            self.signals.failed.emit(self, ex)

    @property
    def result(self):
        return self.proc.result

    @property
    def sigResultReady(self):
        return self.signals.resultReady

    @property
    def sigFailed(self):
        return self.signals.failed


class ThreadedFuncWrapper(QtCore.QThread):
    sigResultReady = QtCore.Signal(object)
    """ThreadedFuncWrapper instance, emmitted on successful function run"""
    sigFailed = QtCore.Signal(object, object)
    """ThreadedFuncWrapper instance, exception instance, emmitted on any exception during function run"""

    def __init__(self, func, **kwargs):
        super().__init__()
        if not isinstance(func, AtomicProcess):
            kwargs.update(interactive=False)
            func = AtomicProcess(func, **kwargs)
        self.proc = func

    def run(self):
        try:
            self.proc.run()
            self.sigResultReady.emit(self)
        except Exception as ex:
            self.sigFailed.emit(self, ex)

    @property
    def result(self):
        return self.proc.result


class RunnableThreadContainer(QtCore.QObject):
    sigTasksUpdated = QtCore.Signal()

    def __init__(self, pool: QtCore.QThreadPool = None, maxThreadCount=1):
        if pool is None:
            pool = QtCore.QThreadPool()
            pool.setMaxThreadCount(maxThreadCount)
        self.pool = pool
        # QThreadPool doesn't expose pending workers, so keep track of these manually
        self.unfinishedRunners = []
        super().__init__()

    def discardUnfinishedRunners(self):
        # Reverse to avoid race condition where first runner is supposed to happen before second, first is deleted,
        # and second runs regardless
        for runner in reversed(self.unfinishedRunners):
            if self.pool.tryTake(runner):
                self._onRunnerFinish(runner)

    def addRunner(self, runner: t.Callable | RunnableFuncWrapper, **kwargs):
        if not isinstance(runner, RunnableFuncWrapper):
            runner = RunnableFuncWrapper(runner, **kwargs)
        self.unfinishedRunners.append(runner)
        runner.signals.resultReady.connect(self._onRunnerFinish)
        self.pool.start(runner)
        self.sigTasksUpdated.emit()
        return runner

    def _onRunnerFinish(self, runner):
        if runner in self.unfinishedRunners:
            self.unfinishedRunners.remove(runner)
        self.sigTasksUpdated.emit()


class AbortableThreadContainer(QtCore.QObject):
    sigThreadsUpdated = QtCore.Signal()

    def __init__(self, maxConcurrentThreads=1):
        self.threads: list[ThreadedFuncWrapper] = []
        self.maxConcurrentThreads = maxConcurrentThreads
        super().__init__()

    def addThread(self, thread: ThreadedFuncWrapper | t.Callable = None, **addedKwargs):
        if not isinstance(thread, ThreadedFuncWrapper):
            thread = ThreadedFuncWrapper(thread, **addedKwargs)
        self.threads.append(thread)
        self.updateThreads()
        return thread

    def _threadFinishedWrapper(self, thread):
        """QThread.finished doesn't have an argument for the thread, so wrap a no-arg slot to accomodate"""

        def slot():
            self.endThreads(thread)

        # Without a reference, there's no way to disconnect() it later
        thread.__dict__["terminationSlot"] = slot
        return slot

    def updateThreads(self):
        for thread in self.threads[: self.maxConcurrentThreads]:
            if thread.isRunning() or thread.isFinished():
                # Either the max number of threads are already active or recycling is falling behind
                continue
            thread.finished.connect(self._threadFinishedWrapper(thread))
            thread.start()
        self.sigThreadsUpdated.emit()

    def endThreads(
        self,
        threads: ThreadedFuncWrapper | t.Iterable[ThreadedFuncWrapper],
        endRunning=False,
    ):
        """
        Abort a thread or thread at an index. Optionally does nothing if the requested thread is in progress.
        :return: Boolean indicating whether the thread was terminated
        """
        returnScalar = False
        if isinstance(threads, ThreadedFuncWrapper):
            threads = [threads]
            returnScalar = True
        returns = []
        for thread in threads:
            if thread not in self.threads or (thread.isRunning() and not endRunning):
                returns.append(False)
            else:
                self._removeThread(
                    thread, endFunc="terminate" if thread.isRunning() else "quit"
                )
                returns.append(True)
        if any(returns):
            self.updateThreads()
        if returnScalar:
            return returns[0]
        return returns

    def _removeThread(self, thread: ThreadedFuncWrapper, endFunc="quit"):
        # Already-connected threads need to be disconnected to avoid infinite waiting for termination
        if "terminationSlot" in thread.__dict__:
            thread.finished.disconnect(thread.__dict__["terminationSlot"])
        if endFunc == "quit":
            thread.quit()
        else:
            thread.terminate()
        thread.wait()
        self.threads.remove(thread)
