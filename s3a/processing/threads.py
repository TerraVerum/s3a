from __future__ import annotations

import time
import typing as t

from pyqtgraph import QtCore

from . import PipelineFunction

# class RunnableFunctionWrapperSignals(QtCore.QObject):
#     resultReady = QtCore.Signal(object)
#     """ThreadedFunctionWrapper instance, emmitted on successful function run"""
#     failed = QtCore.Signal(object, object)
#     """
#     ThreadedFunctionWrapper instance, exception instance, emmitted on any exception during
#     function run
#     """


# class RunnableFunctionWrapper(QtCore.QRunnable):
#     def __init__(self, func, **kwargs):
#         super().__init__()
#         self.signals = RunnableFunctionWrapperSignals()
#         if not isinstance(func, PipelineFunction):
#             func = PipelineFunction(func, **kwargs)
#         self.proc = func
#         self.inProgress = False
#
#     def run(self):
#         self.inProgress = True
#         try:
#             self.proc()
#             # This could go in a "finally" block to avoid duplication below, but now
#             # "inProgress" will be false before the signal is emitted
#             self.inProgress = False
#             self.signals.resultReady.emit(self)
#         except Exception as ex:
#             self.inProgress = False
#             self.signals.failed.emit(self, ex)
#
#     @property
#     def result(self):
#         return self.proc.result
#
#     @property
#     def sigResultReady(self):
#         return self.signals.resultReady
#
#     @property
#     def sigFailed(self):
#         return self.signals.failed
#


class ThreadedFunctionWrapper(QtCore.QThread):
    sigResultReady = QtCore.Signal(object)
    """ThreadedFunctionWrapper instance, emmitted on successful function run"""
    sigFailed = QtCore.Signal(object, object)
    """
    ThreadedFunctionWrapper instance, exception instance, emmitted on any exception 
    during function run
    """

    def __init__(self, func, **kwargs):
        super().__init__()
        if not isinstance(func, PipelineFunction):
            func = PipelineFunction(func, **kwargs)
        self.proc = func

    def run(self):
        try:
            self.proc()
            self.sigResultReady.emit(self)
        except Exception as ex:
            self.sigFailed.emit(self, ex)

    @property
    def result(self):
        return self.proc.result


# class RunnableThreadContainer(QtCore.QObject):
#     sigTasksUpdated = QtCore.Signal()
#
#     def __init__(self, pool: QtCore.QThreadPool = None, maxThreadCount=1):
#         if pool is None:
#             pool = QtCore.QThreadPool()
#         pool.setMaxThreadCount(maxThreadCount)
#         self.pool = pool
#         # QThreadPool doesn't expose pending workers, so keep track of these manually
#         self.unfinishedRunners = []
#         super().__init__()
#
#     def discardUnfinishedRunners(self):
#         # Reverse to avoid race condition where first runner is supposed to happen
#         # before second, first is deleted, and second runs regardless
#         update = False
#         for runner in reversed(self.unfinishedRunners):  # type: RunnableFunctionWrapper
#             if runner in self.unfinishedRunners and self.pool.tryTake(runner):
#                 self.unfinishedRunners.remove(runner)
#                 update = True
#         if update:
#             self.sigTasksUpdated.emit()
#
#     def addRunner(self, runner: t.Callable | RunnableFunctionWrapper, **kwargs):
#         if not isinstance(runner, RunnableFunctionWrapper):
#             runner = RunnableFunctionWrapper(runner, **kwargs)
#         self.unfinishedRunners.append(runner)
#
#         def onFinish(*_):
#             if runner in self.unfinishedRunners:
#                 self.unfinishedRunners.remove(runner)
#
#         for signal in runner.sigResultReady, runner.sigFailed:
#             signal.connect(onFinish)
#         self.pool.start(runner)
#         self.sigTasksUpdated.emit()
#         return runner


class AbortableThreadContainer(QtCore.QObject):
    sigThreadsUpdated = QtCore.Signal()

    def __init__(self, maxConcurrentThreads=1, rateLimitMs=0):
        self.threads: list[ThreadedFunctionWrapper] = []
        self.maxConcurrentThreads = maxConcurrentThreads
        self.rateLimitMs = rateLimitMs
        self.lastThreadStart = time.perf_counter()
        super().__init__()

    def addThread(
        self,
        thread: ThreadedFunctionWrapper | t.Callable = None,
        updateThreads=True,
        **addedKwargs,
    ):
        currentTime = time.perf_counter()
        elapsedMs = (currentTime - self.lastThreadStart) * 1000
        if self.rateLimitMs > 0 and elapsedMs < self.rateLimitMs:
            return None
        self.lastThreadStart = currentTime
        if not isinstance(thread, ThreadedFunctionWrapper):
            thread = ThreadedFunctionWrapper(thread, **addedKwargs)
        self.threads.append(thread)
        if updateThreads:
            self.updateThreads()
        return thread

    def _threadFinishedWrapper(self, thread):
        """
        QThread.finished doesn't have an argument for the thread, so wrap a no-arg
        slot to accomodate
        """

        def slot():
            self.endThreads(thread)

        # Without a reference, there's no way to disconnect() it later
        thread.__dict__["terminationSlot"] = slot
        return slot

    def updateThreads(self):
        for thread in self.threads[: self.maxConcurrentThreads]:
            if thread.isRunning() or thread.isFinished():
                # Either the max number of threads are already active or recycling is
                # falling behind
                continue
            thread.finished.connect(self._threadFinishedWrapper(thread))
            thread.start()
        self.sigThreadsUpdated.emit()

    def endThreads(
        self,
        threads: ThreadedFunctionWrapper | t.Iterable[ThreadedFunctionWrapper],
        endRunning=False,
    ) -> bool | list[bool]:
        """
        Abort a thread or thread at an index. Optionally does nothing if the requested
        thread is in progress.

        Returns
        -------
        bool or list[bool]
            indicates whether the thread was terminated
        """
        returnScalar = False
        if isinstance(threads, ThreadedFunctionWrapper):
            threads = [threads]
            returnScalar = True
        returns = []
        for thread in threads:
            if thread not in self.threads or (thread.isRunning() and not endRunning):
                returns.append(False)
            else:
                self._removeThread(
                    thread, endFunction="terminate" if thread.isRunning() else "quit"
                )
                returns.append(True)
        if any(returns):
            self.updateThreads()
        if returnScalar:
            return returns[0]
        return returns

    def _removeThread(self, thread: ThreadedFunctionWrapper, endFunction="quit"):
        # Already-connected threads need to be disconnected to avoid infinite waiting
        # for termination
        if "terminationSlot" in thread.__dict__:
            thread.finished.disconnect(thread.__dict__["terminationSlot"])
        if endFunction == "quit":
            thread.quit()
        else:
            thread.terminate()
        thread.wait()
        self.threads.remove(thread)
