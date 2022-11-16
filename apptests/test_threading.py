import time

import pytest

# isort: off
from s3a.processing.threads import (
    AbortableThreadContainer,
    # RunnableFunctionWrapper,
    # RunnableThreadContainer,
    ThreadedFunctionWrapper,
)

# isort: on

# class GlobalThreadPoolContainer(RunnableThreadContainer):
#     """
#     See https://github.com/pytest-dev/pytest-qt/issues/199#issuecomment-366518687
#     for why this might work
#     """
#
#     def __init__(self, maxThreadCount=1):
#         super().__init__(QtCore.QThreadPool.globalInstance(), maxThreadCount)


def sleepUntilCallback(cb, timeoutSeconds=5.0):
    startTime = time.time()
    timeout = False
    while not cb() and not timeout:
        timeout = (time.time() - startTime) > timeoutSeconds
        time.sleep(0.1)
    if timeout:
        raise TimeoutError(f"{cb} never occurred")


# Leave as parametrize for now in case GlobalThreadPoolContainer is used
@pytest.mark.parametrize("container", [AbortableThreadContainer])
def test_containers(qtbot, container):
    pool = container()
    # if isinstance(pool, GlobalThreadPoolContainer):
    #     addFunc = pool.addRunner
    #     queue = pool.unfinishedRunners
    # else:
    #     addFunc = pool.addThread
    #     queue = pool.threads
    addFunc = pool.addThread
    queue = pool.threads

    def runner_func(id_):
        values.append(id_)

    values = []
    for ii in range(5):
        addFunc(runner_func, id_=ii)

    qtbot.waitUntil(lambda: len(values) == 5)
    # Clean threads that haven't had a chance to remove themselves
    pool.endThreads(queue)
    assert not len(queue)
    # One thread at a time *should* guarantee concurrency
    assert values == list(range(5))


# def test_terminate_waiting(qtbot):
#     pool = GlobalThreadPoolContainer()
#     end = False
#     runner = pool.addRunner(sleepUntilCallback, cb=lambda: end)
#     pool.addRunner(sleepUntilCallback, cb=lambda: end)
#     pool.discardUnfinishedRunners()
#     assert len(pool.unfinishedRunners) == 1
#     with qtbot.waitSignal(runner.sigResultReady):
#         end = True


def test_abort_during_run(qtbot):
    end = False
    pool = AbortableThreadContainer()
    threads = [pool.addThread(sleepUntilCallback, cb=lambda: end) for _ in range(2)]
    pool.updateThreads()
    # One process is started, the other isn't. The unstarted process should terminate
    assert pool.endThreads(threads, endRunning=False) == [False, True]

    with qtbot.waitSignal(threads[0].sigResultReady):
        end = True
    qtbot.waitUntil(lambda: not pool.threads)


def test_waiting(qtbot):
    """
    Confirms that multiple update() calls don't try to remove the same running
    thread multiple times
    """

    def doneHandle(_):
        nonlocal doneCounter
        doneCounter += 1

    doneCounter = 0
    end = False
    pool = AbortableThreadContainer()
    thread = pool.addThread(sleepUntilCallback, cb=lambda: end)
    thread.sigResultReady.connect(doneHandle)
    assert pool.threads[0].isRunning()
    for _ in range(5):
        pool.updateThreads()
    with qtbot.waitSignal(thread.sigResultReady):
        end = True
    assert doneCounter == 1


# Leave as parametrize for now in case GlobalThreadPoolContainer is used
@pytest.mark.parametrize("threadOrRunnable", [ThreadedFunctionWrapper])
def test_wrappers(qtbot, threadOrRunnable):
    def myfunc(id_, err=False):
        if err:
            raise ValueError(id_)
        return id_

    runnable = threadOrRunnable(myfunc, id_=5)
    with qtbot.waitSignal(
        runnable.sigResultReady, check_params_cb=lambda t: t.result == 5
    ):
        runnable.run()

    runnable = threadOrRunnable(myfunc, id_=5, err=True)
    with qtbot.waitSignal(
        runnable.sigFailed,
        check_params_cb=lambda t, err: isinstance(err, ValueError),
    ):
        runnable.run()
