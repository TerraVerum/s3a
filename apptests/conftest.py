import os

os.environ["S3A_PLATFORM"] = "minimal"
from typing import Type

import pytest

from apptests.helperclasses import CompDfTester
from apptests.testingconsts import (
    NUM_COMPS,
    SAMPLE_IMG,
    SAMPLE_IMG_FNAME,
    SAMPLE_SMALL_IMG,
    SAMPLE_SMALL_IMG_FNAME,
)
from s3a import REQD_TBL_FIELDS, mkQApp
from s3a.constants import PRJ_ENUMS
from s3a.plugins.file import FilePlugin, ProjectData
from s3a.plugins.tablefield import VerticesPlugin
from s3a.views.s3agui import S3A

mkQApp()

dfTester = CompDfTester(NUM_COMPS)
dfTester.fillRandomVerts(imageShape=SAMPLE_IMG.shape)


@pytest.fixture(scope="module")
def sampleComps():
    return dfTester.compDf.copy()


# Assign temporary project directory
@pytest.fixture(scope="session", autouse=True)
def app(tmpdir_factory):
    dummyProject = ProjectData.create(name=str(tmpdir_factory.mktemp("proj")))
    app_ = S3A(
        Image=SAMPLE_IMG_FNAME,
        log=PRJ_ENUMS.LOG_TERM,
        loadLastState=False,
        project=dummyProject.configPath,
    )
    app_.filePlugin.projectData.create(
        name=str(tmpdir_factory.mktemp("proj")), parent=app_.filePlugin.projectData
    )
    app_.appStateEditor.stateManager.moveDirectory(tmpdir_factory.mktemp("settings"))
    return app_


@pytest.fixture(scope="session")
def filePlugin(app):
    plg: FilePlugin = app.filePlugin
    return plg


@pytest.fixture(scope="session")
def mgr(app):
    return app.componentManager


@pytest.fixture(scope="session", autouse=True)
def vertsPlugin(app) -> VerticesPlugin:
    try:
        # False positive, since classPluginMap returns valid subclasses of plugins too
        # noinspection PyTypeChecker
        plg: VerticesPlugin = app.classPluginMap[VerticesPlugin]
    except KeyError:
        raise RuntimeError(
            "Vertices plugin was not provided. Some tests are guaranteed to fail."
        )

    plg.queueActions = False
    plg.processEditor.changeActiveProcessor("Basic Shapes")
    return plg


# Each test can request wheter it starts with components, small image, etc.
# After each test, all components are removed from the app
@pytest.fixture(autouse=True)
def resetAppAndTester(request, app, filePlugin, mgr):
    for img in filePlugin.projectData.images:
        try:
            if img != app.sourceImagePath:
                filePlugin.projectData.removeImage(img)
        except (FileNotFoundError,):
            pass
    app.mainImage.shapeCollection.forceUnlock()
    if "smallimage" in request.keywords:
        app.setMainImage(SAMPLE_SMALL_IMG_FNAME, SAMPLE_SMALL_IMG)
    else:
        app.setMainImage(SAMPLE_IMG_FNAME, SAMPLE_IMG)
    if "withcomps" in request.keywords:
        dfTester.fillRandomVerts(app.mainImage.image.shape)
        dfTester.compDf[REQD_TBL_FIELDS.IMAGE_FILE] = str(app.sourceImagePath)
        mgr.addComponents(dfTester.compDf.copy())
    yield
    app.actionStack.clear()
    app.clearBoundaries()


def assertExInList(exList, typ: Type[Exception] = Exception):
    assert any(issubclass(ex[0], typ) for ex in exList)
