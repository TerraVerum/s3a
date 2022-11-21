import pickle as pkl
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from qtextras import fns

from apptests.conftest import dfTester
from apptests.helperclasses import CompDfTester
from apptests.testingconsts import (
    SAMPLE_IMG_FNAME,
    SAMPLE_SMALL_IMG,
    SAMPLE_SMALL_IMG_FNAME,
)
from s3a import S3A
from s3a.plugins.file import ProjectData, absolutePath
from s3a.tabledata import IOTemplateManager

_autosaveFile = "autosave.parameter"


@pytest.fixture
def tmpProj(tmp_path):
    return ProjectData.create(name=tmp_path / "tmpproj")


@pytest.fixture
def prjWithSavedStuff(tmpProj):
    tester = CompDfTester(50)
    tester.fillRandomVerts(SAMPLE_SMALL_IMG.shape[:2])
    tmpProj.addAnnotation(data=tester.compDf, image=SAMPLE_SMALL_IMG_FNAME)
    tmpProj.addAnnotation(data=tester.compDf, image=SAMPLE_IMG_FNAME)
    return tmpProj


def test_create_project(app, sampleComps, tmpProj):
    dummyAnnFile = SAMPLE_SMALL_IMG_FNAME.with_suffix(
        f"{SAMPLE_SMALL_IMG_FNAME.suffix}.pkl"
    )
    tmpProj.loadConfig(tmpProj.configPath, {"annotation-format": "pkl"}, force=True)
    tmpProj.addAnnotation(dummyAnnFile, dfTester.compDf, image=SAMPLE_SMALL_IMG_FNAME)

    assert tmpProj.configPath.exists()
    assert tmpProj.annotationsPath.exists()
    assert tmpProj.imagesPath.exists()
    tmpProj.addAnnotation(data=sampleComps, image=SAMPLE_SMALL_IMG_FNAME)
    assert len(list(tmpProj.imagesPath.glob(SAMPLE_SMALL_IMG_FNAME.name))) == 1
    assert len(list(tmpProj.annotationsPath.glob(dummyAnnFile.name))) == 1


def test_update_props(filePlugin):
    annFmt = lambda: filePlugin.projectData.config["annotation-format"]
    oldFmt = annFmt()
    filePlugin.updateProjectProperties(annotationFormat="pkl")
    assert annFmt() == "pkl"
    filePlugin.updateProjectProperties(annotationFormat=oldFmt)
    loc = filePlugin.projectData.location / "newcfg.tblcfg"
    newCfg = {"fields": {"Class": ""}}
    fns.hierarchicalUpdate(newCfg, IOTemplateManager.getTableConfig("s3a"))
    fns.saveToFile(newCfg, loc)
    oldName = filePlugin.projectData.tableData.configPath
    filePlugin.updateProjectProperties(tableConfig=loc)
    assert newCfg == filePlugin.projectData.tableData.config
    filePlugin.updateProjectProperties(tableConfig=oldName)


@pytest.mark.withcomps
def test_export(prjWithSavedStuff, tmp_path):
    prj = prjWithSavedStuff

    out = tmp_path / "my-project"
    prj.exportProject(out)
    assert out.exists()
    for fileA, fileB in zip(out.rglob("*.*"), prj.location.rglob("*.*")):
        assert fileA.name == fileB.name


@pytest.mark.withcomps
def test_export_anns(prjWithSavedStuff, tmp_path):
    prj = prjWithSavedStuff
    outpath = tmp_path / "export-anns"
    prj.exportAnnotations(outpath, combine=True)
    assert (outpath / "annotations.csv").exists()
    assert (outpath / "images").exists()
    assert len(list((outpath / "images").iterdir())) == 2

    for typ in ["csv", "pkl"]:
        shutil.rmtree(outpath, ignore_errors=True)
        prj.exportAnnotations(outpath, annotationFormat=typ)
        assert (outpath / "annotations").exists()
        assert sorted(f.stem for f in (outpath / "annotations").iterdir()) == sorted(
            f.stem for f in prj.annotationsPath.iterdir()
        )

    # Make sure self export doesn't break anything
    prj.exportAnnotations(prj.location)


def test_load_startup_img(tmp_path, app, filePlugin):
    prjcfg = {"startup": {"image": str(SAMPLE_SMALL_IMG_FNAME)}}
    oldCfg = filePlugin.projectData.configPath, filePlugin.projectData.config
    filePlugin.open(tmp_path / "test-startup.s3aprj", prjcfg)
    assert (
        app.sourceImagePath
        == filePlugin.projectData.imagesPath / SAMPLE_SMALL_IMG_FNAME.name
    )
    for img in None, filePlugin.projectData.imagesPath / "my-image.jpg":
        app.sourceImagePath = img
        app.appStateEditor.stateFunctionsDf.at["project", "exportFunction"](
            tmp_path / "another"
        )
        assert bool(img) == ("image" in filePlugin.projectData.startup)

    filePlugin.open(*oldCfg)


def test_load_with_plg(monkeypatch, tmp_path):
    # Make separate win to avoid clobbering existing menus/new projs
    app = S3A(loadLastState=False)
    filePlugin = app.filePlugin
    with monkeypatch.context() as m:
        from apptests.files.sample_plg import SamplePlugin

        cfg = {"plugin-config": {"Test": "apptests.files.sample_plg.SamplePlugin"}}
        filePlugin.open(tmp_path / "plgprj.s3aprj", cfg)
        assert SamplePlugin in app.classPluginMap
        assert len(filePlugin.projectData.spawnedPlugins) == 1
        assert filePlugin.projectData.spawnedPlugins[0].window

    # Remove existing plugin
    cfg = {"plugin-config": {"New Name": "nonsense.Plugin"}}
    with pytest.raises(ValueError):
        filePlugin.open(tmp_path / "plgprj2.s3aprj", cfg)
    # Add nonsense plugin
    cfg["plugin-config"]["Test"] = "files.sample_plg.SamplePlugin"
    with pytest.warns(UserWarning):
        filePlugin.open(tmp_path / "plgprj2.s3aprj", cfg)


def test_unique_tblcfg(tmp_path, tmpProj):
    tblCfg = {"fields": {"Test": ""}}
    tblName = tmp_path / "tbl.yml"
    fns.saveToFile(tblCfg, tblName)

    cfg = {"table-config": str(tblName)}
    tmpProj.loadConfig(tmp_path / "myprj.s3aprj", cfg)
    assert tmpProj.tableData.fieldFromName("Test")


def test_img_ops(tmpProj, tmp_path):
    img = {"data": SAMPLE_SMALL_IMG, "name": "my image.png"}
    cfg = {"images": [img]}
    tmpProj.loadConfig(tmp_path / "test.s3aprj", cfg)
    tmpProj._addConfigImages()
    assert len(tmpProj.images) == 1
    assert tmpProj.images[0].name == "my image.png"

    with pytest.raises(IOError):
        tmpProj.addImage("this image does not exist.png", copyToProject=True)


def test_ann_opts(prjWithSavedStuff, sampleComps):
    img, toRemove = next(iter(prjWithSavedStuff.imageAnnotationMap.items()))
    prjWithSavedStuff.removeAnnotation(toRemove)
    assert img not in prjWithSavedStuff.imageAnnotationMap

    with pytest.raises(IOError):
        prjWithSavedStuff.addAnnotation(data=sampleComps, image="garbage.png")


def test_filter_proj_imgs(filePlugin, prjWithSavedStuff):
    for img in prjWithSavedStuff.images:
        filePlugin.projectData.addImage(img)
    fMgr = filePlugin._projectImagePane
    fMgr.completer.setText("hubble")
    assert "*hubble*" in fMgr.fileModel.nameFilters()


def test_load_self_cfg(prjWithSavedStuff):
    assert prjWithSavedStuff.loadConfig(prjWithSavedStuff.configPath) is None


def test_load_independent_tbl_cfg(prjWithSavedStuff, tmpdir):
    tblCfg = {"fields": {"Class": ["a", "b", "c"]}}
    outName = "separate_tbl_cfg.tblcfg"
    file = tmpdir / outName
    fns.saveToFile(tblCfg, file)
    prjCfg = {"table-config": str(file)}
    prjWithSavedStuff.loadConfig(tmpdir / "new_loaded_cfg.s3aprj", prjCfg)

    relativeName = prjWithSavedStuff.location / outName
    assert prjWithSavedStuff.tableData.configPath == relativeName


def test_none_tblinfo(tmpdir):
    cfg = {}
    prj = ProjectData(tmpdir / "none-table.s3aprj", cfg)
    assert prj.tableData.configPath == prj.configPath
    assert prj.tableData.config == IOTemplateManager.getTableConfig("s3a")


def test_change_image_path(tmpProj):
    tmpProj.addImage(SAMPLE_SMALL_IMG_FNAME, copyToProject=False)
    newPath = Path("./ridiculous/but/different") / SAMPLE_SMALL_IMG_FNAME.name
    tmpProj.changeImagePath(SAMPLE_SMALL_IMG_FNAME, newPath)
    assert newPath.absolute() in tmpProj.images
    assert SAMPLE_SMALL_IMG_FNAME not in tmpProj.images

    tmpProj.changeImagePath(newPath, None)
    assert newPath not in tmpProj.images


def test_add_image_folder(tmpProj, tmpdir):
    for fname in SAMPLE_SMALL_IMG_FNAME, SAMPLE_IMG_FNAME:
        shutil.copy(fname, tmpdir / fname.name)
    tmpProj.addImageFolder(Path(tmpdir))
    assert len(tmpProj.images) == 2


def test_base_dir_logic(tmpProj: ProjectData, tmpdir):
    tmpdir = Path(tmpdir)
    shutil.copy(SAMPLE_SMALL_IMG_FNAME, tmpdir / SAMPLE_SMALL_IMG_FNAME.name)
    tmpProj.imageFolders.add(tmpdir)

    assert (
        tmpProj.getFullImgName(SAMPLE_SMALL_IMG_FNAME.name)
        == tmpdir / SAMPLE_SMALL_IMG_FNAME.name
    )

    tmp2 = tmpdir / "another"
    tmp2.mkdir()
    shutil.copy(SAMPLE_SMALL_IMG_FNAME, tmp2)

    tmpProj.imageFolders.add(tmp2)
    with pytest.raises(IOError):
        tmpProj.getFullImgName(SAMPLE_SMALL_IMG_FNAME.name)

    # Since a set is used internally, no gurantee that the order is preserved
    assert tmpProj.getFullImgName(SAMPLE_SMALL_IMG_FNAME.name, thorough=False)


def test_remove_image(prjWithSavedStuff):
    imName = prjWithSavedStuff.getFullImgName(SAMPLE_IMG_FNAME.name)
    assert imName in prjWithSavedStuff.imageAnnotationMap
    assert imName.exists()

    prjWithSavedStuff.removeImage(imName)
    assert imName not in prjWithSavedStuff.imageAnnotationMap
    assert not imName.exists()

    nonexistImg = "garbage.png"
    prjWithSavedStuff.removeImage(nonexistImg)


def test_pkl(prjWithSavedStuff):
    pklBytes = pkl.dumps(prjWithSavedStuff)

    # Test that the pickle is valid
    loaded = pkl.loads(pklBytes)

    assert loaded.config == prjWithSavedStuff.config


def test_add_fmt_annotation(
    prjWithSavedStuff: ProjectData, sampleComps: pd.DataFrame, tmpdir
):
    imname = SAMPLE_SMALL_IMG_FNAME.name
    fpath = Path(tmpdir / imname + ".csv")
    prjWithSavedStuff.componentIo.exportCsv(sampleComps, fpath)
    with pytest.raises(IOError):
        prjWithSavedStuff.addFormattedAnnotation(fpath)

    prjWithSavedStuff.addFormattedAnnotation(fpath, True)
    fullImgName = prjWithSavedStuff.getFullImgName(imname)
    annName = prjWithSavedStuff.imageAnnotationMap[fullImgName]
    cmpData = prjWithSavedStuff.componentIo.importByFileType(annName)
    assert np.array_equal(cmpData.values, sampleComps.values)


def test_abspath_none():
    assert absolutePath(None) is None


def test_load_autosave(app, filePlugin, tmp_path):
    state = app.appStateEditor

    fns.saveToFile({"interval": 10}, tmp_path / _autosaveFile)

    importer = state.stateFunctionsDf.at["autosave", "importFunction"]
    importer(True)
    assert filePlugin.autosaveTimer.isActive()

    importer(tmp_path / _autosaveFile)
    assert filePlugin.autosaveTimer.interval() == 1000 * 60 * 10

    importer(False)
    assert not filePlugin.autosaveTimer.isActive()


def test_export_autosave(app, filePlugin, tmp_path):
    state = app.appStateEditor

    fns.saveToFile({"interval": 10}, tmp_path / _autosaveFile)

    exporter = state.stateFunctionsDf.at["autosave", "exportFunction"]
    for name, function in filePlugin.nameFunctionMap.items():
        if fns.nameFormatter(name) == "Start Autosave":
            break
    else:
        raise ValueError("Autosave process not found")

    assert len(set(function.parameters).intersection(["interval", "backupFolder"])) == 2
    function(interval=10, backupFolder=str(tmp_path))

    outpath = tmp_path / "autosave_export"
    outpath.mkdir()
    exporter(outpath)
    assert list(outpath.iterdir())

    cfg = fns.attemptFileLoad(next(outpath.iterdir()))
    assert "interval" in cfg and cfg["interval"] == 10
    assert "backupFolder" in cfg and cfg["backupFolder"] == str(tmp_path)

    filePlugin.stopAutosave()
    for file in outpath.iterdir():
        file.unlink()
    assert not exporter(outpath)
    assert not list(outpath.iterdir())
