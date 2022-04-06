import shutil
from io import StringIO
from pathlib import Path
import pickle as pkl

import numpy as np
import pandas as pd
import pytest

from apptests.helperclasses import CompDfTester
from apptests.testingconsts import (
    SAMPLE_SMALL_IMG_FNAME,
    SAMPLE_SMALL_IMG,
    SAMPLE_IMG_FNAME,
    TEST_FILE_DIR,
)
from apptests.conftest import dfTester
from s3a import S3A
from s3a.parameditors.table import IOTemplateManager
from s3a.plugins.file import ProjectData, absolutePath
from utilitys import fns


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
    tmpProj.loadCfg(tmpProj.cfgFname, {"annotation-format": "pkl"}, force=True)
    tmpProj.addAnnotation(dummyAnnFile, dfTester.compDf, image=SAMPLE_SMALL_IMG_FNAME)

    assert tmpProj.cfgFname.exists()
    assert tmpProj.annotationsDir.exists()
    assert tmpProj.imagesDir.exists()
    tmpProj.addAnnotation(data=sampleComps, image=SAMPLE_SMALL_IMG_FNAME)
    assert len(list(tmpProj.imagesDir.glob(SAMPLE_SMALL_IMG_FNAME.name))) == 1
    assert len(list(tmpProj.annotationsDir.glob(dummyAnnFile.name))) == 1


def test_update_props(filePlg):
    annFmt = lambda: filePlg.projData.cfg["annotation-format"]
    oldFmt = annFmt()
    filePlg.updateProjectProperties(annotationFormat="pkl")
    assert annFmt() == "pkl"
    filePlg.updateProjectProperties(annotationFormat=oldFmt)
    loc = filePlg.projData.location / "newcfg.tblcfg"
    newCfg = {"fields": {"Class": ""}}
    fns.hierarchicalUpdate(newCfg, IOTemplateManager.getTableCfg("s3a"))
    fns.saveToFile(newCfg, loc)
    oldName = filePlg.projData.tableData.cfgFname
    filePlg.updateProjectProperties(tableConfig=loc)
    assert newCfg == filePlg.projData.tableData.cfg
    filePlg.updateProjectProperties(tableConfig=oldName)


@pytest.mark.withcomps
def test_export(prjWithSavedStuff, tmp_path):
    prj = prjWithSavedStuff

    out = tmp_path / "my-project"
    prj.exportProj(out)
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
            f.stem for f in prj.annotationsDir.iterdir()
        )

    # Make sure self export doesn't break anything
    prj.exportAnnotations(prj.location)


def test_load_startup_img(tmp_path, app, filePlg):
    prjcfg = {"startup": {"image": str(SAMPLE_SMALL_IMG_FNAME)}}
    oldCfg = filePlg.projData.cfgFname, filePlg.projData.cfg
    filePlg.open(tmp_path / "test-startup.s3aprj", prjcfg)
    assert app.srcImgFname == filePlg.projData.imagesDir / SAMPLE_SMALL_IMG_FNAME.name
    for img in None, filePlg.projData.imagesDir / "my-image.jpg":
        app.srcImgFname = img
        app.appStateEditor.stateFuncsDf.at["project", "exportFunc"](
            tmp_path / "another"
        )
        assert bool(img) == ("image" in filePlg.projData.startup)

    filePlg.open(*oldCfg)


def test_load_with_plg(monkeypatch, tmp_path):
    # Make separate win to avoid clobbering existing menus/new projs
    app = S3A(loadLastState=False)
    filePlg = app.filePlg
    with monkeypatch.context() as m:
        m.syspath_prepend(str(TEST_FILE_DIR))
        from files.sample_plg import SamplePlugin

        cfg = {"plugin-cfg": {"Test": "files.sample_plg.SamplePlugin"}}
        filePlg.open(tmp_path / "plgprj.s3aprj", cfg)
        assert SamplePlugin in app.clsToPluginMapping
        assert len(filePlg.projData.spawnedPlugins) == 1
        assert filePlg.projData.spawnedPlugins[0].win

    # Remove existing plugin
    cfg = {"plugin-cfg": {"New Name": "nonsense.Plugin"}}
    with pytest.raises(ValueError):
        filePlg.open(tmp_path / "plgprj2.s3aprj", cfg)
    # Add nonsense plugin
    cfg["plugin-cfg"]["Test"] = "files.sample_plg.SamplePlugin"
    with pytest.warns(UserWarning):
        filePlg.open(tmp_path / "plgprj2.s3aprj", cfg)


def test_unique_tblcfg(tmp_path, tmpProj):
    tblCfg = {"fields": {"Test": ""}}
    tblName = tmp_path / "tbl.yml"
    fns.saveToFile(tblCfg, tblName)

    cfg = {"table-cfg": str(tblName)}
    tmpProj.loadCfg(tmp_path / "myprj.s3aprj", cfg)
    assert tmpProj.tableData.fieldFromName("Test")


def test_img_ops(tmpProj, tmp_path):
    img = {"data": SAMPLE_SMALL_IMG, "name": "my image.png"}
    cfg = {"images": [img]}
    tmpProj.loadCfg(tmp_path / "test.s3aprj", cfg)
    tmpProj._addConfigImages()
    assert len(tmpProj.images) == 1
    assert tmpProj.images[0].name == "my image.png"

    with pytest.raises(IOError):
        tmpProj.addImage("this image does not exist.png", copyToProject=True)


def test_ann_opts(prjWithSavedStuff, sampleComps):
    img, toRemove = next(iter(prjWithSavedStuff.imgToAnnMapping.items()))
    prjWithSavedStuff.removeAnnotation(toRemove)
    assert img not in prjWithSavedStuff.imgToAnnMapping

    with pytest.raises(IOError):
        prjWithSavedStuff.addAnnotation(data=sampleComps, image="garbage.png")


def test_filter_proj_imgs(filePlg, prjWithSavedStuff):
    for img in prjWithSavedStuff.images:
        filePlg.projData.addImage(img)
    fMgr = filePlg._projImgMgr
    fMgr.completer.setText("hubble")
    assert "*hubble*" in fMgr.fileModel.nameFilters()


def test_load_self_cfg(prjWithSavedStuff):
    assert prjWithSavedStuff.loadCfg(prjWithSavedStuff.cfgFname) is None


def test_load_independent_tbl_cfg(prjWithSavedStuff, tmpdir):
    tblCfg = {"fields": {"Class": ["a", "b", "c"]}}
    outName = "separate_tbl_cfg.tblcfg"
    file = tmpdir / outName
    fns.saveToFile(tblCfg, file)
    prjCfg = {"table-cfg": str(file)}
    prjWithSavedStuff.loadCfg(tmpdir / "new_loaded_cfg.s3aprj", prjCfg)

    relativeName = prjWithSavedStuff.location / outName
    assert prjWithSavedStuff.tableData.cfgFname == relativeName


def test_none_tblinfo(tmpdir):
    cfg = {}
    prj = ProjectData(tmpdir / "none-table.s3aprj", cfg)
    assert prj.tableData.cfgFname == prj.cfgFname
    assert prj.tableData.cfg == IOTemplateManager.getTableCfg("s3a")


def test_change_image_path(tmpProj):
    tmpProj.addImage(SAMPLE_SMALL_IMG_FNAME, copyToProject=False)
    newPath = Path("./ridiculous/but/different") / SAMPLE_SMALL_IMG_FNAME.name
    tmpProj.changeImgPath(SAMPLE_SMALL_IMG_FNAME, newPath)
    assert newPath.absolute() in tmpProj.images
    assert SAMPLE_SMALL_IMG_FNAME not in tmpProj.images

    tmpProj.changeImgPath(newPath, None)
    assert newPath not in tmpProj.images


def test_add_image_folder(tmpProj, tmpdir):
    for fname in SAMPLE_SMALL_IMG_FNAME, SAMPLE_IMG_FNAME:
        shutil.copy(fname, tmpdir / fname.name)
    tmpProj.addImageFolder(Path(tmpdir))
    assert len(tmpProj.images) == 2


def test_base_dir_logic(tmpProj: ProjectData, tmpdir):
    tmpdir = Path(tmpdir)
    shutil.copy(SAMPLE_SMALL_IMG_FNAME, tmpdir / SAMPLE_SMALL_IMG_FNAME.name)
    tmpProj.baseImgDirs.add(tmpdir)

    assert (
        tmpProj.getFullImgName(SAMPLE_SMALL_IMG_FNAME.name)
        == tmpdir / SAMPLE_SMALL_IMG_FNAME.name
    )

    tmp2 = tmpdir / "another"
    tmp2.mkdir()
    shutil.copy(SAMPLE_SMALL_IMG_FNAME, tmp2)

    tmpProj.baseImgDirs.add(tmp2)
    with pytest.raises(IOError):
        tmpProj.getFullImgName(SAMPLE_SMALL_IMG_FNAME.name)

    # Since a set is used internally, no gurantee that the order is preserved
    assert tmpProj.getFullImgName(SAMPLE_SMALL_IMG_FNAME.name, thorough=False)


def test_remove_image(prjWithSavedStuff):
    imName = prjWithSavedStuff.getFullImgName(SAMPLE_IMG_FNAME.name)
    assert imName in prjWithSavedStuff.imgToAnnMapping
    assert imName.exists()

    prjWithSavedStuff.removeImage(imName)
    assert imName not in prjWithSavedStuff.imgToAnnMapping
    assert not imName.exists()

    nonexistImg = "garbage.png"
    prjWithSavedStuff.removeImage(nonexistImg)


def test_pkl(prjWithSavedStuff):
    pklBytes = pkl.dumps(prjWithSavedStuff)

    # Test that the pickle is valid
    loaded = pkl.loads(pklBytes)

    assert loaded.cfg == prjWithSavedStuff.cfg


def test_add_fmt_annotation(
    prjWithSavedStuff: ProjectData, sampleComps: pd.DataFrame, tmpdir
):
    imname = SAMPLE_SMALL_IMG_FNAME.name
    fpath = Path(tmpdir / imname + ".csv")
    prjWithSavedStuff.compIo.exportCsv(sampleComps, fpath)
    with pytest.raises(IOError):
        prjWithSavedStuff.addFormattedAnnotation(fpath)

    prjWithSavedStuff.addFormattedAnnotation(fpath, True)
    fullImgName = prjWithSavedStuff.getFullImgName(imname)
    annName = prjWithSavedStuff.imgToAnnMapping[fullImgName]
    cmpData = prjWithSavedStuff.compIo.importByFileType(annName)
    assert np.array_equal(cmpData.values, sampleComps.values)


def test_abspath_none():
    assert absolutePath(None) is None


def test_load_autosave(app, filePlg, tmp_path):
    state = app.appStateEditor

    fns.saveToFile({"interval": 10}, tmp_path / "autosave.param")

    importer = state.stateFuncsDf.at["autosave", "importFunc"]
    importer(True)
    assert filePlg.autosaveTimer.isActive()

    importer(tmp_path / "autosave.param")
    assert filePlg.autosaveTimer.interval() == 1000 * 60 * 10

    importer(False)
    assert not filePlg.autosaveTimer.isActive()


def test_export_autosave(app, filePlg, tmp_path):
    state = app.appStateEditor

    fns.saveToFile({"interval": 10}, tmp_path / "autosave.param")

    exporter = state.stateFuncsDf.at["autosave", "exportFunc"]
    for proc, params in filePlg.toolsEditor.procToParamsMapping.items():
        if proc.name == "Start Autosave":
            break
    else:
        raise ValueError("Autosave proc not found")

    params["interval"] = 10
    params["backupFolder"] = str(tmp_path)
    proc()

    outpath = tmp_path / "autosave_export"
    outpath.mkdir()
    exporter(outpath)
    assert list(outpath.iterdir())

    cfg = fns.attemptFileLoad(next(outpath.iterdir()))
    assert "interval" in cfg and cfg["interval"] == 10
    assert "backupFolder" in cfg and cfg["backupFolder"] == str(tmp_path)

    filePlg.stopAutosave()
    for file in outpath.iterdir():
        file.unlink()
    assert not exporter(outpath)
    assert not list(outpath.iterdir())
