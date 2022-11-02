import pytest
from pyqtgraph.parametertree import interact

from s3a.parameditors.algcollection import AlgorithmCollection
from s3a.processing.pipeline import PipelineParameter, maybeGetFunction


def one():
    return dict(a=1)


def two(a):
    return dict(a=a, b=3)


def final(a=6, b=7):
    return dict(a=a, b=b, c=3)


@pytest.fixture()
def pipeline():
    pipe = PipelineParameter(name="Test")
    for stage in [one, two, final]:
        pipe.addStage(stage)
    return pipe


def test_savestate(pipeline):
    pipeline.child("final").setOpts(enabled=False)
    parent = PipelineParameter.create(name="parent", type="group", children=[pipeline])
    interact(pipeline.activate, parent=parent)

    expected = dict(Test=["One", "Two", dict(Final={}, enabled=False)])
    assert pipeline.saveState() == expected

    func = maybeGetFunction(pipeline.child("final"))
    func.input["a"] = 5
    expected["Test"][2]["Final"]["a"] = 5
    assert pipeline.saveState() == expected

    assert pipeline.activate() == dict(a=1, b=3)

    pipeline.child("final").setOpts(enabled=True)
    assert pipeline.activate() == dict(a=1, b=3, c=3)


def test_algcollection(pipeline):
    collection = AlgorithmCollection()
    collection.addProcess(pipeline, top=True)
    state = collection.saveParameterValues()
    assert state == dict(
        top={"Test": ["One", "Two", "Final"]}, primitive={}, modules=[]
    )
