import pyqtgraph as pg
import pytest

from s3a.constants import PRJ_CONSTS as CNST
from s3a.plugins.usermetrics import UserMetricsPlugin


@pytest.mark.smallimage
def test_metrics_image(app):
    metrics: UserMetricsPlugin = app.classPluginMap[UserMetricsPlugin]
    metrics.props[CNST.PROP_COLLECT_USR_METRICS] = True

    toEmit = [
        dict(
            action=CNST.DRAW_ACT_ADD,
            mouse_pos=pos,
            pixel_size=app.mainImage.imageItem.pixelWidth(),
        )
        for pos in [(50, 50), (40, 40)]
    ]
    for info in toEmit:
        metrics.mainImageMouseFilter.sigMouseMoved.emit(info)
        # last proxy is mouse, first proxy is viewbox
        metrics.collectorProxies[-1].flush()
    # TODO: Check correctness, for now just enough to ensure no crashing
    vb: pg.ViewBox = app.mainImage.getViewBox()
    vb.sigRangeChanged.emit(vb, [[0, 1], [0, 1]], [True, True])
    metrics.collectorProxies[0].flush()

    metrics.props[CNST.PROP_COLLECT_USR_METRICS] = False
    # Ensure no metric collection after this function runs
    assert not metrics.props[CNST.PROP_COLLECT_USR_METRICS]
