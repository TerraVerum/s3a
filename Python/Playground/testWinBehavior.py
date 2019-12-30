import pyqtgraph as pg
import numpy as np

from graphicshelpers import flipHorizontal

# make plot with a line drawn in
app = pg.mkQApp()
widget = pg.PlotWindow()
plt = pg.PlotDataItem([1,5,2,4,3,2], pen='r')

# add an image, scaled
img = pg.ImageItem(np.random.normal(size=(100,100)))
img.scale(0.2, 0.1)
img.setZValue(-100)
widget.addItem(img)
widget.addItem(plt)

txt = pg.LabelItem('test', color='y')

txt.setPos(2,2)
#txt.setTransform(txt.transform().scale(-1,1))
flipHorizontal(txt)
widget.addItem(txt)

app.exec()