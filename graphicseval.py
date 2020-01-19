import pyqtgraph as pg

def overlayImgs(img1, *addtnlImgs):
  win = pg.GraphicsWindow()
  vb = win.addViewBox()
  baseImg = pg.ImageItem(img1)
  vb.addItem(baseImg)
  baseImg.setZValue(1)
  numAddtnlImgs = len(addtnlImgs)
  for ii, extraImg in enumerate(addtnlImgs):
    extraImgItem = pg.ImageItem(extraImg)
    vb.addItem(extraImgItem)
    extraImgItem.setZValue(ii+1)
    extraImgItem.setOpacity(1/(numAddtnlImgs+1))
  return win

def makeImgPieces(img):
  win = pg.PlotWindow()
  vb = pg.ViewBox(invertY=False)
  imgItem = pg.ImageItem(img)
  vb.addItem(imgItem)
  win.addItem(vb)
  return win, vb, imgItem