import cv2 as cv
import numpy as np
from skimage import data
from skimage import io

from s3a.constants import BASE_DIR, REQD_TBL_FIELDS

TEST_AUTHOR = 'testauthor'

NUM_COMPS = 15
TESTS_DIR = BASE_DIR.parent/'apptests'
IMG_DIR = TESTS_DIR/'files'
FIMG_SER_COLS = [REQD_TBL_FIELDS.INST_ID, REQD_TBL_FIELDS.VERTICES]
RND = np.random.default_rng(seed=42)

SAMPLE_IMG = data.hubble_deep_field()
SAMPLE_IMG_FNAME = IMG_DIR/'hubble_deep_field.png'

baseImg = np.zeros((5,5), 'uint8')
baseImg[2, 2] = 255
smallImgSrc = cv.resize(baseImg, (50,50), interpolation=cv.INTER_NEAREST)
smallImgSrc = np.tile(smallImgSrc[:, :, None], (1, 1, 3))
SAMPLE_SMALL_IMG = smallImgSrc
SAMPLE_SMALL_IMG_FNAME = IMG_DIR/'smallimg.png'

for impath, im in zip([SAMPLE_SMALL_IMG_FNAME, SAMPLE_IMG_FNAME],
                      [SAMPLE_SMALL_IMG, SAMPLE_IMG]):
  impath.parent.mkdir(exist_ok=True, parents=True)
  io.imsave(impath, im)

# from warnings import simplefilter
# simplefilter('error', DeprecationWarning)
# simplefilter('error', VisibleDeprecationWarning)
# simplefilter('error', UserWarning)