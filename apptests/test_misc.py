from s3a import generalutils as gu
import numpy as np

def test_resize_pad():
  img = np.zeros((100, 500), 'uint8')

  for sz in (500,500), (100,500), (1000, 100):
    rp = gu.resize_pad(img, sz)
    assert rp.shape == sz