import os
import random
import logging
import sys
import numbers
import math
import sklearn
import datetime
import numpy as np
import cv2

import mxnet as mx
from mxnet import ndarray as nd
from mxnet import io
from mxnet import recordio
from mxnet import gluon
from mxnet.gluon.data import Dataset
from mxnet.gluon.data.vision import transforms
from skimage import transform as trans
from mxnet import np, npx
npx.set_np()

logger = logging.getLogger()


class FaceDataset(Dataset):

    def __init__(self, data_shape, path_imgrec, transform=None):
        super(FaceDataset, self).__init__()
        logging.info('loading recordio %s...',
                     path_imgrec)
        path_imgidx = path_imgrec[0:-4]+".idx"
        self.imgrec = recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')  # pylint: disable=redefined-variable-type
        s = self.imgrec.read_idx(0)
        header, _ = recordio.unpack(s)
        if header.flag>0:
          print('header0 label', header.label)
          self.header0 = (int(header.label[0]), int(header.label[1]))
          #assert(header.flag==1)
          self.imgidx = list(range(1, int(header.label[0])))
          #self.imgidx = []
          #self.id2range = {}
          #self.seq_identity = range(int(header.label[0]), int(header.label[1]))
          #for identity in self.seq_identity:
          #  s = self.imgrec.read_idx(identity)
          #  header, _ = recordio.unpack(s)
          #  a,b = int(header.label[0]), int(header.label[1])
          #  count = b-a
          #  if count<images_filter:
          #    continue
          #  self.id2range[identity] = (a,b)
          #  self.imgidx += range(a, b)
          #print('id2range', len(self.id2range))
        else:
          self.imgidx = list(self.imgrec.keys)
        self.seq = self.imgidx

        self.data_shape = data_shape
        self.transform = transforms.Compose([
          #transforms.RandomBrightness(0.3),
          #transforms.RandomContrast(0.3),
          #transforms.RandomSaturation(0.3),
          transforms.RandomFlipLeftRight(),
          transforms.ToTensor()
          ])

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        s = self.imgrec.read_idx(self.seq[idx])
        header, img = recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
          label = label[0]
        img = mx.image.imdecode(img)
        #print(img.__class__)
        #if np.random.rand()<0.5:
        #    img = img[:,::-1,:]
        if self.transform is not None:
            img = self.transform(img)
            #img -= 0.5
        return img, label

if __name__ == '__main__':
  ds = FaceDataset(data_shape=(3,112,112), path_imgrec = '/gpu/data1/jiaguo/faces_emore/train.rec')
  print(len(ds))
  img, label = ds[0]
  print(img.__class__, label.__class__)
  print(img.shape, label)
  loader = gluon.data.DataLoader(ds, batch_size=512, shuffle=True, num_workers = 8, last_batch='discard')
  for batch_idx, (data, label) in enumerate(loader):
    print(batch_idx,data.shape,label.shape,data.__class__)

