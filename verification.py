
import argparse
import numpy as np
import sklearn
from sklearn import preprocessing
import datetime
import pickle
from mxnet.gluon.data.vision import transforms
import mxnet as mx
from mxnet import gluon


def load_bin(path, image_size):
  transform = transforms.Compose([
    transforms.ToTensor()
    ])
  try:
    with open(path, 'rb') as f:
      bins, issame_list = pickle.load(f) #py2
  except UnicodeDecodeError as e:
    with open(path, 'rb') as f:
      bins, issame_list = pickle.load(f, encoding='bytes') #py3
  data_list = []
  for flip in [0,1]:
    data = mx.np.zeros((len(issame_list)*2, 3, image_size[0], image_size[1]), dtype=np.float32)
    data_list.append(data)
  for i in range(len(issame_list)*2):
    _bin = bins[i]
    img = mx.image.imdecode(_bin)
    img = transform(img)
    img = 2*img - 1
    #img = img.transpose( (2,0,1) )
    #img = nd.transpose(img, axes=(2, 0, 1))
    for flip in [0,1]:
      if flip==1:
        img = img[:,:,::-1]
        #img = mx.ndarray.flip(data=img, axis=2)
      data_list[flip][i][:] = img
    if i%1000==0:
      print('loading bin', i)
  print(data_list[0].shape)
  return (data_list, issame_list)

def easytest(data_set, net, ctx, batch_size):
  print('testing verification..')
  data_list = data_set[0]
  issame_list = data_set[1]
  time_consumed = 0.0
  embeddings = None
  for i in range( len(data_list) ):
    data = data_list[i]
    cx = mx.np.zeros( (batch_size,)+data.shape[1:] )
    ba = 0
    while ba<data.shape[0]:
      bb = min(ba+batch_size, data.shape[0])
      count = bb-ba
      cx[:count,:,:,:] = data[ba:bb, :,:,:]
      #x = data[ba:bb, :,:,:]
      time0 = datetime.datetime.now()
      xs = gluon.utils.split_and_load(cx, ctx_list=ctx, batch_axis=0)
      embs = [net(x).asnumpy() for x in xs]
      time_now = datetime.datetime.now()
      diff = time_now - time0
      time_consumed+=diff.total_seconds()
      embs = np.concatenate(embs, axis=0)
      #print(ba, bb, _embeddings.shape)
      if embeddings is None:
        embeddings = np.zeros( (data.shape[0], embs.shape[1]) )
      embeddings[ba:bb,:] += embs[:count,:]
      ba = bb

  print('infer time', time_consumed)
  embeddings /= len(data_list)
  #print(embeddings.shape)
  xnorm = embeddings*embeddings
  xnorm = np.sum(xnorm, axis=1)
  xnorm = np.sqrt(xnorm)
  xnorm = np.mean(xnorm)
  #print(embeddings[0:5, 0:20])
  embeddings = sklearn.preprocessing.normalize(embeddings)
  print(embeddings.shape)
  emb1 = embeddings[0::2]
  emb2 = embeddings[1::2]
  sim = emb1 * emb2
  sim = np.sum(sim, 1)
  #diff = emb1 - emb2
  #dist = np.sum(np.square(diff),1)
  #thresholds = np.arange(0, 4, 0.01)
  thresholds = np.arange(0, 1, 0.0025)
  actual_issame = issame_list
  acc_max = 0.0
  thresh = 0.0
  for threshold in thresholds:
    #predict_issame = np.less(dist, threshold)
    predict_issame = np.greater(sim, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))
  
    tpr = 0 if (tp+fn==0) else float(tp) / float(tp+fn)
    fpr = 0 if (fp+tn==0) else float(fp) / float(fp+tn)
    acc = float(tp+tn)/predict_issame.size
    if acc>acc_max:
      acc_max = acc
      thresh = threshold
  return xnorm, acc_max, thresh

