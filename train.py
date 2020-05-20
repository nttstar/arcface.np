
from mxnet import np, npx
npx.set_np()

import os
import sys
import math
import random
import logging
import time
import pickle
import numpy as np
import sklearn
from image_iter import FaceDataset
#from image_iter import FaceImageIterList
import mxnet as mx
from mxnet import gluon
from mxnet import profiler
from mxnet.gluon import nn
from mxnet import ndarray as nd
from mxnet import autograd as ag
from mxnet.gluon.metric import Accuracy, TopKAccuracy, CompositeEvalMetric
import argparse
import mxnet.optimizer as optimizer
#sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src', 'eval'))
import verification
#sys.path.append(os.path.join(os.path.dirname(__file__), 'common'))
#sys.path.append(os.path.join(os.path.dirname(__file__), 'eval'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'blocks'))
import fresnet
from loss import *


logger = logging.getLogger()
logger.setLevel(logging.INFO)

args = None


class AccMetric(mx.gluon.metric.EvalMetric):
  def __init__(self):
    self.axis = 1
    super(AccMetric, self).__init__(
        'acc', axis=self.axis,
        output_names=None, label_names=None)
    self.losses = []
    self.count = 0

  def update(self, labels, preds):
    self.count+=1
    #preds = [preds[1]] #use softmax output
    for label, pred_label in zip(labels, preds):
        if pred_label.shape != label.shape:
            pred_label = mx.np.argmax(pred_label, axis=self.axis)
        pred_label = pred_label.asnumpy().astype('int32').flatten()
        label = label.asnumpy()
        if label.ndim==2:
          label = label[:,0]
        label = label.astype('int32').flatten()
        assert label.shape==pred_label.shape
        self.sum_metric += (pred_label.flat == label.flat).sum()
        self.num_inst += len(pred_label.flat)

class LossValueMetric(mx.gluon.metric.EvalMetric):
  def __init__(self):
    self.axis = 1
    super(LossValueMetric, self).__init__(
        'lossvalue', axis=self.axis,
        output_names=None, label_names=None)
    self.losses = []

  def update(self, labels, preds):
    loss = preds[-1].asnumpy()[0]
    self.sum_metric += loss
    self.num_inst += 1.0
    gt_label = preds[-2].asnumpy()
    #print(gt_label)

def parse_args():
  global args
  parser = argparse.ArgumentParser(description='Train face network')
  # general
  parser.add_argument('--data-dir', default='./faces_emore', help='training set directory')
  parser.add_argument('--prefix', default='./models/A', help='directory to save model.')
  parser.add_argument('--pretrained', default='', help='pretrained model to load')
  parser.add_argument('--ckpt', type=int, default=1, help='checkpoint saving option. 0: discard saving. 1: save when necessary. 2: always save')
  parser.add_argument('--loss-type', type=int, default=4, help='loss type')
  parser.add_argument('--verbose', type=int, default=2000, help='do verification testing and model saving every verbose batches')
  parser.add_argument('--max-steps', type=int, default=0, help='max training batches')
  parser.add_argument('--end-epoch', type=int, default=100000, help='training epoch size.')
  parser.add_argument('--network', default='r50', help='specify network')
  parser.add_argument('--lr', type=float, default=0.1, help='start learning rate')
  parser.add_argument('--lr-steps', type=str, default='', help='steps of lr changing')
  parser.add_argument('--wd', type=float, default=0.0005, help='weight decay')
  parser.add_argument('--fc7-wd-mult', type=float, default=1.0, help='weight decay mult for fc7')
  parser.add_argument('--bn-mom', type=float, default=0.9, help='bn mom')
  parser.add_argument('--mom', type=float, default=0.9, help='momentum')
  parser.add_argument('--emb-size', type=int, default=512, help='embedding length')
  parser.add_argument('--per-batch-size', type=int, default=64, help='batch size in each context')
  parser.add_argument('--margin-m', type=float, default=0.5, help='margin for loss')
  parser.add_argument('--margin-s', type=float, default=64.0, help='scale for feature')
  parser.add_argument('--margin-a', type=float, default=1.0, help='')
  parser.add_argument('--margin-b', type=float, default=0.0, help='')
  parser.add_argument('--rand-mirror', type=int, default=1, help='if do random mirror in training')
  parser.add_argument('--cutoff', type=int, default=0, help='cut off aug')
  parser.add_argument('--eval', type=str, default='lfw,cfp_fp,agedb_30', help='verification targets')
  parser.add_argument('--task', type=str, default='', help='')
  parser.add_argument('--mode', type=str, default='gluon', help='')
  parser.add_argument('--gpus', type=str, default='0,1,2,3,4,5,6,7', help='')
  args = parser.parse_args()
  return args


class TrainBlock(gluon.HybridBlock):
    def __init__(self, args, **kwargs):
        super(TrainBlock, self).__init__(**kwargs)
        with self.name_scope():
          self.feat_net = fresnet.get(args.num_layers, args.emb_size)
          initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="out", magnitude=2) #resnet style
          self.feat_net.initialize(init=initializer)
          self.margin_block = ArcMarginBlock(args)
          self.margin_block.initialize(init=mx.init.Normal(0.01))

    def feat_net(self):
      return self.feat_net

    def hybrid_forward(self, F, x, y):
        feat = self.feat_net(x)
        fc7 = self.margin_block(feat,y)
        return fc7
        #print(z[0].shape, z[1].shape)

#
def train_net(args):
    ctx = []
    for ctx_id in [int(x) for x in args.gpus.split(',')]:
      ctx.append(mx.gpu(ctx_id))
    print('gpu num:', len(ctx))
    prefix = args.prefix
    prefix_dir = os.path.dirname(prefix)
    if not os.path.exists(prefix_dir):
      os.makedirs(prefix_dir)
    end_epoch = args.end_epoch
    args.ctx_num = len(ctx)
    args.num_layers = int(args.network[1:])
    print('num_layers', args.num_layers)
    args.batch_size = args.per_batch_size*args.ctx_num
    args.image_channel = 3

    data_dir = args.data_dir
    print('data dir', data_dir)
    path_imgrec = None
    path_imglist = None
    for line in open(os.path.join(data_dir, 'property')):
      vec = line.strip().split(',')
      assert len(vec)==3
      args.num_classes = int(vec[0])
      image_size = [int(vec[1]), int(vec[2])]
    args.image_h = image_size[0]
    args.image_w = image_size[1]
    print('image_size', image_size)
    assert(args.num_classes>0)
    print('num_classes', args.num_classes)
    path_imgrec = os.path.join(data_dir, "train.rec")


    print('Called with argument:', args)
    data_shape = (args.image_channel,image_size[0],image_size[1])
    mean = None
    begin_epoch = 0

    #feat_net = fresnet.get(100, 256)
    #margin_block = ArcMarginBlock(args)
    net = TrainBlock(args)
    net.collect_params().reset_ctx(ctx)
    net.hybridize()

    #initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="out", magnitude=2) #resnet style
    #feat_net.initialize(ctx=ctx, init=initializer)
    #feat_net.hybridize()
    #margin_block.initialize(ctx=ctx, init=mx.init.Normal(0.01))
    #margin_block.hybridize()

    ds = FaceDataset(data_shape=(3,112,112), path_imgrec = path_imgrec)
    #print(len(ds))
    #img, label = ds[0]
    #print(img.__class__, label.__class__)
    #print(img.shape, label)
    loader = gluon.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers = 8, last_batch='discard')

    metric = CompositeEvalMetric([AccMetric()])

    ver_list = []
    ver_name_list = []
    if args.task=='':
      for name in args.eval.split(','):
        path = os.path.join(data_dir,name+".bin")
        if os.path.exists(path):
          data_set = verification.load_bin(path, image_size)
          ver_list.append(data_set)
          ver_name_list.append(name)
          print('ver', name)

    def ver_test(nbatch):
      results = []
      for i in range(len(ver_list)):
        acc1, std1, acc2, std2, xnorm, embeddings_list = verification.test(ver_list[i], net.feat_net, ctx, batch_size = args.batch_size)
        print('[%s][%d]XNorm: %f' % (ver_name_list[i], nbatch, xnorm))
        #print('[%s][%d]Accuracy: %1.5f+-%1.5f' % (ver_name_list[i], nbatch, acc1, std1))
        print('[%s][%d]Accuracy-Flip: %1.5f+-%1.5f' % (ver_name_list[i], nbatch, acc2, std2))
        results.append(acc2)
      return results


    total_time = 0
    num_epochs = 0
    best_acc = [0]
    highest_acc = [0.0, 0.0]  #lfw and target
    global_step = [0]
    save_step = [0]
    lr_steps = [100000, 160000, 220000]
    print('lr_steps', lr_steps)

    kv = mx.kv.create('device')
    #kv = mx.kv.create('local')
    #_rescale = 1.0/args.ctx_num
    #opt = optimizer.SGD(learning_rate=args.lr, momentum=args.mom, wd=args.wd, rescale_grad=_rescale)
    #opt = optimizer.SGD(learning_rate=args.lr, momentum=args.mom, wd=args.wd)
    if args.mode=='gluon':
      trainer = gluon.Trainer(net.collect_params(), 'sgd', 
              {'learning_rate': args.lr, 'wd': args.wd, 'momentum': args.mom, 'multi_precision': True},
              kvstore=kv)
    else:
      _rescale = 1.0/args.ctx_num
      opt = optimizer.SGD(learning_rate=args.lr, momentum=args.mom, wd=args.wd, rescale_grad=_rescale)
      _cb = mx.callback.Speedometer(args.batch_size, 20)
      arg_params = None
      aux_params = None
      data = mx.sym.var('data')
      label = mx.sym.var('softmax_label')
      if args.margin_a>0.0:
        fc7 = net(data, label)
      else:
        fc7 = net(data)
      #sym = mx.symbol.SoftmaxOutput(data=fc7, label = label, name='softmax', normalization='valid')
      ceop = gluon.loss.SoftmaxCrossEntropyLoss()
      loss = ceop(fc7, label) 
      #loss = loss/args.per_batch_size
      loss = mx.sym.mean(loss)
      sym = mx.sym.Group( [mx.symbol.BlockGrad(fc7), mx.symbol.MakeLoss(loss, name='softmax')] )

    def _batch_callback():
      mbatch = global_step[0]
      global_step[0]+=1
      for _lr in lr_steps:
        if mbatch==_lr:
          args.lr *= 0.1
          if args.mode=='gluon':
            trainer.set_learning_rate(args.lr)
          else:
            opt.lr  = args.lr
          print('lr change to', args.lr)
          break

      #_cb(param)
      if mbatch%1000==0:
        print('lr-batch-epoch:',args.lr, mbatch)

      if mbatch>0 and mbatch%args.verbose==0:
        save_step[0]+=1
        msave = save_step[0]
        do_save = False
        is_highest = False
        acc_list = ver_test(mbatch)
        if len(acc_list)>0:
          lfw_score = acc_list[0]
          if lfw_score>highest_acc[0]:
            highest_acc[0] = lfw_score
          if acc_list[-1]>=highest_acc[-1]:
            highest_acc[-1] = acc_list[-1]
        if args.ckpt==0:
          do_save = False
        elif args.ckpt==1:
          do_save = True
          msave = 1
        elif args.ckpt>1:
          do_save = True
        if do_save:
          print('saving', msave)
          #print('saving gluon params')
          fname = args.prefix+"-gluon.params"
          net.feat_net.save_parameters(fname)
          net.feat_net.export(args.prefix, msave)
          #arg, aux = model.get_params()
          #mx.model.save_checkpoint(prefix, msave, model.symbol, arg, aux)
        print('[%d]Accuracy-Highest: %1.5f'%(mbatch, highest_acc[-1]))
      if args.max_steps>0 and mbatch>args.max_steps:
        sys.exit(0)

    def _batch_callback_sym(param):
      _cb(param)
      _batch_callback()


    if args.mode!='gluon':
      model = mx.mod.Module(
          context       = ctx,
          symbol        = sym,
      )
      model.fit(train_iter,
          begin_epoch        = 0,
          num_epoch          = args.end_epoch,
          eval_data          = None,
          eval_metric        = metric,
          kvstore            = 'device',
          optimizer          = opt,
          initializer        = initializer,
          arg_params         = arg_params,
          aux_params         = aux_params,
          allow_missing      = True,
          batch_end_callback = _batch_callback_sym,
          epoch_end_callback = None )
    else:
      loss_weight = 1.0
      #loss = gluon.loss.SoftmaxCrossEntropyLoss(weight = loss_weight)
      #loss = nd.SoftmaxOutput
      loss = gluon.loss.SoftmaxCrossEntropyLoss()
      while True:
          #trainer = update_learning_rate(opt.lr, trainer, epoch, opt.lr_factor, lr_steps)
          tic = time.time()
          #train_iter.reset()
          metric.reset()
          btic = time.time()
          #for i, batch in enumerate(train_iter):
          for batch_idx, (x, y) in enumerate(loader):
              #print(x.shape, y.shape)
              _batch_callback()
              #data = gluon.utils.split_and_load(batch.data[0].astype(opt.dtype), ctx_list=ctx, batch_axis=0)
              #label = gluon.utils.split_and_load(batch.label[0].astype(opt.dtype), ctx_list=ctx, batch_axis=0)
              data = gluon.utils.split_and_load(x, ctx_list=ctx, batch_axis=0)
              label = gluon.utils.split_and_load(y, ctx_list=ctx, batch_axis=0)
              outputs = []
              losses = []
              with ag.record():
                  for x, y in zip(data, label):
                      #print(y.asnumpy())
                      fc7 = net(x, y)
                      #feat = feat_net(x)
                      #if args.margin_a>0.0:
                      #  fc7 = margin_block(feat,y)
                      #else:
                      #  fc7 = margin_block(feat)
                      #print(z[0].shape, z[1].shape)
                      losses.append(loss(fc7, y))
                      outputs.append(fc7)
              for l in losses:
                  l.backward()
              #trainer.step(batch.data[0].shape[0], ignore_stale_grad=True)
              #trainer.step(args.ctx_num)
              n = x.shape[0]
              #print(n,n)
              trainer.step(n)
              metric.update(label, outputs)
              i = batch_idx
              if i>0 and i%20==0:
                  name, acc = metric.get()
                  if len(name)==2:
                    logger.info('Epoch[%d] Batch [%d]\tSpeed: %f samples/sec\t%s=%f, %s=%f'%(
                                   num_epochs, i, args.batch_size/(time.time()-btic), name[0], acc[0], name[1], acc[1]))
                  else:
                    logger.info('Epoch[%d] Batch [%d]\tSpeed: %f samples/sec\t%s=%f'%(
                                   num_epochs, i, args.batch_size/(time.time()-btic), name[0], acc[0]))
                  #metric.reset()
              btic = time.time()

          epoch_time = time.time()-tic

          # First epoch will usually be much slower than the subsequent epics,
          # so don't factor into the average
          if num_epochs > 0:
            total_time = total_time + epoch_time

          #name, acc = metric.get()
          #logger.info('[Epoch %d] training: %s=%f, %s=%f'%(num_epochs, name[0], acc[0], name[1], acc[1]))
          logger.info('[Epoch %d] time cost: %f'%(num_epochs, epoch_time))
          num_epochs = num_epochs + 1
          #name, val_acc = test(ctx, val_data)
          #logger.info('[Epoch %d] validation: %s=%f, %s=%f'%(epoch, name[0], val_acc[0], name[1], val_acc[1]))

          # save model if meet requirements
          #save_checkpoint(epoch, val_acc[0], best_acc)
      if num_epochs > 1:
          print('Average epoch time: {}'.format(float(total_time)/(num_epochs - 1)))



def main():
    #time.sleep(3600*6.5)
    global args
    args = parse_args()
    train_net(args)

if __name__ == '__main__':
    main()

