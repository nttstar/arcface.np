
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
  #parser.add_argument('--data-dir', default='./faces_emore', help='training set directory')
  parser.add_argument('--data-dir', default='./ms1m-retinaface-t2', help='training set directory')
  #parser.add_argument('--data-dir', default='./faces_webface_112x112', help='training set directory')
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
  parser.add_argument('--batch-size', type=int, default=512, help='batch size in all')
  parser.add_argument('--margin-s', type=float, default=64.0, help='scale for feature')
  parser.add_argument('--margin-a', type=float, default=1.0, help='')
  parser.add_argument('--margin-m', type=float, default=0.3, help='margin for loss')
  parser.add_argument('--margin-b', type=float, default=0.2, help='')
  parser.add_argument('--rand-mirror', type=int, default=1, help='if do random mirror in training')
  parser.add_argument('--cutoff', type=int, default=0, help='cut off aug')
  parser.add_argument('--eval', type=str, default='lfw,cfp_fp,agedb_30', help='verification targets')
  #parser.add_argument('--eval', type=str, default='lfw', help='verification targets')
  parser.add_argument('--gpus', type=str, default='0,1,2,3,4,5,6,7', help='')
  parser.add_argument('--hybrid', action='store_true', help='')
  args = parser.parse_args()
  return args


class FeatBlock(gluon.HybridBlock):
    def __init__(self, args, is_train, **kwargs):
        super(FeatBlock, self).__init__(**kwargs)
        with self.name_scope():
          self.feat_net = nn.HybridSequential(prefix='')
          self.feat_net.add(fresnet.get(args.num_layers, args.emb_size, args.use_dropout))
          self.is_train = is_train
          if is_train:
            initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="out", magnitude=2) #resnet style
            self.feat_net.initialize(init=initializer)
          #self.feat_with_bn = feat_with_bn


    def hybrid_forward(self, F, x):
        feat = self.feat_net(x)
        if self.is_train:
          mean = F.np.mean(feat, axis=[0])
          var = F.np.var(feat, axis=[0])
          var = F.np.sqrt(var + 2e-5)
          feat = (feat - mean) / var
        return feat


class NPCache():
  def __init__(self):
    self._cache = {}

  def get(self, context, name, shape):
    key = "%s_%s"%(name, context)
    #print(key)
    if not key in self._cache:
      v = mx.np.zeros( shape=shape, ctx = context)
      self._cache[key] = v
    else:
      v = self._cache[key]
    return v

  def get2(self, context, name, arr):
    key = "%s_%s"%(name, context)
    #print(key)
    if not key in self._cache:
      v = mx.np.zeros( shape=arr.shape, ctx = context)
      self._cache[key] = v
    else:
      v = self._cache[key]
    arr.copyto(v)
    #mx.np.copy(arr, out=v)
    return v

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
    assert args.batch_size%args.ctx_num==0
    args.per_batch_size = args.batch_size//args.ctx_num
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
    global_num_ctx = args.ctx_num
    if args.num_classes%global_num_ctx==0:
      args.ctx_num_classes = args.num_classes//global_num_ctx
    else:
      args.ctx_num_classes = args.num_classes//global_num_ctx+1
    args.local_num_classes = args.ctx_num_classes * args.ctx_num
    args.local_class_start = 0
    args.ctx_class_start = []
    for i in range(args.ctx_num):

      _c = args.local_class_start + i*args.ctx_num_classes
      args.ctx_class_start.append(_c)
    path_imgrec = os.path.join(data_dir, "train.rec")


    print('Called with argument:', args)
    data_shape = (args.image_channel,image_size[0],image_size[1])
    mean = None
    begin_epoch = 0

    #feat_net = fresnet.get(100, 256)
    #net = TrainBlock(args)
    args.use_dropout = True
    if args.num_classes>=20000:
      args.use_dropout = False
    feat_net = FeatBlock(args, is_train=True)
    feat_net.collect_params().reset_ctx(ctx)
    if args.hybrid:
      feat_net.hybridize()
    cls_nets = []
    for i in range(args.ctx_num):
      cls_net = ArcMarginBlock(args)
      #cls_net.initialize(init=mx.init.Normal(0.01))
      cls_net.collect_params().reset_ctx(mx.gpu(i))
      if args.hybrid:
        cls_net.hybridize()
      cls_nets.append(cls_net)


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
    for name in args.eval.split(','):
      path = os.path.join(data_dir,name+".bin")
      if os.path.exists(path):
        print('loading ver-set:', name)
        data_set = verification.load_bin(path, image_size)
        ver_list.append(data_set)
        ver_name_list.append(name)

    def ver_test(nbatch, tnet):
      results = []
      for i in range(len(ver_list)):
        xnorm, acc, thresh = verification.easytest(ver_list[i], tnet, ctx, batch_size = args.batch_size)
        print('[%s][%d]Accuracy-Thresh-XNorm: %.5f - %.5f - %.5f' % (ver_name_list[i], nbatch, acc, thresh, xnorm))
        #print('[%s][%d]Accuracy: %1.5f+-%1.5f' % (ver_name_list[i], nbatch, acc1, std1))
        #print('[%s][%d]Accuracy-Flip: %1.5f+-%1.5f' % (ver_name_list[i], nbatch, acc2, std2))
        results.append(acc)
      return results


    total_time = 0
    num_epochs = 0
    best_acc = [0]
    highest_acc = [0.0, 0.0]  #lfw and target
    global_step = [0]
    save_step = [0]
    lr_steps = [20000, 28000, 32000]
    if args.num_classes>=20000:
      lr_steps = [100000, 160000, 220000]
    print('lr_steps', lr_steps)

    kv = mx.kv.create('device')
    trainer = gluon.Trainer(feat_net.collect_params(), 'sgd', 
            {'learning_rate': args.lr, 'wd': args.wd, 'momentum': args.mom, 'multi_precision': True},
            )

    cls_trainers = []
    for i in range(args.ctx_num):
        _trainer = gluon.Trainer(cls_nets[i].collect_params(), 'sgd', 
                {'learning_rate': args.lr, 'wd': args.wd, 'momentum': args.mom, 'multi_precision': True},
                )
        cls_trainers.append(_trainer)

    def _batch_callback():
      mbatch = global_step[0]
      global_step[0]+=1
      for _lr in lr_steps:
        if mbatch==_lr:
          args.lr *= 0.1
          trainer.set_learning_rate(args.lr)
          for _trainer in cls_trainers:
              _trainer.set_learning_rate(args.lr)
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
        tnet = FeatBlock(args, is_train=False, params = feat_net.collect_params())
        if args.hybrid:
          tnet.hybridize()
        acc_list = ver_test(mbatch, tnet)
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
          tnet.save_parameters(fname)
          if args.hybrid:
            tnet.export(args.prefix, msave)
          #arg, aux = model.get_params()
          #mx.model.save_checkpoint(prefix, msave, model.symbol, arg, aux)
        print('[%d]Accuracy-Highest: %1.5f'%(mbatch, highest_acc[-1]))
      if args.max_steps>0 and mbatch>args.max_steps:
        sys.exit(0)


    #loss_weight = 1.0
    #loss = gluon.loss.SoftmaxCrossEntropyLoss(weight = loss_weight)
    #loss = nd.SoftmaxOutput
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    tmp_ctx = mx.gpu(0)
    cpu_ctx = mx.cpu()
    cache = NPCache()
    #ctx_fc7_max = mx.np.zeros( (args.batch_size, args.ctx_num), dtype=np.float32, ctx=cpu_ctx)
    #global_fc7_max = mx.np.zeros( (args.batch_size, 1), dtype=np.float32, ctx=cpu_ctx)
    #local_fc7_sum = mx.np.zeros((args.batch_size,1), ctx=cpu_ctx)
    #local_fc1_grad = mx.np.zeros( (args.batch_size,args.emb_size), ctx=cpu_ctx)
    while True:
        #trainer = update_learning_rate(opt.lr, trainer, epoch, opt.lr_factor, lr_steps)
        tic = time.time()
        #train_iter.reset()
        metric.reset()
        btic = time.time()
        #for i, batch in enumerate(train_iter):
        for batch_idx, (x, y) in enumerate(loader):
            y = y.astype(np.float32)
            #print(x.shape, y.shape)
            #print(x.dtype, y.dtype)
            _batch_callback()
            #data = gluon.utils.split_and_load(batch.data[0].astype(opt.dtype), ctx_list=ctx, batch_axis=0)
            #label = gluon.utils.split_and_load(batch.label[0].astype(opt.dtype), ctx_list=ctx, batch_axis=0)
            data = gluon.utils.split_and_load(x, ctx_list=ctx, batch_axis=0)
            label = gluon.utils.split_and_load(y, ctx_list=ctx, batch_axis=0)
            #outputs = []
            #losses = []
            fc1_list = []
            fc1_out_list = []
            fc1_list_cpu = []
            fc7_list = []
            with ag.record():
                for _data, _label in zip(data, label):
                    #print(y.asnumpy())
                    fc1 = feat_net(_data)
                    fc1_out_list.append(fc1)
                    #fc1_list.append(fc1)
            for _fc1 in fc1_out_list:
                #fc1_cpu = cache.get2(cpu_ctx, 'fc1_cpu', _fc1)
                fc1_cpu = _fc1.as_in_ctx(cpu_ctx)
                fc1_list_cpu.append(fc1_cpu)
            global_fc1 = cache.get(cpu_ctx, 'global_fc1_cpu', (args.batch_size, args.emb_size))
            mx.np.concatenate(fc1_list_cpu, axis=0, out=global_fc1)
                #mean = mx.np.mean(global_fc1, axis=[0])
                #var = mx.np.var(global_fc1, axis=[0])
                #var = mx.np.sqrt(var + 2e-5)
                #global_fc1 = (global_fc1 - mean) / var
            _xlist = []
            _ylist = []
            for i, cls_net in enumerate(cls_nets):
                _ctx = mx.gpu(i)
                _y = cache.get2(_ctx, 'ctxy', y)
                _y -= args.ctx_class_start[i]
                _x = cache.get2(_ctx, 'ctxfc1', global_fc1)
                _xlist.append(_x)
                _ylist.append(_y)
            with ag.record():
                for i, cls_net in enumerate(cls_nets):
                    _ctx = mx.gpu(i)
                    _x = _xlist[i]
                    _y = _ylist[i]
                    #_y = cache.get2(_ctx, 'ctxy', y)
                    #_y -= args.ctx_class_start[i]
                    #_x = cache.get2(_ctx, 'ctxfc1', global_fc1)
                    #_x = global_fc1.as_in_ctx(_ctx)
                    _x.attach_grad()
                    _fc7 = cls_net(_x, _y)
                    fc7_list.append(_fc7)
                    fc1_list.append(_x)
            #print('log A')
            fc7_grads = [None] * args.ctx_num
            ctx_fc7_max = cache.get(cpu_ctx, 'gctxfc7max', (args.batch_size, args.ctx_num))
            ctx_fc7_max[:,:] = 0.0
            for i, cls_net in enumerate(cls_nets):
                _fc7 = fc7_list[i]
                _max = cache.get(_fc7.context, 'ctxfc7max', (args.batch_size,))
                #_max = cache.get(cpu_ctx, 'ctxfc7max', (args.batch_size, ))
                mx.np.max(_fc7, axis=1, out=_max)
                #_cpumax = cache.get2(cpu_ctx, 'ctxfc7maxcpu', _max)
                _cpumax = _max.as_in_ctx(cpu_ctx)
                ctx_fc7_max[:, i] = _cpumax
                fc7_grads[i] = cache.get2(_fc7.context, 'fc7grad', _fc7)
            #nd.max(ctx_fc7_max, axis=1, keepdims=True, out=local_fc7_max)
            global_fc7_max = cache.get(cpu_ctx, 'globalfc7max', (args.batch_size, 1))
            mx.np.max(ctx_fc7_max, axis=1, keepdims=True, out=global_fc7_max)
            local_fc7_sum = cache.get(cpu_ctx, 'local_fc7_sum', (args.batch_size, 1))
            local_fc7_sum[:,:] = 0.0

            for i, cls_net in enumerate(cls_nets):
              _ctx = mx.gpu(i)
              #_max = global_fc7_max.as_in_ctx(mx.gpu(i))
              _max = cache.get2(_ctx, 'fc7maxgpu', global_fc7_max)
              fc7_grads[i] -= _max
              #mx.np.exp(fc7_grads[i], out=fc7_grads[i])
              fc7_grads[i] = mx.np.exp(fc7_grads[i])
              #_sum = cache.get(cpu_ctx, 'ctxfc7sum', (args.batch_size, 1))
              _sum = cache.get(_ctx, 'ctxfc7sum', (args.batch_size, 1))
              mx.np.sum(fc7_grads[i], axis=1, keepdims=True, out=_sum)
              #_cpusum = cache.get2(cpu_ctx, 'ctxfc7maxcpu', _max)
              _cpusum = _sum.as_in_ctx(cpu_ctx)
              local_fc7_sum += _cpusum
            global_fc7_sum = local_fc7_sum

            #print('log B')
            local_fc1_grad = cache.get(cpu_ctx, 'localfc1grad', (args.batch_size, args.emb_size))
            local_fc1_grad[:,:] = 0.0

            for i, cls_net in enumerate(cls_nets):
              #_sum = global_fc7_sum.as_in_ctx(mx.gpu(i))
              _ctx = mx.gpu(i)
              _sum = cache.get2(_ctx, 'globalfc7sumgpu', global_fc7_sum)
              fc7_grads[i] /= _sum
              a = i*args.ctx_num_classes
              b = (i+1)*args.ctx_num_classes
              _y = cache.get2(_ctx, 'ctxy2', y)
              _y -= args.ctx_class_start[i]
              _yonehot = cache.get(_ctx, 'yonehot', (args.batch_size, args.ctx_num_classes))
              mx.npx.one_hot(_y, depth=args.ctx_num_classes, on_value=1.0, off_value=0.0, out=_yonehot)
              #_label = (y - args.ctx_class_start[i]).as_in_ctx(mx.gpu(i))
              #_label = mx.npx.one_hot(_label, depth=args.ctx_num_classes, on_value=1.0, off_value=0.0)
              fc7_grads[i] -= _yonehot
              fc7_list[i].backward(fc7_grads[i])
              fc1 = fc1_list[i]
              #fc1_grad = cache.get2(cpu_ctx, 'fc1gradcpu', fc1.grad)
              fc1_grad = fc1.grad.as_in_ctx(cpu_ctx)
              #print(fc1.grad.dtype, fc1.grad.shape)
              #print(fc1.grad[0:5,0:5])
              local_fc1_grad += fc1_grad
              cls_trainers[i].step(args.batch_size)
            #print('log C')
            for i in range(args.ctx_num):
                p = args.batch_size//args.ctx_num
                a = p*i
                b = p*(i+1)
                _fc1_grad = local_fc1_grad[a:b, :]
                _grad = cache.get2(mx.gpu(i), 'fc1gradgpu', _fc1_grad)
                #_grad = local_fc1_grad[a:b,:].as_in_ctx(mx.gpu(i))
                #print(i, fc1_out_list[i].shape, _grad.shape)
                fc1_out_list[i].backward(_grad)
            #print('log D')
            trainer.step(args.batch_size)
            #print('after step')
            mx.npx.waitall()
            i = batch_idx
            if i>0 and i%20==0:
                logger.info('Epoch[%d] Batch [%d]\tSpeed: %f samples/sec'%(
                               num_epochs, i, args.batch_size/(time.time()-btic) ))
            #metric.update(label, outputs)
            #if i>0 and i%20==0:
            #    name, acc = metric.get()
            #    if len(name)==2:
            #      logger.info('Epoch[%d] Batch [%d]\tSpeed: %f samples/sec\t%s=%f, %s=%f'%(
            #                     num_epochs, i, args.batch_size/(time.time()-btic), name[0], acc[0], name[1], acc[1]))
            #    else:
            #      logger.info('Epoch[%d] Batch [%d]\tSpeed: %f samples/sec\t%s=%f'%(
            #                     num_epochs, i, args.batch_size/(time.time()-btic), name[0], acc[0]))
            #    #metric.reset()
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

