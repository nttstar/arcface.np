import mxnet as mx
from mxnet import gluon
from mxnet import profiler
from mxnet.gluon import nn
#from mxnet import ndarray as nd


class ArcMarginBlock(gluon.HybridBlock):
    def __init__(self, args, **kwargs):
      super(ArcMarginBlock, self).__init__(**kwargs)
      self.margin_s = args.margin_s
      self.margin_m = args.margin_m
      self.margin_a = args.margin_a
      self.margin_b = args.margin_b
      self.num_classes = args.num_classes
      if args.ctx_num_classes is not None:
          self.num_classes = args.ctx_num_classes
      self.emb_size = args.emb_size
      print('in arc margin', self.num_classes, self.emb_size)
      #self.weight = gluon.Parameter(name = 'fc7_weight', shape = (self.num_classes, self.emb_size))
      #self.weight.initialize()
      #self._weight = nd.empty(shape = (self.num_classes, self.emb_size))
      #if self.margin_a>0.0:
      with self.name_scope():
        self.fc7_weight = self.params.get('fc7_weight', shape=(self.num_classes, self.emb_size))
        self.fc7_weight.initialize(init=mx.init.Normal(0.01))


    def hybrid_forward(self, F, x, y, fc7_weight):
        if self.margin_a==0.0:
          fc7 = F.npx.FullyConnected(x, fc7_weight, no_bias = True, num_hidden=self.num_classes, name='fc7')
          #fc7 = self.dense(feat)
          #with x.context:
          #  _w = self._weight.data()
            #_b = self._bias.data()
          #fc7 = nd.FullyConnected(data=feat, weight=_w, bias = _b, num_hidden=self.num_classes, name='fc7')
          #fc7 = F.softmax_cross_entropy(data = fc7, label=label)
          return fc7

        xnorm = F.np.linalg.norm(x, 'fro', 1, True) + 0.00001
        nx = x / xnorm
        #nx = nx * self.margin_s

        wnorm = F.np.linalg.norm(fc7_weight, 'fro', 1, True) + 0.00001
        nw = fc7_weight / wnorm

        #nx = F.L2Normalization(x, mode='instance', name='fc1n')*self.margin_s
        #nw = F.npx.L2Normalization(fc7_weight, mode='instance')
        fc7 = F.npx.fully_connected(nx, nw, no_bias = True, num_hidden=self.num_classes, name='fc7')
        #fc7 = self.dense(nx)
        if self.margin_a!=1.0 or self.margin_m!=0.0 or self.margin_b!=0.0:
          gt_one_hot = F.npx.one_hot(y, depth = self.num_classes, on_value = 1.0, off_value = 0.0)
          if self.margin_a==1.0 and self.margin_m==0.0:
            _onehot = gt_one_hot*self.margin_b
            fc7 = fc7-_onehot
          else:
            fc7_onehot = fc7 * gt_one_hot
            cos_t = fc7_onehot
            t = F.np.arccos(cos_t)
            if self.margin_a!=1.0:
              t = t*self.margin_a
            if self.margin_m>0.0:
              t = t+self.margin_m
            margin_cos = F.np.cos(t)
            if self.margin_b>0.0:
              margin_cos = margin_cos - self.margin_b
            margin_fc7 = margin_cos
            margin_fc7_onehot = margin_fc7 * gt_one_hot
            diff = margin_fc7_onehot - fc7_onehot
            fc7 = fc7+diff

        fc7 = fc7*self.margin_s
        return fc7

