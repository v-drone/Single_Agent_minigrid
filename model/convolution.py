from abc import ABC

from mxnet.gluon import nn
from mxnet.gluon.nn import BatchNorm
from mxnet.gluon import HybridBlock


def _conv2d(channel, kernel, padding, stride, norm_layer=BatchNorm, norm_kwargs=None):
    """A common conv-bn-leakyrelu cell"""
    cell = nn.HybridSequential(prefix='')
    cell.add(nn.Conv2D(channel, kernel_size=kernel, strides=stride, padding=padding, use_bias=False))
    cell.add(norm_layer(epsilon=1e-5, momentum=0.9, **({} if norm_kwargs is None else norm_kwargs)))
    cell.add(nn.LeakyReLU(0.1))
    return cell


class DetectionBlock(HybridBlock):
    """Based on YOLO V3 Detection Block
    Parameters
    ----------
    channel : int
        Number of channels for 1x1 conv. 3x3 Conv will have 2*channel.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    """

    def __init__(self, channel, norm_kwargs=None, **kwargs):
        super(DetectionBlock, self).__init__(**kwargs)
        assert channel % 2 == 0, "channel {} cannot be divided by 2".format(channel)
        with self.name_scope():
            self.body = nn.HybridSequential(prefix='')
            # 1x1 reduce
            self.body.add(_conv2d(channel, 1, 0, 1))
            # 3x3 conv expand
            self.body.add(_conv2d(channel * 2, 3, 1, 1))

    def hybrid_forward(self, F, x, *args):
        residual = x
        x = self.body(x)
        return x + residual
