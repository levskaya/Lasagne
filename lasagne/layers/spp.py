"""
SPPLayer
from
"Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition"
http://arxiv.org/abs/1406.4729
"""
from __future__ import division #, absolute_import
#from __future__ import print_function, unicode_literals

from .base import Layer
from ..utils import as_tuple


import numpy as np
#import theano
import theano.tensor as T
from theano.tensor.signal import downsample

#fX = theano.config.floatX

__all__ = [
    "SPPLayer"
]

def spp_max_pool_axis_kwargs(in_shape, out_shape):
    int_ceil = lambda x: int(np.ceil(x))
    # eg. if input is 5 and output is 2, each pool size should be 3
    pool_size = int_ceil(in_shape / out_shape)
    # stride should equal pool_size, since we want non-overlapping regions
    stride = pool_size
    # pad as much as possible, since ignore_border=True
    padding = int_ceil((pool_size * out_shape - in_shape) / 2)

    assert padding < pool_size

    return dict(
        ds=pool_size,
        st=stride,
        padding=padding,
    )

def spp_max_pool_kwargs(in_shape, out_shape):
    assert len(in_shape) == len(out_shape)
    axis_res = []
    for i, o in zip(in_shape, out_shape):
        axis_res.append(spp_max_pool_axis_kwargs(i, o))
    return dict(
        ds=tuple([r["ds"] for r in axis_res]),
        st=tuple([r["st"] for r in axis_res]),
        padding=tuple([r["padding"] for r in axis_res]),
        # must be set to true for padding to work
        ignore_border=True,
    )

class SPPLayer(Layer):
    """
    SPP layer

    Parameters
    ----------
    incoming : a :class:`Layer` instance or tuple
        The layer feeding into this layer, or the expected input shape.

    spp_levels:  [(1,1), (2,2)]

    **kwargs
        Any additional keyword arguments are passed to the :class:`Layer`
        superclass.

    Notes
    -----

    """

    def __init__(self, incoming, spp_levels, **kwargs):
        super(SPPLayer, self).__init__(incoming, **kwargs)

        #self.pool_size = as_tuple(pool_size, 2)
        self.spp_levels = spp_levels

        if len(self.input_shape) != 4:
            raise ValueError("Tried to create a 2D pooling layer with "
                             "input shape %r. Expected 4 input dimensions "
                             "(batchsize, channels, 2 spatial dimensions)."
                             % (self.input_shape,))


    def get_output_shape_for(self, input_shape):

        output_shape = (input_shape[0],
                        input_shape[1] * sum(d1 * d2 for d1, d2 in self.spp_levels) )

        return tuple(output_shape)

    def get_output_for(self, input, **kwargs):
        input_shape = tuple(input.shape)

        mp_kwargs_list = [spp_max_pool_kwargs(input_shape[2:], spp_level)
                          for spp_level in self.spp_levels]
        pooled = [downsample.max_pool_2d(input, **kwargs)
                  for kwargs in mp_kwargs_list]
        concat_pooled = T.concatenate([p.flatten(2) for p in pooled], axis=1)

        return concat_pooled
