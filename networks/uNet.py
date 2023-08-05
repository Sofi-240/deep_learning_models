import tensorflow as tf
from tensorflow import keras
from typing import Callable
from networks.configs import BlockConfig, Config


def _base_setup(mode='base'):
    assert mode == 'base' or mode == 'resnet'
    dbl_conv_encoder = BlockConfig('dbl_conv', block='encoder', mode='base', fn=None)
    dbl_conv_encoder.add_layer_config('conv', from_base=True, call_name='conv2d')
    dbl_conv_encoder.add_layer_config('norm', from_base=True, call_name='normalization', norm='batch')
    dbl_conv_encoder.add_layer_config('act', from_base=True, call_name='activation')

    down_sample_encoder = BlockConfig('down_sample', block='encoder', fn=None)
    down_sample_encoder.add_layer_config('pool', from_base=True, call_name='pool', mode='max')
    down_sample_encoder.add_layer_config('dropout', from_base=True, call_name='dropout')

    dbl_conv_middle = BlockConfig('dbl_conv', block='middle', mode='base', fn=None)
    dbl_conv_middle.add_layer_config('conv', from_base=True, call_name='conv2d')
    dbl_conv_middle.add_layer_config('norm', from_base=True, call_name='normalization', norm='batch')
    dbl_conv_middle.add_layer_config('act', from_base=True, call_name='activation')

    up_sample_decoder = BlockConfig('up_sample', block='decoder', fn=None)
    up_sample_decoder.add_layer_config('pool', from_base=True, call_name='conv2dT')
    up_sample_decoder.add_layer_config('dropout', from_base=True, call_name='dropout')

    dbl_conv_decoder = BlockConfig('dbl_conv', block='decoder', mode='base', fn=None)
    dbl_conv_decoder.add_layer_config('conv', from_base=True, call_name='conv2d')
    dbl_conv_decoder.add_layer_config('norm', from_base=True, call_name='normalization', norm='batch')
    dbl_conv_decoder.add_layer_config('act', from_base=True, call_name='activation')

    output_block = BlockConfig('output_conv', block='output', fn=None)
    output_block.add_layer_config('conv', from_base=True, call_name='conv2d', layer_kw=dict(kernel_size=(1, 1)))
    output_block.add_layer_config('act', from_base=True, call_name='activation', layer_kw=dict(activation='softmax'))

    unet_config = Config('unet')
    unet_config.update(
        dbl_conv_encoder=dbl_conv_encoder,
        down_sample_encoder=down_sample_encoder,
        dbl_conv_middle=dbl_conv_middle,
        up_sample_decoder=up_sample_decoder,
        dbl_conv_decoder=dbl_conv_decoder,
        output_block=output_block
    )
    return unet_config
