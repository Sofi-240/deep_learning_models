import tensorflow as tf
from tensorflow import keras
from typing import Union
from networks.configs import copy_layer, base_config, Config
from networks.utils import activation_from_config, normalization_from_config, resize_as, copy_current_name_scope
from keras.layers import Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, Add, concatenate, Identity


def _base_setup(conv_mode: str = 'base'):
    assert conv_mode == 'base' or conv_mode == 'resnet'
    unet_setup = Config('unet_setup')
    dbl_conv_base = Config('dbl_conv', mode=conv_mode)
    dbl_conv_base.update(
        conv=base_config('conv', call_name='conv2d'),
        norm=base_config('norm', call_name='normalization', norm='batch'),
        act=base_config('act', call_name='activation')
    )
    unet_setup['dbl_conv_encoder'] = dbl_conv_base.deepcopy(name='dbl_conv')
    unet_setup['down_sample'] = Config('down_sample')
    unet_setup['down_sample'].update(
        pool=base_config('pool', call_name='pool', mode='max'),
        dropout=base_config('dropout', call_name='dropout')
    )
    unet_setup['dbl_conv_middle'] = dbl_conv_base.deepcopy(name='dbl_conv')
    unet_setup['up_sample'] = Config('up_sample')
    unet_setup['up_sample'].update(
        convT=base_config('convT', call_name='conv2dT'),
        dropout=base_config('dropout', call_name='dropout')
    )
    unet_setup['dbl_conv_decoder'] = dbl_conv_base.deepcopy(name='dbl_conv')
    unet_setup['output_block'] = Config('output_conv')
    unet_setup['output_block'].update(
        conv=base_config('conv', call_name='conv2d', layer_kw=dict(kernel_size=(1, 1))),
        act=base_config('act', call_name='activation', layer_kw=dict(activation='softmax'))
    )
    return unet_setup


def __double_conv(inputs, filters, configs):
    shape = inputs.get_shape()
    scope_name = copy_current_name_scope()

    conv1 = Conv2D(filters=filters, name=scope_name + 'cn1', **configs.get('conv'))
    conv2 = Conv2D(filters=filters, name=scope_name + 'cn2', **configs.get('conv'))

    X_copy = None
    mode = configs.get_from_class('mode', 'base')
    if mode == 'resnet':
        id_configs = configs.get('convID')
        if id_configs is not None or shape[-1] != filters:
            if id_configs is None:
                id_configs = configs['conv'].deepcopy(name='convID', kernel_size=(1, 1))
            X_copy = Conv2D(filters=filters, name=scope_name + 'cnID', **id_configs)(inputs)
            X_copy = normalization_from_config(configs, name=scope_name + 'normID')(X_copy)

    norm1 = normalization_from_config(configs, name=scope_name + 'norm1')
    norm2 = copy_layer(norm1, name=scope_name + 'norm2', include_weights=False)

    act1 = activation_from_config(configs, scope_name + 'act1')
    act2 = copy_layer(act1, name=scope_name + 'act2', include_weights=False)

    X = act1(norm1(conv1(inputs)))
    X = norm2(conv2(X))
    if mode == 'resnet':
        X = Add(name=scope_name + 'add')((X, X_copy if X_copy is not None else inputs))
    X = act2(X)
    return X


def __down_sample(X, configs):
    scope_name = copy_current_name_scope()
    if scope_name:
        scope_name += '/'

    # TODO: Avg, Global....
    max_pool = MaxPooling2D(name=scope_name + 'poll', **configs.get('pool'))

    dropout = None
    drop_configs = configs.get('drop', {'rate': 0.0})
    if drop_configs['rate'] > 0:
        dropout = Dropout(name=scope_name + 'drop', **drop_configs)

    X = max_pool(X)
    if dropout:
        X = dropout(X)
    return X


def __up_sample(X, filters, configs):
    scope_name = copy_current_name_scope()
    if scope_name:
        scope_name += '/'

    conv_T = Conv2DTranspose(filters=filters, name=scope_name + 'cnT', **configs.get('convT'))

    dropout = None
    drop_configs = configs.get('drop', {'rate': 0.0})
    if drop_configs['rate'] > 0:
        dropout = Dropout(name=scope_name + 'drop', **drop_configs)

    X = conv_T(X)
    if dropout:
        X = dropout(X)
    return X


def __output_conv(X, filters, configs):
    scope_name = copy_current_name_scope()
    if scope_name:
        scope_name += '/'
    X = Conv2D(filters=filters, name=scope_name + 'cn', **configs.get('conv'))(X)
    act = activation_from_config(configs, scope_name + 'act')
    X = act(X)
    return X


def UnetModel(
        input_shape: Union[tuple, list, tf.TensorShape],
        unet_levels: int = 5,
        init_filters: int = 64,
        num_classes: int = 1,
        conv_mode: str = 'base',
        name: Union[None, str] = None,
        unet_setup: Union[None, Config] = None):

    input_shape = tf.TensorShape(input_shape)
    input_shape.assert_has_rank(3)
    assert unet_levels >= 2 and init_filters >= 0 and num_classes >= 0

    if unet_setup is None:
        unet_setup = _base_setup(conv_mode=conv_mode)

    filters = tf.cast(init_filters, dtype=tf.int32)
    identity_map = []

    inputs = keras.Input(shape=input_shape, name='input')

    for u in tf.range(unet_levels - 1):
        with tf.name_scope(f'E_U{u + 1}') as unit_name:
            X = __double_conv(inputs if u == 0 else X, filters, unet_setup.get('dbl_conv_encoder'))
            identity_map.append(Identity(name=unit_name + 'X_copy')(X))
            X = __down_sample(X, unet_setup.get('down_sample'))

            filters = filters * 2

    with tf.name_scope('middle_block'):
        X = __double_conv(X, filters, unet_setup.get('dbl_conv_middle'))

    filters = filters // 2

    for u in tf.range(unet_levels - 1, 0, -1):
        with tf.name_scope(f'D_U{u}') as unit_name:
            X = __up_sample(X, filters, unet_setup.get('up_sample'))
            X, X_copy = resize_as(X, identity_map[u - 1], method='crop', interpolation='bilinear', output_as='X1')
            X = concatenate((X, X_copy), name=unit_name + 'concat')
            X = __double_conv(X, filters, unet_setup.get('dbl_conv_decoder'))

            filters = filters // 2

    X = __output_conv(X, num_classes, unet_setup.get('output_block'))

    return keras.Model(inputs, X, name=name)


if __name__ == '__main__':
    # setup = _base_setup()
    model = UnetModel(tf.TensorShape((128, 128, 3)), unet_levels=3, conv_mode='resnet')
    model.summary()
    tf.keras.utils.plot_model(
        model,
        to_file='model.png',
        show_shapes=True,
        show_dtype=False,
        show_layer_names=True,
        rankdir='TB',
        expand_nested=True,
        dpi=120,
        layer_range=None,
        show_layer_activations=True,
        show_trainable=False
    )
