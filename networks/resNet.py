import tensorflow as tf
from tensorflow import keras
from typing import Union
from networks.configs import copy_layer, Config, base_config
from keras.layers import Conv2D, MaxPooling2D, Flatten, Add, AveragePooling2D, Dense
from networks.utils import activation_from_config, normalization_from_config, copy_current_name_scope


def _base_setup(N_layers: int, bottleneck: bool = False):
    default = {
        18: [2, 2, 2, 2],
        34: [3, 4, 6, 3],
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 4, 36, 3],
    }
    conv_rep = default.get(N_layers)
    bottleneck = True if N_layers >= 50 else bottleneck
    if conv_rep is None:
        raise ValueError('Number of layers can be 18, 34, 50, 101, 152')
    resnet_setup = Config('resnet_setup')

    conv1 = Config('conv1')
    conv1.update(
        conv=base_config('conv', call_name='conv2d', layer_kw=dict(kernel_size=(7, 7), strides=(2, 2))),
        norm=base_config('norm', call_name='normalization', norm='batch'),
        act=base_config('act', call_name='activation'),
        pool=base_config('pool', call_name='pool', mode='max',
                         layer_kw=dict(pool_size=(3, 3), padding='same', strides=(2, 2)))
    )

    resnet_setup['conv1'] = conv1
    resnet_setup['convN_x'] = Config('convN_x', rep=conv_rep, bottleneck=bottleneck)
    resnet_setup['convN_x'].update(
        conv=base_config('conv', call_name='conv2d', layer_kw=dict(kernel_size=(3, 3))),
        convID=base_config('conv', call_name='conv2d', layer_kw=dict(kernel_size=(1, 1))),
        norm=base_config('norm', call_name='normalization', norm='batch'),
        act=base_config('act', call_name='activation'),
    )

    resnet_setup['output_block'] = Config('output_block')
    resnet_setup['output_block'].update(
        pool=base_config('pool', call_name='pool', mode='max', layer_kw=dict(pool_size=(7, 7), padding='same')),
        dense=base_config('dense',
                          call_name='dense',
                          layer_kw=dict(
                              kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0., stddev=.01),
                              bias_initializer=tf.keras.initializers.zeros(),
                              kernel_regularizer=tf.keras.regularizers.L2(.01)
                          )
                          ),
        act=base_config('act', call_name='activation', layer_kw=dict(activation='softmax')),
    )
    return resnet_setup


def __conv1(X, filters, configs):
    scope_name = copy_current_name_scope()

    conv = Conv2D(filters=filters, name=scope_name + 'conv', **configs.get('conv'))
    norm = normalization_from_config(configs, name=scope_name + 'norm')
    act = activation_from_config(configs, scope_name + 'act')

    X = act(norm(conv(X)))
    X = __pooling(X, configs)
    return X


def __pooling(X, configs):
    scope_name = copy_current_name_scope()

    pooling_config = configs.get('pool')

    mode = pooling_config.get_from_class('pool', 'max')
    assert mode == 'max' or mode == 'avg'

    pool = MaxPooling2D if mode == 'max' else AveragePooling2D
    X = pool(name=scope_name + f'{mode}_poll', **pooling_config)(X)
    return X


def __fc_block(X, units_out, configs):
    scope_name = copy_current_name_scope()
    X = __pooling(X, configs=configs)

    X = Flatten(name=scope_name + 'flt')(X)

    X = Dense(units=units_out, name=scope_name + 'dense', **configs.get('fc', {}))(X)
    X = activation_from_config(configs, name=scope_name + 'act')(X)
    return X


def __block_ex(inputs, filters, configs, down_sample=True):
    bottleneck = configs.get_from_class('bottleneck')
    if bottleneck:
        return bottleneck_block(inputs, filters, configs, down_sample=down_sample)
    return residual_block(inputs, filters, configs, down_sample=down_sample)


def residual_block(inputs, filters, configs, down_sample=True):
    scope_name = copy_current_name_scope()
    shape = inputs.get_shape()
    conv1 = Conv2D(filters=filters, name=scope_name + 'cn1', **configs.get('conv'))
    if down_sample:
        conv1.strides = (2, 2)

    X_copy = None
    if down_sample or shape[-1] != filters:
        id_configs = configs.get('convID')
        if id_configs is None:
            id_configs = configs['conv'].deepcopy(name='convID', kernel_size=(1, 1))
        conv_id = Conv2D(filters=filters, name=scope_name + 'cnID', **id_configs)
        conv_id.strides = conv1.strides
        X_copy = conv_id(inputs)
        X_copy = normalization_from_config(configs, name=scope_name + 'normID')(X_copy)

    norm1 = normalization_from_config(configs, name=scope_name + 'norm1')
    act1 = activation_from_config(configs, scope_name + 'act1')

    conv2 = copy_layer(conv1, name=scope_name + 'cn2', include_weights=False, strides=(1, 1))
    norm2 = copy_layer(norm1, name=scope_name + 'norm2', include_weights=False)
    act2 = copy_layer(act1, name=scope_name + 'act2', include_weights=False)

    X = act1(norm1(conv1(inputs)))
    X = norm2(conv2(X))
    X = Add(name=scope_name + 'add')((X, X_copy if X_copy is not None else inputs))
    X = act2(X)
    return X


def bottleneck_block(inputs, filters, configs, down_sample=True):
    scope_name = copy_current_name_scope()

    conv1 = Conv2D(filters=filters, name=scope_name + 'cn1', **configs.get('conv'))
    conv1.kernel_size = (1, 1)
    if down_sample:
        conv1.strides = (2, 2)

    filters_out = filters * 4

    id_configs = configs.get('convID', {})
    if not id_configs:
        conv_id = copy_layer(conv1, name=scope_name + 'cnID', include_weights=False, filters=filters_out)
    else:
        conv_id = Conv2D(filters=filters_out, name=scope_name + 'cnID', **id_configs)
        conv_id.strides = conv1.strides

    X_copy = conv_id(inputs)
    X_copy = normalization_from_config(configs, name=scope_name + 'normID')(X_copy)

    norm1 = normalization_from_config(configs, name=scope_name + 'norm1')
    act1 = activation_from_config(configs, scope_name + 'act1')

    conv2 = Conv2D(filters=filters, name=scope_name + 'cn2', **configs.get('conv'))
    conv2.strides = (1, 1)
    norm2 = copy_layer(norm1, name=scope_name + 'norm2', include_weights=False)
    act2 = copy_layer(act1, name=scope_name + 'act2', include_weights=False)

    conv3 = copy_layer(conv1, name=scope_name + 'cn3', include_weights=False, strides=(1, 1), filters=filters_out)
    norm3 = copy_layer(norm1, name=scope_name + 'norm3', include_weights=False)
    act3 = copy_layer(act1, name=scope_name + 'act3', include_weights=False)

    X = act1(norm1(conv1(inputs)))
    X = act2(norm2(conv2(X)))
    X = norm3(conv3(X))
    X = Add(name=scope_name + 'add')((X, X_copy if X_copy is not None else inputs))
    X = act3(X)
    return X


def ResNetModel(
        input_shape: Union[tuple, list, tf.TensorShape],
        N_layers: int = 18,
        num_classes: int = 1,
        bottleneck: bool = False,
        name: Union[None, str] = None):
    input_shape = tf.TensorShape(input_shape)
    input_shape.assert_has_rank(3)
    assert num_classes >= 0

    setup = _base_setup(N_layers, bottleneck=bottleneck)
    inputs = keras.Input(shape=input_shape, name='input')

    with tf.name_scope('conv1'):
        X = __conv1(inputs, filters=64, configs=setup.get('conv1'))

    conv_x_base = setup.get('convN_x')
    rep = conv_x_base.get_from_class('rep')
    filters = 64

    for i in tf.range(4):
        prefix_name = f'conv{i + 2}_'
        down_sample = bool(i)

        for r in tf.range(rep[i]):
            with tf.name_scope(prefix_name + f'{r + 1}'):
                X = __block_ex(X, filters=filters, configs=conv_x_base, down_sample=down_sample)

            down_sample = False
        filters = filters * 2

    with tf.name_scope('fc'):
        X = __fc_block(X, units_out=num_classes, configs=setup.get('output_block'))

    return keras.Model(inputs, X, name=name)


if __name__ == '__main__':
    model = ResNetModel(tf.TensorShape((224, 224, 3)), N_layers=18)
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
