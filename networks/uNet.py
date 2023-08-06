import tensorflow as tf
from tensorflow import keras
from typing import Callable, Union
from networks.configs import BlockConfig, copy_layer
from networks.utils import activation_from_config, normalization_from_config, resize_as, copy_current_name_scope
from keras.layers import Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, Add, concatenate, Identity


def _base_setup(mode: str = 'base'):
    assert mode == 'base' or mode == 'resnet'
    dbl_conv_encoder = BlockConfig('dbl_conv', block='encoder', mode=mode, fn=None)
    dbl_conv_encoder.add_layer_config('conv', from_base=True, call_name='conv2d')
    dbl_conv_encoder.add_layer_config('norm', from_base=True, call_name='normalization', norm='batch')
    dbl_conv_encoder.add_layer_config('act', from_base=True, call_name='activation')

    down_sample_encoder = BlockConfig('down_sample', block='encoder', fn=None)
    down_sample_encoder.add_layer_config('pool', from_base=True, call_name='pool', mode='max')
    down_sample_encoder.add_layer_config('dropout', from_base=True, call_name='dropout')

    dbl_conv_middle = BlockConfig('dbl_conv', block='middle', mode=mode, fn=None)
    dbl_conv_middle.add_layer_config('conv', from_base=True, call_name='conv2d')
    dbl_conv_middle.add_layer_config('norm', from_base=True, call_name='normalization', norm='batch')
    dbl_conv_middle.add_layer_config('act', from_base=True, call_name='activation')

    up_sample_decoder = BlockConfig('up_sample', block='decoder', fn=None)
    up_sample_decoder.add_layer_config('convT', from_base=True, call_name='conv2dT')
    up_sample_decoder.add_layer_config('dropout', from_base=True, call_name='dropout')

    dbl_conv_decoder = BlockConfig('dbl_conv', block='decoder', mode=mode, fn=None)
    dbl_conv_decoder.add_layer_config('conv', from_base=True, call_name='conv2d')
    dbl_conv_decoder.add_layer_config('norm', from_base=True, call_name='normalization', norm='batch')
    dbl_conv_decoder.add_layer_config('act', from_base=True, call_name='activation')

    output_block = BlockConfig('output_conv', block='output', fn=None)
    output_block.add_layer_config('conv', from_base=True, call_name='conv2d', layer_kw=dict(kernel_size=(1, 1)))
    output_block.add_layer_config('act', from_base=True, call_name='activation', layer_kw=dict(activation='softmax'))

    unet_config = dict(
        dbl_conv_encoder=dbl_conv_encoder,
        down_sample_encoder=down_sample_encoder,
        dbl_conv_middle=dbl_conv_middle,
        up_sample_decoder=up_sample_decoder,
        dbl_conv_decoder=dbl_conv_decoder,
        output_block=output_block
    )
    return unet_config


def __double_conv(inputs, filters, configs):
    user_fn = configs.get_from_class('fn')
    if user_fn is not None:
        return user_fn(inputs, filters, configs)

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
                id_configs = configs.get_layer_config('conv', kernel_size=(1, 1))
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
    user_fn = configs.get_from_class('fn')
    if user_fn is not None:
        return user_fn(X, configs)

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
    user_fn = configs.get_from_class('fn')
    if user_fn is not None:
        return user_fn(X, configs)

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
    user_fn = configs.get_from_class('fn')
    if user_fn is not None:
        return user_fn(X, configs)
    scope_name = copy_current_name_scope()
    if scope_name:
        scope_name += '/'
    X = Conv2D(filters=filters, name=scope_name + 'cn', **configs.get('conv'))(X)
    act = activation_from_config(configs, scope_name + 'act')
    X = act(X)
    return X


class UnetSetup:
    def __init__(self,
                 unet_levels: int = 5,
                 init_filters: int = 64,
                 num_classes: int = 1,
                 conv_mode: str = 'base',
                 name: Union[None, str] = None):
        assert unet_levels >= 2 and init_filters >= 0 and num_classes >= 0
        self.name = name or 'UnetSetup'
        self.unet_levels = unet_levels
        self.init_filters = init_filters
        self.output_dim = num_classes
        self.config = _base_setup(mode=conv_mode)

    def change_setup(self,
                     config_name: Union[None, str] = None,
                     sub_model: Union[None, str] = None,
                     **kwargs):
        if config_name is not None and config_name not in self.config.keys():
            raise ValueError(
                f'Unknown sub model name {config_name}'
            )
        con = self.config.get(config_name)

        if sub_model is not None and con is not None:
            assert isinstance(sub_model, str)
            if sub_model not in con.keys():
                raise KeyError(sub_model)
            con.update_layer_config(sub_model, **kwargs)

        change_mode = kwargs.get('mode')
        if change_mode is not None:
            if change_mode != 'resnet' and change_mode != 'base':
                raise ValueError(f'mode for double convolution except base or resnet')

            for item in self.config.values():
                if item.name == 'dbl_conv':
                    item.update_class(**dict(mode=change_mode))

        user_fn = kwargs.get('fn')
        if user_fn is not None and con is not None:
            if not isinstance(user_fn, Callable):
                raise ValueError('costume function need to be callable')
            con.update_class(**kwargs)

    def build_model(self, input_shape):
        return UnetModel(
            input_shape,
            unet_setup=self
        )


def UnetModel(
        input_shape: Union[tuple, list, tf.TensorShape],
        unet_levels: int = 5,
        init_filters: int = 64,
        num_classes: int = 1,
        conv_mode: str = 'base',
        name: Union[None, str] = None,
        unet_setup: Union[None, UnetSetup] = None):
    setup = None
    if unet_setup is not None:
        unet_levels = unet_setup.unet_levels
        init_filters = unet_setup.init_filters
        num_classes = unet_setup.output_dim
        name = unet_setup.name
        setup = unet_setup.config

    input_shape = tf.TensorShape(input_shape)
    input_shape.assert_has_rank(3)
    assert unet_levels >= 2 and init_filters >= 0 and num_classes >= 0

    if setup is None:
        setup = _base_setup(mode=conv_mode)

    filters = tf.cast(init_filters, dtype=tf.int32)
    identity_map = []

    inputs = keras.Input(shape=input_shape, name='input')

    for u in tf.range(unet_levels - 1):
        with tf.name_scope(f'E_U{u + 1}') as unit_name:
            X = __double_conv(inputs if u == 0 else X, filters, setup.get('dbl_conv_encoder'))
            identity_map.append(Identity(name=unit_name + 'X_copy')(X))
            X = __down_sample(X, setup.get('down_sample_encoder'))

            filters = filters * 2

    with tf.name_scope('middle_block'):
        X = __double_conv(X, filters, setup.get('dbl_conv_middle'))

    filters = filters // 2

    for u in tf.range(unet_levels - 1, 0, -1):
        with tf.name_scope(f'D_U{u}') as unit_name:
            X = __up_sample(X, filters, setup.get('up_sample_decoder'))
            X, X_copy = resize_as(X, identity_map[u - 1], method='crop', interpolation='bilinear', output_as='X1')
            X = concatenate((X, X_copy), name=unit_name + 'concat')
            X = __double_conv(X, filters, setup.get('dbl_conv_decoder'))

            filters = filters // 2

    X = __output_conv(X, num_classes, setup.get('output_block'))

    return keras.Model(inputs, X, name=name)


if __name__ == '__main__':
    model_setup = UnetSetup(unet_levels=3)
    # model_setup.change_setup(mode='resnet')
    model = model_setup.build_model(tf.TensorShape((128, 128, 3)))
    # model = UnetModel(tf.TensorShape((128, 128, 3)), unet_levels=3)
    model.summary()
    # tf.keras.utils.plot_model(
    #     model,
    #     to_file='model.png',
    #     show_shapes=True,
    #     show_dtype=False,
    #     show_layer_names=True,
    #     rankdir='TB',
    #     expand_nested=True,
    #     dpi=120,
    #     layer_range=None,
    #     show_layer_activations=True,
    #     show_trainable=False
    # )
