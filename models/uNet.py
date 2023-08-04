import tensorflow as tf
from tensorflow import keras
from typing import Callable
from models.utils import copy_layer, Config
from keras.layers import Conv2D, Activation, BatchNormalization, MaxPooling2D, Dropout, Conv2DTranspose, Identity, \
    concatenate, Resizing, Cropping2D


def base_configs(layer_name, **kwargs):
    if layer_name == 'conv':
        con = Config(
            'conv',
            configs=dict(
                kernel_size=(3, 3),
                strides=(1, 1),
                dilation_rate=(1, 1),
                padding='same',
                kernel_initializer='he_normal',
                bias_initializer='zeros',
                data_format='channels_last',
                activation=None,
                trainable=True
            )
        )
    elif layer_name == 'bn':
        con = Config(
            'bn',
            configs=dict(
                momentum=0.99,
                epsilon=0.001,
                center=True,
                scale=True,
                beta_initializer='zeros',
                gamma_initializer='ones',
                moving_mean_initializer='zeros',
                moving_variance_initializer='ones'
            ),
            norm='batch'
        )
    elif layer_name == 'act':
        con = Config(
            'act',
            configs=dict(
                activation='relu'
            )
        )
    elif layer_name == 'max_pooling':
        con = Config(
            'pool',
            configs=dict(
                pool_size=(2, 2),
                strides=(2, 2),
                padding='valid',
                data_format='channels_last'
            ),
            mode='max'
        )
    elif layer_name == 'dropout':
        con = Config(
            'dropout',
            configs=dict(
                rate=0.0,
                noise_shape=None,
                seed=None
            )
        )
    elif layer_name == 'convT':
        con = Config(
            'convT',
            configs=dict(
                kernel_size=(2, 2),
                strides=(2, 2),
                dilation_rate=(1, 1),
                padding='valid',
                kernel_initializer='he_normal',
                bias_initializer='zeros',
                data_format='channels_last',
                trainable=True
            )
        )
    else:
        con = Config(layer_name, configs={})
    con.update(**kwargs)
    return con


class UNET:
    def __init__(self, unet_levels=5, init_filters=64, num_classes=1, name=None):
        self.name = name
        self.unet_levels = unet_levels
        self.init_filters = init_filters
        self.output_dim = num_classes
        self.__setup_base()
        # TODO: concat function to avoid shape error

    def __setup_base(self):
        self.dbl_conv_encode = Config('dbl_conv', block='encoder', fn=None, mode='base')
        self.dbl_conv_encode.update(
            conv=base_configs('conv'),
            bn=base_configs('bn'),
            act=base_configs('act'),
            conv_id=None
        )

        self.down_sample = Config('down_sample', block='encoder', fn=None)
        self.down_sample.update(
            pool=base_configs('max_pooling'),
            drop=base_configs('dropout')
        )

        self.dbl_conv_middle = Config('dbl_conv', block='middle', fn=None, mode='base')
        self.dbl_conv_middle.update(
            conv=base_configs('conv'),
            bn=base_configs('bn'),
            act=base_configs('act')
        )

        self.up_sample = Config('up_sample', block='decoder', fn=None)
        self.up_sample.update(
            convT=base_configs('convT'),
            drop=base_configs('dropout')
        )

        self.dbl_conv_decoder = Config('dbl_conv', block='decoder', fn=None, mode='base')
        self.dbl_conv_decoder.update(
            conv=base_configs('conv'),
            bn=base_configs('bn'),
            act=base_configs('act')
        )

        self.output_conv = Config('output_conv', block='output', fn=None)
        self.output_conv.update(
            conv=base_configs('conv', kernel_size=(1, 1)),
            act=base_configs('act', activation='softmax')
        )
        self.config_names = [
            'dbl_conv_encode',
            'down_sample',
            'dbl_conv_middle',
            'up_sample',
            'dbl_conv_decoder',
            'output_conv'
        ]

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        input_shape.assert_has_rank(3)

        filters = tf.cast(self.init_filters, dtype=tf.int32)
        identity_map = []

        inputs = keras.Input(shape=input_shape, name='input')

        for u in tf.range(self.unet_levels - 1):
            with tf.name_scope(f'E_U{u + 1}') as unit_name:
                X = self.__double_conv(inputs if u == 0 else X, filters, self.dbl_conv_encode)

                identity_map.append(Identity(name=unit_name + 'X_copy')(X))

                X = self.__down_sample(X, self.down_sample)
                filters = filters * 2

        with tf.name_scope('middle_block'):
            X = self.__double_conv(X, filters, self.dbl_conv_middle)

        filters = filters // 2

        for u in tf.range(self.unet_levels - 1, 0, -1):
            with tf.name_scope(f'D_U{u}') as unit_name:
                X = self.__up_sample(X, filters, self.up_sample)
                X = concatenate((X, identity_map[u - 1]), axis=-1, name=unit_name + 'concat')
                X = self.__double_conv(X, filters, self.dbl_conv_decoder)

                filters = filters // 2

        configs = self.output_conv
        with tf.name_scope('output') as scope_name:
            # TODO: costume function like for middle
            X = Conv2D(filters=self.output_dim, name=scope_name + 'cn', **configs.get('conv', {}))(X)
            act_config = configs.get('act')
            if act_config is None:
                raise ValueError('activation configs is missing')
            act_config = act_config.get('activation')
            if issubclass(type(act_config), tf.keras.layers.Layer):
                act = copy_layer(act_config, name=scope_name + 'act', include_weights=False)
            else:
                act = Activation(name=scope_name + 'act', **configs.get('act', {}))
            X = act(X)
        return keras.Model(inputs, X, name=self.name)

    def change_setup(self, config_name, sub_model=None, **kwargs):
        if config_name not in self.config_names:
            raise ValueError(
                f'Unknown sub model name {config_name}'
            )
        con = self.__getattribute__(config_name)

        if sub_model is not None:
            assert isinstance(sub_model, str)
            sub_con = con.get(sub_model)
            if sub_con is None:
                con.update(sub_model=kwargs)
            else:
                sub_con.update(**kwargs)
            return

        change_mode = kwargs.get('mode')
        if con.name == 'dbl_conv' and change_mode is not None:
            if change_mode != 'resnet' and change_mode != 'base':
                raise ValueError(
                    f'mode for double convolution except base or resnet'
                )
            if change_mode == 'resnet':
                con.update(convID=base_configs('conv', kernel_size=(1, 1)))
        user_fn = kwargs.get('fn')
        if user_fn is not None:
            if not isinstance(user_fn, Callable):
                raise ValueError('costume function need to be callable')

        con.update_class_dict(**kwargs)

    @staticmethod
    def __double_conv(X, filters, configs):
        user_fn = configs.get_from_class_dict('fn')
        if user_fn is not None:
            return user_fn(X, filters, configs)
        with tf.name_scope('double_conv') as scope_name:
            conv1 = Conv2D(filters=filters, name=scope_name + 'cn1', **configs.get('conv'))
            conv2 = Conv2D(filters=filters, name=scope_name + 'cn2', **configs.get('conv'))
            if configs.mode == 'resnet':
                id_config = configs.get('convID', {})
                if not id_config:
                    id_config.update(**configs.get('conv'))
                    id_config['kernel_size'] = (1, 1)
                conv3 = Conv2D(filters=filters, name=scope_name + 'cnId', **id_config)
                X_copy = Identity(name=scope_name + 'Xcopy')(X)
                X_copy = conv3(X_copy)

            norm1 = BatchNormalization(name=scope_name + 'bn1', **configs.get('bn', {}))
            norm2 = BatchNormalization(name=scope_name + 'bn2', **configs.get('bn', {}))

            act_config = configs.get('act')
            if act_config is None:
                raise ValueError('activation configs is missing')
            act_config = act_config.get('activation')
            if issubclass(type(act_config), tf.keras.layers.Layer):
                act1 = copy_layer(act_config, name=scope_name + 'act1', include_weights=False)
                act2 = copy_layer(act_config, name=scope_name + 'act2', include_weights=False)
            else:
                act1 = Activation(name=scope_name + 'act1', **configs.get('act', {}))
                act2 = Activation(name=scope_name + 'act2', **configs.get('act', {}))

            X = act1(norm1(conv1(X)))
            X = norm2(conv2(X))
            if configs.mode == 'resnet':
                X = concatenate((X, X_copy), axis=-1, name=scope_name + 'concat')
            X = act2(X)
            return X

    @staticmethod
    def __down_sample(X, configs):
        user_fn = configs.get_from_class_dict('fn')
        if user_fn is not None:
            return user_fn(X, configs)
        with tf.name_scope('down_sample') as scope_name:
            max_pool = MaxPooling2D(name=scope_name + 'poll', **configs.get('pool', {}))

            dropout = None
            drop_configs = configs.get('drop', {'rate': 0.0})
            if drop_configs['rate'] > 0:
                dropout = Dropout(name=scope_name + 'drop', **drop_configs)

            X = max_pool(X)
            if dropout:
                X = dropout(X)
            return X

    @staticmethod
    def __up_sample(X, filters, configs):
        user_fn = configs.get_from_class_dict('fn')
        if user_fn is not None:
            return user_fn(X, configs)
        with tf.name_scope('up_sample') as scope_name:
            conv_T = Conv2DTranspose(filters=filters, name=scope_name + 'cnT', **configs.get('convT', {}))

            dropout = None
            drop_configs = configs.get('drop', {'rate': 0.0})
            if drop_configs['rate'] > 0:
                dropout = Dropout(name=scope_name + 'drop', **drop_configs)

            X = conv_T(X)
            if dropout:
                X = dropout(X)
            return X


if __name__ == '__main__':
    m = UNET()
    model = m.build((128, 128, 3))
    # m.change_setup('dbl_conv_decoder', mode='resnet')
