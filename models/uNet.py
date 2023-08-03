import tensorflow as tf
from models.utils import Config_node, compute_output_shape, copy_layer
from keras.layers import Conv2D, Activation, BatchNormalization, MaxPooling2D, Dropout, Conv2DTranspose, Identity, \
    concatenate


def _default_setup():
    model_config = Config_node('Unet')
    # TODO: resnet identity convolution layer
    double_conv_config = Config_node(
        'double_conv', mode='base', activation='relu'
    )
    double_conv_config.update(
        'conv',
        dict(
            kernel_size=(3, 3),
            strides=(1, 1),
            dilation_rate=(1, 1),
            padding='same',
            kernel_initializer='he_normal',
            bias_initializer='zeros',
            data_format='channels_last',
            trainable=True
        )
    )

    down_sample_config = Config_node(
        'down_sample', mode='max', dropout_rate=0.0
    )
    down_sample_config.update(
        'max_pooling',
        dict(
            pool_size=(2, 2),
            strides=(2, 2),
            padding='valid',
            data_format='channels_last',
        )
    )

    middle_block_config = Config_node(
        'middle_block', mode='base', activation='relu', path='double_conv'
    )
    middle_block_config.update(
        'conv', double_conv_config['conv'].copy()
    )

    up_sample_config = Config_node(
        'up_sample', mode='conv_trans', dropout_rate=0.0, activation='relu'
    )
    up_sample_config.update(
        'conv',
        dict(
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

    output_conv_config = Config_node(
        'output_conv', activation='softmax'
    )
    conv_base = double_conv_config['conv'].copy()
    conv_base.update(kernel_size=(1, 1), padding='valid')
    output_conv_config.update(
        'conv', conv_base
    )
    for con in [double_conv_config, down_sample_config, middle_block_config, up_sample_config, output_conv_config]:
        model_config.update(con.name, con)
    return model_config.export()


class UNET:
    def __init__(self, unet_levels=4, init_filters=64, num_classes=1, name=None):
        self.name = name or 'UNET_model'
        self.unet_levels = unet_levels
        self.init_filters = init_filters
        self.output_dim = num_classes
        self.configs = _default_setup()
        # TODO: concatenate layers need to go throw function for validate the shapes
        # TODO: validate the unet_levels with the input shape
        # TODO: see maby all ths class need to inherit from keras.Model

    def __double_conv(self, X, filters, middle=False):
        with tf.name_scope('double_conv') as scope_name:
            config = self.configs.double_conv if not middle else self.configs.middle_block

            conv1 = Conv2D(filters=filters, name=scope_name + 'cn1', **config.get('conv'))
            conv2 = Conv2D(filters=filters, name=scope_name + 'cn2', **config.get('conv'))
            if config.mode == 'resnet':
                id_config = config.get('id_conv', {})
                if not id_config:
                    id_config.update(**config.get('conv'))
                    id_config['kernel_size'] = (1, 1)
                conv3 = Conv2D(filters=filters, name=scope_name + 'cnId', **id_config)
                X_copy = Identity(name=scope_name + 'Xcopy')(X)
                X_copy = conv3(X_copy)

            norm1 = BatchNormalization(name=scope_name + 'bn1', **config.get('bn', {}))
            norm2 = BatchNormalization(name=scope_name + 'bn2', **config.get('bn', {}))

            if issubclass(type(config.activation), tf.keras.layers.Layer):
                act1 = copy_layer(config.activation, name=scope_name + 'act1', include_weights=False)
                act2 = copy_layer(config.activation, name=scope_name + 'act2', include_weights=False)
            else:
                act1 = Activation(activation=config.activation, name=scope_name + 'act1', **config.get('act', {}))
                act2 = Activation(activation=config.activation, name=scope_name + 'act2', **config.get('act', {}))

            X = act1(norm1(conv1(X)))
            X = norm2(conv2(X))
            if config.mode == 'resnet':
                X = concatenate((X, X_copy), axis=-1, name=scope_name + 'concat')
            X = act2(X)
            return X

    def __down_sample(self, X):
        with tf.name_scope('down_sample') as scope_name:
            config = self.configs.down_sample
            max_pool = MaxPooling2D(name=scope_name + 'poll', **config.get('max_pooling', {}))

            dropout = None
            if config.dropout_rate > 0:
                dropout = Dropout(rate=config.dropout_rate, name=scope_name + 'drop')

            X = max_pool(X)
            if dropout:
                X = dropout(X)
            return X

    def __up_sample(self, X, filters):
        with tf.name_scope('up_sample') as scope_name:
            config = self.configs.up_sample
            conv_T = Conv2DTranspose(filters=filters, name=scope_name + 'cnT', **config.get('conv', {}))

            dropout = None
            if config.dropout_rate > 0:
                dropout = Dropout(rate=config.dropout_rate, name=scope_name + 'drop')

            X = conv_T(X)
            if dropout:
                X = dropout(X)
            return X

    def __middle_block(self, X, filters):
        config = self.configs.middle_block
        if config.path == 'double_conv':
            return self.__double_conv(X, filters, middle=True)
        # TODO: where set up function?
        return self.__double_conv(X, filters, middle=True)

    def setup(self, config, sub_model=None, **kwargs):
        con = None
        if isinstance(config, int):
            con = self.configs[config]
        if isinstance(config, str):
            con = self.configs.__getattribute__(config)
        if con is None:
            raise ValueError(f'Unknown sub model name {config}')

        if sub_model is not None:
            assert isinstance(sub_model, str)
            sub_con = con.get(sub_model)
            if sub_con is None:
                con.update(sub_model, kwargs)
            else:
                sub_con.update(**kwargs)
        else:
            con.__dict__.update(**kwargs)

    def build(self, input_shape):
        filters = tf.cast(self.init_filters, dtype=tf.int32)
        identity_map = []

        inputs = tf.keras.Input(shape=input_shape, dtype=tf.float32, name='input')

        for u in tf.range(self.unet_levels):
            with tf.name_scope(f'E_U{u + 1}') as unit_name:
                X = self.__double_conv(inputs if u == 0 else X, filters)

                identity_map.append(Identity(name=unit_name + 'X_copy')(X))

                X = self.__down_sample(X)
                filters = filters * 2

        with tf.name_scope('middle_block'):
            X = self.__middle_block(X, filters)

        filters = filters // 2

        for u in tf.range(self.unet_levels, 0, -1):
            with tf.name_scope(f'D_U{u}') as unit_name:
                X = self.__up_sample(X, filters)
                X = concatenate((identity_map[u - 1], X), axis=-1, name=unit_name + 'concat')
                X = self.__double_conv(X, filters)

                filters = filters // 2

        config = self.configs.output_conv
        with tf.name_scope('output') as scope_name:
            # TODO: costume function like for middle
            X = Conv2D(filters=self.output_dim, name=scope_name + 'cn', **config.get('conv', {}))(X)
            if issubclass(type(config.activation), tf.keras.layers.Layer):
                act = copy_layer(config.activation, name=scope_name + 'act', include_weights=False)
            else:
                act = Activation(activation=config.activation, name=scope_name + 'act')
            X = act(X)

        return tf.keras.Model(inputs, X, name=self.name)


setup_model = UNET()

setup_model.setup(0, mode='resnet')
# model = setup_model.build((128, 128, 3))

