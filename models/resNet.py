import tensorflow as tf
from tensorflow import keras
from typing import Callable
from models.configs import copy_layer, Config, base_configs
from keras.layers import Conv2D, MaxPooling2D, Flatten, Add, AveragePooling2D, Dense
from models.utils import activation_from_config, normalization_from_config


class ResNet:
    def __init__(self, N_layers, num_classes=1, bottleneck=False, name=None):
        self.name = name or f'ResNet{N_layers}'
        self.N_layers = N_layers
        self.output_dim = num_classes
        self.__setup_base(bottleneck)

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        input_shape.assert_has_rank(3)

        inputs = keras.Input(shape=input_shape, name='input')

        with tf.name_scope('conv1'):
            X = self.__conv1_block(inputs, filters=64, configs=self.conv1_config)

        with tf.name_scope('pool'):
            X = self.__pooling(X, configs=self.pool_config)

        configs = self.convN_x_config
        rep = configs.get_from_class_dict('rep')
        filters = 64
        conv_x_base = configs.get('base')

        for i in tf.range(4):
            prefix_name = f'conv{i + 2}_'
            down_sample = bool(i)
            conv_x = configs.get(prefix_name + 'x') or conv_x_base

            for r in tf.range(rep[i]):
                with tf.name_scope(prefix_name + f'{r + 1}'):
                    X = self.__block_ex(X, filters=filters, configs=conv_x, down_sample=down_sample)
                down_sample = False
            filters = filters * 2

        with tf.name_scope('fc'):
            X = self.__fc_block(X, units_out=self.output_dim, configs=self.output_block_config)

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

        user_fn = kwargs.get('fn')
        if user_fn is not None:
            if not isinstance(user_fn, Callable):
                raise ValueError('costume function need to be callable')

        con.update_class_dict(**kwargs)

    def __setup_base(self, bottleneck=False):
        default = {
            18: [2, 2, 2, 2],
            34: [3, 4, 6, 3],
            50: [3, 4, 6, 3],
            101: [3, 4, 23, 3],
            152: [3, 4, 36, 3],
        }
        conv_rep = default.get(self.N_layers)
        bottleneck = True if self.N_layers >= 50 else bottleneck
        if conv_rep is None:
            raise ValueError('N can be 18, 34, 50, 101, 152')

        self.conv1_config = Config('conv1', block='input', fn=None)
        self.conv1_config.update(
            conv=base_configs('conv', kernel_size=(7, 7), strides=(2, 2)),
            norm=base_configs('norm', config_kw={'norm': 'batch'}),
            act=base_configs('act')
        )
        self.pool_config = Config('pool', block='input', fn=None)
        self.pool_config.update(
            pool=base_configs('pool', config_kw={'pool': 'max'}, pool_size=(3, 3), padding='same', strides=(2, 2))
        )

        self.convN_x_config = Config('convN_x', block='middle', fn=None, rep=conv_rep)

        convN_x_base = Config('convN_x_base', block='middle', fn=None, bottleneck=bottleneck)

        convN_x_base.update(
            conv=base_configs('conv', kernel_size=(3, 3), strides=(1, 1)),
            convID=base_configs('convID', kernel_size=(1, 1), strides=(1, 1)),
            norm=base_configs('norm', config_kw={'norm': 'batch'}),
            act=base_configs('act')
        )
        self.convN_x_config.update(
            base=convN_x_base, conv2_x=None, conv3_x=None, conv4_x=None, conv5_x=None
        )

        self.output_block_config = Config('output_block', block='output', fn=None)
        self.output_block_config.update(
            pool=base_configs('pool', config_kw={'mode': 'avg'}, pool_size=(7, 7)),
            fc=base_configs(
                'dense',
                kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0., stddev=.01),
                bias_initializer=tf.keras.initializers.zeros(),
                kernel_regularizer=tf.keras.regularizers.L2(.01)
            ),
            act=base_configs('act', activation='softmax')
        )
        self.config_names = [
            'conv1_config',
            'pool_config',
            'convN_x_config',
            'output_block_config'
        ]

    def __conv1_block(self, X, filters, configs):
        scope_name = tf.get_current_name_scope()
        if scope_name:
            scope_name += '/'
        conv = Conv2D(filters=filters, name=scope_name + 'conv', **configs.get('conv'))
        norm = normalization_from_config(configs, name=scope_name + 'norm')
        act = activation_from_config(configs, scope_name + 'act')

        X = act(norm(conv(X)))
        return X

    def __pooling(self, X, configs):
        scope_name = tf.get_current_name_scope()
        if scope_name:
            scope_name += '/'
        pooling_config = configs.get('pool')
        if pooling_config is None:
            return X

        mode = pooling_config.get_from_class_dict('pool', 'max')
        assert mode == 'max' or mode == 'avg'

        pool = MaxPooling2D if mode == 'max' else AveragePooling2D
        X = pool(name=scope_name + 'poll', **pooling_config)(X)
        return X

    def __block_ex(self, inputs, filters, configs, down_sample=True):
        bottleneck = configs.get_from_class_dict('bottleneck')
        if bottleneck:
            return self.__bottleneck_block(inputs, filters, configs, down_sample=down_sample)
        return self.__residual_block(inputs, filters, configs, down_sample=down_sample)

    def __residual_block(self, inputs, filters, configs, down_sample=True):
        scope_name = tf.get_current_name_scope()
        if scope_name:
            scope_name += '/'

        conv1 = Conv2D(filters=filters, name=scope_name + 'cn1', **configs.get('conv'))
        if down_sample:
            conv1.strides = (2, 2)

        X_copy = None
        if down_sample:
            conv_id = Conv2D(filters=filters, name=scope_name + 'cnID', **configs.get('convID'))
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

    def __bottleneck_block(self, inputs, filters, configs, down_sample=True):
        scope_name = tf.get_current_name_scope()
        if scope_name:
            scope_name += '/'

        conv1 = Conv2D(filters=filters, name=scope_name + 'cn1', **configs.get('conv'))
        conv1.kernel_size = (1, 1)
        if down_sample:
            conv1.strides = (2, 2)

        filters_out = filters * 4

        conv_id = Conv2D(filters=filters_out, name=scope_name + 'cnID', **configs.get('convID'))
        conv_id.strides = conv1.strides
        X_copy = conv_id(inputs)
        X_copy = normalization_from_config(configs, name=scope_name + 'normID')(X_copy)

        norm1 = normalization_from_config(configs, name=scope_name + 'norm1')
        act1 = activation_from_config(configs, scope_name + 'act1')

        conv2 = copy_layer(conv1, name=scope_name + 'cn2', include_weights=False, strides=(1, 1))
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

    def __fc_block(self, X, units_out, configs):
        scope_name = tf.get_current_name_scope()
        if scope_name:
            scope_name += '/'
        X = self.__pooling(X, configs=configs)
        X = Flatten(name=scope_name + 'flt')(X)
        X = Dense(units=units_out, name=scope_name + 'dense', **configs.get('fc', {}))(X)
        X = activation_from_config(configs, name=scope_name + 'act')(X)
        return X


if __name__ == '__main__':
    model_setup = ResNet(18)
    model = model_setup.build((224, 224, 3))
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
