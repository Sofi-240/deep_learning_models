import tensorflow as tf


class Config(dict):
    def __init__(self, name, configs=None, **kwargs):
        configs = {} if configs is None else configs
        super(Config, self).__init__(**configs)
        self.name = name
        self.__dict__.update(**kwargs)

    def update_class_dict(self, **kwargs):
        if 'name' in kwargs.keys():
            raise ValueError(
                '"name" is unchangeable value'
            )
        self.__dict__.update(**kwargs)

    def get_from_class_dict(self, key, default=None):
        return self.__dict__.get(key, default)


def copy_layer(layer, name=None, include_weights=False, **updates):
    config = layer.get_config()
    config['name'] = name or config['name'] + '_2'
    config.update(**updates)
    new_layer = type(layer).from_config(config)

    if layer.built and include_weights:
        weights = layer.get_weights()
        new_layer.build(layer.input_shape)
        new_layer.set_weights(weights)

    return new_layer


def base_configs(layer_name, config_kw=None, **kwargs):
    config_kw = config_kw if config_kw is not None else {}
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
            ),
            **config_kw
        )
    elif layer_name == 'norm':
        con = Config(
            'norm',
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
            **config_kw
        )
    elif layer_name == 'act':
        con = Config(
            'act',
            configs=dict(
                activation='relu'
            ),
            **config_kw
        )
    elif layer_name == 'pool':
        con = Config(
            'pool',
            configs=dict(
                pool_size=(2, 2),
                strides=(2, 2),
                padding='valid',
                data_format='channels_last'
            ),
            **config_kw
        )
    elif layer_name == 'dropout':
        con = Config(
            'dropout',
            configs=dict(
                rate=0.0,
                noise_shape=None,
                seed=None
            ),
            **config_kw
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
            ),
            **config_kw
        )
    elif layer_name == 'dense':
        con = Config(
            'dense',
            configs=dict(
                activation=None,
                use_bias=True,
                kernel_initializer='he_normal',
                bias_initializer='zeros',
                kernel_regularizer=None,
                bias_regularizer=None,
                activity_regularizer=None,
                kernel_constraint=None,
                bias_constraint=None,
            ),
            **config_kw
        )
    else:
        con = Config(layer_name, configs={})
    con.update(**kwargs)
    return con
