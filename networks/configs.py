import inspect
from typing import Union
import copy as COPY
import tensorflow as tf


class Config(dict):
    def __init__(self,
                 name: str,
                 **kwargs):
        super(Config, self).__init__()
        self.name = name
        self.__dict__.update(**kwargs)

    def __setattr__(self, key, value):
        if inspect.stack()[1][3] == '__init__':
            return super().__setattr__(key, value)
        if key == 'name':
            raise AttributeError('name is unchangeable')
        if key == '__name':
            key = 'name'
        return super().__setattr__(key, value)

    def update_class(self, **kwargs):
        for key, item in kwargs.items():
            self.__setattr__(key, item)

    def get_from_class(self, key: str, default=None):
        return self.__dict__.get(key, default)

    def deepcopy(self, name, **updates):
        self_copy = COPY.deepcopy(self)
        self_copy.__name = name
        self_copy.update(**updates)
        return self_copy


class BaseConfiguration:

    @staticmethod
    def conv2d(con: Union[type(Config), dict], **layer_kw):
        configs = dict(
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            kernel_initializer='he_normal',
            bias_initializer='zeros',
            data_format='channels_last',
            activation=None,
            trainable=True
        )
        configs.update(**layer_kw)
        con.update(**configs)

    @staticmethod
    def pool(con: Union[type(Config), dict], **layer_kw):
        configs = dict(
            pool_size=(2, 2),
            strides=(2, 2),
            padding='valid',
            data_format='channels_last'
        )
        configs.update(**layer_kw)
        con.update(**configs)

    @staticmethod
    def normalization(con: Union[type(Config), dict], **layer_kw):
        configs = dict(
            momentum=0.99,
            epsilon=0.001,
            center=True,
            scale=True,
            beta_initializer='zeros',
            gamma_initializer='ones',
            moving_mean_initializer='zeros',
            moving_variance_initializer='ones'
        )
        configs.update(**layer_kw)
        con.update(**configs)

    @staticmethod
    def activation(con: Union[type(Config), dict], **layer_kw):
        configs = dict(
            activation='relu'
        )
        configs.update(**layer_kw)
        con.update(**configs)

    @staticmethod
    def dropout(con: Union[type(Config), dict], **layer_kw):
        configs = dict(
            rate=0.0,
            noise_shape=None,
            seed=None
        )
        configs.update(**layer_kw)
        con.update(**configs)

    @staticmethod
    def conv2dT(con: Union[type(Config), dict], **layer_kw):
        configs = dict(
            kernel_size=(2, 2),
            strides=(2, 2),
            dilation_rate=(1, 1),
            padding='valid',
            kernel_initializer='he_normal',
            bias_initializer='zeros',
            data_format='channels_last',
            trainable=True
        )
        configs.update(**layer_kw)
        con.update(**configs)

    @staticmethod
    def dense(con: Union[type(Config), dict], **layer_kw):
        configs = dict(
            activation=None,
            use_bias=True,
            kernel_initializer='he_normal',
            bias_initializer='zeros',
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
        )
        configs.update(**layer_kw)
        con.update(**configs)


BASE_CON_NAMES = inspect.getmembers(BaseConfiguration, predicate=inspect.isfunction)
BASE_CON_NAMES = [i[0] for i in BASE_CON_NAMES]


def base_config(name, call_name, layer_kw=None, **class_kw):
    con = Config(name, **class_kw)
    if call_name not in BASE_CON_NAMES:
        raise KeyError(f'{call_name} not in Base Configuration methods valid keys: {BASE_CON_NAMES}')
    layer_kw = layer_kw if layer_kw is not None else {}
    eval(f'BaseConfiguration.{call_name}')(con, **layer_kw)
    return con


def copy_layer(layer,
               name: Union[None, str] = None,
               include_weights: Union[None, str] = False,
               **updates):
    assert isinstance(layer, tf.keras.layers.Layer)
    config = layer.get_config()
    updates['name'] = name or layer.name + '_2'
    config.update(**updates)
    new_layer = type(layer).from_config(config)

    if layer.built and include_weights:
        weights = layer.get_weights()
        new_layer.build(layer.input_shape)
        new_layer.set_weights(weights)

    return new_layer


