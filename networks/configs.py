class Config(dict):
    def __init__(self, name, **kwargs):
        super(Config, self).__init__()
        self.name = name
        self.__dict__.update(**kwargs)

    def update_class(self, **kwargs):
        if 'name' in kwargs.keys():
            raise ValueError(
                '"name" is unchangeable value'
            )
        self.__dict__.update(**kwargs)

    def get_from_class(self, key, default=None):
        return self.__dict__.get(key, default)


class BaseConfiguration:

    @staticmethod
    def conv2d(con, **layer_kw):
        configs = dict(
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
        configs.update(**layer_kw)
        con.update(**configs)

    @staticmethod
    def pool(con, **layer_kw):
        configs = dict(
            pool_size=(2, 2),
            strides=(2, 2),
            padding='valid',
            data_format='channels_last'
        )
        configs.update(**layer_kw)
        con.update(**configs)

    @staticmethod
    def normalization(con, **layer_kw):
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
    def activation(con, **layer_kw):
        configs = dict(
            activation='relu'
        )
        configs.update(**layer_kw)
        con.update(**configs)

    @staticmethod
    def dropout(con, **layer_kw):
        configs = dict(
            rate=0.0,
            noise_shape=None,
            seed=None
        )
        configs.update(**layer_kw)
        con.update(**configs)

    @staticmethod
    def conv2dT(con, **layer_kw):
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
    def dense(con, **layer_kw):
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


class BlockConfig(Config):
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        self.layers = []

    def add_layer_config(self, name, from_base=False, call_name=None, layer_kw=None, **class_kw):
        con = Config(name, **class_kw)
        self.layers.append(name)
        self.update({name: con})
        if from_base and call_name is not None:
            layer_kw = layer_kw if layer_kw is not None else {}
            eval(f'BaseConfiguration.{call_name}')(con, **layer_kw)

    def update_layer_config(self, name, **layer_kw):
        con = self.get(name)
        if con is None:
            return
        con.update(**layer_kw)

    def update_layer_class(self, name, **class_kw):
        con = self.get(name)
        if con is None:
            return
        con.update_class(**class_kw)

    def del_layer(self, name):
        del_l = None
        for l in range(len(self.layers)):
            if self.layers[l] == name:
                del_l = l
                break
        if del_l is None:
            raise KeyError(name)
        del self.layers[del_l]
        self.pop(name)

