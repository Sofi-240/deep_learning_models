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

    def get_from_class_dict(self, key):
        return self.__dict__.get(key)


def compute_output_shape(
        input_shape,
        kernel_size,
        strides,
        paddings,
        dilation=None,
        filters=None,
        transform=False
):
    assert isinstance(input_shape, tf.TensorShape)
    assert input_shape.ndims == 3

    Hin, Win, Din = tf.unstack(tf.cast(input_shape, dtype=tf.float32))
    paddings = (0, 0) if paddings == 'valid' else (kernel_size[0] // 2, kernel_size[1] // 2)
    dilation = dilation or (1, 1)

    if transform:
        H_out = ((Hin - 1) * strides[0]) - (2 * paddings[0]) + (dilation[0] * (kernel_size[0] - 1)) + 1
        W_out = ((Win - 1) * strides[1]) - (2 * paddings[1]) + (dilation[1] * (kernel_size[1] - 1)) + 1
    else:
        H_out = ((Hin + (2 * paddings[0]) - (dilation[0] * (kernel_size[0] - 1)) - 1) / strides[0]) + 1
        W_out = ((Win + (2 * paddings[1]) - (dilation[1] * (kernel_size[1] - 1)) - 1) / strides[1]) + 1

    D_out = tf.cast(filters, dtype=tf.float32) if filters is not None else Din

    output_shape = tf.cast(tf.floor(tf.stack((H_out, W_out, D_out), axis=0)), dtype=tf.int32)
    return tf.TensorShape(output_shape)


def copy_layer(layer, name=None, include_weights=False):
    config = layer.get_config()
    config['name'] = name or config['name'] + '_2'
    new_layer = type(layer).from_config(config)

    if layer.built and include_weights:
        weights = layer.get_weights()
        new_layer.build(layer.input_shape)
        new_layer.set_weights(weights)

    return new_layer
