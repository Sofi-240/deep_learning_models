import tensorflow as tf
from models.configs import copy_layer
from keras.layers import Activation, BatchNormalization, GroupNormalization


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


def activation_from_config(config, name=None):
    act_config = config.get('act')
    if act_config is None:
        raise ValueError('activation configs is missing')
    activation = act_config.get('activation')
    if issubclass(type(activation), tf.keras.layers.Layer):
        act_layer = copy_layer(activation, name=name, include_weights=False)
    else:
        act_layer = Activation(name=name, **config.get('act', {}))
    return act_layer


def normalization_from_config(config, name=None):
    norm_config = config.get('norm')
    if norm_config is None:
        raise ValueError('normalization configs is missing')
    norm = norm_config.get_from_class_dict('norm')
    if issubclass(type(norm), tf.keras.layers.Layer):
        norm_layer = copy_layer(norm, name=name, include_weights=False)
    else:
        if norm == 'batch':
            norm_layer = BatchNormalization(name=name, **norm_config)
        elif norm == 'instance':
            norm_layer = GroupNormalization(name=name, **norm_config)
            # instance_normalization == GroupNormalization with groups = feature dim
        elif norm == 'group':
            norm_layer = GroupNormalization(name=name, **norm_config)
        else:
            norm_layer = None
    return norm_layer
