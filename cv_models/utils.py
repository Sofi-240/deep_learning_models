import tensorflow as tf
from typing import Union

PI = tf.cast(
    tf.math.angle(tf.constant(-1, dtype=tf.complex64)), tf.float32
)


def gaussian_kernel(
        kernel_size: int,
        sigma: Union[None, float] = None
) -> tf.Tensor:
    if kernel_size == 0 and (sigma is None or sigma < 0.8):
        raise ValueError('minimum kernel need to be size of 3')

    if kernel_size == 0:
        kernel_size = ((tf.math.ceil(3 * (sigma - 0.8)) + 1) * 2) + 1
        kernel_size = kernel_size + 1 if (kernel_size % 2) == 0 else kernel_size

    assert kernel_size % 2 != 0 and kernel_size > 2

    if sigma is None:
        sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8

    ax = tf.range(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
    xx, yy = tf.meshgrid(ax, ax)
    normal = 1 / (2.0 * PI * (sigma ** 2))
    kernel = tf.exp(
        -((xx ** 2) + (yy ** 2)) / (2.0 * (sigma ** 2))
    ) * normal
    return kernel / tf.reduce_sum(kernel)


def clip_to_shape(
        X: tf.Tensor,
        shape: Union[tf.Tensor, list, tuple],
        **kwargs
) -> tf.Tensor:
    shape_ = X.get_shape()
    n_dim_ = len(shape_)

    assert 2 <= n_dim_ <= 4
    assert len(shape) == 2

    b_pad = []
    d_pad = []

    if n_dim_ == 4:
        shape_ = shape_[1:3]
        b_pad, d_pad = [[0, 0], [0, 0]]
    elif n_dim_ == 3:
        shape_ = shape_[:-1]
        d_pad = [0, 0]

    h, w = shape_
    H, W = shape

    pad_h = int(H - h)
    pad_h = [pad_h // 2, pad_h - pad_h // 2]

    pad_w = int(W - w)
    pad_w = [pad_w // 2, pad_w - pad_w // 2]

    paddings = [pad_h, pad_w]
    paddings += [d_pad] if d_pad else []
    paddings = [b_pad] + paddings if b_pad else paddings

    paddings = tf.constant(paddings, dtype=tf.int32)

    return tf.pad(X, paddings, **kwargs)


def make_neighborhood3D(
        init_cords: tf.Tensor,
        con: int = 3,
        origin_shape: Union[None, tuple, list, tf.TensorShape] = None
) -> tf.Tensor:
    B, ndim = init_cords.get_shape()

    assert ndim == 4

    ax = tf.range(-con // 2 + 1, (con // 2) + 1, dtype=tf.int64)

    con_kernel = tf.stack(tf.meshgrid(ax, ax, ax), axis=-1)

    con_kernel = tf.reshape(con_kernel, shape=(1, con ** 3, 3))

    b, yxd = tf.split(init_cords, [1, 3], axis=1)
    yxd = yxd[:, tf.newaxis, ...]

    yxd = yxd + con_kernel

    b = tf.repeat(b[:, tf.newaxis, ...], repeats=con ** 3, axis=1)

    neighbor = tf.concat((b, yxd), axis=-1)
    if origin_shape is None:
        return neighbor

    assert len(origin_shape) == 4
    neighbor = neighbor + 1
    b, y, x, d = tf.unstack(neighbor, num=4, axis=-1)

    y_cast = tf.logical_and(tf.math.greater_equal(y, 1), tf.math.less_equal(y, origin_shape[1]))
    x_cast = tf.logical_and(tf.math.greater_equal(x, 1), tf.math.less_equal(x, origin_shape[2]))
    d_cast = tf.logical_and(tf.math.greater_equal(d, 1), tf.math.less_equal(d, origin_shape[3]))

    valid = tf.cast(tf.logical_and(tf.logical_and(y_cast, x_cast), d_cast), dtype=tf.int32)
    valid = tf.math.reduce_prod(valid, axis=-1)
    cords_valid = tf.where(valid == 1)
    neighbor = tf.gather_nd(neighbor, cords_valid) - 1
    return neighbor


def cast_cords(
        cords: tf.Tensor,
        shape: Union[tf.Tensor, list, tuple]
) -> tf.Tensor:
    cords_shape_ = cords.get_shape()
    assert len(cords_shape_) == 2
    assert cords_shape_[1] == len(shape)

    def cast(arr, min_val, max_val):
        return tf.logical_and(tf.math.greater_equal(arr, min_val), tf.math.less_equal(arr, max_val))

    cords_unstack = tf.unstack(cords, num=4, axis=-1)
    masked_cords = [cast(cords_unstack[c], 0, shape[c] - 1) for c in range(1, len(shape))]

    casted_ = tf.ones(shape=masked_cords[0].shape, dtype=tf.bool)
    for mask in masked_cords:
        casted_ = tf.math.logical_and(casted_, mask)

    casted_ = tf.where(casted_)
    ret = tf.concat([tf.reshape(tf.gather(c, casted_), (casted_.shape[0], 1)) for c in cords_unstack], axis=-1)
    return ret


if __name__ == '__main__':
    case1 = tf.ones(shape=(1, 3, 3, 3), dtype=tf.float32)

    cords_case1 = tf.where(tf.equal(case1, 1.0))
    neighbor1 = make_neighborhood3D(init_cords=cords_case1, con=3, origin_shape=(1, 3, 3, 3))
    assert neighbor1.shape[0] == 1
