import tensorflow as tf

PI = tf.cast(
    tf.math.angle(tf.constant(-1, dtype=tf.complex64)), tf.float32
)


def gaussian_kernel(kernel_size, sigma=None):
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


def pad(X, b=None, h=None, w=None, d=None, **kwargs):
    n_dim = len(X.get_shape())
    assert n_dim == 4
    if not b and not h and not w and not d: return X
    paddings = []
    for arg in [b, h, w, d]:
        arg = int(arg) if arg is not None else [0, 0]
        arg = [arg, arg] if issubclass(type(arg), int) else list(arg)
        paddings.append(arg)

    paddings = tf.constant(paddings, dtype=tf.int32)

    padded = tf.pad(X, paddings, **kwargs)
    return padded


def clip_to_shape(X, shape, **kwargs):
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

