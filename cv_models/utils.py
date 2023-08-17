import tensorflow as tf
from typing import Union
import numpy as np

PI = tf.cast(
    tf.math.angle(tf.constant(-1, dtype=tf.complex64)), tf.float32
)


def gaussian_kernel(
        kernel_size: int,
        sigma: Union[None, float] = None
) -> tf.Tensor:
    if kernel_size == 0 and (sigma is None or sigma < 0.8):
        if sigma is None:
            raise ValueError('need sigma parameter when the kernel size is 0')
        raise ValueError('minimum kernel need to be size of 3 --> sigma > 0.8')

    if kernel_size == 0:
        kernel_size = ((((sigma - 0.8) / 0.3) + 1) * 2) + 1
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


def gaussian_blur(
        X: tf.Tensor,
        kernel_size: Union[tf.Tensor, int] = 0,
        sigma: Union[tf.Tensor, float] = 0.8
) -> tf.Tensor:
    shape_ = X.get_shape()
    n_dim_ = len(shape_)
    assert n_dim_ == 4
    b, h, w, d = shape_
    assert d == 1
    _dtype = X.dtype

    kernel = gaussian_kernel(kernel_size=kernel_size, sigma=sigma)
    kernel = tf.cast(kernel, dtype=_dtype)
    kernel_size = kernel.shape[0]
    kernel = tf.reshape(kernel, shape=(kernel_size, kernel_size, 1, 1))

    k = int(kernel_size // 2)
    paddings = tf.constant([[0, 0], [k, k], [k, k], [0, 0]], dtype=tf.int32)

    X_pad = tf.pad(X, paddings, mode='SYMMETRIC')
    Xg = tf.nn.convolution(X_pad, kernel, padding='VALID')
    return Xg


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


def make_neighborhood2D(
        init_cords: tf.Tensor,
        con: int = 3,
        origin_shape: Union[None, tuple, list, tf.TensorShape] = None
) -> tf.Tensor:
    B, ndim = init_cords.get_shape()
    con = int(con)
    assert ndim == 4

    ax = tf.range(-con // 2 + 1, (con // 2) + 1, dtype=tf.int64)

    con_kernel = tf.stack(tf.meshgrid(ax, ax), axis=-1)

    con_kernel = tf.reshape(con_kernel, shape=(1, con ** 2, 2))

    b, yx, d = tf.split(init_cords, [1, 2, 1], axis=1)
    yx = yx[:, tf.newaxis, ...]

    yx = yx + con_kernel

    b = tf.repeat(b[:, tf.newaxis, ...], repeats=con ** 2, axis=1)
    d = tf.repeat(d[:, tf.newaxis, ...], repeats=con ** 2, axis=1)

    neighbor = tf.concat((b, yx, d), axis=-1)
    if origin_shape is None:
        return neighbor

    assert len(origin_shape) == 4
    neighbor = neighbor + 1
    b, y, x, d = tf.unstack(neighbor, num=4, axis=-1)

    y_cast = tf.logical_and(tf.math.greater_equal(y, 1), tf.math.less_equal(y, origin_shape[1]))
    x_cast = tf.logical_and(tf.math.greater_equal(x, 1), tf.math.less_equal(x, origin_shape[2]))

    valid = tf.cast(tf.logical_and(y_cast, x_cast), dtype=tf.int32)
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


def compute_extrema3D(
        X: tf.Tensor,
        threshold: Union[tf.Tensor, float, None] = None,
        con: Union[tf.Tensor, int, tuple, list] = 3,
        border_width: Union[tf.Tensor, tuple, list, None] = None,
        epsilon: Union[tf.Tensor, float] = 1e-07
) -> tf.Tensor:
    _shape = tf.shape(X)
    _n_dims = len(_shape)
    if _n_dims != 4:
        raise ValueError(
            'expected the inputs to be 4D tensor with size of (None, H, W, C)'
        )
    b, h, w, d = tf.unstack(tf.cast(_shape, dtype=tf.int64), num=4, axis=-1)

    X = tf.cast(X, dtype=tf.float32)

    threshold = tf.cast(threshold, dtype=tf.float32) if threshold is not None else None

    if tf.is_tensor(con):
        con = tf.get_static_value(tf.reshape(con, shape=(-1,)))
        con = tuple(con) if len(con) != 1 else int(con)

    if isinstance(con, int):
        con = (con, con, con)

    if len(con) > 3:
        raise ValueError('con parameter need to be int or iterable with size 3')

    half_con = [c // 2 for c in con]

    x_con = tf.concat((X, X * -1.0), axis=-1)

    ex = tf.nn.max_pool2d(x_con, ksize=(3, 3), strides=[1, 1], padding='VALID')
    ex = tf.transpose(ex, perm=(0, 1, 3, 2))
    ex = tf.nn.max_pool2d(ex, ksize=(1, 3), strides=[1, 1], padding='VALID')
    ex = tf.transpose(ex, perm=(0, 1, 3, 2))

    e_ = int((d * 2) - 2)
    extrema_max, _, extrema_min = tf.split(ex, [(e_ // 2) - 1, 2, (e_ // 2) - 1], axis=-1)
    extrema_min = extrema_min * -1.0

    compare_array = tf.slice(X, [0, *half_con], [b, h - 2 * half_con[0], w - 2 * half_con[1], d - 2 * half_con[2]])

    def _equal_with_epsilon(arr):
        return tf.logical_and(
            tf.math.greater_equal(arr, compare_array - epsilon),
            tf.math.less_equal(arr, compare_array + epsilon)
        )

    extrema_cond = tf.logical_or(
        _equal_with_epsilon(extrema_max),
        _equal_with_epsilon(extrema_min)
    )
    if threshold is not None:
        extrema_cond = tf.logical_and(extrema_cond, tf.math.greater(tf.abs(compare_array), threshold))

    byxd = tf.where(extrema_cond)

    byxd = byxd + tf.constant([[0] + half_con], dtype=tf.int64)

    if border_width is not None:
        if tf.is_tensor(border_width):
            border_width = tf.get_static_value(tf.reshape(border_width, shape=(-1,)))
            border_width = tuple(border_width)
        if len(border_width) != 3:
            raise ValueError('border_width need to be with len of 3')
        cb, cy, cx, cd = tf.unstack(byxd, num=4, axis=-1)
        by, bx, bd = tf.unstack(tf.cast(border_width, dtype=tf.int64), num=3)
        y_cond = tf.logical_and(tf.math.greater_equal(cy, by), tf.math.less_equal(cy, h - by))
        x_cond = tf.logical_and(tf.math.greater_equal(cx, bx), tf.math.less_equal(cx, w - bx))
        d_cond = tf.logical_and(tf.math.greater_equal(cd, bd), tf.math.less_equal(cd, d - bd))

        casted_ = tf.logical_and(tf.logical_and(y_cond, x_cond), d_cond)
        byxd = tf.boolean_mask(byxd, casted_)

    return byxd


def compute_central_gradient3D(
        X: tf.Tensor
) -> tf.Tensor:
    _shape = tf.shape(X)
    _n_dims = len(_shape)
    if _n_dims != 4:
        raise ValueError(
            'expected the inputs to be 4D tensor with size of (None, H, W, C)'
        )

    X = tf.cast(X, dtype=tf.float32)

    kx = tf.constant([[0.0, 0.0, 0.0], [-1.0, 0.0, 1.0], [0.0, 0.0, 0.0]], dtype=tf.float32)
    kx = tf.pad(
        tf.reshape(kx, shape=(3, 3, 1, 1, 1)),
        paddings=tf.constant([[0, 0], [0, 0], [1, 1], [0, 0], [0, 0]]),
        constant_values=0.0
    )
    ky = tf.constant([[0.0, -1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=tf.float32)
    ky = tf.pad(
        tf.reshape(ky, shape=(3, 3, 1, 1, 1)),
        paddings=tf.constant([[0, 0], [0, 0], [1, 1], [0, 0], [0, 0]]),
        constant_values=0.0
    )
    kz = tf.zeros_like(kx)
    kz = tf.tensor_scatter_nd_update(kz, tf.constant([[1, 1, 0, 0, 0], [1, 1, 2, 0, 0]]), tf.constant([-1.0, 1.0]))

    kernels_dx = tf.concat((kx, ky, kz), axis=-1)

    X = tf.expand_dims(X, axis=-1)
    grad = tf.nn.convolution(X, kernels_dx, padding='VALID') * 0.5
    return grad


def compute_central_gradient2D(
        X: tf.Tensor
) -> tf.Tensor:
    _shape = tf.shape(X)
    _n_dims = len(_shape)
    if _n_dims != 4:
        raise ValueError(
            'expected the inputs to be 4D tensor with size of (None, H, W, C)'
        )
    b, h, w, d = tf.unstack(tf.cast(_shape, dtype=tf.int64), num=4, axis=-1)
    X = tf.cast(X, dtype=tf.float32)

    kx = tf.constant([[0.0, 0.0, 0.0], [-1.0, 0.0, 1.0], [0.0, 0.0, 0.0]], dtype=tf.float32)
    kx = tf.reshape(kx, shape=(3, 3, 1, 1))
    ky = tf.constant([[0.0, -1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=tf.float32)
    ky = tf.reshape(ky, shape=(3, 3, 1, 1))

    kernels_dx = tf.concat((kx, ky), axis=-1)
    transpose = False
    if d > 1:
        X = tf.transpose(X, perm=(0, 3, 1, 2))
        X = tf.reshape(X, shape=(b * d, h, w, 1))
        transpose = True
    grad = tf.nn.convolution(X, kernels_dx, padding='VALID')
    if transpose:
        grad = tf.reshape(grad, shape=(b, d, h - 2, w - 2, 2))
        grad = tf.transpose(grad, perm=(0, 2, 3, 1, 4))
    return grad


def compute_hessian_3D(
        X: tf.Tensor
) -> tf.Tensor:
    _shape = tf.shape(X)
    _n_dims = len(_shape)
    if _n_dims != 4:
        raise ValueError(
            'expected the inputs to be 4D tensor with size of (None, H, W, C)'
        )

    X = tf.cast(X, dtype=tf.float32)

    dxx = tf.constant([[0.0, 0.0, 0.0], [1.0, -2.0, 1.0], [0.0, 0.0, 0.0]], dtype=tf.float32)
    dxx = tf.pad(
        tf.reshape(dxx, shape=(3, 3, 1, 1, 1)),
        paddings=tf.constant([[0, 0], [0, 0], [1, 1], [0, 0], [0, 0]]),
        constant_values=0.0
    )
    dyy = tf.constant([[0.0, 1.0, 0.0], [0.0, -2.0, 0.0], [0.0, 1.0, 0.0]], dtype=tf.float32)
    dyy = tf.pad(
        tf.reshape(dyy, shape=(3, 3, 1, 1, 1)),
        paddings=tf.constant([[0, 0], [0, 0], [1, 1], [0, 0], [0, 0]]),
        constant_values=0.0
    )
    dzz = tf.zeros_like(dxx)
    dzz = tf.tensor_scatter_nd_update(
        dzz, tf.constant([[1, 1, 0, 0, 0], [1, 1, 1, 0, 0], [1, 1, 2, 0, 0]]), tf.constant([1.0, -2.0, 1.0])
    )

    kww = tf.concat((dxx, dyy, dzz), axis=-1)

    dxy = tf.constant([[1.0, 0.0, -1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 1.0]], dtype=tf.float32)
    dxy = tf.pad(
        tf.reshape(dxy, shape=(3, 3, 1, 1, 1)),
        paddings=tf.constant([[0, 0], [0, 0], [1, 1], [0, 0], [0, 0]]),
        constant_values=0.0
    )

    dxz = tf.zeros_like(dxy)
    dxz = tf.tensor_scatter_nd_update(
        dxz,
        tf.constant([[1, 0, 0, 0, 0], [1, 2, 2, 0, 0], [1, 0, 2, 0, 0], [1, 2, 0, 0, 0]]),
        tf.constant([1.0, 1.0, -1.0, -1.0])
    )

    dyz = tf.zeros_like(dxy)
    dyz = tf.tensor_scatter_nd_update(
        dyz,
        tf.constant([[0, 1, 0, 0, 0], [2, 1, 2, 0, 0], [0, 1, 2, 0, 0], [2, 1, 0, 0, 0]]),
        tf.constant([1.0, 1.0, -1.0, -1.0])
    )

    kws = tf.concat((dxy, dyz, dxz), axis=-1)

    X = tf.expand_dims(X, axis=-1)

    dFww = tf.nn.convolution(X, kww, padding='VALID') * 0.25

    dFws = tf.nn.convolution(X, kws, padding='VALID') * 0.25

    dxx, dyy, dzz = tf.unstack(dFww, 3, axis=-1)
    dxy, dyz, dxz = tf.unstack(dFws, 3, axis=-1)
    hessian_mat = tf.stack(
        (
            tf.stack((dxx, dxy, dxz), axis=-1),
            tf.stack((dxy, dyy, dyz), axis=-1),
            tf.stack((dxz, dyz, dzz), axis=-1)
        ), axis=-1
    )
    return hessian_mat


if __name__ == '__main__':
    case1 = tf.ones(shape=(1, 3, 3, 3), dtype=tf.float32)

    cords_case1 = tf.where(tf.equal(case1, 1.0))
    neighbor1 = make_neighborhood3D(init_cords=cords_case1, con=3, origin_shape=(1, 3, 3, 3))
    assert neighbor1.shape[0] == 1

    case2 = np.array([
        [
            [0.00, 0.01, 0.02, 0.03, 0.04],
            [0.05, 0.06, 0.07, 0.08, 0.09],
            [0.10, 0.11, 0.12, 0.13, 0.14],
            [0.15, 0.16, 0.17, 0.18, 0.19],
            [0.20, 0.21, 0.22, 0.23, 0.24]
        ],
        [
            [0.25, 0.26, 0.27, 0.28, 0.29],
            [0.30, 0.31, 0.32, 0.33, 0.34],
            [0.35, 0.36, 0.37, 0.38, 0.39],
            [0.40, 0.41, 0.42, 0.43, 0.44],
            [0.45, 0.46, 0.47, 0.48, 0.49]
        ],
        [
            [0.50, 0.51, 0.52, 0.53, 0.54],
            [0.55, 0.56, 0.57, 0.58, 0.59],
            [0.60, 0.61, 0.62, 0.63, 0.64],
            [0.65, 0.66, 0.67, 0.68, 0.69],
            [0.70, 0.71, 0.72, 0.73, 0.74]
        ],
        [
            [0.75, 0.76, 0.77, 0.78, 0.79],
            [0.80, 0.81, 0.82, 0.83, 0.84],
            [0.85, 0.86, 0.87, 0.88, 0.89],
            [0.90, 0.91, 0.92, 0.93, 0.94],
            [0.95, 0.96, 0.97, 0.98, 0.99]
        ],
        [
            [1.00, 1.01, 1.02, 1.03, 1.04],
            [1.05, 1.06, 1.07, 1.08, 1.09],
            [1.10, 1.11, 1.12, 1.13, 1.14],
            [1.15, 1.16, 1.17, 1.18, 1.19],
            [1.20, 1.21, 1.22, 1.23, 1.24]
        ]
    ])
    case2 = tf.expand_dims(case2, axis=0)
    case2 = tf.cast(case2, dtype=tf.float32)

    g_xyz = compute_central_gradient3D(case2)
    g_xyz = tf.reduce_mean(tf.reshape(g_xyz, shape=(-1, 3)), axis=0)

    g_xy = compute_central_gradient2D(case2)
