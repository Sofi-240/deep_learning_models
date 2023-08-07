from typing import Union
import tensorflow as tf
from cv_models.utils import PI, gaussian_kernel, clip_to_shape
from viz import show_images
import numpy as np

math = tf.math


# https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf


def gaussian_bluer(X: tf.Tensor,
                   sigma: Union[tf.Tensor, float]):
    kernel = gaussian_kernel(kernel_size=0, sigma=sigma)
    kernel_size = kernel.shape[0]
    kernel = tf.reshape(kernel, shape=(kernel_size, kernel_size, 1, 1), name='kernel')

    k = int(kernel_size // 2)
    paddings = tf.constant([[0, 0], [k, k], [k, k], [0, 0]], dtype=tf.int32)

    X_pad = tf.pad(X, paddings, mode='SYMMETRIC', name='X_pad')
    Xg = tf.nn.convolution(X_pad, kernel, padding='VALID', name='Xg')
    return Xg


def base_image(
        image: tf.Tensor,
        sigma: Union[tf.Tensor, float],
        image_blur: Union[tf.Tensor, float] = .5):
    shape_ = image.get_shape()
    n_dim_ = len(shape_)
    assert n_dim_ == 4
    b, h, w, d = shape_
    assert d == 1

    X = tf.image.resize(image, size=[h * 2, w * 2], method='bilinear', name='X')

    sigma = tf.cast(sigma, dtype=tf.float32)
    image_blur = tf.cast(image_blur, dtype=tf.float32)

    delta_sigma = (sigma ** 2) - ((2 * image_blur) ** 2)
    delta_sigma = math.sqrt(tf.maximum(delta_sigma, 0.64))
    return gaussian_bluer(X, sigma=delta_sigma)


def n_octaves(image_shape, min_shape=0.0):
    assert len(image_shape) == 4
    b, h, w, d = image_shape

    s_ = float(min([h, w]))
    diff = math.log(s_)
    if min_shape > 1.0:
        diff = diff - math.log(tf.cast(min_shape, tf.float32))

    num_octaves = tf.round(diff / math.log(2.0)) + 1
    return tf.cast(num_octaves, tf.int32)


def intervals_kernels(sigma, intervals):
    images_per_octaves = intervals + 3
    K = 2 ** (1 / intervals)
    sigma = tf.cast(sigma, dtype=tf.float32)

    sigma_prev = (K ** (tf.cast(tf.range(1, images_per_octaves), dtype=tf.float32) - 1.0)) * sigma
    sigmas = math.sqrt((K * sigma_prev) ** 2 - sigma_prev ** 2)

    sigmas = tf.concat((tf.reshape(sigma, shape=(1,)), sigmas), axis=0)

    kernels = gaussian_kernel(kernel_size=0, sigma=sigmas[-1])
    paddings = tf.constant([[0, 0], [0, 0], [images_per_octaves - 1, 0]], dtype=tf.int32)
    kernels = tf.pad(kernels[..., tf.newaxis], paddings, constant_values=0.0)

    h, w, _ = kernels.shape

    con_indices = tf.stack(tf.meshgrid(tf.range(h), tf.range(w)), axis=-1)
    con_indices = tf.reshape(con_indices, (-1, 2))

    for s in tf.range(images_per_octaves - 1):
        temp_kernel = gaussian_kernel(kernel_size=0, sigma=sigmas[s])
        curr_indices = tf.pad(
            con_indices, tf.constant([[0, 0], [0, 1]], dtype=tf.int32), constant_values=tf.cast(s, dtype=tf.int32)
        )
        temp_kernel = clip_to_shape(temp_kernel, [h, w], constant_values=0.0)
        temp_kernel = tf.reshape(temp_kernel, (-1,))

        kernels = tf.tensor_scatter_nd_update(kernels, curr_indices, temp_kernel)
    return sigmas, kernels


def scale_space_pyramid(image, num_octaves, kernels):
    shape_ = image.get_shape()
    n_dim_ = len(shape_)
    assert n_dim_ == 4
    b, h, w, d = shape_
    assert d == 1

    K, _, I = kernels.shape
    kernels = tf.reshape(kernels, shape=(K, K, 1, I), name='kernel')
    paddings = tf.constant([[0, 0], [K // 2, K // 2], [K // 2, K // 2], [0, 0]], dtype=tf.int32)

    pyramid = []

    for _ in tf.range(num_octaves):
        image_pad = tf.pad(image, paddings, mode='SYMMETRIC')

        image_blur = tf.nn.convolution(image_pad, kernels, padding='VALID')

        pyramid.append(image_blur)

        octave_base = tf.expand_dims(image_blur[..., -3], axis=-1)
        _, Oh, Ow, _ = octave_base.get_shape()
        image = tf.image.resize(octave_base, size=[Oh // 2, Ow // 2], method='nearest')

    DOG_pyramid = [p[..., 1:] - p[..., :-1] for p in pyramid]
    return DOG_pyramid


def compute_gradient_xyz(X):
    shape_ = X.get_shape()
    assert len(shape_) == 4

    kx = tf.constant([[0.0, 0.0, 0.0], [-1.0, 0.0, 1.0], [0.0, 0.0, 0.0]], dtype=tf.float32)
    kx = tf.pad(
        tf.reshape(kx, shape=(3, 3, 1, 1)),
        paddings=tf.constant([[0, 0], [0, 0], [1, 1], [0, 0]]),
        constant_values=0.0
    )
    ky = tf.constant([[0.0, -1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=tf.float32)
    ky = tf.pad(
        tf.reshape(ky, shape=(3, 3, 1, 1)),
        paddings=tf.constant([[0, 0], [0, 0], [1, 1], [0, 0]]),
        constant_values=0.0
    )
    kz = tf.zeros_like(kx)
    kz = tf.tensor_scatter_nd_update(kz, tf.constant([[1, 1, 0, 0], [1, 1, 2, 0]]), tf.constant([-1.0, 1.0]))

    kernels_dx = tf.concat((kx, ky, kz), axis=-1)
    grad = tf.nn.convolution(X, kernels_dx, padding='VALID') * 0.5
    return tf.reshape(grad, shape=(-1, 3, 1))


def compute_hessian_xyz(F):
    shape_ = F.get_shape()
    assert len(shape_) == 4

    dxx = tf.constant([[0.0, 0.0, 0.0], [1.0, -2.0, 1.0], [0.0, 0.0, 0.0]], dtype=tf.float32)
    dxx = tf.pad(
        tf.reshape(dxx, shape=(3, 3, 1, 1)),
        paddings=tf.constant([[0, 0], [0, 0], [1, 1], [0, 0]]),
        constant_values=0.0
    )
    dyy = tf.constant([[0.0, 1.0, 0.0], [0.0, -2.0, 0.0], [0.0, 1.0, 0.0]], dtype=tf.float32)
    dyy = tf.pad(
        tf.reshape(dyy, shape=(3, 3, 1, 1)),
        paddings=tf.constant([[0, 0], [0, 0], [1, 1], [0, 0]]),
        constant_values=0.0
    )
    dzz = tf.zeros_like(dxx)
    dzz = tf.tensor_scatter_nd_update(
        dzz, tf.constant([[1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 2, 0]]), tf.constant([1.0, -2.0, 1.0])
    )

    kww = tf.concat((dxx, dyy, dzz), axis=-1)

    dxy = tf.constant([[1.0, 0.0, -1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 1.0]], dtype=tf.float32)
    dxy = tf.pad(
        tf.reshape(dxy, shape=(3, 3, 1, 1)),
        paddings=tf.constant([[0, 0], [0, 0], [1, 1], [0, 0]]),
        constant_values=0.0
    )

    dxz = tf.zeros_like(dxy)
    dxz = tf.tensor_scatter_nd_update(
        dxz,
        tf.constant([[1, 0, 0, 0], [1, 2, 2, 0], [1, 0, 2, 0], [1, 2, 0, 0]]),
        tf.constant([1.0, 1.0, -1.0, -1.0])
    )

    dyz = tf.zeros_like(dxy)
    dyz = tf.tensor_scatter_nd_update(
        dyz,
        tf.constant([[0, 1, 0, 0], [2, 1, 2, 0], [0, 1, 2, 0], [2, 1, 0, 0]]),
        tf.constant([1.0, 1.0, -1.0, -1.0])
    )

    kws = tf.concat((dxy, dyz, dxz), axis=-1)

    dFww = tf.nn.convolution(F, kww, padding='VALID') * 0.25
    dFww = tf.reshape(dFww, shape=(-1, 3))

    dFws = tf.nn.convolution(F, kws, padding='VALID') * 0.25
    dFws = tf.reshape(dFws, shape=(-1, 3))

    dxx, dyy, dzz = tf.unstack(dFww, 3, axis=-1)
    dxy, dyz, dxz = tf.unstack(dFws, 3, axis=-1)
    hessian_mat = tf.stack(
        (
            tf.stack((dxx, dxy, dxz), axis=-1),
            tf.stack((dxy, dyy, dyz), axis=-1),
            tf.stack((dxz, dyz, dzz), axis=-1)
        ), axis=1
    )
    return hessian_mat


if __name__ == '__main__':
    # img = tf.random.uniform((2, 3, 3, 3), 0, 10, dtype=tf.int32)
    # img = tf.cast(img, tf.float32)
    #

    img = tf.keras.utils.load_img('box_in_scene.png', color_mode='grayscale')
    img = tf.convert_to_tensor(tf.keras.utils.img_to_array(img), dtype=tf.float32)
    img = img[tf.newaxis, ...]

    base_img = base_image(img, sigma=1.6, image_blur=0.5)

    sigma_per_interval, kernel_per_interval = intervals_kernels(sigma=1.6, intervals=3)

    N_oc = n_octaves(base_img.shape, kernel_per_interval.shape[0])

    dOg_pyramid = scale_space_pyramid(base_img, N_oc, kernel_per_interval)

    # show_images(tf.transpose(dOg_pyramid[-1], perm=(3, 1, 2, 0)), 2, 3)
    threshold = tf.floor(0.5 * 0.04 / 3.0 * 255.0)
    con = 3
    p = dOg_pyramid[3]
    _, H, W, D = p.shape

    con_kernel = tf.nn.max_pool3d(
        tf.expand_dims(tf.abs(p), axis=0), (3, 3, 3), (1, 1, 1), 'SAME', data_format='NDHWC', name=None
    )
    con_kernel = tf.reshape(con_kernel, p.shape)

    cords = tf.where(
        math.logical_and(math.equal(con_kernel, tf.abs(p)), math.greater(tf.abs(p), threshold))
    )
    b, y, x, d = tf.split(cords, [1, 1, 1, 1], axis=-1)


    def cast(arr, max_val, diff=0):
        arr = tf.reshape(arr, shape=(-1,))
        arr = tf.where(tf.math.greater(arr, max_val - diff), -1, arr)
        arr = tf.where(tf.math.less(arr, diff), -1, arr)
        return arr


    y = cast(y, H - 1, 5)
    x = cast(x, W - 1, 5)
    d = cast(d, D - 1, 1)

    casted = tf.where(math.logical_and(math.logical_and(x > 0, y > 0), d > 0))

    casted_cords = tf.concat([tf.reshape(tf.gather(i, casted), (casted.shape[0], 1)) for i in [b, y, x, d]], axis=-1)

    ax = tf.range(-con // 2 + 1, (con // 2) + 1, dtype=tf.int64)
    con_kernel = tf.stack(tf.meshgrid(ax, ax, ax), axis=-1)
    con_kernel = tf.reshape(con_kernel, shape=(1, con ** 3, 3))
    _, H, W, D = p.shape

    b, yxd = tf.split(casted_cords, [1, 3], axis=1)
    yxd = yxd[:, tf.newaxis, ...]

    yxd = yxd + con_kernel

    b = tf.repeat(b[:, tf.newaxis, ...], repeats=con ** 3, axis=1)

    y, x, d = tf.split(yxd, [1, 1, 1], axis=-1)

    neighbor = tf.concat((b, y, x, d), axis=-1)

    neighbor = tf.reshape(neighbor, (-1, 4))

    values = tf.gather_nd(p, neighbor)
    values = values / 255.0

    values = tf.reshape(values, (-1, con ** 3, 1))
    values = tf.reshape(values, (-1, con, con, con))

    grad = compute_gradient_xyz(values)
    hessian = compute_hessian_xyz(values)
    extrema = - tf.linalg.lstsq(hessian, grad, l2_regularizer=0.0, fast=False)

    # pad_extrema =
    # ex, ey, ez = tf.unstack(tf.reshape(extrema, (-1, 3)), num=3, axis=1)
    # next_cords = tf.where(math.logical_and(math.logical_and(ex > 0.5, ey > 0.5), ez > 0.5))
    # next_cords = tf.gather_nd(casted_cords, next_cords)