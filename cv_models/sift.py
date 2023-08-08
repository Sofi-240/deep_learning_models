from typing import Union
import tensorflow as tf
from cv_models.utils import PI, gaussian_kernel, clip_to_shape, make_neighborhood3D, cast_cords
from viz import show_images
import numpy as np

math = tf.math

# https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf

SIGMA = 1.6
ASSUME_BLUR = .5
INTERVALS = 3
BORDER_WIDTH = 5
CONTRAST_THRESHOLD = 0.04
CON = 3


class KeyPoint:
    def __init__(self):
        self.pt = None
        self.octave = None
        self.size = None
        self.response = None


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
        assume_blur: Union[tf.Tensor, float] = .5):
    shape_ = image.get_shape()
    n_dim_ = len(shape_)
    assert n_dim_ == 4
    b, h, w, d = shape_
    assert d == 1

    X = tf.image.resize(image, size=[h * 2, w * 2], method='bilinear', name='X')

    sigma = tf.cast(sigma, dtype=tf.float32)
    assume_blur = tf.cast(assume_blur, dtype=tf.float32)

    delta_sigma = (sigma ** 2) - ((2 * assume_blur) ** 2)
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
    return kernels


def scale_space_pyramid(image, num_octaves, kernels, border_width=5, contrast_threshold=0.04):
    shape_ = image.get_shape()
    n_dim_ = len(shape_)
    assert n_dim_ == 4
    assert shape_[-1] == 1

    threshold = tf.floor(0.5 * contrast_threshold / 3.0 * 255.0)

    def extrema(p):
        _, H, W, D = p.shape
        p_pad = tf.pad(p, tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]], dtype=tf.int32), constant_values=0.0)
        p_pad = tf.transpose(p_pad, perm=(0, 3, 1, 2))
        p_pad = tf.expand_dims(p_pad, axis=-1)
        p_pad = math.abs(p_pad)

        max_pooling = tf.nn.max_pool3d(p_pad, (3, 3, 3), (1, 1, 1), 'VALID', data_format='NDHWC', name=None)
        max_pooling = tf.squeeze(max_pooling, axis=-1)
        max_pooling = tf.transpose(max_pooling, perm=(0, 2, 3, 1))
        max_pooling = tf.pad(max_pooling, tf.constant([[0, 0], [0, 0], [0, 0], [1, 1]], dtype=tf.int32),
                             constant_values=0.0)

        byxd = tf.where(math.logical_and(math.equal(max_pooling, tf.abs(p)), math.greater(tf.abs(p), threshold)))

        b, y, x, d = tf.split(byxd, [1, 1, 1, 1], axis=-1)

        def cast(arr, max_val, diff=0):
            arr = tf.reshape(arr, shape=(-1,))
            arr = tf.where(tf.math.greater(arr, max_val - diff), -1, arr)
            arr = tf.where(tf.math.less(arr, diff), -1, arr)
            return arr

        y = cast(y, H - 1, border_width)
        x = cast(x, W - 1, border_width)
        b = tf.reshape(b, shape=(-1,))
        d = tf.reshape(d, shape=(-1,))

        casted_ = tf.where(math.logical_and(math.logical_and(x > 0, y > 0), d > 0))

        byxd = tf.concat([tf.reshape(tf.gather(c, casted_), (casted_.shape[0], 1)) for c in [b, y, x, d]], axis=-1)
        return byxd

    K, _, I = kernels.shape
    kernels = tf.reshape(kernels, shape=(K, K, 1, I), name='kernel')
    paddings = tf.constant([[0, 0], [K // 2, K // 2], [K // 2, K // 2], [0, 0]], dtype=tf.int32)

    pyramid = []
    DOG_pyramid = []
    DOG_pyramid_extrema_points = []

    for _ in tf.range(num_octaves):
        image_pad = tf.pad(image, paddings, mode='SYMMETRIC')

        image_blur = tf.nn.convolution(image_pad, kernels, padding='VALID')

        pyramid.append(image_blur)

        dog = image_blur[..., 1:] - image_blur[..., :-1]
        DOG_pyramid.append(dog)

        curr_extrema = extrema(dog)
        DOG_pyramid_extrema_points.append(curr_extrema)

        octave_base = tf.expand_dims(image_blur[..., -3], axis=-1)
        _, Oh, Ow, _ = octave_base.get_shape()
        image = tf.image.resize(octave_base, size=[Oh // 2, Ow // 2], method='nearest')

    return DOG_pyramid, DOG_pyramid_extrema_points


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


def localize_extrema():
    return


if __name__ == '__main__':
    # show_images(tf.transpose(dOg_pyramid[-1], perm=(3, 1, 2, 0)), 2, 3)
    img = tf.keras.utils.load_img('box_in_scene.png', color_mode='grayscale')
    img = tf.convert_to_tensor(tf.keras.utils.img_to_array(img), dtype=tf.float32)
    img = img[tf.newaxis, ...]

    base_img = base_image(img, sigma=SIGMA, assume_blur=ASSUME_BLUR)
    kernel_per_interval = intervals_kernels(sigma=SIGMA, intervals=INTERVALS)
    N_oc = n_octaves(base_img.shape, kernel_per_interval.shape[0])

    dOg_pyramid, extrema_points = scale_space_pyramid(
        base_img, N_oc, kernel_per_interval, border_width=BORDER_WIDTH, contrast_threshold=CONTRAST_THRESHOLD
    )

    poi = extrema_points[0]
    curr_dog = dOg_pyramid[0]

    neighbor = make_neighborhood3D(poi, con=CON, origin_shape=dOg_pyramid[0].shape)
    neighbor = tf.reshape(neighbor, shape=(-1, 4))

    values = tf.gather_nd(curr_dog, neighbor)
    values = values / 255.0

    values = tf.reshape(values, (-1, CON, CON, CON))

    grad = compute_gradient_xyz(values)
    hessian = compute_hessian_xyz(values)
    extrema = - tf.linalg.lstsq(hessian, grad, l2_regularizer=0.0, fast=True)
    extrema = tf.reshape(extrema, (-1, 3))
    ex, ey, ez = tf.unstack(extrema, num=3, axis=1)

    # next cord case:

    # next_cords = tf.where(math.logical_or(math.logical_or(tf.abs(ex) >= 0.5, tf.abs(ey) >= 0.5), tf.abs(ez) >= 0.5))
    # positions_move = tf.gather_nd(tf.stack((ey, ex, ez), axis=-1), next_cords)
    # next_cords = tf.gather_nd(poi, next_cords)
    #
    # positions_move = tf.pad(positions_move, paddings=tf.constant([[0, 0], [1, 0]], dtype=tf.int32), constant_values=0)
    #
    # next_cords = next_cords + tf.cast(tf.round(positions_move), dtype=tf.int64)
    #
    # next_cords = cast_cords(next_cords, shape=curr_dog.shape)

    # key point case:

    maby_key_points = tf.where(math.logical_and(math.logical_and(tf.abs(ex) < 0.5, tf.abs(ey) < 0.5), tf.abs(ez) < 0.5))
    grad = tf.gather_nd(grad, maby_key_points)
    hessian = tf.gather_nd(hessian, maby_key_points)
    extrema = tf.gather_nd(extrema, maby_key_points)
    maby_key_values = tf.gather_nd(values, maby_key_points)
    maby_key_values = maby_key_values[:, 1, 1, 1]

    dot = tf.reduce_sum(
        tf.multiply(
            extrema[:, tf.newaxis, tf.newaxis, ...], tf.transpose(grad, perm=(0, 2, 1))[:, tf.newaxis, ...]
        ),
        axis=-1, keepdims=False
    )
    dot = tf.reshape(dot, shape=(-1,))
    functionValue = maby_key_values + 0.5 * dot
