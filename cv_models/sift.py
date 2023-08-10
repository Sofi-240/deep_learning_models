from typing import Union
import tensorflow as tf
from cv_models.utils import PI, gaussian_kernel, clip_to_shape, make_neighborhood3D, make_neighborhood2D
from tensorflow.python.keras import backend
import numpy as np
from collections import namedtuple

# from viz import show_images

math = tf.math
linalg = tf.linalg

backend.set_floatx('float32')

# https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf

SIGMA = 1.6
ASSUME_BLUR = .5
INTERVALS = 3
BORDER_WIDTH = 5
CONTRAST_THRESHOLD = 0.04
CON = 3
N_ATTEMPTS = 5
EXTREMA_OFFSET = .5
EIGEN_RATIO = 10
DTYPE = backend.floatx()
THRESHOLD = tf.floor(0.5 * CONTRAST_THRESHOLD / 3.0 * 255.0)
SCALE_FACTOR = 1.5
PEAK_RATIO = .8
N_BINS = 36
RADIUS = 3
KeyPoint = namedtuple(f'KeyPoint', 'pt, size, angle, octave, octave_id, response')
Octave = namedtuple('Octave', 'octave_id, base_X, gaus_X, dog_X')


class KeyPointSift:

    def __init__(self):
        self.pt = tf.constant([[]], shape=(0, 4), dtype=DTYPE)
        self.size = tf.constant([[]], shape=(0,), dtype=DTYPE)
        self.angle = tf.constant([[]], shape=(0,), dtype=DTYPE)
        self.octave = tf.constant([[]], shape=(0,), dtype=tf.int32)
        self.octave_id = tf.constant([[]], shape=(0,), dtype=tf.int32)
        self.response = tf.constant([[]], shape=(0,), dtype=DTYPE)

    def __len__(self):
        _shape = self.pt.get_shape()
        return _shape[0]

    def __getitem__(self, index):
        assert isinstance(index, int)
        if index >= self.__len__():
            raise IndexError('Index out of range')
        ret = KeyPoint(
            pt=tf.reshape(self.pt[index], (1, 4)),
            size=self.size[index],
            angle=self.angle[index],
            octave=self.octave[index],
            octave_id=self.octave_id[index],
            response=self.response[index]
        )
        return ret

    def add_key(self,
                pt: Union[tf.Tensor, list, tuple],
                size: Union[tf.Tensor, float],
                angle: Union[tf.Tensor, float] = 0.0,
                octave: Union[tf.Tensor, int] = 0,
                octave_id: Union[tf.Tensor, int] = 0,
                response: Union[tf.Tensor, float] = -1.0):
        if isinstance(size, float):
            size = tf.convert_to_tensor([size])

        n_points_ = max(size.shape)

        def map_args(arg):
            if not isinstance(arg, tf.Tensor):
                return tf.convert_to_tensor([arg] * n_points_)
            shape_ = arg.shape
            n_dim = len(shape_)
            assert n_dim <= 2
            if n_dim == 0 or (n_dim == 1 and shape_[0] == 1):
                arg = tf.get_static_value(arg)
                arg = arg[0] if isinstance(arg, np.ndarray) else arg
                return tf.convert_to_tensor([arg] * n_points_)
            arg = tf.reshape(arg, shape=(-1,))
            assert arg.shape[0] == n_points_
            return arg

        pt = tf.cast(tf.reshape(pt, shape=(-1, 4)), dtype=DTYPE)
        assert pt.shape[0] == n_points_
        self.pt = tf.concat((self.pt, pt), axis=0)

        size = tf.cast(tf.reshape(size, shape=(-1,)), dtype=DTYPE)
        self.size = tf.concat((self.size, size), axis=0)

        args = [angle, octave, octave_id, response]
        angle, octave, octave_id, response = list(map(map_args, args))

        angle = tf.cast(angle, dtype=DTYPE)
        self.angle = tf.concat((self.angle, angle), axis=0)

        octave = tf.cast(octave, dtype=tf.int32)
        self.octave = tf.concat((self.octave, octave), axis=0)

        octave_id = tf.cast(octave_id, dtype=tf.int32)
        self.octave_id = tf.concat((self.octave_id, octave_id), axis=0)

        response = tf.cast(response, dtype=DTYPE)
        self.response = tf.concat((self.response, response), axis=0)


def gaussian_blur(
        X: tf.Tensor,
        sigma: Union[tf.Tensor, float]
) -> tf.Tensor:
    shape_ = X.get_shape()
    n_dim_ = len(shape_)
    assert n_dim_ == 4
    b, h, w, d = shape_
    assert d == 1

    kernel = gaussian_kernel(kernel_size=0, sigma=sigma)
    kernel = tf.cast(kernel, dtype=DTYPE)
    kernel_size = kernel.shape[0]
    kernel = tf.reshape(kernel, shape=(kernel_size, kernel_size, 1, 1))

    k = int(kernel_size // 2)
    paddings = tf.constant([[0, 0], [k, k], [k, k], [0, 0]], dtype=tf.int32)

    X_pad = tf.pad(X, paddings, mode='SYMMETRIC')
    Xg = tf.nn.convolution(X_pad, kernel, padding='VALID')
    return Xg


def blur_base_image(
        image: tf.Tensor,
        sigma: Union[tf.Tensor, float],
        assume_blur: Union[tf.Tensor, float] = .5
) -> tf.Tensor:
    shape_ = image.get_shape()
    n_dim_ = len(shape_)
    assert n_dim_ == 4
    b, h, w, d = shape_
    assert d == 1

    X = tf.image.resize(image, size=[h * 2, w * 2], method='bilinear', name='X')

    sigma = tf.cast(sigma, dtype=DTYPE)
    assume_blur = tf.cast(assume_blur, dtype=DTYPE)

    delta_sigma = (sigma ** 2) - ((2 * assume_blur) ** 2)
    delta_sigma = math.sqrt(tf.maximum(delta_sigma, 0.64))
    return gaussian_blur(X, sigma=delta_sigma)


def intervals_kernels(
        sigma: Union[tf.Tensor, float],
        intervals: Union[tf.Tensor, float]
) -> tf.Tensor:
    images_per_octaves = intervals + 3
    kernels_n = images_per_octaves - 1
    K = 2 ** (1 / intervals)
    sigma = tf.cast(sigma, dtype=DTYPE)

    sigma_prev = (K ** (tf.cast(tf.range(1, images_per_octaves), dtype=DTYPE) - 1.0)) * sigma
    sigmas = math.sqrt((K * sigma_prev) ** 2 - sigma_prev ** 2)

    # sigmas = tf.concat((tf.reshape(sigma, shape=(1,)), sigmas), axis=0)

    # conv with padded kernel == conv with the original kernel.
    kernels = gaussian_kernel(kernel_size=0, sigma=sigmas[-1])
    paddings = tf.constant([[0, 0], [0, 0], [kernels_n - 1, 0]], dtype=tf.int32)
    kernels = tf.pad(kernels[..., tf.newaxis], paddings, constant_values=0.0)

    h, w, _ = kernels.shape

    con_indices = tf.stack(tf.meshgrid(tf.range(h), tf.range(w)), axis=-1)
    con_indices = tf.reshape(con_indices, (-1, 2))

    for s in tf.range(images_per_octaves - 1):
        temp_kernel = gaussian_kernel(kernel_size=0, sigma=sigmas[s])
        curr_indices = tf.pad(
            con_indices,
            tf.constant([[0, 0], [0, 1]], dtype=tf.int32),
            constant_values=tf.cast(s, dtype=tf.int32)
        )
        temp_kernel = clip_to_shape(temp_kernel, [h, w], constant_values=0.0)
        temp_kernel = tf.reshape(temp_kernel, (-1,))

        kernels = tf.tensor_scatter_nd_update(kernels, curr_indices, temp_kernel)
    return kernels


def compute_default_N_octaves(
        image_shape: Union[tf.Tensor, list, tuple],
        min_shape: int = 0
) -> tf.Tensor:
    assert len(image_shape) == 4
    b, h, w, d = image_shape

    s_ = float(min([h, w]))
    diff = math.log(s_)
    if min_shape > 1:
        diff = diff - math.log(tf.cast(min_shape, dtype=DTYPE))

    n_octaves = tf.round(diff / math.log(2.0)) + 1
    return tf.cast(n_octaves, tf.int32)


def pad(
        X: tf.Tensor,
        kernel_size: Union[int, tf.Tensor]
) -> tf.Tensor:
    k_ = int(kernel_size)
    paddings = tf.constant(
        [[0, 0], [k_ // 2, k_ // 2], [k_ // 2, k_ // 2], [0, 0]], dtype=tf.int32
    )
    X_pad = tf.pad(X, paddings, mode='SYMMETRIC')
    return X_pad


def compute_extrema(
        X: tf.Tensor,
        threshold: Union[tf.Tensor, float, None] = None
) -> tf.Tensor:
    threshold = threshold if threshold is not None else THRESHOLD
    _, h, w, d = X.shape

    X_pad = X[..., tf.newaxis]
    X_pad = tf.transpose(X_pad, perm=(0, 3, 1, 2, 4))

    extrema_max = tf.nn.max_pool3d(
        X_pad, (3, 3, 3), (1, 1, 1), 'VALID', data_format='NDHWC'
    )
    extrema_min = tf.nn.max_pool3d(
        X_pad * -1.0, (3, 3, 3), (1, 1, 1), 'VALID', data_format='NDHWC'
    ) * -1.0

    extrema_max = tf.squeeze(tf.transpose(extrema_max, perm=(0, 2, 3, 1, 4)), axis=-1)
    extrema_min = tf.squeeze(tf.transpose(extrema_min, perm=(0, 2, 3, 1, 4)), axis=-1)

    _, compare_array, _ = tf.split(X, [1, 3, 1], axis=-1)
    compare_array = compare_array[:, 1:-1, 1:-1, :]

    byxd = tf.where(
        math.logical_and(
            math.logical_or(
                math.equal(extrema_max, compare_array), math.equal(extrema_min, compare_array)
            ),
            math.greater(tf.abs(compare_array), threshold)
        )
    )

    byxd = byxd + tf.constant([[0, 1, 1, 1]], dtype=tf.int64)

    cb, cy, cx, cd = tf.unstack(byxd, num=4, axis=-1)

    y_cond = tf.logical_and(tf.math.greater_equal(cy, 2), tf.math.less_equal(cy, h - 1 - 2))
    x_cond = tf.logical_and(tf.math.greater_equal(cx, 2), tf.math.less_equal(cx, w - 1 - 2))

    casted_ = tf.logical_and(y_cond, x_cond)

    casted_ = tf.where(casted_)
    byxd = tf.concat([tf.reshape(tf.gather(c, casted_), (casted_.shape[0], 1)) for c in [cb, cy, cx, cd]], axis=-1)
    return byxd


def compute_gradient_xyz(
        X: tf.Tensor
) -> tf.Tensor:
    shape_ = X.get_shape()
    assert len(shape_) == 4

    kx = tf.constant([[0.0, 0.0, 0.0], [-1.0, 0.0, 1.0], [0.0, 0.0, 0.0]], dtype=DTYPE)
    kx = tf.pad(
        tf.reshape(kx, shape=(3, 3, 1, 1)),
        paddings=tf.constant([[0, 0], [0, 0], [1, 1], [0, 0]]),
        constant_values=0.0
    )
    ky = tf.constant([[0.0, -1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=DTYPE)
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


def compute_gradient_xy(
        X: tf.Tensor
) -> tf.Tensor:
    shape_ = X.get_shape()
    assert len(shape_) == 4

    kx = tf.constant([[0.0, 0.0, 0.0], [-1.0, 0.0, 1.0], [0.0, 0.0, 0.0]], dtype=DTYPE)
    kx = tf.reshape(kx, shape=(3, 3, 1, 1))
    ky = tf.constant([[0.0, -1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=DTYPE)
    ky = tf.reshape(ky, shape=(3, 3, 1, 1))

    kernels_dx = tf.concat((kx, ky), axis=-1)
    grad = tf.nn.convolution(X, kernels_dx, padding='VALID')
    return grad


def compute_hessian_xyz(
        F: tf.Tensor
) -> tf.Tensor:
    shape_ = F.get_shape()
    assert len(shape_) == 4

    dxx = tf.constant([[0.0, 0.0, 0.0], [1.0, -2.0, 1.0], [0.0, 0.0, 0.0]], dtype=DTYPE)
    dxx = tf.pad(
        tf.reshape(dxx, shape=(3, 3, 1, 1)),
        paddings=tf.constant([[0, 0], [0, 0], [1, 1], [0, 0]]),
        constant_values=0.0
    )
    dyy = tf.constant([[0.0, 1.0, 0.0], [0.0, -2.0, 0.0], [0.0, 1.0, 0.0]], dtype=DTYPE)
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


def split_extrema_cond(
        extrema: tf.Tensor,
        current_cords: tf.Tensor
) -> tuple[tf.Tensor, tf.Tensor]:
    shape_ = extrema.get_shape()
    assert shape_[-1] == 3
    assert len(shape_) == 2

    cond = math.less(math.reduce_max(math.abs(extrema), axis=-1), EXTREMA_OFFSET)

    next_step_cords_temp = tf.boolean_mask(current_cords, ~cond)

    if next_step_cords_temp.shape[0] == 0:
        return None, cond

    extrema_shift = tf.boolean_mask(extrema, ~cond)

    ex, ey, ez = tf.unstack(extrema_shift, num=3, axis=1)
    positions_shift = tf.pad(
        tf.stack((ey, ex, ez), axis=-1),
        paddings=tf.constant([[0, 0], [1, 0]], dtype=tf.int32),
        constant_values=0.0
    )
    next_step_cords = next_step_cords_temp + tf.cast(tf.round(positions_shift), dtype=tf.int64)

    check_if_change = tf.reduce_max(math.abs(next_step_cords_temp - next_step_cords), axis=-1)

    next_step_cords = tf.boolean_mask(next_step_cords, math.greater(check_if_change, 0))

    return next_step_cords, cond


def localize_extrema(
        middle_pixel_cords: tf.Tensor,
        dOg_array: tf.Tensor,
        octave_index: Union[int, float],
        keyPoint_pointer: Union[KeyPointSift, None] = None
) -> tuple[Union[tf.Tensor, None], Union[tuple, None, KeyPointSift]]:
    dog_shape_ = dOg_array.shape

    cube_neighbor = make_neighborhood3D(middle_pixel_cords, con=CON, origin_shape=dog_shape_)
    if cube_neighbor.shape[0] == 0:
        return None, None

    # the size can change because the make_neighborhood3D return only the valid indexes
    middle_pixel_cords = cube_neighbor[:, (CON ** 3) // 2, :]

    cube_values = tf.gather_nd(dOg_array, tf.reshape(cube_neighbor, shape=(-1, 4))) / 255.0
    cube_values = tf.reshape(cube_values, (-1, CON, CON, CON))

    grad = compute_gradient_xyz(cube_values)
    hess = compute_hessian_xyz(cube_values)
    extrema_update = - linalg.lstsq(hess, grad, l2_regularizer=0.0, fast=False)
    extrema_update = tf.squeeze(extrema_update, axis=-1)

    next_step_cords, kp_cond_1 = split_extrema_cond(extrema_update, middle_pixel_cords)

    dot_ = tf.reduce_sum(
        tf.multiply(
            extrema_update[:, tf.newaxis, tf.newaxis, ...],
            tf.transpose(grad, perm=(0, 2, 1))[:, tf.newaxis, ...]
        ), axis=-1, keepdims=False
    )
    dot_ = tf.reshape(dot_, shape=(-1,))
    update_response = cube_values[:, 1, 1, 1] + 0.5 * dot_

    kp_cond_2 = math.greater_equal(math.abs(update_response) * INTERVALS, CONTRAST_THRESHOLD)

    hess_xy = hess[:, :2, :2]
    hess_xy_trace = linalg.trace(hess_xy)
    hess_xy_det = linalg.det(hess_xy)

    kp_cond_3 = math.logical_and(
        math.greater(hess_xy_det, 0.0),
        math.less(EIGEN_RATIO * (hess_xy_trace ** 2), ((EIGEN_RATIO + 1) ** 2) * hess_xy_det)
    )
    sure_key_points = math.logical_and(math.logical_and(kp_cond_1, kp_cond_2), kp_cond_3)

    # -------------------------------------------------------------------------
    kp_extrema_update = tf.boolean_mask(extrema_update, sure_key_points)

    if kp_extrema_update.shape[0] == 0:
        return next_step_cords, None

    kp_cords = tf.cast(tf.boolean_mask(middle_pixel_cords, sure_key_points), dtype=DTYPE)
    octave_index = tf.cast(octave_index, dtype=DTYPE)

    ex, ey, ez = tf.unstack(kp_extrema_update, num=3, axis=1)
    cd, cy, cx, cz = tf.unstack(kp_cords, num=4, axis=1)

    kp_pt = tf.stack(
        (cd, (cy + ey) * (2 ** octave_index), (cx + ex) * (2 ** octave_index), cz), axis=-1
    )

    kp_octave = octave_index + cz * (2 ** 8) + tf.round((ez + 0.5) * 255.0) * (2 ** 16)
    kp_octave = tf.cast(kp_octave, dtype=tf.int32)

    kp_size = SIGMA * (2 ** ((cz + ez) / tf.cast(INTERVALS, dtype=DTYPE))) * (2 ** (octave_index + 1.0))
    kp_response = math.abs(tf.boolean_mask(update_response, sure_key_points))

    if keyPoint_pointer is not None:
        keyPoint_pointer.add_key(
            pt=kp_pt, octave=kp_octave, size=kp_size, response=kp_response,
            octave_id=tf.cast(octave_index, dtype=tf.int32)
        )
        return next_step_cords, keyPoint_pointer
    kp = namedtuple('KeyPoint', 'pt, octave, o_size, response, octave_index')
    kp = kp(kp_pt, kp_octave, kp_size, kp_response, tf.cast(octave_index, dtype=tf.int32))
    return next_step_cords, kp


def compute_histogram(
        key_point: KeyPoint,
        octave: Octave
) -> tf.Tensor:
    scale = SCALE_FACTOR * key_point.size / (2 ** (tf.cast(key_point.octave_id, dtype=DTYPE) + 1))
    radius = tf.cast(tf.round(RADIUS * scale), dtype=tf.int32)
    weight_factor = -0.5 / (scale ** 2)

    _prob = 1.0 / tf.cast(2 ** key_point.octave_id, dtype=DTYPE)
    _one = tf.ones_like(_prob)
    _prob = tf.stack((_one, _prob, _prob, _one), axis=-1)

    region_center = tf.cast(key_point.pt * _prob, dtype=tf.int64)

    con = (radius * 2) + 3
    block = make_neighborhood2D(region_center, con=con, origin_shape=octave.gaus_X.shape)

    gaus_img = tf.gather_nd(octave.gaus_X, tf.reshape(block, shape=(-1, 4)))
    gaus_img = tf.reshape(gaus_img, shape=(1, con, con, 1))
    gaus_grad = compute_gradient_xy(gaus_img)

    dx, dy = tf.split(gaus_grad, [1, 1], axis=-1)
    magnitude = math.sqrt(dx * dx + dy * dy)
    orientation = math.atan2(dx, dy) * (180.0 / PI)

    neighbor_index = make_neighborhood2D(tf.constant([[0, 0, 0, 0]], dtype=tf.int64), con=con - 2)
    neighbor_index = tf.cast(tf.reshape(neighbor_index, shape=(1, con - 2, con - 2, 4)), dtype=DTYPE)
    _, y, x, _ = tf.split(neighbor_index, [1, 1, 1, 1], axis=-1)

    weight = math.exp(weight_factor * (y ** 2 + x ** 2))
    hist_index = tf.cast(tf.round(orientation * N_BINS / 360.), dtype=tf.int64)
    hist_index = hist_index % N_BINS
    hist_index = tf.reshape(hist_index, (-1, 1))

    hist = tf.zeros(shape=(N_BINS,), dtype=DTYPE)
    hist = tf.tensor_scatter_nd_add(hist, hist_index, tf.reshape(weight * magnitude, (-1,)))
    return hist


def compute_orientation(
        key_points: KeyPointSift,
        octaves: list[Octave]
) -> KeyPointSift:
    histogram = tf.constant([[]], shape=(0, N_BINS), dtype=DTYPE)

    for k in range(len(key_points)):
        curr_kp = key_points[k]
        curr_octave = octaves[curr_kp.octave_id]
        curr_hist = compute_histogram(curr_kp, curr_octave)
        histogram = tf.concat((histogram, tf.reshape(curr_hist, shape=(1, -1))), axis=0)

    gaussian1D = tf.constant([1, 4, 6, 4, 1], dtype=DTYPE) / 16.0
    gaussian1D = tf.reshape(gaussian1D, shape=(-1, 1, 1))

    histogram_pad = tf.pad(
        tf.expand_dims(histogram, axis=-1),
        paddings=tf.constant([[0, 0], [2, 2], [0, 0]], dtype=tf.int32),
        mode='SYMMETRIC'
    )

    smooth_histogram = tf.nn.convolution(histogram_pad, gaussian1D, padding='VALID')
    smooth_histogram = tf.squeeze(smooth_histogram, axis=-1)

    orientation_max = tf.reduce_max(smooth_histogram, axis=-1)
    value_cond = tf.repeat(tf.reshape(orientation_max, shape=(-1, 1)), repeats=N_BINS, axis=-1)
    value_cond = value_cond * PEAK_RATIO

    cond = math.logical_and(
        math.greater(smooth_histogram, tf.roll(smooth_histogram, shift=1, axis=1)),
        math.greater(smooth_histogram, tf.roll(smooth_histogram, shift=-1, axis=1))
    )
    cond = math.logical_and(
        cond, math.greater_equal(smooth_histogram, value_cond)
    )

    peak_index = tf.where(cond)
    peak_value = tf.boolean_mask(smooth_histogram, cond)

    p_id, p_idx = tf.unstack(peak_index, num=2, axis=-1)

    left_index = (p_idx - 1) % N_BINS
    right_index = (p_idx + 1) % N_BINS

    left_value = tf.gather_nd(smooth_histogram, tf.stack((p_id, left_index), axis=-1))
    right_value = tf.gather_nd(smooth_histogram, tf.stack((p_id, right_index), axis=-1))

    interpolated_peak_index = ((tf.cast(p_idx, dtype=DTYPE) + 0.5 * (left_value - right_value)) / (
            left_value - (2 * peak_value) + right_value)) % N_BINS

    orientation = 360. - interpolated_peak_index * 360. / N_BINS

    orientation = tf.where(math.less(math.abs(orientation), 1e-7), 0.0, orientation)

    key_points_new = KeyPointSift()

    p_id = tf.reshape(p_id, (-1, 1))

    key_points_new.add_key(
        pt=tf.gather_nd(key_points.pt, p_id),
        octave=tf.gather_nd(key_points.octave, p_id),
        octave_id=tf.gather_nd(key_points.octave_id, p_id),
        size=tf.gather_nd(key_points.size, p_id),
        angle=orientation,
        response=tf.gather_nd(key_points.response, p_id)
    )
    return key_points_new


if __name__ == '__main__':
    img = tf.keras.utils.load_img('box.png', color_mode='grayscale')
    img = tf.convert_to_tensor(tf.keras.utils.img_to_array(img), dtype=DTYPE)
    img = img[tf.newaxis, ...]

    base_image = blur_base_image(img, sigma=SIGMA, assume_blur=ASSUME_BLUR)
    gaus_kernels = intervals_kernels(sigma=SIGMA, intervals=INTERVALS)
    kernels_shape_ = gaus_kernels.shape
    gaus_kernels = tf.reshape(gaus_kernels, shape=(kernels_shape_[0], kernels_shape_[1], 1, kernels_shape_[-1]))

    num_octaves = compute_default_N_octaves(image_shape=base_image.shape, min_shape=kernels_shape_[0])
    octave_capture = []
    key_points_capture = KeyPointSift()
    octave_base = tf.identity(base_image)

    for i in tf.range(num_octaves):
        octave_base_pad = pad(octave_base, kernels_shape_[0])
        octave_pyramid = tf.nn.convolution(octave_base_pad, gaus_kernels, padding='VALID')

        octave_pyramid = tf.concat((octave_base, octave_pyramid), axis=-1)

        DOG_pyramid = octave_pyramid[..., 1:] - octave_pyramid[..., :-1]

        continue_search = compute_extrema(DOG_pyramid)

        for _ in tf.range(N_ATTEMPTS):
            continue_search, _ = localize_extrema(
                continue_search, DOG_pyramid, i, keyPoint_pointer=key_points_capture
            )

            if continue_search is None:
                break

        octave_capture.append(
            Octave(i, octave_base, octave_pyramid, DOG_pyramid)
        )

        octave_base = tf.expand_dims(octave_pyramid[..., -3], axis=-1)
        _, Oh, Ow, _ = octave_base.get_shape()
        octave_base = tf.image.resize(octave_base, size=[Oh // 2, Ow // 2], method='nearest')

    keyPoints = compute_orientation(key_points_capture, octave_capture)
