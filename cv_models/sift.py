from typing import Union
import tensorflow as tf
from cv_models.utils import PI, gaussian_kernel, clip_to_shape, make_neighborhood3D, cast_cords
from tensorflow.python.keras import backend

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
DTYPE = tf.float32
THRESHOLD = tf.floor(0.5 * CONTRAST_THRESHOLD / 3.0 * 255.0)


class KeyPoint:
    """for now...."""

    def __init__(self, octave_master, extrema, coordinates, response):
        gaussian_images = octave_master.gaussian_images
        octave_index = octave_master.index

        self.response = tf.abs(response)

        self.octave_index = tf.cast(octave_index, dtype=tf.int32)

        ex, ey, ez = tf.unstack(extrema, num=3, axis=1)
        cb, cy, cx, cz = tf.unstack(tf.cast(coordinates, dtype=tf.float32), num=4, axis=1)

        self.pt = (cb, (cy + ey) * (2 ** octave_index), (cx + ex) * (2 ** octave_index))
        self.octave = octave_index + (cz * (2 ** 8)) + tf.round((ez + 0.5) * 255) * (2 ** 16)
        self.size = SIGMA * (2 ** ((cz + ez) / tf.cast(INTERVALS, dtype=tf.float32))) * (2 ** (octave_index + 1))
        self.local_image_index = tf.cast(cz, dtype=tf.int32)
        self.local_image_index_uni = tf.unique(self.local_image_index)
        self.gaussian_images = tf.gather(gaussian_images, self.local_image_index_uni[0], axis=-1)

    def __len__(self):
        return self.local_image_index.shape[0]


class Octave:
    def __init__(self,
                 index: int,
                 base_X: tf.Tensor,
                 gaussian_images: tf.Tensor,
                 dog_images: tf.Tensor,
                 init_extrema: tf.Tensor):
        self.index_ = index
        self.base_X = base_X
        self.gaussian_images = gaussian_images
        self.dog_images = dog_images
        self.init_extrema = init_extrema

    @property
    def index(self) -> tf.Tensor:
        return tf.cast(self.index_, dtype=DTYPE)

    def __split_cond(
            self,
            extrema: tf.Tensor,
            current_cords: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor]:
        shape_ = extrema.get_shape()
        assert shape_[-1] == 3
        assert len(shape_) == 2

        ex, ey, ez = tf.unstack(extrema, num=3, axis=1)
        cond = math.logical_and(
            math.logical_and(tf.abs(ex) < EXTREMA_OFFSET, tf.abs(ey) < EXTREMA_OFFSET),
            tf.abs(ez) < EXTREMA_OFFSET
        )

        cond_by_index = tf.where(cond == False)

        positions_shift = tf.gather_nd(
            tf.stack((ey, ex, ez), axis=-1), cond_by_index
        )
        if positions_shift.shape[0] == 0:
            return None, tf.where(cond == True)

        next_middle_pixel_cords = tf.gather_nd(current_cords, cond_by_index)

        positions_shift = tf.pad(
            positions_shift, paddings=tf.constant([[0, 0], [1, 0]], dtype=tf.int32), constant_values=0
        )
        next_middle_pixel_cords = next_middle_pixel_cords + tf.cast(tf.round(positions_shift), dtype=tf.int64)
        next_middle_pixel_cords = cast_cords(next_middle_pixel_cords, shape=self.dog_images.shape)

        maby_key_point_cords = tf.where(cond == True)

        return next_middle_pixel_cords, maby_key_point_cords

    def __localize_extrema(
            self,
            current_middle_pixel: tf.Tensor
    ) -> tuple[tf.Tensor, Union[KeyPoint, None]]:
        # 3x3x3 cube
        neighbor = make_neighborhood3D(current_middle_pixel, con=CON, origin_shape=self.dog_images.shape)
        neighbor = tf.reshape(neighbor, shape=(-1, 4))

        neighbor_values = tf.gather_nd(self.dog_images, neighbor) / 255.0
        neighbor_values = tf.reshape(neighbor_values, (-1, CON, CON, CON))

        gradient = compute_gradient_xyz(neighbor_values)
        hessian = compute_hessian_xyz(neighbor_values)
        extrema_updates = - tf.linalg.lstsq(hessian, gradient, l2_regularizer=0.0, fast=False)
        extrema_updates = tf.reshape(extrema_updates, (-1, 3))

        next_middle_pixel_cords, maby_key_point_cords = self.__split_cond(extrema_updates, current_middle_pixel)

        # check the key points
        gradient = tf.gather_nd(gradient, maby_key_point_cords)
        hessian = tf.gather_nd(hessian, maby_key_point_cords)
        extrema_updates = tf.gather_nd(extrema_updates, maby_key_point_cords)
        middle_pixel_values = tf.gather_nd(neighbor_values, maby_key_point_cords)
        middle_pixel_values = middle_pixel_values[:, 1, 1, 1]
        middle_pixel_cords = tf.gather_nd(current_middle_pixel, maby_key_point_cords)

        dot = tf.reduce_sum(
            tf.multiply(
                extrema_updates[:, tf.newaxis, tf.newaxis, ...],
                tf.transpose(gradient, perm=(0, 2, 1))[:, tf.newaxis, ...]
            ),
            axis=-1, keepdims=False
        )
        dot = tf.reshape(dot, shape=(-1,))
        functionValue = middle_pixel_values + 0.5 * dot

        mask_for_key_points = math.greater_equal(
            math.abs(functionValue) * INTERVALS, CONTRAST_THRESHOLD
        )

        xy_hess = hessian[:, :2, :2]
        xy_hess_trace = tf.linalg.trace(xy_hess)
        xy_hess_det = tf.linalg.det(xy_hess)

        sure_key_points = math.logical_and(
            math.logical_and(
                xy_hess_det > 0,
                math.less(EIGEN_RATIO * (xy_hess_trace ** 2), ((EIGEN_RATIO + 1) ** 2) * xy_hess_det)
            ), mask_for_key_points
        )

        sure_key_points_cords = tf.where(sure_key_points == True)

        if sure_key_points_cords.shape[0] == 0:
            return next_middle_pixel_cords, None

        kp_extrema = tf.gather_nd(extrema_updates, sure_key_points_cords)
        kp_cords = tf.gather_nd(middle_pixel_cords, sure_key_points_cords)
        kp_functionValue = tf.gather_nd(functionValue, sure_key_points_cords)

        kp = KeyPoint(octave_master=self, extrema=kp_extrema, coordinates=kp_cords, response=kp_functionValue)

        return next_middle_pixel_cords, kp

    def __call__(
            self,
            iterations: Union[None, int] = None
    ) -> list[KeyPoint, ...]:

        iterations = iterations if iterations is not None else N_ATTEMPTS
        current_middle_pixel = self.init_extrema
        keyPoints = []

        for _ in tf.range(iterations):
            if current_middle_pixel is None:
                break
            current_middle_pixel, kp = self.__localize_extrema(current_middle_pixel)
            if kp is not None:
                keyPoints.append(kp)
        return keyPoints


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
    kernel_size = kernel.shape[0]
    kernel = tf.reshape(kernel, shape=(kernel_size, kernel_size, 1, 1), name='kernel')

    k = int(kernel_size // 2)
    paddings = tf.constant([[0, 0], [k, k], [k, k], [0, 0]], dtype=tf.int32)

    X_pad = tf.pad(X, paddings, mode='SYMMETRIC', name='X_pad')
    Xg = tf.nn.convolution(X_pad, kernel, padding='VALID', name='Xg')
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

    sigma = tf.cast(sigma, dtype=tf.float32)
    assume_blur = tf.cast(assume_blur, dtype=tf.float32)

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
    sigma = tf.cast(sigma, dtype=tf.float32)

    sigma_prev = (K ** (tf.cast(tf.range(1, images_per_octaves), dtype=tf.float32) - 1.0)) * sigma
    sigmas = math.sqrt((K * sigma_prev) ** 2 - sigma_prev ** 2)

    # sigmas = tf.concat((tf.reshape(sigma, shape=(1,)), sigmas), axis=0)

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
        diff = diff - math.log(tf.cast(min_shape, tf.float32))

    n_octaves = tf.round(diff / math.log(2.0)) + 1
    return tf.cast(n_octaves, tf.int32)


def compute_gradient_xyz(
        X: tf.Tensor
) -> tf.Tensor:
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


def compute_hessian_xyz(
        F: tf.Tensor
) -> tf.Tensor:
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


def compute_extrema(
        X: tf.Tensor,
        threshold: Union[tf.Tensor, float, None] = None
) -> tf.Tensor:
    threshold = threshold if threshold is not None else THRESHOLD
    _, h, w, d = X.shape
    X_pad = tf.pad(
        X, tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]], dtype=tf.int32), constant_values=0.0
    )
    # TODO: this is not correct if : abs(min(h)) > abs(max(h)) so we throw the local maxima!!!
    X_pad = tf.transpose(math.abs(X_pad), perm=(0, 3, 1, 2))[..., tf.newaxis]

    extrema = tf.nn.max_pool3d(
        X_pad, (3, 3, 3), (1, 1, 1), 'VALID', data_format='NDHWC'
    )

    extrema = tf.transpose(tf.squeeze(extrema, axis=-1), perm=(0, 2, 3, 1))
    extrema = tf.pad(
        extrema, tf.constant([[0, 0], [0, 0], [0, 0], [1, 1]], dtype=tf.int32), constant_values=0.0
    )

    byxd = tf.where(
        math.logical_and(
            math.equal(extrema, tf.abs(X)), math.greater(tf.abs(X), threshold)
        )
    )

    def cast(arr, min_val, max_val):
        return tf.logical_and(tf.math.greater_equal(arr, min_val), tf.math.less_equal(arr, max_val))

    cords_unstack = tf.unstack(byxd, num=4, axis=-1)
    casted_shape = [[BORDER_WIDTH, h - 1 - BORDER_WIDTH],
                    [BORDER_WIDTH, w - 1 - BORDER_WIDTH]]

    masked_cords = [cast(cords_unstack[c], casted_shape[c - 1][0], casted_shape[c - 1][1]) for c in
                    range(1, len(cords_unstack) - 1)]

    casted_ = tf.ones(shape=masked_cords[0].shape, dtype=tf.bool)
    for mask in masked_cords:
        casted_ = tf.math.logical_and(casted_, mask)

    casted_ = tf.where(casted_)
    byxd = tf.concat([tf.reshape(tf.gather(c, casted_), (casted_.shape[0], 1)) for c in cords_unstack], axis=-1)
    return byxd


def pad(X: tf.Tensor, kernel_size: Union[int, tf.Tensor]) -> tf.Tensor:
    k_ = int(kernel_size)
    paddings = tf.constant(
        [[0, 0], [k_ // 2, k_ // 2], [k_ // 2, k_ // 2], [0, 0]], dtype=tf.int32
    )
    X_pad = tf.pad(X, paddings, mode='SYMMETRIC')
    return X_pad


if __name__ == '__main__':
    img = tf.keras.utils.load_img('box.png', color_mode='grayscale')
    img = tf.convert_to_tensor(tf.keras.utils.img_to_array(img), dtype=tf.float32)
    img = img[tf.newaxis, ...]

    base_image = blur_base_image(img, sigma=SIGMA, assume_blur=ASSUME_BLUR)
    gaus_kernels = intervals_kernels(sigma=SIGMA, intervals=INTERVALS)
    kernels_shape_ = gaus_kernels.shape
    gaus_kernels = tf.reshape(gaus_kernels, shape=(kernels_shape_[0], kernels_shape_[1], 1, kernels_shape_[-1]))

    num_octaves = compute_default_N_octaves(image_shape=base_image.shape, min_shape=kernels_shape_[0])
    octave_capture = []
    key_points = []

    octave_base = tf.identity(base_image)

    for i in tf.range(num_octaves):
        octave_base_pad = pad(octave_base, kernels_shape_[0])
        octave_pyramid = tf.nn.convolution(octave_base_pad, gaus_kernels, padding='VALID')

        octave_pyramid = tf.concat((octave_base, octave_pyramid), axis=-1)

        DOG_pyramid = octave_pyramid[..., 1:] - octave_pyramid[..., :-1]

        current_octave = Octave(
            index=i, base_X=octave_base, gaussian_images=octave_pyramid,
            dog_images=DOG_pyramid, init_extrema=compute_extrema(DOG_pyramid)
        )

        octave_base = tf.expand_dims(octave_pyramid[..., -3], axis=-1)
        _, Oh, Ow, _ = octave_base.get_shape()
        octave_base = tf.image.resize(octave_base, size=[Oh // 2, Ow // 2], method='nearest')

        key_points += current_octave(iterations=N_ATTEMPTS)
        octave_capture.append(current_octave)
