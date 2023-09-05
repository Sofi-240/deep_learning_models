from typing import Union
import tensorflow as tf
from cv_models.utils import PI, gaussian_kernel, compute_extrema3D, \
    make_neighborhood2D, compute_central_gradient3D, compute_hessian_3D
from tensorflow.python.keras import backend
from collections import namedtuple
from dataclasses import dataclass, field

# https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf

backend.set_floatx('float32')
linalg_ops = tf.linalg
math_ops = tf.math
bitwise_ops = tf.bitwise
image_ops = tf.image


def load_image(name: str) -> tf.Tensor:
    im = tf.keras.utils.load_img(name, color_mode='grayscale')
    im = tf.convert_to_tensor(tf.keras.utils.img_to_array(im), dtype=tf.float32)
    return im[tf.newaxis, ...]


def show_key_points(key_points, image):
    from viz import show_images

    cords = tf.cast(key_points.pt, dtype=tf.int64) * tf.constant([1, 1, 1, 0], dtype=tf.int64)

    marks = make_neighborhood2D(cords, con=3, origin_shape=image.shape)
    marks = tf.reshape(marks, shape=(-1, 4))

    # r = 3
    # ax = tf.range(-r, r + 1, dtype=tf.int64)
    # x, y = tf.meshgrid(ax, ax)
    # ax = tf.cast(tf.where(x ** 2 + y ** 2 <= r ** 2, 1, 0), tf.float32)

    marked = tf.scatter_nd(marks, tf.ones(shape=(marks.get_shape()[0],), dtype=tf.float32) * 255.0, shape=image.shape)
    marked_del = math_ops.abs(marked - 255.0) / 255.0
    img_mark = image * marked_del

    marked = tf.concat((marked, tf.zeros_like(image), tf.zeros_like(image)), axis=-1)

    img_mark = img_mark + marked
    show_images(img_mark, 1, 1)


class OctavePyramid(list):
    def __init__(self):
        super().__init__()
        kx = tf.constant([[0.0, 0.0, 0.0], [-1.0, 0.0, 1.0], [0.0, 0.0, 0.0]], shape=(3, 3, 1, 1, 1), dtype=tf.float32)
        ky = tf.constant([[0.0, -1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]], shape=(3, 3, 1, 1, 1), dtype=tf.float32)
        self.gradient_kernel = tf.concat((kx, ky), axis=-1)

    def append(self, L_yxs: tf.Tensor) -> None:
        dL_yxs = tf.nn.convolution(tf.expand_dims(L_yxs, -1), self.gradient_kernel, padding='VALID')
        dx, dy = tf.unstack(dL_yxs, 2, axis=-1)
        # M[batch, x, y, s]
        magnitude = math_ops.sqrt(dx * dx + dy * dy)
        # Theta[batch, x, y, s]
        orientation = math_ops.atan2(dy, dx) * (180.0 / PI)
        oc = self.Octave(len(self), L_yxs, dx, dy, magnitude, orientation)
        super().append(oc)

    @dataclass(eq=False, order=False)
    class Octave:
        __slots__ = ('index', 'gaussian', 'dx', 'dy', 'magnitude', 'orientation')
        index: int
        gaussian: tf.Tensor
        dx: tf.Tensor
        dy: tf.Tensor
        magnitude: tf.Tensor
        orientation: tf.Tensor

        def __post_init__(self):
            if not isinstance(self.gaussian, tf.Tensor):
                raise ValueError('All the fields need to be type of Tensor')
            _batch_shape = self.gaussian.get_shape()[0]
            for X in [self.dx, self.dy, self.magnitude, self.orientation]:
                if not isinstance(X, tf.Tensor) or X.get_shape()[0] != _batch_shape:
                    raise ValueError('All the fields need to be type of Tensor with same batch size')

        @property
        def shape(self):
            return self.gaussian.shape.as_list()


@dataclass(eq=False, order=False)
class KeyPoints:
    pt: tf.Tensor = tf.constant([[]], shape=(0, 4), dtype=tf.float32)
    size: tf.Tensor = tf.constant([[]], shape=(0, 1), dtype=tf.float32)
    angle: tf.Tensor = tf.constant([[]], shape=(0, 1), dtype=tf.float32)
    octave: tf.Tensor = tf.constant([[]], shape=(0, 1), dtype=tf.float32)
    response: tf.Tensor = tf.constant([[]], shape=(0, 1), dtype=tf.float32)

    def __add__(self, other):
        if not isinstance(other, KeyPoints): raise TypeError
        if other.shape[0] == 0: return KeyPoints(self.pt, self.size, self.angle, self.octave, self.response)
        if self.shape[0] == 0: return KeyPoints(other.pt, other.size, other.angle, other.octave, other.response)
        key_points = KeyPoints(
            pt=tf.concat((self.pt, other.pt), 0),
            size=tf.concat((self.size, other.size), 0),
            angle=tf.concat((self.angle, other.angle), 0),
            octave=tf.concat((self.octave, other.octave), 0),
            response=tf.concat((self.response, other.response), 0)
        )
        return key_points

    def __post_init__(self):
        if not isinstance(self.pt, tf.Tensor):
            raise ValueError('All the fields need to be type of tf.Tensor')
        _shape = self.pt.get_shape().as_list()
        if len(_shape) > 2:
            self.pt = tf.squeeze(self.pt)
            _shape = self.pt.get_shape().as_list()
        if len(_shape) != 2 or _shape[-1] != 4: raise ValueError('expected "pt" to be 2D tensor with size of (None, 4)')
        for f in [self.size, self.angle, self.octave, self.response]:
            if not isinstance(f, tf.Tensor) or f.shape[0] != _shape[0]:
                raise ValueError('All the fields need to be type of Tensor with the same first dim size')

    @property
    def shape(self) -> tuple:
        _len = (self.pt.shape[0],)
        return _len

    def to_image_size(self, unpack_octave: bool = False):
        if self.shape[0] == 9: return None
        pt_unpack = self.pt * tf.constant([1.0, 0.5, 0.5, 1.0], dtype=tf.float32)
        size_unpack = self.size * 0.5
        octave_unpack = bitwise_ops.bitwise_xor(tf.cast(self.octave, dtype=tf.int64), 255)

        unpack_key_points = KeyPoints(
            pt_unpack, size_unpack, self.angle, tf.cast(octave_unpack, dtype=tf.float32), self.response
        )
        if not unpack_octave:
            return unpack_key_points

        octave = bitwise_ops.bitwise_and(octave_unpack, 255)
        layer = bitwise_ops.right_shift(octave_unpack, 8)
        layer = bitwise_ops.bitwise_and(layer, 255)
        octave = tf.where(octave >= 128, bitwise_ops.bitwise_or(octave, -128), octave)
        scale = tf.where(
            octave >= 0,
            tf.cast(1 / tf.bitwise.left_shift(1, octave), dtype=tf.float32),
            tf.cast(tf.bitwise.left_shift(1, -octave), dtype=tf.float32)
        )
        unpacked_octave = namedtuple('unpacked_octave', 'octave, layer, scale')
        unpacked_octave = unpacked_octave(tf.cast(octave, dtype=tf.float32), tf.cast(layer, dtype=tf.float32), scale)
        return unpack_key_points, unpacked_octave

    def stack(self) -> tf.Tensor:
        _stack = tf.concat(([self.pt, self.size, self.angle, self.octave, self.response]), axis=-1)
        return _stack


class Detector:
    def __init__(self, sigma: float = 1.6, assume_blur_sigma: float = 0.5, n_intervals: int = 3,
                 n_octaves: Union[int, None] = None, border_width: int = 5, convergence_iter: int = 5):
        self.__inputs_shape = None
        self.sigma = sigma
        self.assume_blur_sigma = assume_blur_sigma
        self.n_intervals = n_intervals
        self.n_octaves = n_octaves
        self.border_width = (border_width, border_width, 0)
        self.convergence_N = convergence_iter
        self.pyramid = OctavePyramid()
        self.pyramid_kernels = []
        self.key_points = KeyPoints()

    def __call__(self, inputs: tf.Tensor):
        return self.call(inputs)

    def __validate_input(self, inputs: tf.Tensor) -> tf.Tensor:
        _shape = inputs.get_shape().as_list()
        _ndims = len(_shape)
        if _ndims != 4 or _shape[-1] != 1:
            raise ValueError(
                'expected the inputs to be grayscale images with size of (None, h, w, 1)'
            )
        self.__inputs_shape = _shape
        inputs = tf.cast(inputs, dtype=tf.float32)
        return inputs

    def __init_graph(self, inputs_shape: Union[tf.Tensor, list, tuple]):
        self.pyramid = OctavePyramid()
        self.key_points = KeyPoints()
        _, h_, w_, _ = inputs_shape
        delta_sigma = (self.sigma ** 2) - ((2 * self.assume_blur_sigma) ** 2)
        delta_sigma = math_ops.sqrt(tf.maximum(delta_sigma, 0.64))

        base_kernel = gaussian_kernel(kernel_size=0, sigma=delta_sigma)
        base_kernel = tf.expand_dims(tf.expand_dims(base_kernel, axis=-1), axis=-1)

        images_per_octaves = self.n_intervals + 3
        K = 2 ** (1 / self.n_intervals)
        K = tf.cast(K, dtype=tf.float32)

        self.pyramid_kernels = [base_kernel]

        for i in range(1, images_per_octaves):
            s_prev = self.sigma * (K ** (i - 1))
            s = math_ops.sqrt((K * s_prev) ** 2 - s_prev ** 2)
            kernel_ = gaussian_kernel(kernel_size=0, sigma=s)
            self.pyramid_kernels.append(tf.expand_dims(tf.expand_dims(kernel_, axis=-1), axis=-1))

        min_shape = int(self.pyramid_kernels[-1].get_shape()[0])
        max_n_octaves = self.compute_default_N_octaves(height=h_ * 2, weight=w_ * 2, min_shape=min_shape)

        if self.n_octaves is not None and max_n_octaves > self.n_octaves:
            max_n_octaves = self.n_octaves
        self.n_octaves = max_n_octaves

    def localize_extrema(self, octave: OctavePyramid.Octave) -> KeyPoints:
        dim = octave.shape[-1]
        con, extrema_offset, contrast_threshold, eigen_ration = 3, 0.5, 0.03, 10
        octave_index = octave.index

        """Extract all the extrema point in the octave scale space"""
        # D(batch, y, x, s)
        dog = math_ops.subtract(tf.split(octave.gaussian, [1, dim - 1], -1)[1],
                                tf.split(octave.gaussian, [dim - 1, 1], -1)[0])
        dog_shape = dog.get_shape().as_list()

        # e = (batch, y, x, s) (local extrema)
        extrema = compute_extrema3D(
            tf.round(dog), con=con, border_width=[w - 2 if w != 0 else 0 for w in self.border_width]
        )

        dog = dog / 255.0

        """Compute the key points conditions for all the image"""
        # DD / Dx
        grad = compute_central_gradient3D(dog)
        grad = tf.expand_dims(grad, -1)

        # D^2D / Dx^2
        hess = compute_hessian_3D(dog)

        # X' = - (D^2D / Dx^2) * (DD / Dx)
        extrema_update = - linalg_ops.lstsq(hess, grad, l2_regularizer=0.0, fast=False)
        extrema_update = tf.squeeze(extrema_update, axis=-1)

        # (DD / Dx) * X'
        dot_ = linalg_ops.matmul(tf.expand_dims(extrema_update, 4), grad)
        dot_ = tf.squeeze(tf.squeeze(dot_, -1), -1)

        mid_cube_values = tf.slice(dog, [0, 1, 1, 1],
                                   [dog_shape[0], dog_shape[1] - 2, dog_shape[2] - 2, dog_shape[3] - 2])

        # D(X') = D + 0.5 * (DD / Dx) * X'
        update_response = mid_cube_values + 0.5 * dot_

        hess_shape = hess.get_shape().as_list()
        # H[[Dxx, Dxy], [Dyx, Dyy]]
        hess_xy = tf.slice(hess, [0, 0, 0, 0, 0, 0], [*hess_shape[:-2], 2, 2])
        # Dxx + Dyy
        hess_xy_trace = linalg_ops.trace(hess_xy)
        # Dxx * Dyy - Dxy * Dyx
        hess_xy_det = linalg_ops.det(hess_xy)

        # |X'| <= 0.5
        # (X' is larger than 0.5 in any dimension, means that the extreme lies closer to a different sample point)
        kp_cond1 = math_ops.less_equal(math_ops.reduce_max(math_ops.abs(extrema_update), axis=-1), extrema_offset)

        # |D(X')| >= 0.03 (threshold on minimum contrast)
        kp_cond2 = math_ops.greater_equal(math_ops.abs(update_response), contrast_threshold)

        # (Dxx + Dyy) ^ 2 / Dxx * Dyy - Dxy * Dyx < (r + 1) ^ 2 / r
        # ---> ((Dxx + Dyy) ^ 2) * r < (Dxx * Dyy - Dxy * Dyx) * ((r + 1) ^ 2)
        # (threshold on ratio of principal curvatures)
        kp_cond3 = math_ops.logical_and(
            eigen_ration * (hess_xy_trace ** 2) < ((eigen_ration + 1) ** 2) * hess_xy_det, hess_xy_det != 0
        )
        cond = tf.where(kp_cond1 & kp_cond2 & kp_cond3, True, False)

        kp_cond4 = tf.scatter_nd(extrema, tf.ones((extrema.shape[0],), dtype=tf.bool), dog_shape)
        kp_cond4 = tf.slice(kp_cond4, [0, 1, 1, 1],
                            [dog_shape[0], dog_shape[1] - 2, dog_shape[2] - 2, dog_shape[3] - 2])

        """Localize the extrema points"""
        sure_key_points = math_ops.logical_and(cond, kp_cond4)
        attempts = math_ops.logical_and(kp_cond4, ~sure_key_points)

        shape_ = sure_key_points.get_shape().as_list()

        for _ in range(self.convergence_N):
            attempts_cords = tf.where(attempts)
            if attempts_cords.shape[0] == 0: break
            # if ist only one point the shape will bw (4, )
            attempts_cords = tf.reshape(attempts_cords, (-1, 4))
            attempts_update = tf.gather_nd(extrema_update, attempts_cords)

            ex, ey, ez = tf.unstack(attempts_update, num=3, axis=-1)
            cd, cy, cx, cz = tf.unstack(tf.cast(attempts_cords, tf.float32), num=4, axis=1)
            attempts_next = [cd, cy + ey, cx + ex, cz + ez]

            # check that the new cords will lie within the image shape
            cond_next = tf.where(
                (attempts_next[1] >= 0) & (attempts_next[1] < shape_[1]) & (attempts_next[2] > 0) & (
                        attempts_next[2] < shape_[2]) & (attempts_next[3] > 0) & (
                        attempts_next[3] < shape_[3]))

            attempts_next = tf.stack(attempts_next, -1)
            attempts_next = tf.cast(tf.gather(attempts_next, tf.squeeze(cond_next)), dtype=tf.int64)
            if attempts_next.shape[0] == 0: break
            attempts_next = tf.reshape(attempts_next, (-1, 4))

            attempts_mask = tf.scatter_nd(attempts_next, tf.ones((attempts_next.shape[0],), dtype=tf.bool), shape_)

            # add new key points
            new_cords = tf.where(attempts_mask & ~sure_key_points & cond)
            sure_key_points = tf.tensor_scatter_nd_update(sure_key_points, new_cords,
                                                          tf.ones((new_cords.shape[0],), dtype=tf.bool))
            # next points
            attempts = math_ops.logical_and(attempts_mask, ~sure_key_points)

        """Construct the key points"""
        cords = tf.where(sure_key_points)
        if cords.shape[0] == 0: return KeyPoints()
        kp_cords = cords + tf.constant([[0, 1, 1, 1]], dtype=tf.int64)

        # X' = - (D^2D / Dx^2) * (DD / Dx)
        extrema_update = tf.gather_nd(extrema_update, cords)
        octave_index = tf.cast(octave_index, dtype=tf.float32)

        # x', y', s'
        ex, ey, ez = tf.unstack(extrema_update, num=3, axis=1)

        # batch, y, x, s
        cd, cy, cx, cz = tf.unstack(tf.cast(kp_cords, tf.float32), num=4, axis=1)

        # pt = (batch, y = (y + y') * (1 << octave), (x + x') * (1 << octave), s) points in size of octave 0
        kp_pt = tf.stack(
            (cd, (cy + ey) * (2 ** octave_index), (cx + ex) * (2 ** octave_index), cz), axis=-1
        )
        # octave = octave_index + s * (1 << 8) + round((s' + 0.5) * 255) * (1 << 16)
        kp_octave = octave_index + cz * (2 ** 8) + tf.round((ez + 0.5) * 255.0) * (2 ** 16)

        # size = (sigma << ((s + s') / sn)) << (octave_index + 1)
        kp_size = self.sigma * (2 ** ((cz + ez) / tf.cast(self.n_intervals, dtype=tf.float32))) * (
                2 ** (octave_index + 1.0))

        # D(X') = D + 0.5 * (DD / Dx) * X'
        kp_response = math_ops.abs(tf.gather_nd(update_response, cords))

        key_points = KeyPoints(
            pt=tf.reshape(kp_pt, (-1, 4)),
            size=tf.reshape(kp_size, (-1, 1)),
            angle=tf.reshape(tf.ones_like(kp_size) * -1.0, (-1, 1)),
            octave=tf.reshape(kp_octave, (-1, 1)),
            response=tf.reshape(kp_response, (-1, 1))
        )
        return key_points

    def orientation_assignment(self, octave: OctavePyramid.Octave, key_points: KeyPoints) -> KeyPoints:
        orientation_N_bins, scale_factor, radius_factor = 36, 1.5, 3
        histogram = tf.zeros((key_points.shape[0], orientation_N_bins), dtype=tf.float32)

        # scale = 1.5 * sigma  * (1 << ((s + s') / sn)
        scale = scale_factor * key_points.size / (2 ** (octave.index + 1))

        # r[N_points, ] = 3 * scale
        radius = tf.cast(tf.round(radius_factor * scale), dtype=tf.int64)

        # wf[N_points, ]
        weight_factor = -0.5 / (scale ** 2)

        # points back to octave resolution
        _prob = 1.0 / (1 << octave.index)
        _one = tf.ones_like(_prob)
        _prob = tf.stack((_one, _prob, _prob, _one), axis=-1)
        _prob = tf.squeeze(_prob)

        # [batch, x + x', y + y', s] * N_points
        region_center = tf.cast(key_points.pt * _prob, dtype=tf.int64)

        # check that the radius in the image size
        _, y, x, _ = tf.split(region_center, [1] * 4, -1)
        radius = math_ops.minimum(
            math_ops.minimum(octave.shape[1] - 3 - y, octave.shape[2] - 3 - x),
            math_ops.minimum(math_ops.minimum(y, x), radius)
        )
        radius = tf.reshape(radius, (-1,))

        # parallel computation
        parallel = tf.unique(radius)
        split_region = tf.dynamic_partition(
            tf.concat((tf.cast(region_center, tf.float32), weight_factor), -1), parallel.idx,
            tf.reduce_max(parallel.idx) + 1
        )
        index = tf.dynamic_partition(tf.reshape(tf.range(key_points.shape[0], dtype=tf.int64), (-1, 1)),
                                     parallel.idx, tf.reduce_max(parallel.idx) + 1)

        for region_weight, r, hist_index in zip(split_region, parallel.y, index):
            region, weight = tf.split(region_weight, [4, 1], -1)

            neighbor = make_neighborhood2D(tf.constant([[0, 0, 0, 0]], dtype=tf.int64), con=(r * 2) + 1)
            block = tf.expand_dims(tf.cast(region, tf.int64), axis=1) + neighbor

            magnitude = tf.gather_nd(octave.magnitude, tf.reshape(block, (-1, 4)))
            orientation = tf.gather_nd(octave.orientation, tf.reshape(block, (-1, 4)))

            _, curr_y, curr_x, _ = tf.unstack(tf.cast(neighbor, dtype=tf.float32), 4, axis=-1)
            weight = tf.reshape(math_ops.exp(weight * (curr_y ** 2 + curr_x ** 2)), (-1,))

            hist_deg = tf.cast(tf.round(orientation * orientation_N_bins / 360.), dtype=tf.int64) % orientation_N_bins

            hist_index = tf.ones(block.get_shape()[:-1], dtype=tf.int64) * tf.reshape(hist_index, (-1, 1))
            hist_index = tf.stack((tf.reshape(hist_index, (-1,)), hist_deg), -1)
            histogram = tf.tensor_scatter_nd_add(histogram, hist_index, weight * magnitude)

        """ find peaks in the histogram """
        # histogram smooth
        gaussian1D = tf.constant([1, 4, 6, 4, 1], dtype=tf.float32) / 16.0
        gaussian1D = tf.reshape(gaussian1D, shape=(-1, 1, 1))

        pad_ = tf.split(tf.expand_dims(histogram, axis=-1), [2, orientation_N_bins - 4, 2], 1)
        pad_ = tf.concat([pad_[-1], *pad_, pad_[0]], 1)

        smooth_histogram = tf.nn.convolution(pad_, gaussian1D, padding='VALID')
        smooth_histogram = tf.squeeze(smooth_histogram, axis=-1)

        orientation_max = tf.reduce_max(smooth_histogram, axis=-1)

        peak = tf.nn.max_pool1d(tf.expand_dims(smooth_histogram, -1), ksize=3, padding="SAME", strides=1)
        peak = tf.squeeze(peak, -1)

        value_cond = tf.repeat(tf.reshape(orientation_max, shape=(-1, 1)), repeats=36, axis=-1) * 0.8

        peak = tf.where((peak == smooth_histogram) & (smooth_histogram > value_cond))

        p_idx, p_deg = tf.unstack(peak, num=2, axis=-1)

        # interpolate the peak position - parabola
        kernel = tf.constant([1., 0, -1.], shape=(3, 1, 1))
        kernel = tf.concat((kernel, tf.constant([1., -2., 1.], shape=(3, 1, 1))), -1)

        pad_ = tf.split(smooth_histogram, [1, 34, 1], -1)
        pad_ = tf.concat([pad_[-1], *pad_, pad_[0]], -1)

        interp = tf.unstack(tf.nn.convolution(tf.expand_dims(pad_, -1), kernel, padding="VALID"), 2, -1)
        interp = 0.5 * (interp[0] / interp[1]) % 36
        interp = tf.cast(p_deg, tf.float32) + tf.gather_nd(interp, peak)

        orientation = 360. - interp * 360. / 36

        orientation = tf.where(orientation - 360. < 1e-7, 0.0, orientation)

        wrap = key_points.stack()
        wrap = tf.gather(wrap, p_idx)
        pt, size, _, octave, response = tf.split(wrap, [4, 1, 1, 1, 1, ], axis=-1)

        key_points_new = KeyPoints(
            pt=pt,
            size=size,
            angle=tf.reshape(orientation, (-1, 1)),
            octave=octave,
            response=response
        )

        return key_points_new

    @staticmethod
    def compute_default_N_octaves(height: int, weight: int, min_shape: int = 0) -> int:
        s_ = tf.cast(min([height, weight]), dtype=tf.float32)
        diff = math_ops.log(s_)
        if min_shape > 1:
            diff = diff - math_ops.log(tf.cast(min_shape, dtype=tf.float32))

        n_octaves = tf.round(diff / math_ops.log(2.0)) + 1
        return int(n_octaves)

    def call(self, inputs: tf.Tensor):
        # I[batch, y, x, 1]
        inputs = self.__validate_input(inputs)
        self.__init_graph(self.__inputs_shape)
        _, h_, w_, _ = self.__inputs_shape

        def conv_with_pad(x, h):
            k_ = h.get_shape()[0] // 2
            x = tf.pad(x, tf.constant([[0, 0], [k_, k_], [k_, k_], [0, 0]], tf.int32), 'SYMMETRIC')
            return tf.nn.convolution(x, h, padding='VALID')

        G_yxs = self.pyramid_kernels

        # I[batch, H, W, 1] --> I[batch, 2 * H, 2 * W, 1]
        I = image_ops.resize(inputs, size=[h_ * 2, w_ * 2], method='bilinear')
        I = conv_with_pad(I, G_yxs[0])

        size_ = [h_, w_]
        for oc_id in range(self.n_octaves):
            oc_cap = [I]
            for kernel in G_yxs[1:]:
                I = conv_with_pad(I, kernel)
                oc_cap.append(I)
            if oc_id < self.n_octaves - 1:
                I = image_ops.resize(oc_cap[-3], size=size_, method='nearest')
                size_[0] //= 2
                size_[1] //= 2
            self.pyramid.append(tf.concat(oc_cap, -1))
            oc = self.pyramid[-1]
            oc_kp = self.localize_extrema(oc)
            if oc_kp.shape[0] == 0: continue
            self.key_points += self.orientation_assignment(oc, oc_kp)
        return self.key_points


if __name__ == '__main__':
    image = load_image('luka1.jpeg')
    alg = Detector()

    kp = alg(image)

    # show_key_points(kp.to_image_size(), image)
