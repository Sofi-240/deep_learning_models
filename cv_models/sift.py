from typing import Union
import tensorflow as tf
from cv_models.utils import PI, gaussian_kernel, compute_extrema3D, make_neighborhood3D, \
    make_neighborhood2D, compute_central_gradient3D, compute_hessian_3D
from tensorflow.python.keras import backend
from collections import namedtuple
from dataclasses import dataclass, field


backend.set_floatx('float32')
linalg_ops = tf.linalg
math_ops = tf.math
bitwise_ops = tf.bitwise
image_ops = tf.image


@dataclass(eq=False, order=False)
class KeyPoints:
    __slots__ = ('pt', 'size', 'angle', 'octave', 'octave_id', 'response')
    pt: tf.Tensor
    size: tf.Tensor
    angle: tf.Tensor
    octave: tf.Tensor
    octave_id: tf.Tensor
    response: tf.Tensor

    def __post_init__(self):
        if not isinstance(self.pt, tf.Tensor):
            raise ValueError('All the fields need to be type of Tensor')
        _shape = self.pt.get_shape().as_list()
        if len(_shape) > 2:
            self.pt = tf.squeeze(self.pt)
            _shape = self.pt.get_shape().as_list()
        if len(_shape) != 2 or _shape[-1] != 4: raise ValueError('expected "pt" to be 2D tensor with size of (None, 4)')
        for f in [self.size, self.angle, self.octave, self.octave_id, self.response]:
            if not isinstance(f, tf.Tensor) or f.shape[0] != _shape[0]:
                raise ValueError('All the fields need to be type of Tensor with the same first dim size')

    @property
    def shape(self):
        _len = (self.pt.shape[0],)
        return _len

    def to_image_size(self, unpack_octave: bool = False):
        pt_unpack = self.pt * tf.constant([1.0, 0.5, 0.5, 1.0], dtype=tf.float32)
        size_unpack = self.size * 0.5
        octave_unpack = bitwise_ops.bitwise_xor(tf.cast(self.octave, dtype=tf.int64), 255)

        unpack_key_points = KeyPoints(
            pt_unpack, size_unpack, self.angle, tf.cast(octave_unpack, dtype=tf.float32), self.octave_id, self.response
        )
        if not unpack_octave:
            return unpack_key_points

        octave = bitwise_ops.bitwise_and(octave_unpack, 255)
        layer = bitwise_ops.right_shift(octave_unpack, 8)
        layer = bitwise_ops.bitwise_and(layer, 255)
        octave = tf.where(octave >= 128, bitwise_ops.bitwise_or(octave, -128), octave)
        # scale = 1.0 / tf.cast(tf.bitwise.left_shift(1, math_ops.abs(octave)), dtype=tf.float32)
        # scale = tf.where(octave == -1, 1 / scale, scale)
        scale = tf.where(
            octave >= 1,
            tf.cast(1 / tf.bitwise.left_shift(1, octave), dtype=tf.float32),
            tf.cast(tf.bitwise.left_shift(1, -octave), dtype=tf.float32)
        )
        unpacked_octave = namedtuple('unpacked_octave', 'octave, layer, scale')
        unpacked_octave = unpacked_octave(tf.cast(octave, dtype=tf.float32), tf.cast(layer, dtype=tf.float32), scale)
        return unpack_key_points, unpacked_octave

    def stack(self):
        _stack = tf.concat(([self.pt, self.size, self.angle, self.octave, self.octave_id, self.response]), axis=-1)
        return _stack


@dataclass(eq=False, order=False, frozen=True)
class Octave:
    __slots__ = ('gaussian', 'dx', 'dy', 'magnitude', 'orientation')
    gaussian: tf.Tensor
    dx: tf.Tensor
    dy: tf.Tensor
    magnitude: tf.Tensor
    orientation: tf.Tensor

    def __post_init__(self):
        if not isinstance(self.gaussian, tf.Tensor):
            raise ValueError('All the fields need to be type of Tensor')
        _batch_shape = self.gaussian.get_shape()[0]
        for field in [self.dx, self.dy, self.magnitude, self.orientation]:
            if not isinstance(field, tf.Tensor) or field.get_shape()[0] != _batch_shape:
                raise ValueError('All the fields need to be type of Tensor with same batch size')

    @property
    def shape(self):
        return self.gaussian.shape.as_list()


@dataclass(eq=False, order=False)
class Argumentor:
    sigma: float = 1.4
    assume_blur_sigma: float = 0.5
    n_intervals: int = 4
    n_iterations: int = 5
    n_octaves: int = 4
    border_width: tuple = (5, 5, 0)
    orientation_N_bins: int = field(default=36, init=False)
    eigen_ration: int = field(default=10, init=False)
    peak_ratio: float = field(default=0.8, init=False)
    contrast_threshold: float = field(default=0.04, init=False)
    scale_factor: float = field(default=1.5, init=False)
    extrema_offset: float = field(default=0.5, init=False)
    radius_factor: int = field(default=3, init=False)
    con: int = field(default=3, init=False)
    descriptors_N_bins: int = field(default=8, init=False)
    window_width: int = field(default=4, init=False)
    scale_multiplier: int = field(default=3, init=False)
    descriptor_max_value: float = field(default=0.2, init=False)


def uint8(X: tf.Tensor) -> tf.Tensor:
    return tf.cast(X, dtype=tf.uint8)


def float32(X: tf.Tensor) -> tf.Tensor:
    return tf.cast(X, dtype=tf.float32)


class SIFT:

    def __init__(self,
                 sigma: float = 1.4, assume_blur_sigma: float = 0.5, n_intervals: int = 4,
                 n_octaves: Union[int, None] = 4, n_iterations: int = 5, name: Union[str, None] = None):
        self.__inputs_shape = None
        self.__build = False
        self.name = name or 'SIFT'
        self.epsilon = 1e-07
        self.graph_args = Argumentor(
            sigma=sigma, assume_blur_sigma=assume_blur_sigma, n_intervals=n_intervals,
            n_iterations=n_iterations, n_octaves=n_octaves
        )
        self.base_kernel = None
        self.gradient_kernel = None
        self.gaussian_kernels = []
        self.octave_pyramid = []
        self.key_points = None
        self.descriptors_vectors = None

    def __validate_input(self, inputs: tf.Tensor) -> tf.Tensor:
        _shape = inputs.get_shape()
        _ndims = len(_shape)
        if _ndims != 4 or _shape[-1] != 1:
            raise ValueError(
                'expected the inputs to be grayscale images with size of (None, h, w, 1)'
            )
        self.__inputs_shape = _shape
        inputs = tf.cast(inputs, dtype=tf.float32)
        return inputs

    def __init_graph(self):
        _, _h, _w, _ = self.__inputs_shape
        self.key_points = None
        self.descriptors_vectors = None
        self.octave_pyramid = []
        self.gaussian_kernels = []

        delta_sigma = (self.graph_args.sigma ** 2) - ((2 * self.graph_args.assume_blur_sigma) ** 2)
        delta_sigma = math_ops.sqrt(tf.maximum(delta_sigma, 0.64))

        kernel = gaussian_kernel(kernel_size=0, sigma=delta_sigma)
        kernel = tf.expand_dims(tf.expand_dims(kernel, axis=-1), axis=-1)
        self.base_kernel = tf.Variable(kernel, trainable=False, name='base_kernel')

        images_per_octaves = self.graph_args.n_intervals + 3
        K = 2 ** (1 / self.graph_args.n_intervals)
        K = tf.cast(K, dtype=tf.float32)

        sigma_prev = (K ** tf.cast(tf.range(images_per_octaves - 1), dtype=tf.float32)) * self.graph_args.sigma
        sigmas = math_ops.sqrt((K * sigma_prev) ** 2 - sigma_prev ** 2)
        sigmas = math_ops.maximum(sigmas, 0.8)

        for i, s in enumerate(sigmas):
            kernel_ = gaussian_kernel(kernel_size=0, sigma=s)
            kernel_ = tf.expand_dims(tf.expand_dims(kernel_, axis=-1), axis=-1)
            self.gaussian_kernels.append(tf.Variable(kernel_, trainable=False, name=f'kernel{i}'))

        kx = tf.constant([[0.0, 0.0, 0.0], [-1.0, 0.0, 1.0], [0.0, 0.0, 0.0]], shape=(3, 3, 1, 1), dtype=tf.float32)
        ky = tf.constant([[0.0, -1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]], shape=(3, 3, 1, 1), dtype=tf.float32)
        self.gradient_kernel = tf.Variable(tf.concat((kx, ky), axis=-1), trainable=False, name='gradient_kernel')

        min_shape = int(self.gaussian_kernels[-1].get_shape()[0])
        max_n_octaves = self.compute_default_N_octaves(height=_h * 2, weight=_w * 2, min_shape=min_shape)

        if self.graph_args.n_octaves is not None and max_n_octaves > self.graph_args.n_octaves:
            max_n_octaves = tf.cast(self.graph_args.n_octaves, dtype=tf.int32)
        self.graph_args.n_octaves = max_n_octaves

    def __localize_extrema(self, middle_pixel_cords: tf.Tensor, dog: tf.Tensor,
                           octave_index: Union[int, float]) -> tuple[Union[tf.Tensor, None], Union[tf.Tensor, None]]:

        dog_shape_ = dog.shape
        args = self.graph_args
        cube_neighbor = make_neighborhood3D(middle_pixel_cords, con=args.con, origin_shape=dog_shape_)
        if cube_neighbor.shape[0] == 0:
            return None, None

        # the size can change because the make_neighborhood3D return only the valid indexes
        mid_index = (args.con ** 3) // 2
        _, middle_pixel_cords, _ = tf.split(cube_neighbor, [mid_index, 1, mid_index], axis=1)
        middle_pixel_cords = tf.reshape(middle_pixel_cords, shape=(-1, 4))

        cube_values = tf.gather_nd(dog, tf.reshape(cube_neighbor, shape=(-1, 4))) / 255.0
        cube_values = tf.reshape(cube_values, (-1, args.con, args.con, args.con))

        cube_len_ = cube_values.get_shape()[0]

        mid_cube_values = tf.slice(cube_values, [0, 1, 1, 1], [cube_len_, 1, 1, 1])
        mid_cube_values = tf.reshape(mid_cube_values, shape=(-1,))

        grad = compute_central_gradient3D(cube_values)
        grad = tf.reshape(grad, shape=(-1, 3, 1))

        hess = compute_hessian_3D(cube_values)
        hess = tf.reshape(hess, shape=(-1, 3, 3))

        extrema_update = - linalg_ops.lstsq(hess, grad, l2_regularizer=0.0, fast=False)
        extrema_update = tf.squeeze(extrema_update, axis=-1)

        next_step_cords, kp_cond_1 = self.__split_extrema_cond(extrema_update, middle_pixel_cords)

        dot_ = tf.reduce_sum(
            tf.multiply(
                extrema_update[:, tf.newaxis, tf.newaxis, ...],
                tf.transpose(grad, perm=(0, 2, 1))[:, tf.newaxis, ...]
            ), axis=-1, keepdims=False
        )
        dot_ = tf.reshape(dot_, shape=(-1,))
        update_response = mid_cube_values + 0.5 * dot_

        kp_cond_2 = math_ops.greater_equal(
            math_ops.abs(update_response) * int(args.n_intervals), args.contrast_threshold
        )

        hess_xy = tf.slice(hess, [0, 0, 0], [cube_len_, 2, 2])
        hess_xy_trace = linalg_ops.trace(hess_xy)
        hess_xy_det = linalg_ops.det(hess_xy)

        kp_cond_3 = math_ops.logical_and(
            math_ops.greater(hess_xy_det, 0.0),
            math_ops.less(args.eigen_ration * (hess_xy_trace ** 2), ((args.eigen_ration + 1) ** 2) * hess_xy_det)
        )
        sure_key_points = math_ops.logical_and(math_ops.logical_and(kp_cond_1, kp_cond_2), kp_cond_3)

        # -------------------------------------------------------------------------
        kp_extrema_update = tf.boolean_mask(extrema_update, sure_key_points)

        if kp_extrema_update.shape[0] == 0:
            return next_step_cords, None

        kp_cords = tf.cast(tf.boolean_mask(middle_pixel_cords, sure_key_points), dtype=tf.float32)
        octave_index = tf.cast(octave_index, dtype=tf.float32)

        ex, ey, ez = tf.unstack(kp_extrema_update, num=3, axis=1)
        cd, cy, cx, cz = tf.unstack(kp_cords, num=4, axis=1)

        kp_pt = tf.stack(
            (cd, (cy + ey) * (2 ** octave_index), (cx + ex) * (2 ** octave_index), cz), axis=-1
        )

        kp_octave = octave_index + cz * (2 ** 8) + tf.round((ez + 0.5) * 255.0) * (2 ** 16)

        kp_size = args.sigma * (2 ** ((cz + ez) / tf.cast(args.n_intervals, dtype=tf.float32))) * (
                2 ** (octave_index + 1.0))
        kp_response = math_ops.abs(tf.boolean_mask(update_response, sure_key_points))

        octave_index = tf.ones_like(kp_size) * octave_index
        angle = tf.ones_like(kp_size) * -1.0
        key_point_wrap = [
            tf.reshape(kp_pt, (-1, 4)),
            tf.reshape(kp_size, (-1, 1)),
            tf.reshape(angle, (-1, 1)),
            tf.reshape(kp_octave, (-1, 1)),
            tf.reshape(octave_index, (-1, 1)),
            tf.reshape(kp_response, (-1, 1))

        ]
        key_point_wrap = tf.concat(key_point_wrap, axis=-1)
        return next_step_cords, key_point_wrap

    def __split_extrema_cond(self, extrema: tf.Tensor, current_cords: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        shape_ = extrema.get_shape()
        assert shape_[-1] == 3
        assert len(shape_) == 2

        cond = math_ops.less(math_ops.reduce_max(math_ops.abs(extrema), axis=-1), self.graph_args.extrema_offset)
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
        return next_step_cords, cond

    def __compute_histogram(self, key_points: KeyPoints, octaves: list[Octave]) -> tf.Tensor:
        args = self.graph_args

        histogram = tf.zeros(shape=(key_points.pt.shape[0], args.orientation_N_bins), dtype=tf.float32)

        scale = args.scale_factor * key_points.size / (2 ** key_points.octave_id + 1)

        radius = tf.cast(tf.round(args.radius_factor * scale), dtype=tf.int32)
        weight_factor = -0.5 / (scale ** 2)

        _prob = 1.0 / (2 ** key_points.octave_id)
        _one = tf.ones_like(_prob)
        _prob = tf.stack((_one, _prob, _prob, _one), axis=-1)
        _prob = tf.squeeze(_prob)

        region_center = tf.cast(key_points.pt * _prob, dtype=tf.int64) - tf.constant([[0, 1, 1, 0]], dtype=tf.int64)

        con = tf.reshape((radius * 2) + 1, (-1,))
        octave_id = tf.reshape(tf.cast(key_points.octave_id, tf.int32), (-1,))

        oct_shapes = [tf.reshape(a.shape, shape=(1, 4)) for a in octaves]
        oct_shapes = tf.gather(tf.concat(oct_shapes, axis=0), octave_id)

        maximum_con = self.__compute_max_con(region_center, oct_shapes)
        con = math_ops.minimum(con, maximum_con)
        parallel = self.__compute_parallel_jobs(con, octave_id)

        for curr_parallel in parallel:
            curr_octave = octaves[curr_parallel.octave_id]
            curr_neighbor = make_neighborhood2D(tf.constant([[0, 0, 0, 0]], dtype=tf.int64), con=curr_parallel.con)

            curr_region = tf.gather(region_center, curr_parallel.idx)
            curr_block = tf.expand_dims(curr_region, axis=1) + curr_neighbor

            curr_magnitude = tf.gather_nd(curr_octave.magnitude, tf.reshape(curr_block, (-1, 4)))
            curr_orientation = tf.gather_nd(curr_octave.orientation, tf.reshape(curr_block, (-1, 4)))

            curr_neighbor = tf.cast(curr_neighbor, dtype=tf.float32)
            _, curr_y, curr_x, _ = tf.split(curr_neighbor, [1, 1, 1, 1], axis=-1)

            curr_weight = tf.gather(weight_factor, curr_parallel.idx)
            curr_weight = math_ops.exp(curr_weight * tf.squeeze((curr_y ** 2 + curr_x ** 2), axis=-1))

            curr_hist_index = tf.cast(tf.round(curr_orientation * args.orientation_N_bins / 360.), dtype=tf.int64)
            curr_hist_index = curr_hist_index % args.orientation_N_bins

            curr_hist_id = tf.ones_like(curr_weight, dtype=tf.int64) * tf.reshape(curr_parallel.idx, (-1, 1))
            curr_weight = tf.reshape(curr_weight, (-1,))

            curr_hist_index = tf.concat((tf.reshape(curr_hist_id, (-1, 1)), tf.reshape(curr_hist_index, (-1, 1))),
                                        axis=-1)

            histogram = tf.tensor_scatter_nd_add(histogram, curr_hist_index, curr_weight * curr_magnitude)

        gaussian1D = tf.constant([1, 4, 6, 4, 1], dtype=tf.float32) / 16.0
        gaussian1D = tf.reshape(gaussian1D, shape=(-1, 1, 1))

        histogram_pad = tf.pad(
            tf.expand_dims(histogram, axis=-1), paddings=tf.constant([[0, 0], [2, 2], [0, 0]], dtype=tf.int32),
            constant_values=0.0
        )

        smooth_histogram = tf.nn.convolution(histogram_pad, gaussian1D, padding='VALID')
        smooth_histogram = tf.squeeze(smooth_histogram, axis=-1)

        orientation_max = tf.reduce_max(smooth_histogram, axis=-1)
        value_cond = tf.repeat(tf.reshape(orientation_max, shape=(-1, 1)), repeats=args.orientation_N_bins, axis=-1)
        value_cond = value_cond * args.peak_ratio

        cond = math_ops.logical_and(
            math_ops.greater(smooth_histogram, tf.roll(smooth_histogram, shift=1, axis=1)),
            math_ops.greater(smooth_histogram, tf.roll(smooth_histogram, shift=-1, axis=1))
        )
        cond = math_ops.logical_and(
            cond, math_ops.greater_equal(smooth_histogram, value_cond)
        )

        peak_index = tf.where(cond)
        peak_value = tf.boolean_mask(smooth_histogram, cond)

        p_id, p_idx = tf.unstack(peak_index, num=2, axis=-1)

        left_index = (p_idx - 1) % args.orientation_N_bins
        right_index = (p_idx + 1) % args.orientation_N_bins

        left_value = tf.gather_nd(smooth_histogram, tf.stack((p_id, left_index), axis=-1))
        right_value = tf.gather_nd(smooth_histogram, tf.stack((p_id, right_index), axis=-1))

        interpolated_peak_index = ((tf.cast(p_idx, dtype=tf.float32) + 0.5 * (left_value - right_value)) / (
                left_value - (2 * peak_value) + right_value)) % args.orientation_N_bins

        orientation = 360. - interpolated_peak_index * 360. / args.orientation_N_bins

        orientation = tf.where(math_ops.less(math_ops.abs(orientation), 1e-7), 0.0, orientation)

        p_id = tf.reshape(p_id, (-1,))

        prev_key_stack = key_points.stack()
        prev_key_stack = tf.gather(prev_key_stack, p_id)
        prev_key_split = tf.split(prev_key_stack, [4, 1, 1, 1, 1, 1], axis=-1)

        key_points = KeyPoints(
            pt=prev_key_split[0],
            size=prev_key_split[1],
            angle=tf.reshape(orientation, (-1, 1)),
            octave=prev_key_split[3],
            octave_id=prev_key_split[4],
            response=prev_key_split[5]
        )

        return histogram, key_points

    def __compute_max_con(self, points, shapes):
        shapes = tf.cast(shapes, dtype=tf.int32)
        points = tf.cast(points, dtype=tf.int32)

        _, points_h, points_w, _ = tf.unstack(points, 4, axis=-1)
        _, limit_h, limit_w, _ = tf.unstack(shapes, 4, axis=-1)

        valid_h = math_ops.minimum(points_h, limit_h - 1 - points_h)
        valid_w = math_ops.minimum(points_w, limit_w - 1 - points_w)

        max_con = math_ops.minimum(valid_h, valid_w)
        max_con = math_ops.maximum(max_con, 0)
        return max_con

    def __compute_parallel_jobs(self, con_shape, octave_ids):
        uni_con = tf.unique(con_shape)
        uni_oct = tf.unique(octave_ids)
        parallel_jobs = []
        parallel_tup = namedtuple('parallel_jobs', 'idx, con, octave_id')

        for ci, c in enumerate(uni_con.y):
            mask = math_ops.equal(uni_con.idx, ci)
            for oi, o in enumerate(uni_oct.y):
                mask_ = math_ops.logical_and(mask, math_ops.equal(uni_oct.idx, oi))
                mask_ = tf.where(mask_)
                if mask_.shape[0] > 0:
                    parallel_jobs.append(parallel_tup(tf.reshape(mask_, (-1,)), int(c), int(o)))
        return parallel_jobs

    def __trilinear_interpolation(self, yxz: tf.Tensor, values: tf.Tensor) -> tuple:
        yxz = tf.cast(yxz, dtype=values.dtype)

        y, x, z = tf.unstack(yxz, num=3, axis=-1)

        # interpolation in x direction
        _C0 = values * (1 - x)
        _C1 = values * x

        # interpolation in y direction
        _C00 = _C0 * (1 - y)
        _C01 = _C0 * y

        _C10 = _C1 * (1 - y)
        _C11 = _C1 * y

        # interpolation in z direction
        _C000 = _C00 * (1 - z)
        _C001 = _C00 * z
        _C010 = _C01 * (1 - z)
        _C011 = _C01 * z
        _C100 = _C10 * (1 - z)
        _C101 = _C10 * z
        _C110 = _C11 * (1 - z)
        _C111 = _C11 * z

        points = namedtuple('points', 'C000, C001, C010, C011, C100, C101, C110, C111')
        out = points(_C000, _C001, _C010, _C011, _C100, _C101, _C110, _C111)
        return out

    def __unpack_tri_to_histogram(self, yxz: tf.Tensor, points: tuple, window_width: int, n_bins: int = 8) -> tf.Tensor:
        y, x, z = tf.unstack(yxz, num=3, axis=-1)
        histogram = tf.zeros((yxz.shape[0], window_width, window_width, n_bins), dtype=points[0].dtype)

        b = tf.ones([*yxz.shape[:-1]], dtype=tf.int32) * tf.reshape(tf.range(yxz.shape[0], dtype=tf.int32), (-1, 1))
        b = tf.reshape(b, (-1, 1))

        _add_one = lambda c: c + 1
        _add_two = lambda c: c + 2
        _modulus = lambda c: (c + 1) % n_bins
        _none = lambda c: c

        z_pack = [_none, _modulus]
        x_pack = [_add_one, _add_two]
        y_pack = [_add_one, _add_two]

        for i, _C in enumerate(points):
            zi = z_pack[0](z)
            z_pack[0], z_pack[1] = z_pack[1], z_pack[0]

            if (i % 2) == 0: x_pack[0], x_pack[1] = x_pack[1], x_pack[0]
            xi = x_pack[0](x)

            if (i % 4) == 0: y_pack[0], y_pack[1] = y_pack[1], y_pack[0]
            yi = y_pack[0](y)

            mask_ = math_ops.logical_and(
                tf.logical_and(xi >= 0, xi < window_width), tf.logical_and(yi >= 0, yi < window_width)
            )
            mask_ = tf.reshape(mask_, (-1,))

            cords = tf.stack((yi, xi, zi), axis=-1)
            cords = tf.cast(cords, dtype=tf.int32)
            cords = tf.reshape(cords, (-1, 3))
            cords = tf.concat((b, cords), axis=-1)

            cords = tf.boolean_mask(cords, mask_)
            _C = tf.boolean_mask(tf.reshape(_C, (-1,)), mask_)

            histogram = tf.tensor_scatter_nd_add(histogram, cords, _C)
        return histogram

    def __remove_duplicates(self, key_points: KeyPoints) -> KeyPoints:
        def map_1(args):
            i = 0
            for itm in args:
                if itm != 0: break
                i += 1
            return args[i] if i != len(args) else args[i - 1]

        def map_2(args):
            arg1, arg2 = tf.split(args, [1, 1], -1)
            if tf.reduce_max(arg2 - arg1) == 0:
                return False
            return True

        diff_kernel = tf.constant([-1, 1], shape=(2, 1, 1, 1), dtype=tf.float32)
        kp_stack = [key_points.pt, key_points.size, key_points.angle, key_points.response, key_points.octave,
                    key_points.octave_id]
        kp_stack = tf.concat(kp_stack, axis=-1)

        partitions = tf.split(key_points.pt, [1, 3], axis=-1)[0]
        partitions = tf.cast(tf.squeeze(partitions), tf.int32)

        split_batch = tf.dynamic_partition(kp_stack, partitions, tf.reduce_max(partitions) + 1)

        clean_kp = None

        for curr_batch in split_batch:
            curr_diff = tf.nn.convolution(tf.expand_dims(tf.expand_dims(curr_batch, -1), 0), diff_kernel,
                                          padding='VALID')
            curr_diff = tf.squeeze(curr_diff)

            curr_diff = tf.map_fn(map_1, curr_diff)

            values, indices = math_ops.top_k(curr_diff, k=len(curr_diff))

            curr_compare = tf.split(curr_batch, [6, 3], axis=-1)[0]
            first_ = tf.expand_dims(tf.gather(curr_compare, indices[:-1]), -1)
            next_ = tf.expand_dims(tf.gather(curr_compare, indices[1:]), -1)

            curr_map = tf.map_fn(map_2, tf.concat((first_, next_), axis=-1))

            index_in = tf.boolean_mask(indices[1:], curr_map)
            index_in = tf.concat((tf.reshape(indices[0], (1,)), index_in), axis=0)
            curr_clean = tf.gather(curr_batch, index_in)
            if clean_kp is None:
                clean_kp = curr_clean
            else:
                clean_kp = tf.concat((clean_kp, curr_clean), axis=0)
        pt, size, angle, response, octave, octave_id = tf.split(clean_kp, [4, 1, 1, 1, 1, 1], axis=-1)
        out = KeyPoints(pt=pt, size=size, angle=angle, response=response, octave=octave, octave_id=octave_id)
        return out

    def build_graph(self, inputs: tf.Tensor) -> tuple[KeyPoints, tf.Tensor]:
        inputs = self.__validate_input(inputs)
        self.__init_graph()
        args = self.graph_args
        _b, _h, _w, _ = self.__inputs_shape

        kernel = self.base_kernel
        _k = kernel.get_shape()[0] // 2
        paddings = tf.constant([[0, 0], [_k, _k], [_k, _k], [0, 0]], dtype=tf.int32)

        with tf.name_scope('Xbase'):
            Xbase = image_ops.resize(inputs, size=[_h * 2, _w * 2], method='bilinear', name='Xb_up')
            Xbase = tf.pad(Xbase, paddings, mode='SYMMETRIC')
            Xbase = tf.nn.conv2d(Xbase, kernel, strides=[1, 1, 1, 1], padding='VALID', name='Xb_blur')

        with tf.name_scope('octave_pyramid'):
            maximum_iterations = args.n_octaves * len(self.gaussian_kernels)
            cap = {0: [Xbase, tf.constant([], dtype=tf.float32), tf.constant([], dtype=tf.float32)]}
            for i in range(1, args.n_octaves): cap[i] = [tf.constant([], dtype=tf.float32)] * 3
            threshold = tf.floor(0.5 * args.contrast_threshold / 3.0 * 255.0)
            key_points_warp = tf.constant([[]], shape=(0, 9), dtype=tf.float32)

            def body(iter_, gaussian_kernels, grad_kernels, capture, shapes):
                kernel_pointer = iter_ % len(gaussian_kernels)
                cap_pointer = iter_ // len(gaussian_kernels)

                gauss, dog, grad = capture.get(int(cap_pointer))
                prev_state = tf.split(gauss, [kernel_pointer, 1], axis=-1)[-1]

                kernel_ = gaussian_kernels[kernel_pointer]
                k_pad = int(kernel_.get_shape()[0]) // 2
                prev_state_pad = tf.pad(prev_state,
                                        tf.constant([[0, 0], [k_pad, k_pad], [k_pad, k_pad], [0, 0]], dtype=tf.int32),
                                        mode='SYMMETRIC')
                curr_state = tf.nn.convolution(prev_state_pad, kernel_, padding='VALID')

                curr_grad = tf.nn.convolution(curr_state, grad_kernels, padding='VALID')

                if kernel_pointer == 0: dog = math_ops.subtract(curr_state, prev_state)
                if kernel_pointer > 0: dog = tf.concat((dog, math_ops.subtract(curr_state, prev_state)), axis=-1)
                if iter_ == 0: grad = tf.expand_dims(tf.nn.convolution(prev_state, grad_kernels, padding='VALID'),
                                                     axis=3)

                gauss = tf.concat((gauss, curr_state), axis=-1)
                grad = tf.concat((grad, tf.expand_dims(curr_grad, axis=3)), axis=3)

                capture[int(cap_pointer)] = [gauss, dog, grad]
                iter_ = iter_ + 1

                if (iter_ % len(gaussian_kernels)) == 0 and (iter_ // len(gaussian_kernels)) < len(capture):
                    iter_h, iter_w = shapes
                    iter_d = len(gaussian_kernels) + 1

                    _, next_base, _ = tf.split(gauss, [iter_d - 3, 1, iter_d - (iter_d - 3) - 1], axis=-1)
                    next_base = image_ops.resize(next_base, size=[iter_h, iter_w], method='nearest')

                    capture[int(cap_pointer) + 1][0] = next_base
                    capture[int(cap_pointer) + 1][-1] = tf.expand_dims(
                        tf.nn.convolution(next_base, grad_kernels, padding='VALID'), axis=3
                    )

                    shapes = (iter_h // 2, iter_w // 2)
                return [iter_, gaussian_kernels, grad_kernels, capture, shapes]

            def cond(iter_, gaussian_kernels, grad_kernels, capture, shapes):
                cap_pointer = iter_ // len(gaussian_kernels)
                return cap_pointer < len(capture)

            _, _, _, cap, _ = tf.while_loop(
                cond=cond, body=body, loop_vars=[0, self.gaussian_kernels, self.gradient_kernel, cap, (_h, _w)],
                maximum_iterations=maximum_iterations, parallel_iterations=1
            )

            def body(cords, x, oc_id, static_keys):
                cords, keys = self.__localize_extrema(cords, x, oc_id)
                if cords is None:
                    cords = tf.constant([[]], shape=(0, 4), dtype=tf.int64)
                if keys is None:
                    return [cords, x, oc_id, static_keys]
                static_keys = tf.concat((static_keys, keys), axis=0)
                return [cords, x, oc_id, static_keys]

            def cond(cords, x, oc_id, static_keys):
                return cords.get_shape()[0] != 0

            for oc_key, oc_tup in cap.items():
                oc_gaussian, oc_dog, oc_grad = oc_tup
                dx, dy = tf.unstack(oc_grad * 0.5, 2, axis=-1)
                magnitude = math_ops.sqrt(dx * dx + dy * dy)
                orientation = math_ops.atan2(dx, dy) * (180.0 / PI)
                self.octave_pyramid.append(Octave(oc_gaussian, dx, dy, magnitude, orientation))
                oc_dog = float32(uint8(oc_dog))
                continue_search = compute_extrema3D(
                    float32(uint8(oc_dog)), threshold=threshold, con=args.con, border_width=args.border_width
                )
                if continue_search.shape[0] != 0:
                    params = tf.while_loop(
                        cond=cond,
                        body=body,
                        loop_vars=[continue_search, oc_dog, oc_key, key_points_warp],
                        maximum_iterations=args.n_iterations
                    )
                    key_points_warp = params[-1]

        key_points = KeyPoints(*tf.split(key_points_warp, [4, 1, 1, 1, 1, 1], axis=-1))
        histogram, key_points = self.__compute_histogram(key_points, self.octave_pyramid)
        self.key_points = self.__remove_duplicates(key_points)
        self.descriptors_vectors = self.write_descriptors(self.key_points, self.octave_pyramid)
        return self.key_points, self.descriptors_vectors

    def write_descriptors(self, key_points: KeyPoints, octaves: list[Octave]) -> tf.Tensor:
        def recompute(oc):
            dx = oc.dx * 2
            dy = oc.dy * 2
            magnitude = math_ops.sqrt(dx * dx + dy * dy)
            orientation = (math_ops.atan2(dx, dy) * (180.0 / PI)) % 360
            return Octave(oc.gaussian, dx, dy, magnitude, orientation)

        args = self.graph_args
        octaves = [recompute(oc) for oc in octaves]

        unpack_kp, unpack_oct = key_points.to_image_size(unpack_octave=True)

        bins_per_degree = args.descriptors_N_bins / 360.
        weight_multiplier = -0.5 / ((0.5 * args.window_width) ** 2)

        unpack_oct = unpack_oct._replace(octave=tf.cast(unpack_oct.octave + 1, dtype=tf.int32))
        unpack_oct = unpack_oct._replace(
            octave=tf.where(unpack_oct.octave < 0, len(octaves) + unpack_oct.octave, unpack_oct.octave)
        )
        unpack_oct = unpack_oct._replace(layer=tf.cast(unpack_oct.layer, dtype=tf.int32))

        scale_pad = tf.pad(tf.repeat(unpack_oct.scale, 2, axis=1), paddings=tf.constant([[0, 0], [1, 0]]),
                           constant_values=1.0)
        scale_pad = tf.pad(scale_pad, paddings=tf.constant([[0, 0], [0, 1]]), constant_values=0.0)

        point = tf.cast(tf.round(unpack_kp.pt * scale_pad), dtype=tf.int64)
        point, _ = tf.split(point, [3, 1], axis=-1)
        point = tf.concat((point, tf.cast(unpack_oct.layer, dtype=tf.int64)), axis=-1) - tf.constant([[0, 1, 1, 0]],
                                                                                                     dtype=tf.int64)

        angle = 360. - unpack_kp.angle
        cos_sin_angle = namedtuple('cos_sin_angle', 'cos, sin')
        cos_sin_angle = cos_sin_angle(math_ops.cos((PI / 180) * angle), math_ops.sin((PI / 180) * angle))

        hist_width = args.scale_multiplier * 0.5 * unpack_oct.scale * unpack_kp.size
        hist_con = tf.round(hist_width * math_ops.sqrt(2.0) * (args.window_width + 1.0) * 0.5)
        hist_con = tf.cast(hist_con, dtype=tf.int32)

        hist_con = tf.reshape((hist_con * 2) + 1, (-1,))

        oct_shapes = [tf.reshape(tf.shape(a.dx), (1, 4)) for a in octaves]
        oct_shapes = tf.gather(tf.concat(oct_shapes, axis=0), tf.squeeze(unpack_oct.octave))

        maximum_hist_con = self.__compute_max_con(point, oct_shapes)

        con = math_ops.minimum(tf.squeeze(hist_con), maximum_hist_con)
        parallel = self.__compute_parallel_jobs(con, tf.squeeze(unpack_oct.octave))

        descriptors = tf.zeros((angle.shape[0], (args.window_width ** 2) * args.descriptors_N_bins))

        for i_job in parallel:
            if i_job.con == 0:
                continue
            i_octave = octaves[i_job.octave_id]
            i_hist_width = tf.gather(hist_width, i_job.idx)

            i_block = make_neighborhood2D(tf.constant([[0, 0, 0, 0]] * len(i_job.idx), dtype=tf.int64), con=i_job.con)
            i_point = tf.gather(point, i_job.idx)

            i_cords = i_block + tf.expand_dims(i_point, axis=1)

            i_magnitude = tf.gather_nd(i_octave.magnitude, tf.reshape(i_cords, (-1, 4)))
            i_orientation = tf.gather_nd(i_octave.orientation, tf.reshape(i_cords, (-1, 4)))
            i_angle = tf.gather(angle, i_job.idx)

            i_cos = tf.gather(cos_sin_angle.cos, i_job.idx)
            i_sin = tf.gather(cos_sin_angle.sin, i_job.idx)

            i_block = tf.cast(i_block, dtype=tf.float32)
            _, i_y, i_x, _ = tf.unstack(i_block, 4, axis=-1)

            i_orientation = tf.reshape(i_orientation, shape=i_y.shape)
            i_magnitude = tf.reshape(i_magnitude, shape=i_y.shape)

            i_z = (i_orientation - i_angle) * bins_per_degree

            i_yr = i_x * i_sin + i_y * i_cos
            i_xr = i_x * i_cos - i_y * i_sin

            i_y = (i_yr / i_hist_width) + 0.5 * 4 - 0.5
            i_x = (i_xr / i_hist_width) + 0.5 * 4 - 0.5

            i_weight = math_ops.exp(weight_multiplier * ((i_yr / i_hist_width) ** 2 + (i_x / i_hist_width) ** 2))
            i_values = i_magnitude * i_weight

            i_yxz = [i_y, i_x, i_z]

            i_yxz_floor = [math_ops.floor(c) for c in i_yxz]
            i_yxz_frac = [c1 - c2 for c1, c2 in zip(i_yxz, i_yxz_floor)]

            i_yxz_floor[-1] = tf.where(
                math_ops.logical_and(i_yxz_floor[-1] < 0,
                                     i_yxz_floor[-1] + args.descriptors_N_bins < args.descriptors_N_bins),
                i_yxz_floor[-1] + args.descriptors_N_bins,
                i_yxz_floor[-1]
            )

            i_yxz_frac = tf.stack(i_yxz_frac, axis=-1)
            i_yxz_floor = tf.stack(i_yxz_floor, axis=-1)

            i_intr = self.__trilinear_interpolation(i_yxz_frac, i_values)

            i_histogram = self.__unpack_tri_to_histogram(yxz=i_yxz_floor, points=i_intr,
                                                         window_width=args.window_width + 2,
                                                         n_bins=args.descriptors_N_bins)

            i_descriptor_vector = tf.slice(
                i_histogram, [0, 1, 1, 0],
                [i_histogram.shape[0], args.window_width, args.window_width, args.descriptors_N_bins]
            )
            i_descriptor_vector = tf.reshape(i_descriptor_vector, (i_histogram.shape[0], -1))

            threshold = tf.norm(i_descriptor_vector) * args.descriptor_max_value

            i_descriptor_vector = tf.where(i_descriptor_vector > threshold, threshold, i_descriptor_vector)
            i_descriptor_vector = i_descriptor_vector / tf.maximum(tf.norm(i_descriptor_vector), self.epsilon)
            i_descriptor_vector = tf.round(i_descriptor_vector * 512)
            i_descriptor_vector = tf.maximum(i_descriptor_vector, 0)
            i_descriptor_vector = tf.minimum(i_descriptor_vector, 255)

            axs = tf.reshape(tf.range(i_descriptor_vector.shape[1], dtype=tf.int64), shape=(-1, 1))
            axs = tf.repeat(axs, i_descriptor_vector.shape[0], axis=0)
            i_id = tf.ones_like(i_descriptor_vector, dtype=tf.int64) * tf.reshape(i_job.idx, (-1, 1))
            i_id = tf.reshape(i_id, (-1, 1))
            axs = tf.concat((i_id, axs), axis=-1)

            descriptors = tf.tensor_scatter_nd_add(descriptors, axs, tf.reshape(i_descriptor_vector, (-1,)))
        return descriptors

    def compute_default_N_octaves(self, height: int, weight: int, min_shape: int = 0) -> tf.Tensor:
        s_ = tf.cast(min([height, weight]), dtype=tf.float32)
        diff = math_ops.log(s_)
        if min_shape > 1:
            diff = diff - math_ops.log(tf.cast(min_shape, dtype=tf.float32))

        n_octaves = tf.round(diff / math_ops.log(2.0)) + 1
        return tf.cast(n_octaves, tf.int32)

    def split_by_batch(self, key_points: KeyPoints, descriptors: Union[tf.Tensor, None] = None) -> Union[
                    KeyPoints, tuple[KeyPoints, tf.Tensor]]:
        kp_stack = key_points.stack()
        partitions = tf.split(key_points.pt, [1, 3], axis=-1)[0]
        partitions = tf.cast(tf.squeeze(partitions), tf.int32)

        split_batch = tf.dynamic_partition(kp_stack, partitions, tf.reduce_max(partitions) + 1)
        out = []
        for batch in split_batch:
            curr_kp = KeyPoints(*tf.split(batch, [4, 1, 1, 1, 1, 1], axis=-1))
            out.append(curr_kp)

        if descriptors is None: return out

        descriptors_split = tf.dynamic_partition(descriptors, partitions, tf.reduce_max(partitions) + 1)
        return out, descriptors_split


def show_key_points(key_points, image):
    from viz import show_images

    cords = tf.cast(key_points.pt, dtype=tf.int64) * tf.constant([1, 1, 1, 0], dtype=tf.int64)

    marks = make_neighborhood2D(cords, con=3, origin_shape=image.shape)
    marks = tf.reshape(marks, shape=(-1, 4))
    marked = tf.scatter_nd(marks, tf.ones(shape=(marks.get_shape()[0],), dtype=tf.float32) * 255.0, shape=image.shape)
    marked_del = math_ops.abs(marked - 255.0) / 255.0
    img_mark = image * marked_del

    marked = tf.concat((marked, tf.zeros_like(image), tf.zeros_like(image)), axis=-1)

    img_mark = img_mark + marked
    show_images(img_mark, 1, 1)


if __name__ == '__main__':
    img1 = tf.keras.utils.load_img('luka2.jpg', color_mode='grayscale')
    img1 = tf.convert_to_tensor(tf.keras.utils.img_to_array(img1), dtype=tf.float32)
    img1 = img1[tf.newaxis, ...]

    img2 = tf.keras.utils.load_img('luka1.jpg', color_mode='grayscale')
    img2 = tf.convert_to_tensor(tf.keras.utils.img_to_array(img2), dtype=tf.float32)
    img2 = img2[tf.newaxis, ...]

    alg = SIFT(sigma=1.6, n_octaves=4, n_intervals=3)
    kp1, disc1 = alg.build_graph(img1)
    kp2, disc2 = alg.build_graph(img2)


    # kp_up = kp.to_image_size()
    # show_key_points(kp_up, img1)
    # sp_kp, sp_disc = alg.split_by_batch(kp, disc)



